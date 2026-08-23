"""Quantify how much the pre-CV HVG selection inflates the Task 1 fold accuracy.

The cached ``data/dataset_hv.h5ad`` feature set (1,000 highly variable genes plus
29 paper markers) was selected on **all** 216,431 cells, i.e. before the
cross-validation split.  The fold accuracies reported by
``notebooks/task1_feature_importance_cv.ipynb`` therefore use a feature set that
saw the held-out cells.

This script measures the size of that effect for **fold 0** only, which is enough
to bound it:

1. accumulate CP10K mean/variance over the fold-0 **training cells only** with the
   streaming CSC pass (the full matrix is never resident);
2. reselect the top 1,000 seurat-flavour HVGs from those moments, add the same 29
   paper markers;
3. pull exactly those gene columns out of the raw file, CP10K + log1p them with
   the same convention;
4. refit XGBoost (6 rounds, the students' setting) and the MLP on fold-0
   training cells, score on fold-0 held-out cells;
5. print the difference against the same two models fitted on the cached,
   all-cell-selected feature set and scored on the same held-out cells.

Result is written to ``data/task1_hvg_leakage_fold0.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import balanced_accuracy_score

import pseudobulk as pb
import task1_cv as T

PAPER_MARKERS = [
    "KRT17", "TEX29", "CXCL9", "CD7", "CD74", "IDO1", "HLA-DRB1", "CXCL11",
    "HLA-DPA1", "PTGES", "HAPLN3", "TXNIP", "BATF3", "CD274", "HLA-DRA",
    "DDIT4", "CYP2S1", "SCN9A", "DOCK10", "HLA-B", "PRSS23", "RNF213",
    "GBP1", "GDF15", "SECTM1", "AC016831.1", "VSNL1", "GBP5", "HLA-DPB1",
    "PDCD1LG2",
]


def main() -> None:
    import h5py
    import scanpy as sc

    raw = REPO / "data" / "frangieh" / "rna.h5ad"
    cached = REPO / "data" / "dataset_hv.h5ad"

    # ---- labels and folds, identical to the notebook ---------------------- #
    adata = sc.read_h5ad(cached)
    X_cached = adata.X if sp.isspmatrix_csr(adata.X) else adata.X.tocsr()
    y = adata.obs["condition"].cat.codes.to_numpy().astype(np.int64)
    cached_genes = pd.Index(adata.var_names)
    folds = T.fold_splits(y)
    train_idx, test_idx = folds[0]

    # ---- fold-0 training cells, expressed as a mask over the RAW file ----- #
    meta = pb.read_cell_metadata(raw)
    qc = pb.qc_mask(meta)
    qc_pos = np.flatnonzero(qc)
    assert qc.sum() == adata.n_obs, "QC mask does not match the cached cell count"
    fold_mask = np.zeros(len(meta), dtype=bool)
    fold_mask[qc_pos[train_idx]] = True

    # ---- streaming moments over those cells only -------------------------- #
    t0 = time.time()
    groups = pb.build_group_index(meta, fold_mask)
    stats = pb.stream_gene_stats(raw, fold_mask, groups, progress=False)
    n = float(stats.n_cells_qc)
    means = stats.global_sum_cp10k / n
    variances = np.maximum(
        (stats.global_sumsq_cp10k - stats.global_sum_cp10k ** 2 / n) / (n - 1), 0.0
    )
    stream_s = time.time() - t0
    assert stats.n_cells_qc == len(train_idx), (stats.n_cells_qc, len(train_idx))

    with h5py.File(raw, "r") as f:
        ncells_gene = f["var"]["ncells"][:]
    gene_filter = ncells_gene >= 3  # same filter Phase 0 applied before binning

    hv = T.seurat_hvg_from_moments(
        means[gene_filter], variances[gene_filter],
        np.asarray(stats.gene_names)[gene_filter], n_top_genes=1000,
    )
    fold_hvg = set(hv.index[hv["highly_variable"]])
    markers_present = [g for g in PAPER_MARKERS if g in set(stats.gene_names)]
    fold_genes = sorted(fold_hvg | set(markers_present))

    all_names = pd.Index(stats.gene_names)
    gene_pos = all_names.get_indexer(fold_genes)
    assert (gene_pos >= 0).all()

    # ---- pull those columns, CP10K + log1p, restrict to QC cells ---------- #
    t0 = time.time()
    counts, names = pb.slice_gene_columns(raw, gene_pos)
    counts = counts[qc]
    ncounts = meta["ncounts"].to_numpy()[qc]
    X_fold = counts.astype(np.float32)
    inv = (1e4 / ncounts).astype(np.float32)
    X_fold = sp.diags(inv).dot(X_fold).tocsr()
    X_fold.data = np.log1p(X_fold.data)
    slice_s = time.time() - t0

    # ---- fit both models on both feature sets, same fold ------------------ #
    dev = T.get_device()
    out = {
        "fold": 0,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "stream_seconds": round(stream_s, 1),
        "slice_seconds": round(slice_s, 1),
        "cached_feature_set": {"n_genes": int(len(cached_genes))},
        "fold_feature_set": {"n_genes": int(len(fold_genes))},
        "hvg_overlap_with_cached": int(len(set(fold_genes) & set(cached_genes))),
        "hvg_jaccard": round(
            len(set(fold_genes) & set(cached_genes)) / len(set(fold_genes) | set(cached_genes)), 4
        ),
    }

    for tag, Xm, gl in (("cached", X_cached, cached_genes), ("fold", X_fold, pd.Index(names))):
        bst = T.fit_xgb_fold(Xm, y, train_idx, n_estimators=6, feature_names=gl)
        acc_x = balanced_accuracy_score(y[test_idx], bst.predict(Xm[test_idx]))
        mlp, _ = T.fit_mlp(Xm, y, train_idx, dev, n_epochs=8)
        acc_m = balanced_accuracy_score(y[test_idx], T.predict_mlp(mlp, Xm, test_idx, dev))
        out[f"{tag}_xgb6_balanced_accuracy"] = float(acc_x)
        out[f"{tag}_mlp_balanced_accuracy"] = float(acc_m)
        print(f"{tag:7s} xgb6 {acc_x:.4f}  mlp {acc_m:.4f}", flush=True)

    out["xgb6_optimism"] = out["cached_xgb6_balanced_accuracy"] - out["fold_xgb6_balanced_accuracy"]
    out["mlp_optimism"] = out["cached_mlp_balanced_accuracy"] - out["fold_mlp_balanced_accuracy"]

    dest = REPO / "data" / "task1_hvg_leakage_fold0.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
