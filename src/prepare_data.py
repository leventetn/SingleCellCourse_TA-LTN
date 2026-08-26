"""Build the shared AnnData cache and pseudo-bulk tables from the raw files.

Run once from any directory:

    python src/prepare_data.py

The RNA matrix is 5.5 GB and stored gene-by-gene (CSC). The functions in
``pseudobulk.py`` stream it without loading the full matrix into memory.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Scanpy/Numba need writable cache directories in restricted environments.
CACHE = Path(tempfile.gettempdir()) / "sc-course-2026-cache"
for folder in (CACHE / "matplotlib", CACHE / "numba"):
    folder.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import pseudobulk as pb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNA = DATA / "frangieh" / "rna.h5ad"
PROTEIN = DATA / "frangieh" / "protein.h5ad"

CONDITION_FILE = {"Co-culture": "Co_culture", "Control": "Control", "IFNγ": "IFNg"}
PAPER_MARKERS = [
    "KRT17", "TEX29", "CXCL9", "CD7", "CD74", "IDO1", "HLA-DRB1",
    "CXCL11", "HLA-DPA1", "PTGES", "HAPLN3", "TXNIP", "BATF3",
    "CD274", "HLA-DRA", "DDIT4", "CYP2S1", "SCN9A", "DOCK10",
    "HLA-B", "PRSS23", "RNF213", "GBP1", "GDF15", "SECTM1",
    "AC016831.1", "VSNL1", "GBP5", "HLA-DPB1", "PDCD1LG2",
]


def write_rna_tables(stats: pb.StreamedStats, hvg: pd.DataFrame) -> None:
    """Write the RNA tables read by the Task 3 notebook."""
    means = stats.group_mean_log()
    means.to_parquet(DATA / "rna_group_mean_log1p.parquet")
    stats.groups.frame().to_parquet(DATA / "group_cell_counts.parquet", index=False)
    hvg.to_parquet(DATA / "hvg_table.parquet")

    results = pb.log2fc_per_condition(stats)
    for condition, result in results.items():
        name = CONDITION_FILE[condition]
        result["log2fc"].to_parquet(DATA / f"rna_log2fc_{name}.parquet")
        result["se"].to_parquet(DATA / f"rna_log2fc_se_{name}.parquet")
        result["n_cells"].rename("n_cells").to_frame().to_parquet(
            DATA / f"rna_ncells_{name}.parquet"
        )
        result["control_mean_log1p"].rename("control_mean_log1p").to_frame().to_parquet(
            DATA / f"rna_control_mean_{name}.parquet"
        )

    control = means.xs("control", level="perturbation")
    ((control.loc["IFNγ"] - control.loc["Control"]) / np.log(2)).rename(
        "log2fc_IFNg_vs_Control"
    ).to_frame().to_parquet(DATA / "condition_log2fc_IFNg_vs_Control.parquet")

    # The diagonal records the effect of knocking out a gene on that same gene.
    self_effect = {}
    for condition, result in results.items():
        matrix = result["log2fc"]
        genes = matrix.index.intersection(matrix.columns)
        self_effect[condition] = pd.Series(
            [matrix.loc[gene, gene] for gene in genes], index=genes
        )
    pd.DataFrame(self_effect).dropna().to_parquet(DATA / "self_knockdown.parquet")


def write_protein_tables(meta: pd.DataFrame, groups: pb.GroupIndex) -> None:
    """Compute the small 24-protein CLR and log2FC tables in memory."""
    protein = ad.read_h5ad(PROTEIN)
    if list(protein.obs_names.astype(str)) != list(meta["cell_name"].astype(str)):
        raise ValueError("RNA and protein cell orders do not match")

    values = protein.X.toarray() if hasattr(protein.X, "toarray") else np.asarray(protein.X)
    values = pb.clr(values)
    n_groups, n_proteins = groups.n_groups, values.shape[1]
    sums = np.zeros((n_groups, n_proteins))
    sums_sq = np.zeros_like(sums)
    for group in range(n_groups):
        rows = groups.group_of_cell == group
        sums[group] = values[rows].sum(axis=0)
        sums_sq[group] = np.square(values[rows]).sum(axis=0)

    n = groups.group_sizes[:, None].astype(float)
    means = sums / n
    variances = np.maximum((sums_sq - sums**2 / n) / np.maximum(n - 1, 1), 0)
    index = pd.MultiIndex.from_tuples(
        [groups.key(i) for i in range(n_groups)], names=["perturbation", "condition"]
    )
    mean_table = pd.DataFrame(means, index=index, columns=protein.var_names)
    var_table = pd.DataFrame(variances, index=index, columns=protein.var_names)
    mean_table.to_parquet(DATA / "adt_clr_group_means.parquet")

    group_counts = groups.frame().set_index(["perturbation", "condition"])["n_cells"]
    for condition, file_name in CONDITION_FILE.items():
        condition_mean = mean_table.xs(condition, level="condition")
        condition_var = var_table.xs(condition, level="condition")
        condition_n = group_counts.xs(condition, level="condition")
        control_mean = condition_mean.loc["control"]
        control_var = condition_var.loc["control"]
        control_n = condition_n.loc["control"]
        perturbations = condition_mean.index.difference(["control", "control_impure"])

        log2fc = (condition_mean.loc[perturbations] - control_mean) / np.log(2)
        se = np.sqrt(
            condition_var.loc[perturbations].div(condition_n.loc[perturbations], axis=0)
            + control_var / control_n
        ) / np.log(2)
        log2fc.astype("float32").to_parquet(DATA / f"adt_log2fc_{file_name}.parquet")
        se.astype("float32").to_parquet(DATA / f"adt_log2fc_se_{file_name}.parquet")

    panel = pd.DataFrame(
        {
            "adt": protein.var_names,
            "target": protein.var["Target"].astype(str).values,
            "isotype_control_used": protein.var["Isotype_control"].astype(str).values,
        }
    )
    panel["is_isotype_control"] = panel["isotype_control_used"].eq("nan")
    panel.to_parquet(DATA / "adt_panel.parquet", index=False)


def write_anndata_cache(
    meta: pd.DataFrame,
    cell_mask: np.ndarray,
    hvg: pd.DataFrame,
    gene_names: np.ndarray,
) -> None:
    """Write the 1,000-HVG-plus-marker AnnData used by Task 1 and Task 2."""
    selected_names = set(hvg.index[hvg["highly_variable"]]) | set(PAPER_MARKERS)
    selected = np.array([i for i, gene in enumerate(gene_names) if gene in selected_names])
    counts, selected_names = pb.slice_gene_columns(RNA, selected)
    counts = counts[cell_mask].astype("float32")

    obs = meta.loc[cell_mask].copy()
    obs["is_single_pert"] = (
        (obs["n_targeting_guides"] == 1) & (obs["n_guide_tokens"] == 1)
    )
    obs["is_clean_control"] = obs["n_targeting_guides"] == 0
    obs.index = obs["cell_name"].astype(str)
    obs.index.name = "cell_name"

    var = pd.DataFrame(index=pd.Index(selected_names, name="gene_symbol"))
    var["highly_variable"] = hvg["highly_variable"].reindex(var.index).fillna(False).astype(bool)
    var["paper_marker"] = var.index.isin(PAPER_MARKERS)
    var["dispersions_norm"] = hvg["dispersions_norm"].reindex(var.index)

    # Preserve the existing Task 1 cache convention: normalize the selected
    # feature matrix to 10,000 counts per cell, then apply natural log1p.
    # The pseudo-bulk tables above use genome-wide ncounts instead.
    totals = np.asarray(counts.sum(axis=1)).ravel()
    normalized = counts.multiply((1e4 / totals)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    adata = ad.AnnData(X=normalized, obs=obs, var=var)
    adata.layers["counts"] = counts
    adata.uns["log1p"] = {"base": None}

    # PCA is calculated on a scaled copy; the stored X remains log1p(CP10K).
    scaled = adata.copy()
    sc.pp.scale(scaled, max_value=10)
    adata.var["mean"] = scaled.var["mean"]
    adata.var["std"] = scaled.var["std"]
    sc.pp.pca(scaled, n_comps=50)
    adata.obsm["X_pca"] = scaled.obsm["X_pca"]
    adata.varm["PCs"] = scaled.varm["PCs"]
    adata.uns["pca"] = scaled.uns["pca"]
    sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.umap(adata, random_state=0)
    adata.uns["pseudobulk_provenance"] = {
        "normalisation": "CP10K within the selected feature matrix, then natural log1p",
        "qc": "ngenes>=200, percent_mito<15, ncounts<50000",
        "genes": "1000 Seurat HVGs plus paper markers",
    }
    adata.write_h5ad(DATA / "dataset_hv.h5ad", compression="gzip")


def main() -> None:
    for path in (RNA, PROTEIN):
        if not path.exists():
            raise FileNotFoundError(path)

    print("1/4 Reading cell metadata")
    meta = pb.read_cell_metadata(RNA)
    cell_mask = pb.qc_mask(meta)
    groups = pb.build_group_index(meta, cell_mask)

    print("2/4 Streaming RNA gene statistics")
    stats = pb.stream_gene_stats(RNA, cell_mask, groups)
    raw = ad.read_h5ad(RNA, backed="r")
    gene_names = np.asarray(raw.var_names)
    gene_mask = raw.var["ncells"].to_numpy() >= 3
    raw.file.close()
    hvg = pb.seurat_hvg_from_stats(stats, n_top_genes=1000, gene_mask=gene_mask)
    write_rna_tables(stats, hvg)

    print("3/4 Building the small AnnData cache")
    write_anndata_cache(meta, cell_mask, hvg, gene_names)

    print("4/4 Building protein tables")
    write_protein_tables(meta, groups)

    provenance = {
        "command": "python src/prepare_data.py",
        "rna": str(RNA.relative_to(ROOT)),
        "protein": str(PROTEIN.relative_to(ROOT)),
        "rna_tables": "CP10K using genome-wide obs['ncounts'], then natural log1p",
        "dataset_hv_X": "CP10K within the selected 1,011-gene matrix, then natural log1p",
        "qc": "ngenes>=200, percent_mito<15, ncounts<50000",
        "single_perturbation": "one targeting guide and one total guide token",
        "clean_control": "zero targeting guides",
    }
    (DATA / "CACHE_PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Preparation complete")


if __name__ == "__main__":
    main()
