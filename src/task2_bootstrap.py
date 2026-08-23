"""Bootstrap resampling of cells within perturbation groups for Task 2 clustering stability.

Recomputes the perturbation x gene log2FC signature matrix from cell-level counts with cells
resampled with replacement inside each (condition, perturbation) group, then re-runs every
clustering method. Used to propagate cell-sampling noise into the cluster labels.

Requires the raw gene-major store data/frangieh/rna.h5ad and takes ~10 min for 25 replicates
per condition on 8 cores; results are cached in data/task2_bootstrap_labels.pkl.
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np, pandas as pd, scipy.sparse as sp
import pseudobulk as pb

CONDS = ["Co_culture", "Control", "IFNg"]
COND_MAP = {"Co_culture": "Co-culture", "Control": "Control", "IFNg": "IFN\u03b3"}
H5AD = "data/frangieh/rna.h5ad"


def load_cells(union_genes, all_genes):
    """Load a log1p CP10K matrix restricted to `union_genes`, plus group assignments per cell.

    Groups are guide-derived: a cell belongs to a perturbation group only if it carries exactly
    one targeting guide token; cells with zero targeting guides form the clean control group.
    """
    meta = pb.read_cell_metadata(H5AD)
    # slice_gene_columns returns columns in ascending index order, not in the order requested
    gidx = np.sort(np.array([all_genes.index(g) for g in union_genes]))
    X, gnames = pb.slice_gene_columns(H5AD, gidx)
    gnames = list(gnames)
    assert set(gnames) == set(union_genes), 'requested genes not all returned'

    qc = pb.qc_mask(meta)
    tg = meta.guide_id.map(pb.guide_targets)
    ntg = tg.map(len)
    single = (ntg == 1) & (meta.n_guide_tokens == 1)
    clean_ctl = ntg == 0
    target = tg.map(lambda s: next(iter(s)) if len(s) == 1 else None)

    # CP10K must divide by each cell's GENOME-WIDE total, not the row sum of the gene slice.
    # meta['ncounts'] carries the precomputed per-cell total from the full matrix.
    tot = meta["ncounts"].to_numpy(dtype=np.float64)
    inv = np.where(tot > 0, 1e4 / np.maximum(tot, 1), 0.0)
    Xl = X.astype(np.float64).multiply(inv[:, None]).tocsr()
    Xl.data = np.log1p(Xl.data)
    Xl = Xl.astype(np.float32)

    out = {}
    for c in CONDS:
        mc = (meta.condition.values == COND_MAP[c]) & qc
        mp = mc & single.values & meta.perturbation.notna().values
        mch = mc & clean_ctl.values
        rows = np.where(mp | mch)[0]
        grp = np.where(mch[rows], "__CONTROL__", target.values[rows].astype(object))
        order = np.argsort(grp.astype(str), kind="stable")
        rows, grp = rows[order], grp[order].astype(str)
        starts = np.r_[0, np.flatnonzero(grp[1:] != grp[:-1]) + 1, len(grp)]
        out[c] = dict(X=Xl[rows], genes=gnames, names=[grp[s] for s in starts[:-1]],
                      slices=[(starts[i], starts[i + 1]) for i in range(len(starts) - 1)])
    return out


def group_mean(cd, rng=None):
    """Mean log1p CP10K per group; with `rng`, cells are resampled with replacement in-group."""
    X, rows = cd["X"], []
    for a, b in cd["slices"]:
        idx = np.arange(a, b) if rng is None else rng.integers(a, b, size=b - a)
        rows.append(np.asarray(X[idx].mean(axis=0)).ravel())
    return pd.DataFrame(rows, index=cd["names"], columns=cd["genes"])
