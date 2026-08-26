"""Bootstrap resampling of cells within perturbation groups for Task 2 clustering stability.

Recomputes the perturbation x gene log2FC signature matrix with every QC-passing cell grouped
by its raw ``obs['perturbation']`` label within condition. Cells are resampled with replacement
inside those groups before every clustering method is rerun.

The canonical Task 2 notebook performs this operation directly on `dataset_hv.h5ad`; this
small helper keeps the same aggregation available to other scripts.
"""
import os
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import anndata as ad
import numpy as np, pandas as pd

CONDS = ["Co_culture", "Control", "IFNg"]
COND_MAP = {"Co_culture": "Co-culture", "Control": "Control", "IFNg": "IFN\u03b3"}
H5AD = "data/dataset_hv.h5ad"


def load_cells(union_genes, all_genes):
    """Load a log1p CP10K matrix restricted to `union_genes`, plus group assignments per cell.

    Every QC-passing cell is assigned by ``obs['perturbation']``. The raw ``control`` label is
    renamed ``__CONTROL__`` only to keep the subtraction step explicit.
    """
    adata = ad.read_h5ad(H5AD)
    gnames = sorted(set(union_genes) & set(adata.var_names))
    if set(gnames) != set(union_genes):
        raise ValueError("requested genes are not all present in dataset_hv.h5ad")
    Xl = adata[:, gnames].X.tocsr()
    meta = adata.obs

    out = {}
    for c in CONDS:
        mc = meta.condition.astype(str).values == COND_MAP[c]
        rows = np.where(mc & meta.perturbation.notna().values)[0]
        raw_group = meta.perturbation.values[rows].astype(object)
        grp = np.where(raw_group == "control", "__CONTROL__", raw_group)
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
