"""Memory-bounded pseudo-bulk aggregation for the Frangieh Perturb-CITE-seq data.

Why this module exists
----------------------
``data/frangieh/rna.h5ad`` is 218,331 cells x 23,712 genes with 740,736,244
stored non-zeros.  ``sc.read_h5ad`` materialises ~5.9 GB of matrix, and the
normalise + log1p that follows transiently needs a second copy; on a 16 GB
machine that reliably kills the kernel.

The matrix is stored ``csc_matrix`` -- **gene-major**.  That means one gene is
one contiguous slice of ``X/data`` / ``X/indices``, so a chunk of gene columns
can be pulled straight out of the HDF5 arrays with ``h5py`` and reduced to
per-group sums immediately.  The full matrix is never resident; peak RAM is set
by the gene-chunk size plus the accumulators, both bounded and both small.

Normalisation convention
-----------------------------
1. Per cell, ``cp10k = count / ncounts * 1e4`` where ``ncounts`` is
   ``obs['ncounts']``.  That column was verified to equal the exact row sum of
   ``X`` (max abs deviation 0.0 over sampled cells), so no extra pass is needed.
   This is identical to ``sc.pp.normalize_total(adata, target_sum=1e4)``.
2. ``y = log1p(cp10k)``, natural log -- identical to ``sc.pp.log1p``.
3. The per-group value is the **mean of log1p(CP10K)** over all cells in the
   group, i.e. ``mean-of-log``, *not* ``log-of-mean``.  This is the same
   quantity as ``normalize_total -> log1p -> groupby(...).mean()`` in scanpy.
   The distinction matters: mean-of-log down-weights high-expressing outlier
   cells and is the convention used by scanpy's ``rank_genes_groups`` inputs.
4. Zeros are implicit in sparse storage.  Every group mean therefore divides the
   accumulated sum by the **full group cell count**, never by the number of
   stored non-zeros.  ``validate_against_dense`` exists specifically to catch a
   regression here, because that is the easiest bug to introduce and the hardest
   to notice.

log2 fold change
----------------
Because step 3 already put values on a natural-log scale, the fold change is a
*difference of means* rescaled into log2 units::

    log2FC(gene, pert, condition)
        = ( mean_log1p[pert, condition] - mean_log1p[control, condition] ) / ln 2

This is a log-scale mean difference, which is what limma/edgeR-style pseudo-bulk
contrasts report, and it is what the downstream tasks predict.  It is *not*
``log2(mean_cp10k_pert / mean_cp10k_ctrl)``; those differ whenever the
expression distribution is skewed, and mixing them would silently corrupt the
Task 3 targets.  Controls are always taken from the **same condition** -- the
three conditions are separate experiments and are never pooled.

Cell-set semantics in this dataset (do not trust the column names)
----------------------------------------------------------------------------
``obs['nperts']`` is **not** a perturbation count.  It only ever takes the
values 0 and 1, and it is exactly the indicator ``perturbation != 'control'``
(0 for all 57,605 control-labelled cells, 1 for the other 160,726).  Filtering
on ``nperts == 1`` therefore does *not* give single-perturbation cells; it gives
every perturbed cell including cells carrying up to 19 guides.

``obs['MOI']`` is the real multiplicity: it equals the number of ``;``-separated
tokens in ``obs['guide_id']`` for 99.98% of cells.  Guide tokens are
``<GENE>_<n>`` for targeting guides and ``NO_SITE_<n>`` / ``ONE_NON-GENE_SITE_<n>``
for non-targeting controls; ``guide_id == 'nan'`` means no guide was called.

``obs['perturbation']`` names only **one** target per cell: for every cell with
MOI >= 2 the label is one arbitrary member of that cell's target set.  Grouping
on it without a multiplicity filter mixes single- and multi-knockout cells.

This module therefore derives the cell sets from ``guide_id``:

* ``single``  -- exactly one targeting guide and no other guide token
  (111,345 cells; identical to ``MOI == 1 & perturbation != 'control'``).
* ``control`` -- **zero** targeting guides (39,347 cells: 23,028 with no guide
  call plus 16,319 carrying only non-targeting guides).
* ``control_impure`` -- the 18,258 cells that ``obs['perturbation']`` labels
  ``'control'`` but which carry >= 1 targeting guide.  These are *excluded* from
  the control baseline and accumulated as their own group so a downstream step
  can quantify how much the naive label-based baseline would have been
  contaminated.  Using the raw label as the control would put real knockouts in
  the denominator of every fold change.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

LN2 = np.log(2.0)

#: Guide-token prefixes that denote a non-targeting (control) guide.
CONTROL_GUIDE_PREFIXES = ("NO_SITE", "ONE_NON-GENE_SITE")

#: Label used for the clean control group.
CONTROL_LABEL = "control"

#: Label for cells the raw obs column calls 'control' but which carry a real guide.
IMPURE_CONTROL_LABEL = "control_impure"

_GUIDE_SUFFIX = re.compile(r"_\d+$")


# --------------------------------------------------------------------------- #
# obs / var metadata
# --------------------------------------------------------------------------- #
def _read_categorical(obs: h5py.Group, name: str) -> np.ndarray:
    """Materialise an AnnData categorical column as a string array."""
    grp = obs[name]
    if isinstance(grp, h5py.Group):
        cats = np.asarray(
            [c.decode() if isinstance(c, bytes) else str(c) for c in grp["categories"][:]]
        )
        return cats[grp["codes"][:]]
    raw = grp[:]
    return np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in raw])


def guide_targets(guide_id: str) -> frozenset[str]:
    """Gene symbols targeted by a ``;``-separated ``guide_id`` string."""
    if guide_id == "nan" or not guide_id:
        return frozenset()
    return frozenset(
        _GUIDE_SUFFIX.sub("", tok)
        for tok in guide_id.split(";")
        if not tok.startswith(CONTROL_GUIDE_PREFIXES)
    )


def n_guide_tokens(guide_id: str) -> int:
    """Total guide tokens called in a cell (targeting + non-targeting)."""
    if guide_id == "nan" or not guide_id:
        return 0
    return guide_id.count(";") + 1


def read_cell_metadata(h5ad_path: str | Path) -> pd.DataFrame:
    """Read ``obs`` plus derived guide/QC columns without touching ``X``.

    Adds
    ----
    n_targeting_guides, n_guide_tokens, pert_label
    """
    with h5py.File(str(h5ad_path), "r") as f:
        obs = f["obs"]
        meta = pd.DataFrame(
            {
                "cell_name": _read_categorical(obs, "cell_name"),
                "condition": _read_categorical(obs, "perturbation_2"),
                "perturbation": _read_categorical(obs, "perturbation"),
                "guide_id": _read_categorical(obs, "guide_id"),
                "MOI": obs["MOI"][:],
                "nperts": obs["nperts"][:],
                "ncounts": obs["ncounts"][:].astype(np.float64),
                "ngenes": obs["ngenes"][:],
                "percent_mito": obs["percent_mito"][:],
            }
        )

    # Guide parsing is done on the *categories* and then indexed back out, so the
    # regex runs 58k times rather than 218k times.
    uniq = pd.Index(meta["guide_id"].unique())
    tgt_sets = {g: guide_targets(g) for g in uniq}
    n_tgt = {g: len(tgt_sets[g]) for g in uniq}
    n_tok = {g: n_guide_tokens(g) for g in uniq}

    meta["n_targeting_guides"] = meta["guide_id"].map(n_tgt).astype(np.int16)
    meta["n_guide_tokens"] = meta["guide_id"].map(n_tok).astype(np.int16)

    single = (meta["n_targeting_guides"] == 1) & (meta["n_guide_tokens"] == 1)
    clean_ctrl = meta["n_targeting_guides"] == 0

    label = np.full(len(meta), "", dtype=object)
    label[clean_ctrl.to_numpy()] = CONTROL_LABEL
    # For a single-target cell obs['perturbation'] was verified to equal the sole
    # guide target for 100% of cells, so reuse it rather than re-deriving.
    label[single.to_numpy()] = meta.loc[single, "perturbation"].to_numpy()
    impure = (meta["perturbation"] == CONTROL_LABEL) & (meta["n_targeting_guides"] >= 1)
    label[impure.to_numpy()] = IMPURE_CONTROL_LABEL
    meta["pert_label"] = label  # "" == multi-target cell, excluded from groups

    return meta


def qc_mask(
    meta: pd.DataFrame,
    min_genes: int = 200,
    total_cutoff: int = 50_000,
    mt_cutoff: float = 15.0,
) -> np.ndarray:
    """Cell-level QC mask matching ``src.preprocessing.preprocess`` semantics.

    ``>= min_genes`` genes, ``< mt_cutoff`` percent mito, ``< total_cutoff``
    total counts.  Reuses the AnnData's precomputed QC columns, which is why no
    matrix pass is required.
    """
    return (
        (meta["ngenes"].to_numpy() >= min_genes)
        & (meta["percent_mito"].to_numpy() < mt_cutoff)
        & (meta["ncounts"].to_numpy() < total_cutoff)
    )


def verify_ncounts(h5ad_path: str | Path, n_cells: int = 12, seed: int = 0) -> dict:
    """Check ``obs['ncounts']`` really is the row sum of ``X``.

    ``ncounts`` may have been computed before gene filtering, in which case it
    would be a wrong CP10K denominator.  This scans every gene column but keeps
    only the sampled rows, so it is cheap in RAM.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(str(h5ad_path), "r") as f:
        X = f["X"]
        n_obs, n_var = X.attrs["shape"]
        indptr = X["indptr"][:]
        ncounts = f["obs/ncounts"][:].astype(np.float64)

        cells = np.sort(rng.choice(int(n_obs), n_cells, replace=False))
        pos = -np.ones(int(n_obs), dtype=np.int64)
        pos[cells] = np.arange(n_cells)
        totals = np.zeros(n_cells, dtype=np.float64)

        for g0 in range(0, int(n_var), 4000):
            g1 = min(g0 + 4000, int(n_var))
            a, b = int(indptr[g0]), int(indptr[g1])
            idx = X["indices"][a:b]
            p = pos[idx]
            hit = p >= 0
            if hit.any():
                np.add.at(totals, p[hit], X["data"][a:b][hit].astype(np.float64))

    diff = np.abs(totals - ncounts[cells])
    return {
        "cells": cells.tolist(),
        "rowsums": totals.tolist(),
        "obs_ncounts": ncounts[cells].tolist(),
        "max_abs_diff": float(diff.max()),
        "matches": bool(diff.max() < 1e-6),
    }


# --------------------------------------------------------------------------- #
# group index
# --------------------------------------------------------------------------- #
@dataclass
class GroupIndex:
    """Maps each cell to a (perturbation-label x condition) group, or -1."""

    pert_levels: list[str]
    cond_levels: list[str]
    group_of_cell: np.ndarray  # int32, -1 = excluded
    group_sizes: np.ndarray  # int64, full cell count per group
    n_groups: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_groups = len(self.pert_levels) * len(self.cond_levels)

    def key(self, group: int) -> tuple[str, str]:
        return self.pert_levels[group // len(self.cond_levels)], self.cond_levels[
            group % len(self.cond_levels)
        ]

    def frame(self) -> pd.DataFrame:
        """Group -> (perturbation, condition, n_cells) table."""
        rows = [
            (p, c, int(self.group_sizes[pi * len(self.cond_levels) + ci]))
            for pi, p in enumerate(self.pert_levels)
            for ci, c in enumerate(self.cond_levels)
        ]
        return pd.DataFrame(rows, columns=["perturbation", "condition", "n_cells"])


def build_group_index(meta: pd.DataFrame, cell_mask: np.ndarray) -> GroupIndex:
    """Assign every QC-passing, unambiguously-labelled cell to a group."""
    pert_levels = sorted(set(meta["pert_label"]) - {""})
    cond_levels = sorted(meta["condition"].unique())
    p_of = {p: i for i, p in enumerate(pert_levels)}
    c_of = {c: i for i, c in enumerate(cond_levels)}

    pi = meta["pert_label"].map(lambda x: p_of.get(x, -1)).to_numpy()
    ci = meta["condition"].map(c_of).to_numpy()

    group = np.where(
        (pi >= 0) & cell_mask, pi * len(cond_levels) + ci, -1
    ).astype(np.int32)
    sizes = np.bincount(
        group[group >= 0], minlength=len(pert_levels) * len(cond_levels)
    ).astype(np.int64)
    return GroupIndex(pert_levels, cond_levels, group, sizes)


# --------------------------------------------------------------------------- #
# the streaming pass
# --------------------------------------------------------------------------- #
@dataclass
class StreamedStats:
    """Everything the downstream steps need, from one pass over ``X``.

    Group-level accumulators are over ``y = log1p(cp10k)``:
        ``sum_log`` (n_groups, n_genes), ``sumsq_log``, ``nnz`` (stored non-zeros)

    Global accumulators are over ``cp10k`` itself, restricted to QC-passing
    cells.  Seurat-flavour HVG selection needs the mean/variance of CP10K (it
    does ``expm1`` internally), so these are what make a faithful HVG
    reproduction possible without a second pass.
    """

    gene_names: np.ndarray
    groups: GroupIndex
    sum_log: np.ndarray
    sumsq_log: np.ndarray
    nnz: np.ndarray
    global_sum_cp10k: np.ndarray
    global_sumsq_cp10k: np.ndarray
    global_nnz: np.ndarray
    n_cells_qc: int

    def group_mean_log(self) -> pd.DataFrame:
        """Mean of log1p(CP10K) per group -- divides by FULL group size."""
        sizes = self.groups.group_sizes.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = self.sum_log / sizes[:, None]
        means[sizes == 0] = np.nan
        idx = pd.MultiIndex.from_tuples(
            [self.groups.key(g) for g in range(self.groups.n_groups)],
            names=["perturbation", "condition"],
        )
        return pd.DataFrame(means, index=idx, columns=self.gene_names)

    def group_var_log(self, ddof: int = 1) -> pd.DataFrame:
        """Within-group variance of log1p(CP10K), zeros included."""
        n = self.groups.group_sizes.astype(np.float64)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (self.sumsq_log - self.sum_log**2 / n) / np.maximum(n - ddof, 1e-12)
        var = np.maximum(var, 0.0)
        var[np.broadcast_to(n <= ddof, var.shape)] = np.nan
        idx = pd.MultiIndex.from_tuples(
            [self.groups.key(g) for g in range(self.groups.n_groups)],
            names=["perturbation", "condition"],
        )
        return pd.DataFrame(var, index=idx, columns=self.gene_names)


def stream_gene_stats(
    h5ad_path: str | Path,
    cell_mask: np.ndarray,
    groups: GroupIndex,
    gene_chunk: int = 256,
    genes: np.ndarray | None = None,
    progress: bool = True,
) -> StreamedStats:
    """One pass over the CSC gene columns, accumulating group and gene stats.

    Parameters
    ----------
    cell_mask
        QC mask over all cells; only these contribute to the global gene stats.
    groups
        Group assignment; ``group_of_cell == -1`` cells contribute to the global
        gene stats but to no group.
    gene_chunk
        Number of gene columns read per HDF5 hyperslab.  Peak transient RAM is
        roughly ``gene_chunk * mean_nnz_per_gene * 28`` bytes.
    genes
        Optional gene-index subset (used by the validation helper).

    Notes
    -----
    The inner reduction folds the gene offset into the bin index
    (``bin = gene_local * n_groups + group``) so one ``np.bincount`` handles a
    whole chunk instead of one call per gene.
    """
    cell_mask = np.asarray(cell_mask, dtype=bool)
    grp = groups.group_of_cell
    ng_groups = groups.n_groups

    with h5py.File(str(h5ad_path), "r") as f:
        X = f["X"]
        enc = X.attrs.get("encoding-type")
        if enc != "csc_matrix":
            raise ValueError(
                f"stream_gene_stats needs gene-major CSC storage, got {enc!r}. "
                "A CSR file would have to be streamed over cells instead."
            )
        n_obs, n_var = (int(v) for v in X.attrs["shape"])
        if cell_mask.shape[0] != n_obs:
            raise ValueError(f"cell_mask has {cell_mask.shape[0]} entries, X has {n_obs} rows")
        indptr = X["indptr"][:]
        gene_names = _read_categorical(f["var"], f["var"].attrs["_index"])
        ncounts = f["obs/ncounts"][:].astype(np.float64)
        if (ncounts <= 0).any():
            raise ValueError("obs['ncounts'] contains non-positive values")
        inv_lib = 1e4 / ncounts

        gene_sel = np.arange(n_var) if genes is None else np.asarray(genes, dtype=np.int64)
        n_out = gene_sel.size

        sum_log = np.zeros((ng_groups, n_out), dtype=np.float64)
        sumsq_log = np.zeros((ng_groups, n_out), dtype=np.float64)
        nnz = np.zeros((ng_groups, n_out), dtype=np.int64)
        g_sum = np.zeros(n_out, dtype=np.float64)
        g_sumsq = np.zeros(n_out, dtype=np.float64)
        g_nnz = np.zeros(n_out, dtype=np.int64)

        starts = range(0, n_out, gene_chunk)
        it = starts
        if progress:
            try:
                from tqdm.auto import tqdm

                it = tqdm(list(starts), desc="streaming gene columns", unit="chunk")
            except ImportError:
                pass

        for c0 in it:
            block = gene_sel[c0 : c0 + gene_chunk]
            # Contiguous gene runs let one hyperslab serve the whole block.
            contiguous = block.size > 1 and int(block[-1] - block[0]) == block.size - 1
            if contiguous:
                a, b = int(indptr[block[0]]), int(indptr[block[-1] + 1])
                idx_all = X["indices"][a:b]
                dat_all = X["data"][a:b].astype(np.float64)
                bounds = indptr[block[0] : block[-1] + 2] - indptr[block[0]]
                pieces = [
                    (idx_all[int(bounds[k]) : int(bounds[k + 1])],
                     dat_all[int(bounds[k]) : int(bounds[k + 1])])
                    for k in range(block.size)
                ]
            else:
                pieces = []
                for g in block:
                    a, b = int(indptr[g]), int(indptr[g + 1])
                    pieces.append((X["indices"][a:b], X["data"][a:b].astype(np.float64)))

            # Flatten the block into (bin, value) pairs and reduce once.
            local_bins, local_vals, local_gene = [], [], []
            for k, (rows, vals) in enumerate(pieces):
                cp10k = vals * inv_lib[rows]
                keep = cell_mask[rows]
                if keep.any():
                    kv = cp10k[keep]
                    g_sum[c0 + k] += kv.sum()
                    g_sumsq[c0 + k] += (kv * kv).sum()
                    g_nnz[c0 + k] += int(keep.sum())
                gr = grp[rows]
                ok = gr >= 0
                if ok.any():
                    local_bins.append(gr[ok].astype(np.int64) + k * ng_groups)
                    local_vals.append(np.log1p(cp10k[ok]))
                    local_gene.append(k)

            if local_bins:
                bins = np.concatenate(local_bins)
                y = np.concatenate(local_vals)
                width = block.size * ng_groups
                sum_log[:, c0 : c0 + block.size] += (
                    np.bincount(bins, weights=y, minlength=width)
                    .reshape(block.size, ng_groups)
                    .T
                )
                sumsq_log[:, c0 : c0 + block.size] += (
                    np.bincount(bins, weights=y * y, minlength=width)
                    .reshape(block.size, ng_groups)
                    .T
                )
                nnz[:, c0 : c0 + block.size] += (
                    np.bincount(bins, minlength=width).reshape(block.size, ng_groups).T
                )

    return StreamedStats(
        gene_names=gene_names[gene_sel],
        groups=groups,
        sum_log=sum_log,
        sumsq_log=sumsq_log,
        nnz=nnz,
        global_sum_cp10k=g_sum,
        global_sumsq_cp10k=g_sumsq,
        global_nnz=g_nnz,
        n_cells_qc=int(cell_mask.sum()),
    )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_against_dense(
    h5ad_path: str | Path,
    cell_mask: np.ndarray,
    groups: GroupIndex,
    stats: StreamedStats,
    n_genes: int = 30,
    seed: int = 7,
) -> dict:
    """Recompute group means the naive dense way for a random gene sample.

    This is the guard for the implicit-zero bug: the dense path divides by the
    full group size by construction (it builds a length-``n_cells`` vector with
    real zeros), so if the streamed path ever divided by the non-zero count
    instead, the discrepancy would be large and this would fail.
    """
    rng = np.random.default_rng(seed)
    name_to_col = {n: i for i, n in enumerate(stats.gene_names)}

    with h5py.File(str(h5ad_path), "r") as f:
        X = f["X"]
        n_obs, n_var = (int(v) for v in X.attrs["shape"])
        indptr = X["indptr"][:]
        gene_names = _read_categorical(f["var"], f["var"].attrs["_index"])
        inv_lib = 1e4 / f["obs/ncounts"][:].astype(np.float64)

        pick = rng.choice(n_var, n_genes, replace=False)
        grp = groups.group_of_cell
        sizes = groups.group_sizes.astype(np.float64)
        streamed = stats.group_mean_log().to_numpy()

        max_err = 0.0
        worst = None
        checked = 0
        for g in pick:
            name = gene_names[g]
            if name not in name_to_col:
                continue
            a, b = int(indptr[g]), int(indptr[g + 1])
            dense = np.zeros(n_obs, dtype=np.float64)
            rows = X["indices"][a:b]
            dense[rows] = X["data"][a:b].astype(np.float64) * inv_lib[rows]
            y = np.log1p(dense)  # explicit zeros retained -> log1p(0) == 0

            naive = np.full(groups.n_groups, np.nan)
            ok = grp >= 0
            tot = np.bincount(grp[ok], weights=y[ok], minlength=groups.n_groups)
            nz = sizes > 0
            naive[nz] = tot[nz] / sizes[nz]

            col = name_to_col[name]
            both = np.isfinite(naive) & np.isfinite(streamed[:, col])
            err = np.abs(naive[both] - streamed[both, col])
            checked += 1
            if err.size and err.max() > max_err:
                max_err = float(err.max())
                worst = name

    return {
        "genes_checked": checked,
        "max_abs_error": max_err,
        "worst_gene": worst,
        "passed": bool(max_err < 1e-8),
    }


# --------------------------------------------------------------------------- #
# log2 fold change
# --------------------------------------------------------------------------- #
def log2fc_per_condition(
    stats: StreamedStats,
    control_label: str = CONTROL_LABEL,
    min_cells: int = 1,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Per-condition perturbation x gene log2FC, cell counts and standard error.

    Returns ``{condition: {"log2fc": df, "se": df, "n_cells": series}}``.

    ``log2fc = (mean_log1p_pert - mean_log1p_control) / ln2``, control taken from
    the same condition only.  The standard error is the delta-method SE of that
    difference, also rescaled to log2 units::

        se = sqrt(var_pert / n_pert + var_ctrl / n_ctrl) / ln2

    with ``var`` the within-group variance of log1p(CP10K) including implicit
    zeros.  It treats cells as independent, which ignores guide-level and
    sequencing-batch structure and so is optimistic; downstream uncertainty
    claims should say so.
    """
    means = stats.group_mean_log()
    varis = stats.group_var_log()
    sizes = stats.groups.frame().set_index(["perturbation", "condition"])["n_cells"]

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for cond in stats.groups.cond_levels:
        m = means.xs(cond, level="condition")
        v = varis.xs(cond, level="condition")
        n = sizes.xs(cond, level="condition")

        if control_label not in m.index or n.get(control_label, 0) < 1:
            raise ValueError(f"no control cells for condition {cond!r}")
        ctrl_mu = m.loc[control_label]
        ctrl_var = v.loc[control_label]
        n_ctrl = float(n.loc[control_label])

        keep = [
            p for p in m.index
            if p not in (control_label, IMPURE_CONTROL_LABEL) and n.get(p, 0) >= min_cells
        ]
        lfc = (m.loc[keep] - ctrl_mu) / LN2
        se = (
            np.sqrt(
                v.loc[keep].div(n.loc[keep].astype(float), axis=0)
                + ctrl_var / n_ctrl
            )
            / LN2
        )
        out[cond] = {
            "log2fc": lfc.astype(np.float32),
            "se": se.astype(np.float32),
            "n_cells": n.loc[keep].astype(np.int64),
            "n_control_cells": int(n_ctrl),
            "control_mean_log1p": ctrl_mu.astype(np.float32),
        }
    return out


# --------------------------------------------------------------------------- #
# Seurat-flavour HVG from streamed stats
# --------------------------------------------------------------------------- #
def seurat_hvg_from_stats(
    stats: StreamedStats,
    n_top_genes: int = 1000,
    gene_mask: np.ndarray | None = None,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Reproduce ``sc.pp.highly_variable_genes(flavor='seurat')`` from the pass.

    scanpy's seurat flavour undoes the log (``expm1``) before computing
    mean/variance, so it needs exactly the mean and variance of CP10K -- which
    is what ``global_sum_cp10k`` / ``global_sumsq_cp10k`` hold.  The remaining
    steps (dispersion = var/mean, log-dispersion, 20 equal-width bins of
    ``log1p(mean)``, z-score of dispersion within bin, take the top
    ``n_top_genes``) are reimplemented here and match scanpy's source.

    ``gene_mask`` must reproduce the gene filter the caller applied before
    calling scanpy (``ncells >= min_cells``); binning is global over the
    surviving genes, so including filtered-out genes would shift every bin.
    """
    n = float(stats.n_cells_qc)
    mean = stats.global_sum_cp10k / n
    var = (stats.global_sumsq_cp10k - stats.global_sum_cp10k**2 / n) / (n - 1)
    var = np.maximum(var, 0.0)

    df = pd.DataFrame({"means": mean, "variances": var}, index=stats.gene_names)
    if gene_mask is not None:
        df = df.loc[np.asarray(gene_mask, dtype=bool)]

    safe_mean = df["means"].to_numpy().copy()
    safe_mean[safe_mean == 0] = 1e-12
    disp = df["variances"].to_numpy() / safe_mean
    disp = np.where(disp == 0, np.nan, disp)
    df["dispersions"] = np.log(disp)
    df["means_log1p"] = np.log1p(df["means"])

    df["mean_bin"] = pd.cut(df["means_log1p"], bins=n_bins)
    grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    binstats = grouped.agg(avg="mean", dev="std")
    # scanpy: a bin holding one gene has NaN std -> normalised dispersion := 1
    one = binstats["dev"].isna()
    binstats.loc[one, "dev"] = binstats.loc[one, "avg"]
    binstats.loc[one, "avg"] = 0.0
    aligned = binstats.loc[df["mean_bin"]].set_index(df.index)
    df["dispersions_norm"] = (df["dispersions"] - aligned["avg"]) / aligned["dev"]

    ranked = df["dispersions_norm"].fillna(-np.inf).sort_values(ascending=False)
    df["highly_variable"] = False
    df.loc[ranked.index[:n_top_genes], "highly_variable"] = True
    return df.drop(columns=["mean_bin"])


# --------------------------------------------------------------------------- #
# gene-column slicing (CSC -> CSR)
# --------------------------------------------------------------------------- #
def slice_gene_columns(h5ad_path: str | Path, gene_idx: np.ndarray):
    """Pull selected gene columns out of the CSC file as a CSR matrix.

    Cheap in CSC: each gene is one contiguous slice.  Returns
    ``(csr_matrix, gene_names)`` of raw counts; normalise afterwards.
    """
    import scipy.sparse as sp

    gene_idx = np.sort(np.asarray(gene_idx, dtype=np.int64))
    with h5py.File(str(h5ad_path), "r") as f:
        X = f["X"]
        n_obs, _ = (int(v) for v in X.attrs["shape"])
        indptr = X["indptr"][:]
        gene_names = _read_categorical(f["var"], f["var"].attrs["_index"])[gene_idx]

        counts = (indptr[gene_idx + 1] - indptr[gene_idx]).astype(np.int64)
        new_indptr = np.zeros(gene_idx.size + 1, dtype=np.int64)
        np.cumsum(counts, out=new_indptr[1:])
        total = int(new_indptr[-1])

        data = np.empty(total, dtype=np.float32)
        indices = np.empty(total, dtype=np.int32)
        for k, g in enumerate(gene_idx):
            a, b = int(indptr[g]), int(indptr[g + 1])
            lo, hi = int(new_indptr[k]), int(new_indptr[k + 1])
            data[lo:hi] = X["data"][a:b]
            indices[lo:hi] = X["indices"][a:b]

    csc = sp.csc_matrix((data, indices, new_indptr), shape=(n_obs, gene_idx.size))
    return csc.tocsr(), gene_names


# --------------------------------------------------------------------------- #
# protein / CLR
# --------------------------------------------------------------------------- #
def clr(mat: np.ndarray) -> np.ndarray:
    """Centred log ratio

    ``log1p`` then subtract the per-cell mean across features.  With only 24
    ADTs -- four of which are isotype controls -- the geometric-mean reference is
    estimated from very few features, so this is a coarse approximation to a
    proper CLR and shifts with panel composition.  It is reused here purely for
    consistency with the project's existing convention.
    """
    logm = np.log1p(np.asarray(mat, dtype=np.float64))
    return logm - logm.mean(axis=1, keepdims=True)


def write_provenance(path: str | Path, payload: dict) -> None:
    """Write the JSON sidecar recording exactly how each cache was produced."""
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
