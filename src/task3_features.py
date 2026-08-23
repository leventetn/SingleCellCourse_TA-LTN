"""Task 3 gene-level feature construction (leakage-free by construction).

The unit of observation is (perturbation, condition).  Every feature describes
the PERTURBED GENE itself, never the perturbation's own expression response, so
a held-out perturbation can be featurised without any information about its
log2FC profile.

Sources used
------------
* ``data/frangieh/rna.h5ad`` -- CSC counts.  Only the columns of the ~50 target
  genes plus a fixed marker panel are read (cheap column slicing), and only rows
  belonging to CLEAN CONTROL cells (zero targeting guides) are used.
* ``data/rna_control_mean_{cond}.parquet`` -- per-gene control mean log1p(CP10K).
* ``data/pathway_labels.csv`` -- external pathway/module annotation.
* ``data/rna_ncells_{cond}.parquet`` -- single-perturbation cell counts.

Nothing here touches ``rna_log2fc_*`` (the target) for any perturbation.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd

CONDS = ("Co_culture", "Control", "IFNg")

#: Curated programme genes used as the co-expression reference panel.  Chosen
#: from prior biological knowledge of the programmes discussed in Frangieh 2021
#: (IFN-gamma response, MHC-I antigen presentation, cell cycle, mitochondrial
#: respiration, translation), NOT from any perturbation log2FC statistic.
MARKER_PANEL = [
    "STAT1", "IRF1", "GBP1", "TAP1", "IDO1", "HLA-E", "B2M", "HLA-A", "HLA-B",
    "HLA-C", "CD74", "PSMB9", "NLRC5", "CD274", "SOCS1", "WARS",
    "MKI67", "CCND1", "CDK4", "E2F1", "PCNA", "TOP2A",
    "MT-CO1", "NDUFA4", "COX7C", "ATP5F1E",
    "RPL13A", "RPS6", "EIF4A1", "ACTB", "GAPDH", "PPIA",
    "MITF", "PMEL", "MLANA", "TYR", "AXL", "SOX10", "NGFR", "VIM",
]


def _clean_control_mask(raw_obs: pd.DataFrame, hv_obs: pd.DataFrame) -> pd.Series:
    """Boolean over RAW rows: QC-passing cells carrying zero targeting guides.

    The clean-control definition is taken from the Phase-0 cache
    (``dataset_hv.h5ad`` obs) and mapped back onto the raw cell order by
    ``cell_name``, so it is identical to the definition used for every log2FC
    baseline rather than a re-derivation.
    """
    keep = hv_obs.loc[hv_obs["is_clean_control"].astype(bool), ["cell_name", "condition"]]
    cond_of = dict(zip(keep["cell_name"].astype(str), keep["condition"].astype(str)))
    return raw_obs["cell_name"].astype(str).map(cond_of)


def read_columns(h5_path: str, gene_names, var_names, row_mask: np.ndarray) -> np.ndarray:
    """Read the given gene columns of a CSC h5ad for the masked rows only."""
    pos = {g: i for i, g in enumerate(var_names)}
    rows = np.flatnonzero(row_mask)
    row_lookup = -np.ones(len(row_mask), dtype=np.int64)
    row_lookup[rows] = np.arange(len(rows))
    out = np.zeros((len(rows), len(gene_names)), dtype=np.float32)
    with h5py.File(h5_path, "r") as f:
        X = f["X"]
        indptr = X["indptr"][:]
        for j, g in enumerate(gene_names):
            c = pos[g]
            lo, hi = int(indptr[c]), int(indptr[c + 1])
            if hi <= lo:
                continue
            idx = X["indices"][lo:hi]
            dat = X["data"][lo:hi]
            keep = row_lookup[idx] >= 0
            out[row_lookup[idx[keep]], j] = dat[keep]
    return out


def cp10k_log1p(counts: np.ndarray, ncounts: np.ndarray) -> np.ndarray:
    return np.log1p(counts / ncounts[:, None] * 1e4)


def build_features(selected, h5_path, raw_obs, hv_obs, var_names, pathway_labels,
                   ncells_by_cond, control_mean_by_cond, marker_panel=None):
    """Return the (perturbation, condition) feature table."""
    marker_panel = [g for g in (marker_panel or MARKER_PANEL) if g in set(var_names)]
    cond_of_cell = _clean_control_mask(raw_obs, hv_obs)
    needed = sorted(set(list(selected) + marker_panel) & set(var_names))
    mask = cond_of_cell.notna().values
    mat = read_columns(h5_path, needed, var_names, mask)
    nc = raw_obs.loc[mask, "ncounts"].values.astype(np.float64)
    lognorm = cp10k_log1p(mat, nc)
    cond_vec = cond_of_cell[mask].values
    cmap = {"Co-culture": "Co_culture", "Control": "Control", "IFNγ": "IFNg",
            "Co_culture": "Co_culture", "IFNg": "IFNg"}
    cond_vec = np.array([cmap.get(c, c) for c in cond_vec])
    gidx = {g: i for i, g in enumerate(needed)}

    rows = []
    for cond in CONDS:
        sub = lognorm[cond_vec == cond]
        raw_sub = mat[cond_vec == cond]
        mk = np.array([gidx[g] for g in marker_panel])
        M = sub[:, mk]
        Mz = (M - M.mean(0)) / np.maximum(M.std(0), 1e-8)
        cmean = control_mean_by_cond[cond]
        for p in selected:
            r = {"perturbation": p, "condition": cond}
            if p in gidx:
                v = sub[:, gidx[p]]
                r["self_mean_expr"] = float(v.mean())
                r["self_detection_rate"] = float((raw_sub[:, gidx[p]] > 0).mean())
                r["self_expr_var"] = float(v.var())
                vz = (v - v.mean()) / max(v.std(), 1e-8)
                corr = (Mz * vz[:, None]).mean(0)
                for g, c in zip(marker_panel, corr):
                    r[f"coexpr_{g}"] = float(0.0 if not np.isfinite(c) else c)
                r["coexpr_absmean"] = float(np.nanmean(np.abs(corr)))
                r["coexpr_max"] = float(np.nanmax(corr))
                r["coexpr_min"] = float(np.nanmin(corr))
                r["measured_in_rna"] = 1.0
            else:
                r["self_mean_expr"] = 0.0
                r["self_detection_rate"] = 0.0
                r["self_expr_var"] = 0.0
                for g in marker_panel:
                    r[f"coexpr_{g}"] = 0.0
                r["coexpr_absmean"] = r["coexpr_max"] = r["coexpr_min"] = 0.0
                r["measured_in_rna"] = 0.0
            for c2 in CONDS:
                r[f"ctrlmean_{c2}"] = float(control_mean_by_cond[c2].get(p, 0.0))
            r["ctrlmean_across_cond_var"] = float(np.var([r[f"ctrlmean_{c2}"] for c2 in CONDS]))
            r["ctrlmean_this_cond"] = float(cmean.get(p, 0.0))
            r["log_ncells"] = float(np.log10(ncells_by_cond[cond].get(p, 1.0)))
            rows.append(r)

    feat = pd.DataFrame(rows)
    lab = pathway_labels.set_index("perturbation")["label"]
    feat["pathway_label"] = feat["perturbation"].map(lab).fillna("UNLABELLED")
    feat = pd.concat([feat, pd.get_dummies(feat["pathway_label"], prefix="pw").astype(float)], axis=1)
    feat = pd.concat([feat, pd.get_dummies(feat["condition"], prefix="cond").astype(float)], axis=1)
    return feat
