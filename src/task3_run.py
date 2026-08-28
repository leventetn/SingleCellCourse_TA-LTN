"""Task 3 experiment driver: split -> fit -> LOPO hyperparameter search -> evaluate.

Every fitted object (feature scaler, target PCA, floor means) is fitted inside
``run_split`` on training perturbations only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import task3_model as M


def _prep(X, Y_rna, Y_adt, tr, te, n_pc):
    """Fit scaler + target PCA on training rows; transform both sides."""
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    pca = PCA(n_components=min(n_pc, len(tr) - 1, Y_rna.shape[1])).fit(Y_rna[tr])
    Ztr = pca.transform(Y_rna[tr])
    zsc = StandardScaler().fit(Ztr)
    Ztr = zsc.transform(Ztr)
    asc = StandardScaler().fit(Y_adt[tr])
    Atr = asc.transform(Y_adt[tr])
    return Xtr, Xte, Ztr, Atr, pca, zsc, asc


def _inverse(pred_z, pred_a, pca, zsc, asc):
    return pca.inverse_transform(zsc.inverse_transform(pred_z)), asc.inverse_transform(pred_a)


def lopo_cv(X, Y_rna, Y_adt, perts_row, train_perts, grid, n_pc, seed=0):
    """Leave-one-perturbation-out CV inside the training set only."""
    rows = []
    for g in grid:
        errs = []
        for hp in train_perts:
            tr = np.flatnonzero(np.isin(perts_row, [p for p in train_perts if p != hp]))
            te = np.flatnonzero(perts_row == hp)
            if len(te) == 0:
                continue
            Xtr, Xte, Ztr, Atr, pca, zsc, asc = _prep(X, Y_rna, Y_adt, tr, te, n_pc)
            net = M.train_net(Xtr, Ztr, Atr, seed=seed, **g)
            pz, pa = M.predict_net(net, Xte)
            pr, _ = _inverse(pz, pa, pca, zsc, asc)
            errs.append(np.mean((pr - Y_rna[te]) ** 2))
        rows.append({**g, "n_pc": n_pc, "cv_mse": float(np.mean(errs))})
    return pd.DataFrame(rows).sort_values("cv_mse").reset_index(drop=True)


def lopo_cv_ridge(X, Y_rna, feats, perts_row, train_perts, alphas, seed=123):
    """Leave-one-perturbation-out CV for ridge regression."""

    X_sub = X[:, feats]
    rows = []
    for a in alphas:
        errs = []
        for hp in train_perts:      # hold out one perturbation (in all conditions) at a time
            tr = np.flatnonzero(np.isin(perts_row, [p for p in train_perts if p != hp]))
            te = np.flatnonzero(perts_row == hp)
            if len(te) == 0:
                continue

            scaler = StandardScaler().fit(X_sub[tr])
            Xtr, Xte = scaler.transform(X_sub[tr]), scaler.transform(X_sub[te])
            ridge = M.train_ridge(Xtr, Y_rna[tr], a, seed=seed).predict(Xte)
            errs.append(np.mean((ridge - Y_rna[te]) ** 2))

        rows.append({'alpha': a, 'cv_mse': float(np.mean(errs))})

    return pd.DataFrame(rows).sort_values("cv_mse").reset_index(drop=True)



def run_split(name, X, Y_rna, Y_adt, perts_row, cond_row, train_perts, test_perts,
              best_hp, n_pc, ridge_alpha, ridge_feats, seeds=(0, 1, 2, 3, 4), k=100,):
    """Train the NN (several seeds), ridge regression, and both floors; return per-row metrics."""
    tr = np.flatnonzero(np.isin(perts_row, train_perts))
    te = np.flatnonzero(np.isin(perts_row, test_perts))
    Xtr, Xte, Ztr, Atr, pca, zsc, asc = _prep(X, Y_rna, Y_adt, tr, te, n_pc)

    preds_rna, preds_adt = [], []
    for s in seeds:
        net = M.train_net(Xtr, Ztr, Atr, seed=s, **best_hp)
        pz, pa = M.predict_net(net, Xte)
        pr, padt = _inverse(pz, pa, pca, zsc, asc)
        preds_rna.append(pr)
        preds_adt.append(padt)
        
    nn_rna = np.mean(preds_rna, axis=0)
    nn_adt = np.mean(preds_adt, axis=0)

    # ridge features
    X_sub = X[:, ridge_feats]
    scaler = StandardScaler().fit(X_sub[tr])
    rXtr, rXte = scaler.transform(X_sub[tr]), scaler.transform(X_sub[te])
    # ridge regression
    ridge_rna = M.train_ridge(rXtr, Y_rna[tr], ridge_alpha, s).predict(rXte)
    
    fl_rna = M.floor_train_mean(Y_rna[tr], cond_row[tr], cond_row[te])
    fl_adt = M.floor_train_mean(Y_adt[tr], cond_row[tr], cond_row[te])
    z_rna = M.floor_zero(len(te), Y_rna.shape[1])
    z_adt = M.floor_zero(len(te), Y_adt.shape[1])
    
    out, ridge_residues = [], []
    for i, ridx in enumerate(te):
        base = {"split": name, "perturbation": perts_row[ridx], "condition": cond_row[ridx]}
        for mname, pr, pa in (("nn", nn_rna[i], nn_adt[i]),
                              ("floor_trainmean", fl_rna[i], fl_adt[i]),
                              ("floor_zero", z_rna[i], z_adt[i]),
                              ("ridge", ridge_rna[i], None),):
            rec = {**base, "model": mname}
            rec.update({f"rna_{a}": b for a, b in M.row_metrics(Y_rna[ridx], pr, k=k).items()})
            if mname != "ridge":
                rec.update({f"adt_{a}": b for a, b in M.row_metrics(Y_adt[ridx], pa, k=5).items()})
            else:
                ridge_residues.append(np.abs(Y_rna[ridx] - pr))
            out.append(rec)
    seed_spread = float(np.std([np.sqrt(np.mean((p - Y_rna[te]) ** 2)) for p in preds_rna]))
    ridge_q95 = np.percentile(np.concatenate(ridge_residues), 95)
    return pd.DataFrame(out), {"nn_rna": nn_rna, "nn_adt": nn_adt, "ridge_rna": ridge_rna,
                               "fl_rna": fl_rna, "fl_adt": fl_adt,
                               "te_idx": te, "seed_rmse_sd": seed_spread, "ridge_q95": ridge_q95,
                               "pca_evr": float(pca.explained_variance_ratio_.sum())}
