"""Task 3 models: two reference floors and one multi-task neural network.

Design constraints
------------------
* Unit of observation = (perturbation, condition).  With 50 perturbations x 3
  conditions there are 150 rows, of which ~120-135 are training rows.
* Every quantity that is fitted -- feature scaler, target PCA, floor means --
  is fitted on TRAINING PERTURBATIONS ONLY.  Held-out perturbations are
  transformed by the fitted objects, never used to fit them.
* The gene space itself (``expressed`` mask) is defined from CONTROL cells only
  and so is independent of every perturbation's response.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge


# --------------------------------------------------------------------------- #
# reference floors
# --------------------------------------------------------------------------- #

def floor_train_mean(Y_train, cond_train, cond_test):
    """Per-condition mean log2FC of the TRAINING perturbations.

    Returns one prediction row per test row, taken from the training mean of
    the matching condition.
    """
    out = np.zeros((len(cond_test), Y_train.shape[1]), dtype=np.float64)
    for i, c in enumerate(cond_test):
        m = cond_train == c
        out[i] = Y_train[m].mean(0) if m.any() else Y_train.mean(0)
    return out


def floor_zero(n_rows, n_cols):
    """No-effect predictor."""
    return np.zeros((n_rows, n_cols), dtype=np.float64)


# --------------------------------------------------------------------------- #
# multi-task network
# --------------------------------------------------------------------------- #

class MultiTaskNet(nn.Module):
    """Shared trunk, two heads: RNA (PCA components) and ADT (protein log2FC)."""

    def __init__(self, n_in, n_rna, n_adt, hidden=32, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
        )
        self.rna_head = nn.Linear(hidden // 2, n_rna)
        self.adt_head = nn.Linear(hidden // 2, n_adt)

    def forward(self, x):
        h = self.trunk(x)
        return self.rna_head(h), self.adt_head(h)


def train_net(Xtr, Ytr_rna, Ytr_adt, *, hidden=32, dropout=0.3, lam_adt=0.5,
              lr=1e-3, weight_decay=1e-3, epochs=400, seed=0,
              Xval=None, Yval_rna=None, Yval_adt=None, patience=60):
    """Train one network.  Early stopping on the validation RNA+ADT loss when a
    validation set is supplied, otherwise a fixed epoch budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MultiTaskNet(Xtr.shape[1], Ytr_rna.shape[1], Ytr_adt.shape[1],
                       hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    xt = torch.tensor(Xtr, dtype=torch.float32)
    yr = torch.tensor(Ytr_rna, dtype=torch.float32)
    ya = torch.tensor(Ytr_adt, dtype=torch.float32)
    have_val = Xval is not None and len(Xval) > 0
    if have_val:
        xv = torch.tensor(Xval, dtype=torch.float32)
        yrv = torch.tensor(Yval_rna, dtype=torch.float32)
        yav = torch.tensor(Yval_adt, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    mse = nn.MSELoss()
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        pr, pa = net(xt)
        loss = mse(pr, yr) + lam_adt * mse(pa, ya)
        loss.backward()
        opt.step()
        if have_val:
            net.eval()
            with torch.no_grad():
                pr, pa = net(xv)
                vl = (mse(pr, yrv) + lam_adt * mse(pa, yav)).item()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if have_val and best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


def predict_net(net, X):
    with torch.no_grad():
        pr, pa = net(torch.tensor(X, dtype=torch.float32))
    return pr.numpy().astype(np.float64), pa.numpy().astype(np.float64)


def train_ridge(Xtr, Ytr, alpha, seed=123):
    '''Train a ridge regression model with alpha=alpha'''
    return Ridge(alpha=alpha, random_state=seed).fit(Xtr, Ytr)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def top_gene_jaccard(y_true, y_pred, k=100):
    a = set(np.argsort(-np.abs(y_true))[:k])
    b = set(np.argsort(-np.abs(y_pred))[:k])
    return len(a & b) / len(a | b)


def row_metrics(y_true, y_pred, k=100):
    """Pearson r, Spearman rho, RMSE and top-k |log2FC| Jaccard for one row."""
    if np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        r = rho = np.nan
    else:
        r = pearsonr(y_true, y_pred)[0]
        rho = spearmanr(y_true, y_pred)[0]
    return {
        "pearson_r": r,
        "spearman_rho": rho,
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "top%d_jaccard" % k: top_gene_jaccard(y_true, y_pred, k),
    }


def bootstrap_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """Percentile CI of the mean of ``values`` (NaNs dropped)."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))
