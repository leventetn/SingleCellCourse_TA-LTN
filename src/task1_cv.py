"""Cross-fold model fitting and feature importance for Task 1 (condition classification).

Scope
-----
Everything here operates on the cached ``data/dataset_hv.h5ad`` matrix
(216,431 QC-passing cells x 1,011 genes, CP10K + log1p, CSR) and the treatment
condition in ``obs['condition']``.  The full RNA file is never touched.

Design decisions that matter for the numbers this module produces
----------------------------------------------------------------
* **One split object for every model.**  ``fold_splits`` returns the
  ``StratifiedKFold(n_splits=5, shuffle=True, random_state=123)`` indices once,
  and both the XGBoost and the MLP path consume that same list.  Fold-wise
  importances from the two models are therefore computed on identical training
  cells and identical held-out cells, which is what makes a rank comparison
  between them meaningful.
* **One importance subsample per fold, shared by both models.**
  ``fold_subsample`` draws the cells from the fold's *test* part with a
  fold-dependent but deterministic seed.  TreeSHAP over all ~43k held-out cells
  of a several-hundred-tree multiclass ensemble is not affordable on this
  machine; ~5,000 cells is.  The MLP gradient is evaluated on exactly those
  cells rather than on all of them, so the two importance vectors are estimates
  of the same population quantity on the same sample.
* **Feature names survive into the booster.**  XGBoost only learns column names
  from objects that expose them, which a scipy sparse matrix does not.  The
  names are attached to the fitted booster explicitly, exactly as
  ``src.models.fit_xgb`` does, because ``shap`` and every importance table below
  read them from there.
* **Both importance scales are magnitudes, not signed effects.**  Mean
  ``|SHAP|`` (averaged over cells and over the three class outputs) and mean
  ``|gradient x input|`` both answer "how much does this gene move the model's
  output", not "in which direction".  They are not on a common unit; only ranks
  are compared across models.

Known limitation, stated here because every number downstream inherits it
------------------------------------------------------------------------
The 1,011 genes in ``dataset_hv.h5ad`` were selected by highly-variable-gene
ranking computed on **all** cells, i.e. before the cross-validation split.  The
fold-wise accuracies below are therefore mildly optimistic: the feature set saw
the held-out cells.  The standalone script ``tools/hvg_inside_fold.py`` (not a
function in this module) quantifies the size of that effect for fold 0 by redoing
the selection on that fold's training cells only; it writes
``data/task1_hvg_leakage_fold0.json``.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

cache_dir = Path(tempfile.gettempdir()) / "sc-course-2026-cache"
(cache_dir / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_dir / "numba").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

SEED = 123

# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #


def fold_splits(y, n_splits: int = 5, seed: int = SEED):
    """The 5 stratified folds, materialised once and reused by every model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dummy = np.zeros((len(y), 1), dtype=np.int8)
    return [(tr, te) for tr, te in skf.split(dummy, y)]


def fold_subsample(test_idx, n_cells: int = 5000, fold: int = 0, seed: int = SEED):
    """Deterministic sorted subsample of a fold's held-out cells for importance."""
    test_idx = np.asarray(test_idx)
    if len(test_idx) <= n_cells:
        return np.sort(test_idx)
    rng = np.random.default_rng(seed + 1000 * fold)
    return np.sort(rng.choice(test_idx, n_cells, replace=False))


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #


def xgb_params(n_estimators: int, n_classes: int, max_depth: int = 6, **over):
    """The students' Task 1 XGBoost configuration, with capacity as a parameter."""
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        importance_type="weight",
        tree_method="hist",
        n_jobs=os.cpu_count() or 2,
        random_state=SEED,
    )
    if n_classes > 2:
        params.update(objective="multi:softprob", num_class=n_classes)
    else:
        params.update(objective="binary:logistic")
    params.update(over)
    return params


def fit_xgb_fold(
    X,
    y,
    train_idx,
    n_estimators: int,
    feature_names,
    max_depth: int = 6,
    early_stopping_rounds: int | None = None,
    val_frac: float = 0.15,
):
    """Fit one XGBoost model on ``train_idx``.

    With ``early_stopping_rounds`` set, a stratified ``val_frac`` slice is carved
    out of the *training* cells for the stopping criterion.  The fold's held-out
    cells are never seen, so the reported fold accuracy stays honest.
    """
    n_classes = int(len(np.unique(y)))
    fit_idx = np.asarray(train_idx)
    eval_set = None

    if early_stopping_rounds is not None:
        fit_idx, val_idx = train_test_split(
            fit_idx, test_size=val_frac, stratify=y[fit_idx], random_state=SEED
        )
        fit_idx = np.sort(fit_idx)
        val_idx = np.sort(val_idx)
        eval_set = [(X[val_idx], y[val_idx])]

    params = xgb_params(n_estimators, n_classes, max_depth=max_depth)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
        params["eval_metric"] = "mlogloss"

    bst = XGBClassifier(**params)
    bst.fit(X[fit_idx], y[fit_idx], eval_set=eval_set, verbose=False)
    # Names do not reach the booster through a sparse matrix; attach explicitly.
    bst.get_booster().feature_names = list(feature_names)
    return bst


def shap_importance(bst, X_dense, n_classes: int):
    """Mean ``|SHAP|`` per feature, averaged over cells and over class outputs.

    ``shap.TreeExplainer`` on a multiclass booster returns
    ``(n_cells, n_features, n_classes)``.  Averaging the absolute value over both
    the cell axis and the class axis gives one non-negative number per gene:
    the mean magnitude of that gene's contribution to any class logit.
    """
    import shap

    explainer = shap.TreeExplainer(bst)
    values = explainer.shap_values(X_dense)
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[0] == n_classes:  # (classes, cells, features)
            values = np.transpose(values, (1, 2, 0))
        per_gene = np.abs(values).mean(axis=(0, 2))
    else:
        per_gene = np.abs(values).mean(axis=0)
    return per_gene


# --------------------------------------------------------------------------- #
# MLP -- architecture copied verbatim from notebooks/mlp_classifier_task1.ipynb
# --------------------------------------------------------------------------- #


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLPClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def make_batch(X, y_values, idx, device):
    X_batch = X[idx]
    if sp.issparse(X_batch):
        X_batch = X_batch.toarray()
    else:
        X_batch = np.asarray(X_batch)
    X_batch = torch.as_tensor(X_batch, dtype=torch.float32, device=device)
    y_batch = None if y_values is None else torch.as_tensor(
        y_values[idx], dtype=torch.long, device=device
    )
    return X_batch, y_batch


def batches(indices, batch_size, shuffle=True, seed=SEED):
    indices = np.asarray(indices)
    if shuffle:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def fit_mlp(
    X,
    y,
    train_idx,
    device,
    hidden=128,
    dropout=0.2,
    n_epochs=8,
    batch_size=4096,
    lr=1e-3,
    weight_decay=1e-4,
):
    """The notebook's MLP: 1011 -> 128 -> 64 -> 3, dropout 0.2, Adam, CE loss."""
    torch.manual_seed(SEED)
    model = MLPClassifier(X.shape[1], len(np.unique(y)), hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for epoch in range(n_epochs):
        model.train()
        train_loss, n_seen = 0.0, 0
        for batch_idx in batches(train_idx, batch_size, shuffle=True, seed=SEED + epoch):
            X_batch, y_batch = make_batch(X, y, batch_idx, device)
            opt.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(batch_idx)
            n_seen += len(batch_idx)
        history.append({"epoch": epoch + 1, "train_loss": train_loss / n_seen})
    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_mlp(model, X, indices, device, batch_size=8192):
    model.eval()
    preds = []
    for batch_idx in batches(indices, batch_size, shuffle=False):
        X_batch, _ = make_batch(X, None, batch_idx, device)
        preds.append(model(X_batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def grad_x_input_importance(model, X, y, indices, device, batch_size=2048):
    """Mean ``|gradient x input|`` per gene on the given cells.

    The gradient is taken of the *true-class* logit, summed over the batch, so
    each cell contributes the sensitivity of the logit it should have received.
    Identical to the function in ``notebooks/mlp_classifier_task1.ipynb`` except
    that the cell subsample is supplied by the caller instead of drawn inside,
    which is what lets the XGBoost and MLP importances share one sample.
    """
    model.eval()
    indices = np.asarray(indices)
    importance = np.zeros(X.shape[1], dtype=np.float64)
    n_seen = 0
    for batch_idx in batches(indices, batch_size, shuffle=False):
        X_batch, y_batch = make_batch(X, y, batch_idx, device)
        X_batch.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(X_batch)
        logits.gather(1, y_batch[:, None]).sum().backward()
        importance += (X_batch.grad * X_batch).abs().detach().cpu().numpy().sum(axis=0)
        n_seen += X_batch.shape[0]
    return importance / n_seen


# --------------------------------------------------------------------------- #
# per-gene summary across folds
# --------------------------------------------------------------------------- #


@dataclass
class FoldImportance:
    """Per-fold importance vectors for one model, plus the across-fold summary."""

    model: str
    genes: pd.Index
    per_fold: dict = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        """Long frame: one row per (fold, gene)."""
        rows = []
        for fold, vec in sorted(self.per_fold.items()):
            r = pd.DataFrame({"gene": self.genes, "importance": np.asarray(vec)})
            r["fold"] = fold
            r["model"] = self.model
            r["rank"] = r["importance"].rank(ascending=False, method="min").astype(int)
            rows.append(r)
        return pd.concat(rows, ignore_index=True)

    def summary(self, top_n: int = 20) -> pd.DataFrame:
        """mean / SD / mean rank / number of folds in the model's own top ``top_n``."""
        long = self.frame()
        wide = long.pivot(index="gene", columns="fold", values="importance")
        ranks = long.pivot(index="gene", columns="fold", values="rank")
        out = pd.DataFrame(
            {
                "model": self.model,
                "mean_importance": wide.mean(axis=1),
                "sd_importance": wide.std(axis=1, ddof=1),
                "mean_rank": ranks.mean(axis=1),
                "sd_rank": ranks.std(axis=1, ddof=1),
                "best_rank": ranks.min(axis=1).astype(int),
                "worst_rank": ranks.max(axis=1).astype(int),
                f"n_folds_in_top{top_n}": (ranks <= top_n).sum(axis=1).astype(int),
                "n_folds": wide.notna().sum(axis=1).astype(int),
            }
        )
        out["cv_importance"] = out["sd_importance"] / out["mean_importance"].where(
            out["mean_importance"] > 0
        )
        return out.sort_values("mean_importance", ascending=False)


def seurat_hvg_from_moments(means, variances, gene_names, n_top_genes=1000, n_bins=20):
    """Seurat-flavour HVG selection from CP10K mean/variance vectors.

    Same arithmetic as ``src.pseudobulk.seurat_hvg_from_stats`` (which was
    validated to reproduce ``sc.pp.highly_variable_genes`` exactly, 200/200 genes
    identical), factored out so it can be applied to moments accumulated over an
    arbitrary cell subset -- here, one fold's training cells only.
    """
    df = pd.DataFrame(
        {"means": np.asarray(means, dtype=np.float64),
         "variances": np.asarray(variances, dtype=np.float64)},
        index=pd.Index(gene_names),
    )
    safe_mean = df["means"].to_numpy().copy()
    safe_mean[safe_mean == 0] = 1e-12
    disp = df["variances"].to_numpy() / safe_mean
    disp = np.where(disp == 0, np.nan, disp)
    df["dispersions"] = np.log(disp)
    df["means_log1p"] = np.log1p(df["means"])

    df["mean_bin"] = pd.cut(df["means_log1p"], bins=n_bins)
    binstats = df.groupby("mean_bin", observed=True)["dispersions"].agg(avg="mean", dev="std")
    one = binstats["dev"].isna()
    binstats.loc[one, "dev"] = binstats.loc[one, "avg"]
    binstats.loc[one, "avg"] = 0.0
    aligned = binstats.loc[df["mean_bin"]].set_index(df.index)
    df["dispersions_norm"] = (df["dispersions"] - aligned["avg"]) / aligned["dev"]

    ranked = df["dispersions_norm"].fillna(-np.inf).sort_values(ascending=False)
    df["highly_variable"] = False
    df.loc[ranked.index[:n_top_genes], "highly_variable"] = True
    return df.drop(columns=["mean_bin"])


def baselines(y, seed: int = SEED, n_draws: int = 200):
    """Balanced-accuracy reference points for a 3-class imbalanced problem.

    * ``majority_class`` -- always predict the largest class.  Balanced accuracy
      is 1/n_classes by construction (one class perfect, the others zero).
    * ``stratified_random`` -- draw the label from the empirical class frequencies.
      Its expected balanced accuracy is also 1/n_classes; the spread over
      ``n_draws`` resamples is reported because it sets the scale on which a real
      model's fold-to-fold SD should be read.
    * ``uniform_random`` -- draw uniformly over classes.
    """
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    freq = counts / counts.sum()
    rng = np.random.default_rng(seed)

    maj = balanced_accuracy_score(y, np.full_like(y, classes[np.argmax(counts)]))
    strat = [
        balanced_accuracy_score(y, rng.choice(classes, size=len(y), p=freq))
        for _ in range(n_draws)
    ]
    unif = [
        balanced_accuracy_score(y, rng.choice(classes, size=len(y)))
        for _ in range(n_draws)
    ]
    return {
        "majority_class": float(maj),
        "stratified_random_mean": float(np.mean(strat)),
        "stratified_random_sd": float(np.std(strat, ddof=1)),
        "uniform_random_mean": float(np.mean(unif)),
        "uniform_random_sd": float(np.std(unif, ddof=1)),
        "chance_1_over_k": 1.0 / len(classes),
        "class_frequencies": {int(c): float(f) for c, f in zip(classes, freq)},
    }
