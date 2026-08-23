import numpy as np
import pandas as pd
import scipy.sparse as sp
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

seed = 123


class FitResult(tuple):
    """A 4-tuple ``(bst, score, X_train, y_train)`` carrying extra attributes.
    bst, b_acc, X, y = fit_xgb(adata_hv, .2, 6, 6)

    Callers can read ``res.X_test`` / ``res.y_test`` ::

        res = fit_xgb(adata_hv, .2, 6, 6)
        bst, b_acc, X, y = res            
        preds = res.preds                 # predictions on res.X_test
    """

    def __new__(cls, bst, score, X_train, y_train, X_test, y_test, preds, feature_names):
        obj = super().__new__(cls, (bst, score, X_train, y_train))
        obj.X_test = X_test
        obj.y_test = y_test
        obj.preds = preds
        obj.feature_names = feature_names
        return obj


class NamedCSR(sp.csr_matrix):
    """csr_matrix that reliably carries a ``.columns`` pandas Index.
    """

    @property
    def columns(self):
        return getattr(self, '_columns', None)

    @columns.setter
    def columns(self, value):
        self._columns = pd.Index(value)


def _feature_names(data):
    """Validated, XGBoost-compatible feature names taken from ``data.var_names``."""
    names = pd.Index(data.var_names).astype(str).tolist()
    if len(set(names)) != len(names):
        dupes = pd.Series(names).value_counts()
        dupes = dupes[dupes > 1].index.tolist()[:5]
        raise ValueError(
            f'var_names must be unique for XGBoost feature names; duplicates include {dupes}'
        )
    return names


def fit_xgb(data, test_size, n_estimators, max_depth):
    """Fit an XGBClassifier on ``data.X`` predicting ``data.obs['label']``.

    Returns a :class:`FitResult`, which unpacks as the historical
    ``(bst, score, X_train, y_train)`` 4-tuple.

    Feature names: XGBoost only learns column names from objects that actually
    expose them (pandas DataFrame).  A scipy sparse matrix does not, so the
    names are attached to the fitted booster explicitly afterwards, which is
    what makes ``bst.feature_names_in_`` resolvable.  The sparse input is kept
    sparse.
    """
    feature_names = _feature_names(data)

    X = data.X
    if sp.issparse(X):
        # Row-slicing a CSC matrix (this dataset is stored gene-major) is slow;
        # convert once before the split rather than twice after it.  The split
        # itself is index-based, so this does not change which cells land where.
        X = X.tocsr()

    X_train, X_test, y_train, y_test = train_test_split(X, data.obs['label'], test_size=test_size,
                                                        stratify=data.obs['label'], random_state=seed)
    n_classes = data.obs['label'].nunique()

    if sp.issparse(X_train):
        X_train = NamedCSR(X_train)
        X_test = NamedCSR(X_test)
        X_train.columns = pd.Index(feature_names)
        X_test.columns = pd.Index(feature_names)
    else:
        X_train = pd.DataFrame(np.asarray(X_train), columns=feature_names)
        X_test = pd.DataFrame(np.asarray(X_test), columns=feature_names)
    
    # create model instance
    xgb_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.05,
        importance_type='weight',
        tree_method='hist',
        n_jobs=2,
    )
    if n_classes > 2:
        xgb_params.update(objective='multi:softprob', num_class=n_classes)
    else:
        xgb_params.update(objective='binary:logistic')

    bst = XGBClassifier(**xgb_params)
    # fit model
    bst.fit(X_train, y_train)
    # Attach feature names to the fitted booster.  This is the only route that
    # makes `bst.feature_names_in_` (a read-only property that reads straight
    # from the booster) resolvable when X was sparse, and it is also what
    # xgb.plot_importance and shap use for axis labels.
    bst.get_booster().feature_names = feature_names
    assert list(bst.feature_names_in_) == feature_names, 'feature names did not reach the booster'
    # make predictions
    preds = bst.predict(X_test)

    score = round(
        balanced_accuracy_score(
            y_test, 
            preds, 
        ), 
        3
    )
    return FitResult(bst, score, X_train, y_train, X_test, y_test, preds, feature_names)
