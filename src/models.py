import pandas as pd
import scipy.sparse as sp
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

seed = 123


def fit_xgb(data, test_size, n_estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.obs['label'], test_size=test_size,
                                                        stratify=data.obs['label'], random_state=seed)
    n_classes = data.obs['label'].nunique()

    if sp.issparse(X_train):
        X_train = X_train.tocsr()
        X_test = X_test.tocsr()
        X_train.columns = pd.Index(data.var_names)
        X_test.columns = pd.Index(data.var_names)
    else:
        X_train = pd.DataFrame(X_train, columns=data.var_names)
        X_test = pd.DataFrame(X_test, columns=data.var_names)
    
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
    bst.get_booster().feature_names = data.var_names.astype(str).tolist()
    # make predictions
    preds = bst.predict(X_test)

    score = round(
        balanced_accuracy_score(
            y_test, 
            preds, 
        ), 
        3
    )
    return(bst, score, X_train, y_train)
