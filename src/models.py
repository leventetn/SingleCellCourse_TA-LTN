import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

seed = 123


def fit_xgb(data, test_size, n_estimators, max_depth):
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.obs['label'], test_size=test_size,
                                                        stratify=data.obs['label'], random_state=seed)

    X_train = pd.DataFrame.sparse.from_spmatrix(X_train, columns=data.var_names)
    X_test = pd.DataFrame.sparse.from_spmatrix(X_test, columns=data.var_names)
    
    # create model instance
    bst = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05, objective='multi:softprob', importance_type='weight')
    # fit model
    bst.fit(X_train, y_train)
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
