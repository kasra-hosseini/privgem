#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc as sklearn_auc

from typing import Union

def performance_classification(X_train: Union[list, np.ndarray], 
                               y_train: Union[list, np.ndarray], 
                               X_test: Union[list, np.ndarray], 
                               y_test: Union[list, np.ndarray], 
                               model_imp=RandomForestClassifier()):
    """Compute various performance metrics for classification

    Parameters
    ----------
    X_train : Union[list, np.ndarray]
    y_train : Union[list, np.ndarray]
    X_test : Union[list, np.ndarray]
    y_test : Union[list, np.ndarray]
    """
    model_imp.fit(X_train, y_train)
    
    probs = model_imp.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    yhat = model_imp.predict(X_test)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    f1 = f1_score(y_test, yhat)
    auc = sklearn_auc(recall_curve, precision_curve)

    return f1, auc


def feature_importance(X: Union[list, np.ndarray], 
                       y: Union[list, np.ndarray], 
                       model_imp=RandomForestClassifier(), 
                       repeats: int=10,
                       random_state=42,
                       n_jobs=-1):
    """Feature importance (builtin and permutation)

    Parameters
    ----------
    X : Union[list, np.ndarray]
    y : Union[list, np.ndarray]
    model_imp : model to compute feature importance
    repeats : int, optional
    """
    model_imp.fit(X, y)

    try:
        print(f"[INFO] Extract feature_importances_")
        imp_builtin = model_imp.feature_importances_
    except:
        print(f"[INFO] feature_importances_ was not found.")
        print(f"[INFO] Extract coef_[0]")
        imp_builtin = model_imp.coef_[0]
    
    imp_perm = permutation_importance(
        model_imp, 
        X, y, 
        n_repeats=repeats, 
        random_state=random_state, 
        n_jobs=n_jobs)
    
    return imp_builtin, imp_perm

def cosine_sim(l1: Union[list, np.ndarray], 
               l2: Union[list, np.ndarray]):
    """Cosine similarity between l1 and l2

    Parameters
    ----------
    l1 : Union[list, np.ndarray]
    l2 : Union[list, np.ndarray]
    """
    return (1. - cosine_dist(l1, l2))

def cosine_dist(l1: Union[list, np.ndarray], 
                l2: Union[list, np.ndarray]):
    """Cosine distance between l1 and l2

    Parameters
    ----------
    l1 : Union[list, np.ndarray]
    l2 : Union[list, np.ndarray]
    """
    return distance.cosine(l1, l2)

def L2_norm_dist(l1: Union[list, np.ndarray],
                 l2: Union[list, np.ndarray]):
    """L2 norm distance between l1 and l2

    Parameters
    ----------
    l1 : Union[list, np.ndarray]
    l2 : Union[list, np.ndarray]
    """

    if isinstance(l1, list):
        l1 = np.array(l1)
    if isinstance(l2, list):
        l2 = np.array(l2)
    
    l1d = np.sqrt(np.sum(l1**2))
    l2d = np.sqrt(np.sum(l2**2))

    if (l1d == 0) or (l2d == 0):
        return None
    
    l1_norm = l1/l1d
    l2_norm = l2/l2d
    
    return np.sqrt(np.sum((l1_norm - l2_norm)**2))

def extract_X_y(df, label_col="label"):
    """Extract X and y from a dataframe
    """
    
    # read list of columns and remove label_col
    list_X_cols = list(df.columns)
    list_X_cols.remove(label_col)

    X = df[list_X_cols].to_numpy()
    y = df[label_col].to_numpy()
    return X, y