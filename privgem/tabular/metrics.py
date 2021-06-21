#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance
from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import auc as sklearn_auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

import shap 

import torch
import torch.nn.functional as F

def performance_classification(X_train: Union[list, np.ndarray], 
                               y_train: Union[list, np.ndarray], 
                               X_test: Union[list, np.ndarray], 
                               y_test: Union[list, np.ndarray], 
                               model_imp=RandomForestClassifier(),
                               pipe_classifier_name: Union[str, None]=None,
                               feature_importance_methods: Union[list]=["builtin", "permutation", "shap"]):
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

    try:
        if not "builtin" in feature_importance_methods: raise
        if pipe_classifier_name is None:
            features_imp = model_imp.feature_importances_
        else:
            features_imp = model_imp[pipe_classifier_name].feature_importances_
    except Exception:
        features_imp = None
    
    try:
        if not "permutation" in feature_importance_methods: raise
        feature_imp_perm = permutation_importance(model_imp, X_test, y_test)
        feature_imp_perm = feature_imp_perm["importances_mean"]
    except Exception:
        feature_imp_perm = None

    try:
        if not "shap" in feature_importance_methods: raise
        shap_explainer = shap.TreeExplainer(model=model_imp["classifier"])
        shap_values = shap_explainer.shap_values(X_test)
    
        # version that uses the interventional perturbation option (takes into account a background dataset
        # fm_train) - throws errors in some cases which can be suppressed by setting check_additivity=False
        # in explainer.shap_values(). It is also slower.
        # explainer = shap.TreeExplainer(model=clf, data=fm_train, feature_perturbation='interventional')
        # shap_values = explainer.shap_values(fm_test, check_additivity=False)
        feature_imp_shap = sum(np.abs(shap_values).mean(0))
    except Exception:
        feature_imp_shap = None

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    f1 = f1_score(y_test, yhat)
    auc = sklearn_auc(recall_curve, precision_curve)
    roc_auc = roc_auc_score(y_test, probs)

    return f1, auc, roc_auc, features_imp, feature_imp_perm, feature_imp_shap


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

def kl_div(p_test: Union[tuple, list, np.ndarray], 
           p_targ: Union[tuple, list, np.ndarray]):
    """Compute KL divergence between two lists/arrays

    Parameters
    ----------
    p_test : Union[tuple, list, np.ndarray]
    p_targ : Union[tuple, list, np.ndarray]
    """
    p_test = F.softmax(torch.tensor(p_test), dim=0)
    p_targ = F.softmax(torch.tensor(p_targ), dim=0)
    return F.kl_div(p_test.log(), p_targ, reduction="batchmean")

def create_pipeline(num_columns: list,
                    cat_columns: list,
                    categories: Union[list, tuple],
                    inp_classifer=RandomForestClassifier(),
                    inp_cat_encoder: str="ordinal",
                    inp_cat_imputer: str="most_frequent"):
    """Create a pipeline

    Parameters
    ----------
    num_columns : list
        Numeric columns
    cat_columns : list
    categories : Union[list, tuple]
    inp_classifer : [type], optional
    inp_cat_encoder : str, optional
    inp_cat_imputer : str, optional
    """

    if inp_cat_encoder == "ordinal":
        cat_encoder = OrdinalEncoder(categories=categories)
    elif inp_cat_encoder == "onehot":
        cat_encoder = OneHotEncoder(categories=categories, 
                                    handle_unknown='ignore')
    else:
        raise NotImplementedError(f"Method: {inp_cat_encoder} is not implemented")
        
    # preprocessing
    preproc = Pipeline([
        ("cat_imputer", SimpleImputer(strategy=inp_cat_imputer)),
        ("scaler", StandardScaler())
    ])

    ct = ColumnTransformer([
        # Transformer for categorical variables
        ("cat_encoder", cat_encoder, cat_columns),
        # Transformer for numeric variables
        ("num_preproc", preproc, num_columns)
        ], 
        # variables not specified in the indices?
        remainder="drop")

    pipe = Pipeline([
        ("preprocessing", ct),
        ("classifier", inp_classifer)
    ])
    
    return pipe
