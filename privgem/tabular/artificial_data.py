#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import string
from typing import Union

from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer

def make_table(n_samples: int=1000,
               n_classes: int=2,
               class_weights: Union[list, tuple]=[0.5, 0.5],
               n_clusters_per_class: int=1,
               n_features: int=5, 
               n_informative: int=5, 
               n_redundant: int=0, 
               n_repeated: int=0,
               n_categorical: int=5,
               n_categorical_bins: Union[list, tuple, None]=None,
               flip_y: float=0.0, 
               class_sep: float=1.0, 
               hypercube: bool=True, 
               shift: float=0.0, 
               scale: float=1.0, 
               shuffle: bool=False,
               categorical_encode: str='ordinal',
               categorical_strategy: str='uniform',
               categorical_map_string: str='digits',
               random_state: int=123):
    """Make artifical data with mixed type columns

    Parameters
    ----------
    n_samples : int, optional
    n_classes : int, optional
    class_weights : Union[list, tuple], optional
    n_clusters_per_class : int, optional
    n_features : int, optional
    n_informative : int, optional
    n_redundant : int, optional
    n_repeated : int, optional
    n_categorical : int, optional
    n_categorical_bins : Union[list, tuple, None], optional
    flip_y : float, optional
    class_sep : float, optional
    hypercube : bool, optional
    shift : float, optional
    scale : float, optional
    shuffle : bool, optional
    categorical_encode : str, optional
    categorical_strategy : str, optional
    categorical_map_string : str, optional
    random_state : int, optional
    """

    X, y = make_classification(n_samples=n_samples, 
                               n_classes=n_classes,
                               weights=class_weights, 
                               n_clusters_per_class=n_clusters_per_class,
                               n_features=n_features, 
                               n_informative=n_informative, 
                               n_redundant=n_redundant, 
                               n_repeated=n_repeated,
                               flip_y=flip_y, 
                               class_sep=class_sep, 
                               hypercube=hypercube, 
                               shift=shift, 
                               scale=scale, 
                               shuffle=shuffle,
                               random_state=random_state)
    
    # number of continuous columns
    n_continuous = n_features - n_categorical
    all_categories = []
    
    if n_categorical > 0:
        if n_categorical_bins is None:
            n_categorical_bins = [5]*n_categorical
            
        for icol in range(n_categorical):
            binner = KBinsDiscretizer(n_bins=n_categorical_bins[icol], 
                                      encode=categorical_encode, 
                                      strategy=categorical_strategy)
            # make categorical columns after n_continuous columns
            X[:, n_continuous+icol] = \
                binner.fit_transform(X[:, n_continuous+icol].astype(float).reshape(-1, 1)).reshape(-1)
            
            # collect all categories
            all_categories.append([str(cat) for cat in range(n_categorical_bins[icol])])
    
    # Create a dataframe
    X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    X.iloc[:, n_continuous:] = X.iloc[:, n_continuous:].astype(int)
    
    if n_categorical > 0:
        if categorical_map_string == "ascii":
            map2str = string.ascii_letters
            # map values to another value with .map(dict)
            X.iloc[:, n_continuous:] = \
                X.iloc[:, n_continuous:].apply(lambda x: x.map({i:letter for i,letter in enumerate(map2str)}))
        else:
            X.iloc[:, n_continuous:] = X.iloc[:, n_continuous:].astype("str")

        X.iloc[:, n_continuous:] = X.iloc[:, n_continuous:].astype("category")
    
    X.sort_index(axis=1, inplace=True)
    
    return X, y, all_categories