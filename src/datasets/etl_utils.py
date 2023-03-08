from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler


def preprocess_continous_features(cont_feature_names: List[str],
                                  cat_features_name: List[str],
                                  n_quantiles: int,
                                  output_distribution: str,
                                  df_train: pd.DataFrame,
                                  df_test: Optional[pd.DataFrame]=None,
                                  discretize: Optional[bool]=True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """ Continuous features preprocessing

    :param cont_feature_names: List of str, the names of the continuous features
    :param cat_features_name: List of str, the names of the categorical features
    :param n_quantiles: int
    :param output_distribution: str, 'normal' or 'uniform' -> output distribution of  QuantileTransformer
    :param df_train: pd.DataFrame to discretize
    :param df_test: pd.DataFrame to discretize
    :param discretize: bool, if True the continuous fatures are discretized
    :return: Tuple[pd.DataFrame, pd.DataFrame], the new version of df_train and df_test
    """
    # Change in the distribution
    if n_quantiles > 0:
        tr = QuantileTransformer(n_quantiles=n_quantiles, random_state=0, output_distribution=output_distribution)
    else:
        tr = StandardScaler()
    tr.fit(df_train[cont_feature_names])
    cont_train = tr.transform(df_train[cont_feature_names])
    if df_test is not None:
        cont_test = tr.transform(df_test[cont_feature_names])

    if len(cat_features_name) > 0:
        new_train = np.concatenate((cont_train, df_train[cat_features_name].values), axis=1).tolist()
        if df_test is not None:
            new_test = np.concatenate((cont_test, df_test[cat_features_name].values), axis=1).tolist()
        features_names = cont_feature_names + cat_features_name
    else:
        new_train = cont_train.tolist()
        if df_test is not None:
            new_test = cont_test.tolist()
        features_names = cont_feature_names

    x_train = pd.DataFrame(new_train, columns=features_names)
    if df_test is not None:
        x_test = pd.DataFrame(new_test, columns=features_names)

    # One value for each quantile
    if discretize:
        for c in cont_feature_names:
            c_min = x_train[c].values.min()
            c_max = x_train[c].values.max()
            cuant = (c_max - c_min) / n_quantiles
            for i in range(n_quantiles):
                for df in ([x_train, x_test] if df_test is not None else [x_train]):
                    df[c][(df[c] >= c_min + i * cuant) & (df[c] <= c_min + (i + 1) * cuant)] = round(c_min + i * cuant, 2)
    if df_test is not None:
        return x_train, x_test
    else:
        return x_train, None
