"""Utility functions for processing nucml's various datasets."""

import logging
import numpy as np
import pandas as pd
from joblib import load
from scipy.optimize import curve_fit
from sklearn import preprocessing

from nucml.general_utilities import func  # pylint: disable=import-error

pd.options.mode.chained_assignment = None  # default='warn'


def impute_values(df):
    """Imputes feature values using linear interpolation element-wise.

    The passed dataframe must contain both the number of protons and mass number as "Z" and "A" respetively.

    Args:
        df (pd.DataFrame): DataFrame to impute values off. All missing values will be filled.

    Returns:
        pd.DataFrame: New imputed DataFrame.
    """
    for i in range(0, 119):
        df[df["Z"] == i] = df[df["Z"] == i].sort_values(by="A").interpolate()

        if len(df[df["Z"] == i]) > 1:
            fit_df_original = df[df["Z"] == i].sort_values(by="A").reset_index(drop=True).copy()
            fit_df = fit_df_original.copy()

            col_params = {}
            guess = (0.5, 0.5)

            # Curve fit each column
            for col in fit_df.select_dtypes(np.number).columns:
                if len(fit_df[col].dropna()) > 1:  # SHOULD IT BE 0?
                    # Get x & y
                    x = fit_df[col].dropna().index.astype(float).values
                    y = fit_df[col].dropna().values
                    # Curve fit column and get curve parameters
                    params = curve_fit(func, x, y, guess)
                    # Store optimized parameters
                    col_params[col] = params[0]

            # Extrapolate each column
            for col in col_params.keys():
                # Get the index values for NaNs in the column
                x = fit_df_original[pd.isnull(fit_df_original[col])].index.astype(float).values
                # Extrapolate those points with the fitted function
                fit_df_original[col][x] = func(x, *col_params[col])

            df[df["Z"] == i] = fit_df_original.values
    return df


def normalize_features(df, to_scale, scaling_type="standard", scaler_dir=None):
    """Apply a transformer or normalizer to a set of specific features in the provided dataframe.

    Args:
        df (pd.DataFrame): DataFrame to normalize/transform.
        to_scale (list): List of columns to apply the normalization to.
        scaling_type (str): Scaling or transformer to use. Options include "poweryeo", "standard",
            "minmax", "maxabs", "robust", and "quantilenormal". See the scikit-learn documentation
            for more information on each of these.
        scaler_dir (str): Path-like string to a previously saved scaler. If provided, this overides
            any other parameter by loading the scaler from the provided path and using it to
            transform the provided dataframe. Defaults to None.

    Returns:
        object: Scikit-learn scaler object.
    """
    if scaler_dir is not None:
        logging.info("Using previously saved scaler.")
        scaler_object = load(open(scaler_dir, 'rb'))
    else:
        logging.info("Fitting new scaler.")
        if scaling_type == "poweryeo":
            scaler_object = preprocessing.PowerTransformer().fit(df[to_scale])
        elif scaling_type == "standard":
            scaler_object = preprocessing.StandardScaler().fit(df[to_scale])
        elif scaling_type == "minmax":
            scaler_object = preprocessing.MinMaxScaler().fit(df[to_scale])
        elif scaling_type == "maxabs":
            scaler_object = preprocessing.MaxAbsScaler().fit(df[to_scale])
        elif scaling_type == 'robust':
            scaler_object = preprocessing.RobustScaler().fit(df[to_scale])
        elif scaling_type == 'quantilenormal':
            scaler_object = preprocessing.QuantileTransformer(output_distribution='normal').fit(df[to_scale])
    return scaler_object
