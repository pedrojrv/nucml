"""Utility functions for processing nucml's various datasets."""

import numpy as np
import pandas as pd
from joblib import load
from scipy.optimize import curve_fit
from sklearn import preprocessing

from pathlib import Path
from typing import Optional, List
from nucml.general_utilities import func

pd.options.mode.chained_assignment = None  # default='warn'


def _fit_curve_function_to_numerical_cols(fit_df: pd.DataFrame) -> dict:
    """Fits a linear function to each column of the passed DataFrame.

    Args:
        fit_df (pd.DataFrame): DataFrame to fit the function to.

    Returns:
        dict: Dictionary of column names and their fitted parameters.
    """
    # Curve fit each column
    col_params = {}
    guess = (0.5, 0.5)
    for col in fit_df.select_dtypes(np.number).columns:
        if len(fit_df[col].dropna()) <= 1:  # SHOULD IT BE 0?
            continue

        # Get x & y
        x = fit_df[col].dropna().index.astype(float).values
        y = fit_df[col].dropna().values
        # Curve fit column and get curve parameters
        params = curve_fit(func, x, y, guess)
        # Store optimized parameters
        col_params[col] = params[0]
    return col_params


def _interpolate_numerical_using_params(fit_df_original: pd.DataFrame, col_params: dict) -> pd.DataFrame:
    """Interpolates the passed DataFrame using the fitted parameters.

    Columns to be interpolated are specified in the col_params dictionary.

    Args:
        fit_df_original (pd.DataFrame): DataFrame to interpolate.
        col_params (dict): Dictionary of column names and their fitted parameters.

    Returns:
        pd.DataFrame: Interpolated DataFrame.
    """
    # Extrapolate each column
    for col in col_params.keys():
        # Get the index values for NaNs in the column
        x = fit_df_original[pd.isnull(fit_df_original[col])].index.astype(float).values
        # Extrapolate those points with the fitted function
        fit_df_original[col][x] = func(x, *col_params[col])
    return fit_df_original


def impute_values(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes feature values using linear interpolation element-wise.

    The passed dataframe must contain both the number of protons and mass number as "Z" and "A" respetively.

    Args:
        df (pd.DataFrame): DataFrame to impute values off. All missing values will be filled.

    Returns:
        pd.DataFrame: New imputed DataFrame.
    """
    for i in range(0, df.Z.max()):
        df[df["Z"] == i] = df[df["Z"] == i].sort_values(by="A").interpolate()

        if len(df[df["Z"] == i]) <= 1:
            continue

        fit_df_original = df[df["Z"] == i].sort_values(by="A").reset_index(drop=True).copy()
        fit_df = fit_df_original.copy()

        col_params = _fit_curve_function_to_numerical_cols(fit_df)
        fit_df_original = _interpolate_numerical_using_params(fit_df_original, col_params)
        df[df["Z"] == i] = fit_df_original.values

    return df


scalers = {
    "poweryeo": preprocessing.PowerTransformer().fit,
    "standard": preprocessing.StandardScaler().fit,
    "minmax": preprocessing.MinMaxScaler().fit,
    "maxabs": preprocessing.MaxAbsScaler().fit,
    'robust': preprocessing.RobustScaler().fit,
    'quantilenormal': preprocessing.QuantileTransformer(output_distribution='normal').fit
}


def normalize_features(
        df: pd.DataFrame, to_scale: List[str], scaling_type: Optional[str] = "standard",
        scaler_dir: Optional[Path] = None):
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
    scalers = {
        "poweryeo": preprocessing.PowerTransformer(),
        "standard": preprocessing.StandardScaler(),
        "minmax": preprocessing.MinMaxScaler(),
        "maxabs": preprocessing.MaxAbsScaler(),
        'robust': preprocessing.RobustScaler(),
        'quantilenormal': preprocessing.QuantileTransformer(output_distribution='normal'),
    }
    if scaler_dir is not None:
        scaler_object = load(open(scaler_dir, 'rb'))
    else:
        if scaling_type not in scalers.keys():
            raise ValueError(f"Scaling type not supported. Only {list(scalers.keys())} are supported.")
        scaler_fn = scalers[scaling_type]
        scaler_object = scaler_fn.fit(df[to_scale])
    return scaler_object
