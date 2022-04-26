"""Error calculation relative to ML or Evalutions."""
import pandas as pd
import numpy as np

from nucml.exfor import data_utilities, querying_utils
import nucml.model.utilities as model_utils
from nucml.evaluation.data_utilities import get_for_exfor


def get_mt_errors_exfor_ml(df, Z, A, scaler, to_scale, model):
    """Calculate the error between EXFOR and ML predictions for a given isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.
        model (object): Machine learning object.

    Returns:
        DataFrame
    """
    exfor_isotope_cols = data_utilities._get_isotope_df_cols(df, Z, A, scaler, to_scale)
    error_results = pd.DataFrame(columns=['MT', 'MAE', 'MSE', 'EVS', 'MAE_M', 'R2'])
    for col in exfor_isotope_cols.columns:
        if "MT" not in col:
            continue
        exfor_sample = querying_utils.load_samples(
            df, Z, A, col, nat_iso="I", one_hot=True, scaler=scaler, to_scale=to_scale)
        error_dict = model_utils.regression_error_metrics(
            model.predict(exfor_sample.drop(columns=["Data"])), exfor_sample.Data)
        error_results = error_results.append(pd.DataFrame({
            "MT": [col], "MAE": [error_dict["mae"]],
            "MSE": [error_dict["mse"]], "EVS": [error_dict["evs"]],
            "MAE_M": [error_dict["mae_m"]], "R2": [error_dict["r2"]]}))
    return error_results


def get_mt_error_exfor_endf(df, Z, A, scaler, to_scale):
    """Calculate the error between EXFOR and ENDF for a given isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.

    Returns:
        DataFrame
    """
    # TODO: FIND OUT IF WE NEED SCALER OR TO SCALE
    # We don't care if its scaled or not since we only care abuot the Data feature, but we still need it?
    exfor_isotope_cols = data_utilities._get_isotope_df_cols(df, Z, A, scaler, to_scale)
    error_results = pd.DataFrame(columns=['id', 'mae', 'mse', 'evs', 'mae_m', 'r2', 'MT'])
    for col in exfor_isotope_cols.columns:
        if "MT" not in col or col in ["MT_101", "MT_9000"]:
            continue
        exfor_sample = querying_utils.load_samples(
            df, Z, A, col, nat_iso="I", one_hot=True, scaler=scaler, to_scale=to_scale)
        endf_data = get_for_exfor(Z, A, col)
        _, error_exfor_endf = get_error_endf_exfor(endf_data, exfor_sample)
        error_exfor_endf["MT"] = col
        error_results = error_results.append(error_exfor_endf)
    return error_results


def get_error_endf_exfor(endf, df_sample, filter_energy=True):
    """Calculate the error between a given dataframe of experimental datapoints to ENDF.

    Args:
        endf (DataFrame): DataFrame containing the ENDF datapoints.
        df_sample (DataFrame): DataFrame containing the new experimental data points.

    Returns:
        DataFrame, DataFrame: first dataframe contains original values while the second one
            the calculated errors.
    """
    endf_copy = endf.copy()
    df = df_sample.copy()
    if filter_energy:
        df = df[df.Energy > endf_copy.Energy.min()]
    indexes = np.arange(len(endf), len(endf) + len(df))  # start our index numbering after len(endf)
    df.index = indexes  # This will return a dataframe with non zero index
    energy_interest = df[["Energy"]]  # energy_interest will carry previous indexes
    energy_interest["Data"] = np.nan
    endf_copy = endf_copy.append(energy_interest, ignore_index=False).sort_values(by=['Energy'])
    endf_copy["Data"] = endf_copy["Data"].interpolate(limit_direction="forward")

    # Measuring metrics on predictions.
    error_endf_exfor = model_utils.regression_error_metrics(df["Data"], endf_copy[["Data"]].loc[indexes])
    error_endf_exfor_df = model_utils.create_error_df("EXFOR VS ENDF", error_endf_exfor)

    exfor_endf = pd.DataFrame({
        "Energy": df.Energy.values,
        "EXFOR": df["Data"].values, "ENDF": endf_copy["Data"].loc[indexes].values})
    return exfor_endf, error_endf_exfor_df
