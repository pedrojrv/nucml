"""Data manipulation utilities for ENSDF data."""
import pandas as pd
import numpy as np
import logging
from sklearn import linear_model
import os
from joblib import dump
from functools import partial

import nucml.ensdf.plot as ensdf_plot
import nucml.general_utilities as gen_utils
from nucml.data_utils import copy_data_from_df_to_df
from nucml.data_utils import _filter_df_with_za


def _filter_df_with_za_and_sort_by_levels(df, Z, A):
    filtered = _filter_df_with_za(df, Z, A)
    filtered = filtered.sort_values(by='Level_Number', ascending=True)
    return filtered


def load_ensdf_samples(df, Z, A, scale=False, scaler=None, to_scale=[]):
    """Load ENSDF data for a particular isotope (Z, A).

    Args:
        df (DataFrame): DataFrame containing all necessary information for Z, A.
        Z (int): Number of protons.
        A (int): Mass Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted isotope sample.
    """
    sample = _filter_df_with_za_and_sort_by_levels(df, Z, A)
    if scale:
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def load_ensdf_element(df, Z, scale=False, scaler=None, to_scale=[]):
    """Load ENSDF data for a given element (Z).

    Args:
        df (DataFrame): DataFrame containing all necessary information for Z, A.
        Z (int): Number of protons.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    sample = df[(df["Z"] == Z)]
    if scale:
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def append_ensdf_levels(tot_num_levels, df, Z, A, log=False, scale=False, scaler=None, to_scale=[]):
    """Expand the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample.

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample.
        Z (int): Number of protons.
        A (int): Mass Number.
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number": np.arange(1, tot_num_levels + 1)})
    isotope_exfor = load_ensdf_samples(df, Z, A)
    new_data = copy_data_from_df_to_df(isotope_exfor, new_data, start=2)
    if scale:
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data


def append_ensdf_levels_nodata(tot_num_levels, df, log=False, scaler=None, to_scale=[]):
    """Expand the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample.

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample.
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number": np.arange(1, tot_num_levels + 1)})
    isotope_exfor = df.copy()
    if "Energy" in isotope_exfor.columns:
        isotope_exfor = isotope_exfor.drop(columns="Energy")
    new_data = copy_data_from_df_to_df(isotope_exfor, new_data, start=1)
    if scaler is not None:
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data


def append_ensdf_levels_range(tot_num_levels, df, Z, A, steps=1, log=False, scale=False, scaler=None, to_scale=[]):
    """Expand the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample.

    It uses a range with n steps rather than linear.

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample.
        Z (int): Number of protons.
        A (int): Mass Number.
        steps (int): Number of intermediate steps between 1 and tot_num_levels to create.
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number": np.arange(1, tot_num_levels + 1, steps)})
    isotope_exfor = load_ensdf_samples(df, Z, A)
    new_data = copy_data_from_df_to_df(isotope_exfor, new_data, start=2)
    if scale:
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data


def _extrapolate_to_upper_level(model, append_data_fn, pred, tot_num_levels, upper_energy_mev, it_limit):
    last_energy = pred.Energy.values[-1]
    number_levels = tot_num_levels
    upper_limit = np.log10(upper_energy_mev)
    x = 0

    while last_energy < upper_limit:
        number_levels = number_levels + 100
        simple = append_data_fn(number_levels)
        pred = pd.DataFrame()
        pred["Level_Number"] = simple.Level_Number
        pred["Energy"] = model.predict(pred)
        last_energy = pred.Energy.values[-1]
        x = x + 1
        if x == it_limit:
            logging.info("Iteration limit reached. Target energy not reached.")
            break

    return pred


def generate_level_density_csv(df, Z, A, nodata=False, upper_energy_mev=20, get_upper=False, tot_num_levels=0,
                               it_limit=500, plot=False, save=False, saving_dir=""):
    """Fit a linear model to the isotopic sample provided.

    It can save a CSV file with the linear model values for each energy level avaliable. If get_upper is True, then a
    new level number will be appended until the linear model predicts a value above the upper_energy_mev value.

    Args:
        df (DataFrame): DataFrame containing the needed data for isotope Z, A.
        Z (int): Number of protons.
        A (int): Mass number.
        nodata (bool, optional): If True, it assumes there is no avaliable data for the queried isotope. Defaults to
            False.
        upper_energy_mev (int, optional): If get_upper is True, the algorithm will iterate until this energy is reached.
            Defaults to 20 MeV.
        get_upper (bool, optional): If True, more levels will be added until the level energy in the level density
            reaches the 20 MeV mark. Defaults to False.
        tot_num_levels (int, optional): If any value other than 0 is given, it will append the remaining energy levels
            until reaching tot_num_levels. Defaults to 0.
        it_limit (int, optional): Sets the iteration limits for the linear model to reach the upper_energy_mev value.
            Defaults to 500.
        plot (bool, optional): If True, a plot of the linear model along the experimental levels will be rendered.
            Defaults to False.
        save (bool, optional): If True, the resulting DataFrame using the linear model will be saved. Defaults to False.
        saving_dir (str, optional): Path-like string pointing towars the directory where the DataFrame will be saved.
            Defaults to "".

    Returns:
        DataFrame: New DataFrame with Level Number and Level Energy as predicted by the linear model.
    """
    original = df.copy() if nodata else load_ensdf_samples(df, Z, A)
    if nodata:
        append_data_fn = partial(append_ensdf_levels_nodata, df.copy(), log=True)
    else:
        append_data_fn = partial(append_ensdf_levels, df.copy(), Z, A, log=True)

    element = original.Element_w_A.values[0]
    simple = append_data_fn(tot_num_levels) if tot_num_levels else original.copy()

    original = original[["Level_Number", "Energy"]]
    simple = simple[["Level_Number"]]

    reg = linear_model.LinearRegression()
    reg.fit(original.drop("Energy", 1), original.Energy)

    pred = pd.DataFrame()
    pred["Level_Number"] = simple.Level_Number
    pred["Energy"] = reg.predict(pred)

    if get_upper:
        _extrapolate_to_upper_level(reg, append_data_fn, pred, tot_num_levels, upper_energy_mev, it_limit)

    if plot:
        ensdf_plot.level_density_ml(original, pred, log_sqrt=False, log=True)

    if save:
        pred["A"] = A
        pred["Z"] = Z
        pred["Element_w_A"] = element
        pred["Level_Number"] = 10**pred.Level_Number.values
        pred["Energy"] = 10**pred.Energy.values
        pred["Level_Number"] = pred.Level_Number.astype(int)
        gen_utils.initialize_directories(saving_dir)
        pred.to_csv(os.path.join(saving_dir, "{}_Level_Density.csv".format(element)), index=False)
        dump(reg, os.path.join(saving_dir, '{}_NLD_linear_model.joblib'.format(element)))
    return pred


def get_level_density(energy_mev, df):
    """Given an energy density DataFrame, the level density at a wanted energy is returned.

    Args:
        energy_mev (float): Energy point at which the level density is to be returned.
        df (DataFrame): Level Density DataFrame to interpolate at energy_mev.

    Returns:
        float: Level density at "energy_mev".
    """
    to_append = pd.DataFrame({"Level_Number": [np.nan], "Energy": [np.log10(energy_mev)], "N": [np.nan]})
    to_interpolate = df.append(to_append, ignore_index=True)
    to_interpolate = to_interpolate.sort_values(by="Energy")
    new_index = len(to_interpolate) - 1
    to_interpolate = to_interpolate.interpolate()
    level_density = to_interpolate.loc[new_index]["N"]
    return level_density
