"""Querying utilities for the EXFOR database."""
import logging
import numpy as np
import pandas as pd

import nucml.general_utilities as gen_utils
from nucml.exfor.data_utilities import _copy_over_data_and_scale


def _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, MT=None, scaler=None, to_scale=[]):
    if one_hot:
        sample = df[(df["Z"] == Z) & (df["A"] == A) & (df["Element_Flag_" + nat_iso] == 1)]
    else:
        sample = df[(df["Z"] == Z) & (df["A"] == A) & (df["Element_Flag"] == nat_iso)]

    if MT is not None:
        sample = sample[sample[MT] == 1] if one_hot else sample[sample["MT"] == MT]
    if scaler:
        sample[to_scale] = scaler.transform(sample[to_scale])
    return sample.sort_values(by='Energy', ascending=True)


def load_samples(df, Z, A, MT, nat_iso="I", one_hot=False, scaler=None, to_scale=[], mt_for="EXFOR"):
    """Extract datapoints belonging to a particular isotope-reaction channel pair.

    Args:
        df (DataFrame): DataFrame containing all avaliable datapoints from where to extract the information.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (ENDF-coded).
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    MT = gen_utils.parse_mt(MT, mt_for=mt_for, one_hot=one_hot)
    return _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, MT=MT, scaler=scaler, to_scale=to_scale)


def load_isotope(df, Z, A, nat_iso="I", one_hot=False, scaler=None, to_scale=[]):
    """Load all datapoints avaliable for a particular isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    return _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, scaler=scaler, to_scale=to_scale)


def load_element(df, Z, nat_iso="I", one_hot=False, scale=False, scaler=None, to_scale=[]):
    """Load all datapoints avaliable for a particular element.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    if one_hot:
        sample = df[(df["Z"] == Z) & (df["Element_Flag_" + nat_iso] == 1)].sort_values(by='Energy', ascending=True)
    else:
        sample = df[(df["Z"] == Z) & (df["Element_Flag"] == nat_iso)].sort_values(by='Energy', ascending=True)
    if scale:
        sample[to_scale] = scaler.transform(sample[to_scale])
    return sample


def load_newdata(datapath, df, Z, A, MT, nat_iso="I", one_hot=False, log=False, scaler=None, to_scale=[]):
    """Load new measurments and appends the appropiate EXFOR isotopic data.

    Assumes new data only have two columns: Energy and Data.

    Args:
        datapath (str): Path-like string of the new data file.
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        log (bool, optional): If True, the log of both the Energy and Data features will be taken.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    new_data = pd.read_csv(datapath)
    if log:
        new_data["Energy"] = np.log10(new_data["Energy"])
        new_data["Data"] = np.log10(new_data["Data"])
    isotope_exfor = load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)
    new_data = _copy_over_data_and_scale(isotope_exfor, new_data, scaler, to_scale)
    logging.info("EXFOR extracted DataFrame has shape: {}".format(new_data.shape))
    return new_data
