"""Data maniuplation utilities for evaluation datasets."""
import pandas as pd
import logging

from nucml import general_utilities
import nucml.datasets as nuc_data

elements_dict = nuc_data.elements_dict


def load_new(datapath, mev_to_ev=False):
    """Load new ENDF data from a given filepath.

    The function assumes the file is readable with pd.read_csv() and that the new ENDF file contains a column named
    "Energy".

    Args:
        datapath (str): Path-like string to new ENDF data file.
        mev_to_ev (bool, optional): Converts the energy column to eV. Defaults to False.

    Returns:
        DataFrame: Contains the new ENDF data.
    """
    endf = pd.read_csv(datapath)
    if mev_to_ev:
        endf["Energy"] = endf["Energy"]*1E6
    logging.info("Finish reading ENDF data with shape: {}".format(endf.shape))
    return endf


def get_for_exfor(ZZZAAA, MT, mode="neutrons", library="endfb8.0", mev_to_ev=True, mb_to_b=True, log=True):
    """Get the queried ENDF data for EXFOR functions.

    Note: Internal Function.

    Args:
        Z (int): Number of protons
        A (int): Mass number
        MT (int): Reaction type as an ENDF MT code integer.
        mode (str): Projectile of the reaction of interest. Only "neutrons" and "protons" is allowed for now.
        library (str): Evaluation library to query. Allowed options include endfb8.0, jendl4.0, jeff3.3, and tendl.2019.
        mev_to_ev (bool): If True, it converts the energy from MeV to eV.
        mb_to_b (bool): If True, it converts the cross sections from millibarns to barns.
        log (bool, optional): Apply log transformation to the Energy and Data features. Defaults to True.

    Returns:
        DataFrame: Contains the evaluation dataframe.
    """
    Z, A = general_utilities.parse_zzzaaa(ZZZAAA)
    element_for_endf = list(elements_dict.keys())[list(elements_dict.values()).index(Z)] + str(A).zfill(3)
    endf = nuc_data.load_evaluation(
        element_for_endf, MT, mode=mode, library=library, mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=log, drop_u=True)
    return endf
