import os
import pandas as pd
import sys
import logging

# This allows us to import the nucml utilities
sys.path.append("..")
sys.path.append("../..")

import nucml.datasets as nuc_data                   
import nucml.general_utilities as gen_utils        

elements_dict = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/Element_AAA.pkl'))


def load_new(datapath, mev_to_ev=False):
    """Loads new ENDF data from a given filepath. The function assumes the file is readable with 
    pd.read_csv() and that the new ENDF file contains a column named "Energy".

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


def get_for_exfor(Z, A, MT, mode="neutrons", library="endfb8.0", mev_to_ev=True, mb_to_b=True, log=True, drop_u=True):
    """Gets the queried ENDF data for EXFOR functions.

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
    element_for_endf = list(elements_dict.keys())[list(elements_dict.values()).index(Z)] + str(A).zfill(3)
    endf = nuc_data.load_evaluation(element_for_endf, MT, mode=mode, library=library, mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=log, drop_u=drop_u)
    return endf
