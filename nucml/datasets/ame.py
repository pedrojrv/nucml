"""Data loading functions.

Contains the main utility functions to load different datasets including EXFOR, AME, ENDF, ENSDF, and more.
"""

import os
import logging
import pandas as pd

import nucml.config as config
import nucml.general_utilities as gen_utils

logging.basicConfig(level=logging.INFO)

ame_dir_path = config.ame_dir_path
evaluations_path = config.evaluations_path
ensdf_path = config.ensdf_path
exfor_path = config.exfor_path

dtype_exfor = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/EXFOR_AME_dtypes.pkl'))
exfor_elements = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/exfor_elements_list.pkl'))
elements_dict = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/Element_AAA.pkl'))


def load_ame(natural=False, imputed_nan=False, file="merged"):
    """Load the Atomic Mass Evaluation 2016 data generated by NucML using the parsing utilities.

    For file="merged", there are four AME dataset versions:
    1. AME_all_merged (natural=False, imputed_nan=False): Contains all avaliable AME information from the mass, rct1,
        and rct2 files.
    2. AME_all_merged_no_NaN (natural=False, imputed_nan=True): Same as 1, except all missing values are imputed
        linearly and element-wise.
    3. AME_Natural_Properties_w_NaN (natural=True, imputed_nan=False): Similar to 2, except data for natural abundance
        elements is included.
    4. AME_Natural_Properties_no_NaN (natural=True, imputed_nan=True): Same as 3. except all missing values are imputed
        linearly and element-wise.

    Args:
        natural (bool): if True, the AME data containing natural element data will be loaded. Only applicable when
            file='merged'.
        imputed_nan (bool): If True, the dataset loaded will not contain any missing values (imputed version will be
            loaded).
        file (str): Dataset to extract. Options include 'merged', 'mass16', 'rct1', and 'rct2'.

    Returns:
        DataFrame: a pandas dataframe cantaining the queried AME data.
    """
    if file.lower() == "merged":
        suffix = "_Natural_Properties" if natural else "_all_merged"
        suffix += "_no_NaN" if imputed_nan else "_w_NaN"
        ame_file_path = os.path.join(ame_dir_path, f"AME_{suffix}.csv")
        logging.info("AME: Reading and loading Atomic Mass Evaluation files from: \n {}".format(ame_file_path))
        ame = pd.read_csv(ame_file_path)
        ame[["N", "Z", "A"]] = ame[["N", "Z", "A"]].astype(int)
    elif file.lower() in ["mass16", "rct1", "rct2"]:
        ame_file_path = os.path.join(ame_dir_path, "AME_{}.csv".format(file))
        logging.info("AME: Reading and loading the Atomic Mass Evaluation file from: \n {}".format(ame_file_path))
        ame = pd.read_csv(ame_file_path)
    return ame