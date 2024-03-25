"""Data loading functions.

Contains the main utility functions to load different datasets including EXFOR, AME, ENDF, ENSDF, and more.
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import nucml.general_utilities as gen_utils
import nucml.processing as nuc_proc
from nucml import configure
from nucml.datasets.ame import load_ame


config = configure._get_config()
ensdf_path = config['DATA_PATHS']['ENSDF']


def load_ensdf_headers():
    """Load ENSDF headers from RIPL .dat files.

    Returns:
        DataFrame
    """
    csv_file = os.path.join(ensdf_path, "CSV_Files/all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["Element_w_A", "A", "Z", "Number_of_Levels", "Number_of_Gammas", "N_max", "N_c", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    return ensdf_index


def load_ensdf_isotopic(isotope, filetype="levels"):
    """Load level or gamma records for a given isotope (i.e. U235).

    Args:
        isotope (str): Isotope to query (i.e. u235, cl35, 239Pu)
        filetype (str, optional): Specifies if level or gamma records are to be extracted. Options
            include "levels" and "gammas". Defaults to "levels".

    Returns:
        DataFrame
    """
    isotope = gen_utils.parse_isotope(isotope, parse_for="ENSDF")
    file = Path(ensdf_path) / f"Elemental_ENSDF/{isotope}.txt"
    elemental = pd.read_csv(file, header=None, sep="|")
    elemental[0] = pd.to_numeric(elemental[0].astype(str).str.strip())

    if filetype.lower() == "levels":
        elemental_records = elemental[elemental[0].notna()]
        elemental_records = elemental_records.reset_index(drop=True).drop(columns=[7, 8])
        elemental_records.columns = [
            "Level_Number", "Energy", "Spin", "Parity", "Half_Life", "Gammas", "Flag", "ENSDF_Spin", "Num_Decay_Modes",
            "Decay_Info"]
        elemental_records.Num_Decay_Modes = elemental_records.Num_Decay_Modes.replace("0+#", -1)

        for col in elemental_records.columns:
            elemental_records[col] = elemental_records[col].astype(str).str.strip()
            if col not in ["Flag", "ENSDF_Spin", "Decay_Info"]:
                elemental_records[col] = pd.to_numeric(elemental_records[col])

    elif filetype.lower() == "gammas":
        elemental[0] = elemental[0].fillna(method='ffill')
        elemental[1] = pd.to_numeric(elemental[1].str.strip())
        elemental_records = elemental[~elemental[1].notna()]
        new_columns = elemental_records[11].str.split(expand=True)
        elemental_records = elemental_records.drop(columns=[1, 2, 3, 4, 5, 6, 10, 11])
        elemental_records = pd.concat([elemental_records, new_columns], axis=1)
        elemental_records.columns = [
            "Level_Record", "Final_State", "Energy", "Gamma_Decay_Prob", "Electromag_Decay_Prob", "ICC"
        ]
        elemental_records = pd.to_numeric(elemental_records)

    return elemental_records


def load_ensdf_ground_states():
    """Load the ENSDF file. Only ground state information.

    Returns:
        DataFrame
    """
    df = pd.read_csv(os.path.join(ensdf_path, "CSV_Files/ensdf_stable_state_formatted.csv"), header=None, sep='|')
    df = df.drop(columns=[1, 2, 6])
    df.columns = [
        "Element_w_A", "Spin", "Parity", "Half_Life", "Flag", "ENSDF_Spin", "Num_Decay_Modes", "Modifier", "Decay_Info"]
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df.Num_Decay_Modes = df.Num_Decay_Modes.replace("0+#", -1)
    for col in df.columns:
        if col not in ["Element_w_A", "Flag", "ENSDF_Spin", "Modifier", "Decay_Info"]:
            df[col] = pd.to_numeric(df[col])
    return df


def load_ripl_parameters():
    """Load the RIPL level cut-off parameters file.

    Returns:
        DataFrame
    """
    ripl_params = pd.read_csv(os.path.join(ensdf_path, "CSV_Files/ripl_cut_off_energies.csv"))
    return ripl_params


def load_ensdf(cutoff=False, append_ame=False):
    """Load the Evalauted Nuclear Structure Data File structure levels data generated through NucML parsings utilities.

    Args:
        cutoff (bool, optional): If True, the excited levels are cut-off according to the RIPL cutoof parameters.
            Defaults to False.
        append_ame (bool, optional) If True, AME isotopic properties will be appended to the loaded ENSDF data.
            Defaults to False.

    Returns:
        DataFrame
    """
    if cutoff:
        datapath = os.path.join(ensdf_path, "CSV_Files/ensdf_cutoff.csv")
    else:
        datapath = os.path.join(ensdf_path, "CSV_Files/ensdf.csv")
    logging.info("Reading data from {}".format(datapath))
    df = pd.read_csv(datapath)
    df["Level_Number"] = df["Level_Number"].astype(int)
    df[["Element_w_A"]] = df[["Element_w_A"]].astype('category')
    if append_ame:
        ame = load_ame(imputed_nan=True)
        df = pd.merge(df, ame, on='Element_w_A')
    return df


def load_ensdf_ml_ready(ensdf_df, log_sqrt=False, log=False, frac=0.3, scaling_type="standard", scaler_dir=None,
                        normalize=True):
    """EXPERIMENTAL (NOT MEANT FOR USE).

    Args:
        cutoff (bool, optional): [description]. Defaults to False.
        log_sqrt (bool, optional): [description]. Defaults to False.
        log (bool, optional): [description]. Defaults to False.
        append_ame (bool, optional): [description]. Defaults to False.
        basic (int, optional): [description]. Defaults to -1.
        num (bool, optional): [description]. Defaults to False.
        frac (float, optional): [description]. Defaults to 0.3.
        scaling_type (str, optional): [description]. Defaults to "standard".
        scaler_dir ([type], optional): [description]. Defaults to None.
        normalize (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    if log_sqrt:
        ensdf_df["Energy"] = np.sqrt(ensdf_df["Energy"])
        ensdf_df["Level_Number"] = np.log10(ensdf_df["Level_Number"])
    if log:
        logging.info("Dropping Ground State...")
        ensdf_df = ensdf_df[(ensdf_df["Energy"] != 0)]
        ensdf_df["Energy"] = np.log10(ensdf_df["Energy"])
        ensdf_df["Level_Number"] = np.log10(ensdf_df["Level_Number"])

    logging.info("Dropping unnecessary features and one-hot encoding categorical columns...")

    # We need to keep track of columns to normalize excluding categorical data.
    ensdf_df = ensdf_df.fillna(value=0)
    logging.info("Splitting dataset into training and testing...")
    x_train, x_test, y_train, y_test = train_test_split(
        ensdf_df.drop(["Energy"], axis=1), ensdf_df["Energy"], test_size=frac)

    if normalize:
        logging.info("Normalizing dataset...")
        to_scale = list(x_train.columns)
        if log_sqrt or log:
            to_scale.remove("Level_Number")
        scaler = nuc_proc.normalize_features(x_train, to_scale, scaling_type=scaling_type, scaler_dir=scaler_dir)
        x_train[to_scale] = scaler.transform(x_train[to_scale])
        x_test[to_scale] = scaler.transform(x_test[to_scale])

    logging.info(f"Finished. Resulting dataset has shape {ensdf_df.shape}, Training and Testing dataset shapes are "
                 f"{x_train.shape} and {x_test.shape} respesctively.")
    return ensdf_df, x_train, x_test, y_train, y_test, to_scale, scaler
