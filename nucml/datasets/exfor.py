"""Data loading functions.

Contains the main utility functions to load different datasets including EXFOR, AME, ENDF, ENSDF, and more.
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nucml.config as config
import nucml.general_utilities as gen_utils
import nucml.processing as nuc_proc
from nucml._constants import MAGIC_NUMBERS

logging.basicConfig(level=logging.INFO)

ame_dir_path = config.ame_dir_path
evaluations_path = config.evaluations_path
ensdf_path = config.ensdf_path
exfor_path = config.exfor_path

dtype_exfor = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/EXFOR_AME_dtypes.pkl'))
exfor_elements = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/exfor_elements_list.pkl'))
elements_dict = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/Element_AAA.pkl'))

SUPPORTED_MODES = ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions", "all"]
SUPPORTED_MT_CODINGS = ["one_hot", "particle_coded"]
NEUTRONS_DATAPATH = os.path.join(exfor_path, 'EXFOR_neutrons\\EXFOR_neutrons_MF3_AME_no_RawNaN.csv')
PROTONS_DATAPATH = os.path.join(exfor_path, 'EXFOR_protons\\EXFOR_protons_MF3_AME_no_RawNaN.csv')
ALPHAS_DATAPATH = os.path.join(exfor_path, 'EXFOR_alphas\\EXFOR_alphas_MF3_AME_no_RawNaN.csv')
DEUTERONS_DATAPATH = os.path.join(exfor_path, 'EXFOR_deuterons\\EXFOR_deuterons_MF3_AME_no_RawNaN.csv')
GAMMAS_DATAPATH = os.path.join(exfor_path, 'EXFOR_gammas\\EXFOR_gammas_MF3_AME_no_RawNaN.csv')
HELIONS_DATAPATH = os.path.join(exfor_path, 'EXFOR_helions\\EXFOR_helions_MF3_AME_no_RawNaN.csv')
ALL_DATAPATHS = [
    NEUTRONS_DATAPATH, PROTONS_DATAPATH, ALPHAS_DATAPATH, DEUTERONS_DATAPATH, GAMMAS_DATAPATH, HELIONS_DATAPATH
]
CAT_COLS = ["MT", "Center_of_Mass_Flag", "Element_Flag", "N_tag", "Z_tag", "NZ_tag"]


def load_exfor_raw(mode="neutrons"):
    """Load the original EXFOR library.

    Args:
        mode (str, optional): Projectile type to load data for. Defaults to "neutrons". Options also include "alphas",
            "deuterons", "gammas", "helions", and "protons".

    Returns:
        pd.DataFrame
    """
    if mode == "all":
        alphas = pd.read_csv(os.path.join(exfor_path, "EXFOR_alphas/EXFOR_alphas_ORIGINAL.csv"))
        deuterons = pd.read_csv(os.path.join(exfor_path, "EXFOR_deuterons/EXFOR_deuterons_ORIGINAL.csv"))
        gammas = pd.read_csv(os.path.join(exfor_path, "EXFOR_gammas/EXFOR_gammas_ORIGINAL.csv"))
        helions = pd.read_csv(os.path.join(exfor_path, "EXFOR_helions/EXFOR_helions_ORIGINAL.csv"))
        neutrons = pd.read_csv(os.path.join(exfor_path, "EXFOR_neutrons/EXFOR_neutrons_ORIGINAL.csv"))
        protons = pd.read_csv(os.path.join(exfor_path, "EXFOR_protons/EXFOR_protons_ORIGINAL.csv"))

        data = alphas.append(deuterons).append(gammas).append(helions).append(neutrons).append(protons)
    else:
        data_path = os.path.join(exfor_path, 'EXFOR_' + mode + '/EXFOR_' + mode + '_ORIGINAL.csv')
        data = pd.read_csv(data_path)
    data.MT = data.MT.astype(int)
    return data


def _get_valence_number(particle_num):
    return abs(particle_num - min(MAGIC_NUMBERS, key=lambda x: abs(x - particle_num)))


def load_exfor(log=False, basic=-1, mode="neutrons", filters=False, max_en=2.0E7):
    """Load the EXFOR dataset in its varius forms.

    This function helps load ML-ready EXFOR datasets for different particle induce reactions or all of them.

    Args:
        log (bool, optional): If True, the log of the Energy and Cross Section is taken. Defaults to False.
        low_en (bool, optional): If True, an upper limit in energy is applied given by the max_en argument.
            Defaults to False.
        basic (int, optional): Indicates how many features to load. -1 means all avaliable features. Defaults to -1.
        num (bool, optional): If True, only numerical and relevant categorical features are loaded. Defaults to False.
        frac (float, optional): Fraction of the dataset for test set. Defaults to 0.1.
        mode (str, optional): Dataset to load. Options include neutrons, gammas, and protons. Defaults to "neutrons".
        scaling_type (str, optional): Type of scaler to use for normalizing the dataset. Defaults to "standard".
        scaler_dir (str, optional): Directory in which to store the trained scaler. Defaults to None.
        filters (bool, optional): If True, a variety of filters are applied that help discard irregular data.
            Defaults to False.
        max_en (float, optional): Maximum energy threshold by which the dataset is filtered. Defaults to 2.0E7.
        mt_coding (str, optional): Method used to process the MT reaction channel codes. Defaults to "one_hot".
        scale_energy (bool, optional): If True, the energy will be normalized along with all other features.
            Defaults to False.
        projectile_coding (str, optional): Method used to process the type of projectile. Defaults to "one_hot".
        pedro (bool, optional): Personal settings. Defaults to False.

    Raises:
        FileNotFoundError: If mode is all and one of the files is missing.
        FileNotFoundError: If the selected mode file does not exist.

    Returns:
        DataFrame: Only returns one dataset if num=False.
        DataFrames: Multiple dataframes and objects if num=True.
    """
    if mode not in SUPPORTED_MODES:
        msg = ' '.join([str(v) for v in SUPPORTED_MODES])
        raise ValueError("Specified MODE not supported. Supporte modes include: {}".format(msg))

    if mode == "all":
        if not gen_utils.check_if_files_exist(ALL_DATAPATHS):
            raise FileNotFoundError("One ore more files are missing. Check directories.")
        df = pd.read_csv(NEUTRONS_DATAPATH, dtype=dtype_exfor).dropna()
        protons = pd.read_csv(PROTONS_DATAPATH, dtype=dtype_exfor).dropna()
        alphas = pd.read_csv(ALPHAS_DATAPATH, dtype=dtype_exfor).dropna()
        deuterons = pd.read_csv(DEUTERONS_DATAPATH, dtype=dtype_exfor).dropna()
        gammas = pd.read_csv(GAMMAS_DATAPATH, dtype=dtype_exfor).dropna()
        helions = pd.read_csv(HELIONS_DATAPATH, dtype=dtype_exfor).dropna()
        df = df.append([protons, alphas, deuterons, gammas, helions])
    else:
        # datapath = os.path.join(exfor_path, 'EXFOR_' + mode + '\\EXFOR_' + mode + '_MF3_AME_no_RawNaN.csv')
        datapath = os.path.join(exfor_path, 'EXFOR_' + mode + '/EXFOR_' + mode + '_MF3_AME_no_RawNaN.csv')
        if not os.path.exists(datapath):
            raise FileNotFoundError("CSV file does not exists. Check given path: {}".format(datapath))
        logging.info("Reading data from {}".format(datapath))
        df = pd.read_csv(datapath, dtype=dtype_exfor).dropna()

    if filters:
        df = df[~((
            df.Reaction_Notation.str.contains("WTR")) | (
                df.Title.str.contains("DERIV")) | (df.Energy == 0) | (df.Data == 0))]
        df = df[(df.MT != "203") & (df.MT != "1003") & (df.MT != "1108") & (df.MT != "2103")]
    if max_en:
        df = df[df.Energy < max_en]
    if log:
        if (df[df.Energy == 0].shape[0] != 0) or (df[df.Data == 0].shape[0] != 0):
            logging.error("Cannot take log. Either Energy or Data contain zeros. Ignoring log.")
        else:
            df["Energy"] = np.log10(df["Energy"])
            df["Data"] = np.log10(df["Data"])

    df["N_valence"] = df.N.apply(lambda neutrons: _get_valence_number(neutrons))
    df["Z_valence"] = df.Z.apply(lambda protons: _get_valence_number(protons))
    df["P_factor"] = (df["N_valence"] * df["Z_valence"]) / (df["N_valence"] + df["Z_valence"])
    df.P_factor = df.P_factor.fillna(0)
    df["N_tag"] = df.N_valence.apply(lambda neutrons: "even" if neutrons % 2 == 0 else "odd")
    df["Z_tag"] = df.Z_valence.apply(lambda protons: "even" if protons % 2 == 0 else "odd")
    df["NZ_tag"] = df["N_tag"] + "_" + df["Z_tag"]

    if basic != -1:
        if basic == 0:
            basic_cols = ["Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag"]
        elif basic == 1:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag",
                "Atomic_Mass_Micro", "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio"]
        elif basic == 2:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)"]
        elif basic == 3:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)", "N_valence", "Z_valence", "P_factor", "N_tag", "Z_tag", "NZ_tag"]
        elif basic == 4:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)", "N_valence", "Z_valence", "P_factor", "N_tag", "Z_tag", "NZ_tag"]
            to_append = [x for x in df.columns if x.startswith("Q") or x.startswith("dQ") or x.startswith("dS")]
            basic_cols.extend(to_append)
        df = df[basic_cols]
    return df


def load_exfor_ml_ready(exfor_df, frac=0.1, scaling_type="standard", scaler_dir=None, mt_coding="one_hot",
                        scale_energy=False):
    """Load the EXFOR dataset in its varius forms.

    This function helps load ML-ready EXFOR datasets for different particle induce reactions or all of them.

    Args:
        log (bool, optional): If True, the log of the Energy and Cross Section is taken. Defaults to False.
        low_en (bool, optional): If True, an upper limit in energy is applied given by the max_en argument.
            Defaults to False.
        basic (int, optional): Indicates how many features to load. -1 means all avaliable features. Defaults to -1.
        num (bool, optional): If True, only numerical and relevant categorical features are loaded. Defaults to False.
        frac (float, optional): Fraction of the dataset for test set. Defaults to 0.1.
        mode (str, optional): Dataset to load. Options include neutrons, gammas, and protons. Defaults to "neutrons".
        scaling_type (str, optional): Type of scaler to use for normalizing the dataset. Defaults to "standard".
        scaler_dir (str, optional): Directory in which to store the trained scaler. Defaults to None.
        filters (bool, optional): If True, a variety of filters are applied that help discard irregular data.
            Defaults to False.
        max_en (float, optional): Maximum energy threshold by which the dataset is filtered. Defaults to 2.0E7.
        mt_coding (str, optional): Method used to process the MT reaction channel codes. Defaults to "one_hot".
        scale_energy (bool, optional): If True, the energy will be normalized along with all other features.
            Defaults to False.
        projectile_coding (str, optional): Method used to process the type of projectile. Defaults to "one_hot".
        pedro (bool, optional): Personal settings. Defaults to False.

    Raises:
        FileNotFoundError: If mode is all and one of the files is missing.
        FileNotFoundError: If the selected mode file does not exist.

    Returns:
        DataFrame: Only returns one dataset if num=False.
        DataFrames: Multiple dataframes and objects if num=True.
    """
    if mt_coding not in SUPPORTED_MT_CODINGS:
        msg = ' '.join([str(v) for v in SUPPORTED_MT_CODINGS])
        raise ValueError("Specified mt_coding not supported. Supported codings include: {}".format(msg))

    cat_cols = CAT_COLS
    if mt_coding == "particle_coded":
        cat_cols.remove("MT")
        mt_codes_df = pd.read_csv(
            os.path.join(exfor_path, 'CSV_Files/mt_codes.csv')).drop(columns=["MT_Tag", "MT_Reaction_Notation"])
        mt_codes_df["MT"] = mt_codes_df["MT"].astype(str)
        # We need to keep track of columns to normalize excluding categorical data.
        norm_columns = len(exfor_df.columns) - len(cat_cols) - 2
        exfor_df = pd.concat([exfor_df, pd.get_dummies(exfor_df[cat_cols])], axis=1).drop(columns=cat_cols)
        exfor_df = pd.merge(exfor_df, mt_codes_df, on='MT').drop(columns=["MT"])
    elif mt_coding == "one_hot":
        # We need to keep track of columns to normalize excluding categorical data.
        norm_columns = len(exfor_df.columns) - len(cat_cols) - 1
        exfor_df = pd.concat([exfor_df, pd.get_dummies(exfor_df[cat_cols])], axis=1).drop(columns=cat_cols)

    logging.info("Splitting dataset into training and testing...")
    x_train, x_test, y_train, y_test = train_test_split(
        exfor_df.drop(["Data"], axis=1), exfor_df["Data"], test_size=frac)

    if scaling_type:
        logging.info("Normalizing dataset...")
        to_scale = list(x_train.columns)[:norm_columns]
        if not scale_energy:
            to_scale.remove("Energy")
        scaler = nuc_proc.normalize_features(x_train, to_scale, scaling_type=scaling_type, scaler_dir=scaler_dir)
        x_train[to_scale] = scaler.transform(x_train[to_scale])
        x_test[to_scale] = scaler.transform(x_test[to_scale])
        return exfor_df, x_train, x_test, y_train, y_test, to_scale, scaler
    else:
        return exfor_df, x_train, x_test, y_train, y_test
