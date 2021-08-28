"""Data loading functions.

Contains the main utility functions to load different datasets including EXFOR, AME, ENDF, ENSDF, and more.
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


import nucml.config as config
import nucml.general_utilities as gen_utils
import nucml.processing as nuc_proc
import nucml.exfor.parsing as exfor_parsing

logging.basicConfig(level=logging.INFO)

ame_dir_path = config.ame_dir_path
evaluations_path = config.evaluations_path
ensdf_path = config.ensdf_path
exfor_path = config.exfor_path

dtype_exfor = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/EXFOR_AME_dtypes.pkl'))
exfor_elements = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/exfor_elements_list.pkl'))
elements_dict = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/Element_AAA.pkl'))


def generate_exfor_dataset(user_path, modes=["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"]):
    """Generate all EXFOR datasets for neutron-, proton-, alpha-, deuterons-, gammas-, and helion-induce reactions.

    Beware, NucML configuration needs to be performed first. See nucml.configure. The `modes` argument can be modified
    for the function to generate only user-defined datasets.

    Args:
        user_path (str): path-like string where all information including the datasets will be stored.
        modes (list, optional): Type of projectile for which to generate the datasets.
            Defaults to ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"].

    Returns:
        None
    """
    user_abs_path = os.path.abspath(user_path)
    tmp_dir = os.path.join(user_abs_path, "EXFOR/tmp/")
    heavy_dir = os.path.join(user_abs_path, "EXFOR/CSV_Files/")
    for mode in modes:
        tmp_dir = os.path.join(user_abs_path, "EXFOR/tmp/")
        heavy_dir = os.path.join(user_abs_path, "EXFOR/CSV_Files/")
        exfor_directory = os.path.join(user_abs_path, "EXFOR/C4_Files/{}".format(mode))

        exfor_parsing.get_all(exfor_parsing.get_c4_names(exfor_directory), heavy_dir, tmp_dir, mode=mode)
        exfor_parsing.csv_creator(heavy_dir, tmp_dir, mode, append_ame=True)
        exfor_parsing.impute_original_exfor(heavy_dir, tmp_dir, mode)
    return None


def generate_bigquery_csv():
    """Create a single EXFOR data file to update Google BigQuery database.

    Returns:
        None
    """
    alphas = pd.read_csv(os.path.join(exfor_path, "EXFOR_alphas/EXFOR_alphas_ORIGINAL.csv"))
    deuterons = pd.read_csv(os.path.join(exfor_path, "EXFOR_deuterons/EXFOR_deuterons_ORIGINAL.csv"))
    gammas = pd.read_csv(os.path.join(exfor_path, "EXFOR_gammas/EXFOR_gammas_ORIGINAL.csv"))
    helions = pd.read_csv(os.path.join(exfor_path, "EXFOR_helions/EXFOR_helions_ORIGINAL.csv"))
    neutrons = pd.read_csv(os.path.join(exfor_path, "EXFOR_neutrons/EXFOR_neutrons_ORIGINAL.csv"))
    protons = pd.read_csv(os.path.join(exfor_path, "EXFOR_protons/EXFOR_protons_ORIGINAL.csv"))

    final = alphas.append(deuterons).append(gammas).append(helions).append(neutrons).append(protons)

    NEW_NAMES = {"Cos/LO": "Cos_LO", "dCos/LO": "dCos_LO", "ELV/HL": "ELV_HL", "dELV/HL": "dELV_HL"}
    final = final.rename(NEW_NAMES, axis=1)

    final.to_csv(os.path.join(exfor_path, "EXFOR_original.csv"), index=False)
    return None


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
        if natural:
            if imputed_nan:
                ame_file_path = os.path.join(ame_dir_path, "AME_Natural_Properties_no_NaN.csv")
            else:
                ame_file_path = os.path.join(ame_dir_path, "AME_Natural_Properties_w_NaN.csv")
        else:
            if imputed_nan:
                ame_file_path = os.path.join(ame_dir_path, "AME_all_merged_no_NaN.csv")
            else:
                ame_file_path = os.path.join(ame_dir_path, "AME_all_merged.csv")

        logging.info("AME: Reading and loading Atomic Mass Evaluation files from: \n {}".format(ame_file_path))
        ame = pd.read_csv(ame_file_path)
        ame[["N", "Z", "A"]] = ame[["N", "Z", "A"]].astype(int)
    elif file.lower() in ["mass16", "rct1", "rct2"]:
        ame_file_path = os.path.join(ame_dir_path, "AME_{}.csv".format(file))
        logging.info("AME: Reading and loading the Atomic Mass Evaluation file from: \n {}".format(ame_file_path))
        ame = pd.read_csv(ame_file_path)
    return ame


def load_evaluation(isotope, MT, mode="neutrons", library="endfb8.0", mev_to_ev=True, mb_to_b=True, log=True,
                    drop_u=True):
    """Read an evaluation file for a specific isotope, reaction channel, and evaluated library.

    It is important to inspect the returned data since it queries a local database of an external source which
    extracted data from ENDF using an extraction script. It has been found that some particular reactions are not
    included. These can be added manually for future loading.

    Args:
        isotope (str): Isotope to query (i.e. U233, Cl35).
        MT (int): Reaction channel ENDF code. Must be an integer (i.e. 1, 2, 3)
        mode (str): Type of projectile. Only "neutrons" and "protons" are supported for now.
        library (str): Evaluation library to query. Allowed options include endfb8.0, jendl4.0, jeff3.3, and tendl.2019.
        mev_to_ev (bool): If True, it converts the energy from MeV to eV.
        mb_to_b (bool): If True, it converts the cross sections from millibarns to barns.
        log (bool): If True, it applies the log10 to both the Energy and the Cross Section.
        drop_u (bool): Sometimes, evaluation files contain uncertainty values. If True, these features are removed.
    Returns:
        evaluation (DataFrame): pandas DataFrame containing the ENDF datapoints.
    """
    MT = gen_utils.parse_mt(MT)
    isotope = gen_utils.parse_isotope(isotope, parse_for='endf')

    if mode == "protons":
        projectile = 'p'
    elif mode == "neutrons":
        projectile = 'n'

    path = os.path.join(evaluations_path, f'{mode}/{isotope}/{library}/tables/xs/{projectile}-{isotope}-{MT}.{library}')

    file = Path(path)
    if file.is_file():
        logging.info("EVALUATION: Extracting data from {}".format(path))
        evaluation = pd.read_csv(
            path, skiprows=5, header=None, names=["Energy", "Data", "dDataLow", "dDataUpp"],
            delim_whitespace=True)
        if mev_to_ev:
            logging.info("EVALUATION: Converting MeV to eV...")
            evaluation["Energy"] = evaluation["Energy"]*1E6
        if mb_to_b:
            logging.info("EVALUATION: Converting mb to b...")
            evaluation["Data"] = evaluation["Data"]*0.001
        if log:
            evaluation["Energy"] = np.log10(evaluation["Energy"])
            evaluation["Data"] = np.log10(evaluation["Data"])
            evaluation["dDataLow"] = np.log10(evaluation["dDataLow"])
            evaluation["dDataUpp"] = np.log10(evaluation["dDataUpp"])
        if drop_u:
            if "dData" in list(evaluation.columns):
                evaluation = evaluation.drop(columns=["dDataLow"])
            if "dData2" in list(evaluation.columns):
                evaluation = evaluation.drop(columns=["dDataUpp"])
            if "dDataLow" in list(evaluation.columns):
                evaluation = evaluation.drop(columns=["dDataLow"])
            if "dDataUpp" in list(evaluation.columns):
                evaluation = evaluation.drop(columns=["dDataUpp"])
        logging.info("EVALUATION: Finished. ENDF data contains {} datapoints.".format(evaluation.shape[0]))
        return evaluation
    else:
        raise FileNotFoundError('Evaluation file does not exists at {}'.format(path))


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
    file = os.path.join(ensdf_path, "Elemental_ENSDF/Elemental_ENSDF_no_Header_F/{}.txt".format(isotope))
    elemental = pd.read_csv(file, header=None, sep="|")
    elemental[0] = pd.to_numeric(elemental[0].astype(str).str.strip())

    if filetype.lower() == "levels":
        elemental_level_records = elemental[elemental[0].notna()]
        elemental_level_records = elemental_level_records.reset_index(drop=True).drop(columns=[7, 8])
        elemental_level_records.columns = [
            "Level_Number", "Energy", "Spin", "Parity", "Half_Life", "Gammas", "Flag", "ENSDF_Spin", "Num_Decay_Modes",
            "Decay_Info"]
        elemental_level_records.Num_Decay_Modes = elemental_level_records.Num_Decay_Modes.replace("0+#", -1)

        for col in elemental_level_records.columns:
            elemental_level_records[col] = elemental_level_records[col].astype(str).str.strip()

        for col in elemental_level_records.columns:
            if col not in ["Flag", "ENSDF_Spin", "Decay_Info"]:
                elemental_level_records[col] = pd.to_numeric(elemental_level_records[col])

        return elemental_level_records
    elif filetype.lower() == "gammas":
        elemental[0] = elemental[0].fillna(method='ffill')
        elemental[1] = pd.to_numeric(elemental[1].str.strip())
        elemental_gamma_records = elemental[~elemental[1].notna()]
        new_columns = elemental_gamma_records[11].str.split(expand=True)
        elemental_gamma_records = elemental_gamma_records.drop(columns=[1, 2, 3, 4, 5, 6, 10, 11])
        elemental_gamma_records = pd.concat([elemental_gamma_records, new_columns], axis=1)
        elemental_gamma_records.columns = [
            "Level_Record", "Final_State", "Energy", "Gamma_Decay_Prob", "Electromag_Decay_Prob", "ICC"
        ]
        for col in elemental_gamma_records.columns:
            elemental_gamma_records[col] = pd.to_numeric(elemental_gamma_records[col])
        return elemental_gamma_records


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
    if append_ame:
        ame = load_ame(imputed_nan=True)
        df = pd.merge(df, ame, on='Element_w_A')
    return df


def load_ensdf_ml(cutoff=False, log_sqrt=False, log=False, append_ame=False, basic=-1, num=False, frac=0.3,
                  scaling_type="standard", scaler_dir=None, normalize=True):
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
    if cutoff:
        datapath = os.path.join(ensdf_path, "CSV_Files/ensdf_cutoff.csv")
    else:
        datapath = os.path.join(ensdf_path, "CSV_Files/ensdf.csv")
    df = pd.read_csv(datapath)
    df["Level_Number"] = df["Level_Number"].astype(int)
    df[["Element_w_A"]] = df[["Element_w_A"]].astype('category')
    if log_sqrt:
        df["Energy"] = np.sqrt(df["Energy"])
        df["Level_Number"] = np.log10(df["Level_Number"])
    if log:
        logging.info("Dropping Ground State...")
        df = df[(df["Energy"] != 0)]
        df["Energy"] = np.log10(df["Energy"])
        df["Level_Number"] = np.log10(df["Level_Number"])
    if append_ame:
        ame = load_ame(imputed_nan=True, natural=False)
        df = pd.merge(df, ame, on='Element_w_A')
    if basic == 0:
        basic_cols = ["Level_Number", "Energy", "Z", "N", "A", "Atomic_Mass_Micro"]
        df = df[basic_cols]
    elif basic == 1:
        basic_cols = [
            "Level_Number", "Energy", "Z", "N", "A", "Atomic_Mass_Micro", 'Mass_Excess', 'Binding_Energy',
            'B_Decay_Energy', 'S(2n)', 'S(n)', 'S(p)']
        df = df[basic_cols]
    if num:
        logging.info("Dropping unnecessary features and one-hot encoding categorical columns...")

        # We need to keep track of columns to normalize excluding categorical data.
        df = df.fillna(value=0)
        logging.info("Splitting dataset into training and testing...")
        x_train, x_test, y_train, y_test = train_test_split(df.drop(["Energy"], axis=1), df["Energy"], test_size=frac)

        if normalize:
            logging.info("Normalizing dataset...")
            to_scale = list(x_train.columns)
            if log_sqrt or log:
                to_scale.remove("Level_Number")
            scaler = nuc_proc.normalize_features(x_train, to_scale, scaling_type=scaling_type, scaler_dir=scaler_dir)
            x_train[to_scale] = scaler.transform(x_train[to_scale])
            x_test[to_scale] = scaler.transform(x_test[to_scale])

        logging.info(f"Finished. Resulting dataset has shape {df.shape}, Training and Testing dataset shapes are "
                     f"{x_train.shape} and {x_test.shape} respesctively.")
        return df, x_train, x_test, y_train, y_test, to_scale, scaler
    else:
        logging.info("Finished. Resulting dataset has shape {}".format(df.shape))
        return df


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


supported_modes = ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions", "all"]
supported_mt_coding = ["one_hot", "particle_coded"]


def load_exfor(log=False, low_en=False, basic=-1, num=False, frac=0.1, mode="neutrons", scaling_type="standard",
               scaler_dir=None, filters=False, max_en=2.0E7, mt_coding="one_hot", scale_energy=False,
               projectile_coding="one_hot", normalize=True, pedro=False, pedro_v2=False):
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
    if pedro:
        log = low_en = num = filters = normalize = True
    if pedro_v2:
        log = low_en = num = filters = True
    if mode not in supported_modes:
        msg = ' '.join([str(v) for v in supported_modes])
        return logging.error("Specified MODE not supported. Supporte modes include: {}".format(msg))
    if mt_coding not in supported_mt_coding:
        msg = ' '.join([str(v) for v in supported_mt_coding])
        return logging.error("Specified mt_coding not supported. Supported codings include: {}".format(msg))

    logging.info(" MODE: {}".format(mode))
    logging.info(" LOW ENERGY: {}".format(low_en))
    logging.info(" LOG: {}".format(log))
    logging.info(" BASIC: {}".format(basic))
    # logging.info(" SCALER: {}".format(scaling_type.upper()))

    if mode == "all":
        neutrons_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "neutrons" + '\\EXFOR_' + "neutrons" + '_MF3_AME_no_RawNaN.csv')
        protons_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "protons" + '\\EXFOR_' + "protons" + '_MF3_AME_no_RawNaN.csv')
        alphas_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "alphas" + '\\EXFOR_' + "alphas" + '_MF3_AME_no_RawNaN.csv')
        deuterons_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "deuterons" + '\\EXFOR_' + "deuterons" + '_MF3_AME_no_RawNaN.csv')
        gammas_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "gammas" + '\\EXFOR_' + "gammas" + '_MF3_AME_no_RawNaN.csv')
        helions_datapath = os.path.join(
            exfor_path, 'EXFOR_' + "helions" + '\\EXFOR_' + "helions" + '_MF3_AME_no_RawNaN.csv')
        all_datapaths = [
            neutrons_datapath, protons_datapath, alphas_datapath, deuterons_datapath, gammas_datapath, helions_datapath]
        if gen_utils.check_if_files_exist(all_datapaths):
            df = pd.read_csv(neutrons_datapath, dtype=dtype_exfor).dropna()
            protons = pd.read_csv(protons_datapath, dtype=dtype_exfor).dropna()
            alphas = pd.read_csv(alphas_datapath, dtype=dtype_exfor).dropna()
            deuterons = pd.read_csv(deuterons_datapath, dtype=dtype_exfor).dropna()
            gammas = pd.read_csv(gammas_datapath, dtype=dtype_exfor).dropna()
            helions = pd.read_csv(helions_datapath, dtype=dtype_exfor).dropna()
            df = df.append([protons, alphas, deuterons, gammas, helions])
        else:
            raise FileNotFoundError("One ore more files are missing. Check directories.")
    else:
        # datapath = os.path.join(exfor_path, 'EXFOR_' + mode + '\\EXFOR_' + mode + '_MF3_AME_no_RawNaN.csv')
        datapath = os.path.join(exfor_path, 'EXFOR_' + mode + '/EXFOR_' + mode + '_MF3_AME_no_RawNaN.csv')
        if os.path.exists(datapath):
            logging.info("Reading data from {}".format(datapath))
            df = pd.read_csv(datapath, dtype=dtype_exfor).dropna()
        else:
            raise FileNotFoundError("CSV file does not exists. Check given path: {}".format(datapath))

    if filters:
        df = df[~((
            df.Reaction_Notation.str.contains("WTR")) | (
                df.Title.str.contains("DERIV")) | (df.Energy == 0) | (df.Data == 0))]
        df = df[(df.MT != "203") & (df.MT != "1003") & (df.MT != "1108") & (df.MT != "2103")]
    if low_en:
        df = df[df.Energy < max_en]
    if log:
        if (df[df.Energy == 0].shape[0] != 0) or (df[df.Data == 0].shape[0] != 0):
            logging.error("Cannot take log. Either Energy or Data contain zeros. Ignoring log.")
        else:
            df["Energy"] = np.log10(df["Energy"])
            df["Data"] = np.log10(df["Data"])

    magic_numbers = [2, 8, 20, 28, 40, 50, 82, 126, 184]
    df["N_valence"] = df.N.apply(
        lambda neutrons: abs(neutrons - min(magic_numbers, key=lambda x: abs(x - neutrons))))  # noqa
    df["Z_valence"] = df.Z.apply(lambda protons: abs(protons - min(magic_numbers, key=lambda x: abs(x-protons))))
    df["P_factor"] = (df["N_valence"] * df["Z_valence"]) / (df["N_valence"] + df["Z_valence"])
    df.P_factor = df.P_factor.fillna(0)
    df["N_tag"] = df.N_valence.apply(lambda neutrons: "even" if neutrons % 2 == 0 else "odd")
    df["Z_tag"] = df.Z_valence.apply(lambda protons: "even" if protons % 2 == 0 else "odd")
    df["NZ_tag"] = df["N_tag"] + "_" + df["Z_tag"]

    if basic != -1:
        if basic == 0:
            basic_cols = ["Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag"]
            cat_cols = ["MT", "Center_of_Mass_Flag", "Element_Flag"]
        elif basic == 1:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag",
                "Atomic_Mass_Micro", "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio"]
            cat_cols = ["MT", "Center_of_Mass_Flag", "Element_Flag"]
        elif basic == 2:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)"]
            cat_cols = ["MT", "Center_of_Mass_Flag", "Element_Flag"]
        elif basic == 3:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)", "N_valence", "Z_valence", "P_factor", "N_tag", "Z_tag", "NZ_tag"]
            cat_cols = ["MT", "Center_of_Mass_Flag", "Element_Flag", "N_tag", "Z_tag", "NZ_tag"]
        elif basic == 4:
            basic_cols = [
                "Energy", "Data", "Z", "N", "A", "MT", "Center_of_Mass_Flag", "Element_Flag", "Atomic_Mass_Micro",
                "Nucleus_Radius", "Neutron_Nucleus_Radius_Ratio", "Mass_Excess", "Binding_Energy", "B_Decay_Energy",
                "S(n)", "S(p)", "S(2n)", "S(2p)", "N_valence", "Z_valence", "P_factor", "N_tag", "Z_tag", "NZ_tag"]
            to_append = [x for x in df.columns if x.startswith("Q") or x.startswith("dQ") or x.startswith("dS")]
            basic_cols.extend(to_append)
            cat_cols = ["MT", "Center_of_Mass_Flag", "Element_Flag", "N_tag", "Z_tag", "NZ_tag"]
        if mode == "all":
            if projectile_coding == "particle_coded":
                basic_cols.extend([
                    "Projectile_Z", "Projectile_A", "Projectile_N", "Target_Metastable_State",
                    "Product_Metastable_State"])
            elif projectile_coding == "one_hot":
                basic_cols.extend(["Projectile", "Target_Metastable_State", "Product_Metastable_State"])
                cat_cols.extend(["Projectile"])
        df = df[basic_cols]

    logging.info("Data read into dataframe with shape: {}".format(df.shape))
    if num:
        if mt_coding == "particle_coded":
            cat_cols.remove("MT")
            mt_codes_df = pd.read_csv(
                os.path.join(exfor_path, 'CSV_Files/mt_codes.csv')).drop(columns=["MT_Tag", "MT_Reaction_Notation"])
            mt_codes_df["MT"] = mt_codes_df["MT"].astype(str)
            # We need to keep track of columns to normalize excluding categorical data.
            norm_columns = len(df.columns) - len(cat_cols) - 2
            df = pd.concat([df, pd.get_dummies(df[cat_cols])], axis=1).drop(columns=cat_cols)
            df = pd.merge(df, mt_codes_df, on='MT').drop(columns=["MT"])
        elif mt_coding == "one_hot":
            # We need to keep track of columns to normalize excluding categorical data.
            norm_columns = len(df.columns) - len(cat_cols) - 1
            df = pd.concat([df, pd.get_dummies(df[cat_cols])], axis=1).drop(columns=cat_cols)

        logging.info("Splitting dataset into training and testing...")
        x_train, x_test, y_train, y_test = train_test_split(df.drop(["Data"], axis=1), df["Data"], test_size=frac)

        if normalize:
            logging.info("Normalizing dataset...")
            to_scale = list(x_train.columns)[:norm_columns]
            if not scale_energy:
                to_scale.remove("Energy")
            scaler = nuc_proc.normalize_features(x_train, to_scale, scaling_type=scaling_type, scaler_dir=scaler_dir)
            x_train[to_scale] = scaler.transform(x_train[to_scale])
            x_test[to_scale] = scaler.transform(x_test[to_scale])
            return df, x_train, x_test, y_train, y_test, to_scale, scaler
        else:
            return df, x_train, x_test, y_train, y_test
    else:
        logging.info("Finished. Resulting dataset has shape {}".format(df.shape))
        return df
