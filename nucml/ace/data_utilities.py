"""Data manipulation utilities for ACE datasets."""
import math
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

import nucml.general_utilities as gen_utils
import nucml.config as config
import nucml.ace.ml_utilities as ml_utils


empty_df = pd.DataFrame()
ace_dir = config.ace_path
template_path = config.bench_template_path


def get_to_skip_lines(isotope, temp="03c"):
    """Return the path and parameteres to the corresponding ACE file for a given isotope at a given temperature.

    The number of lines to skip to get to the data block belonging to the given energy and the number of lines
    corresponding to that block are also return.

    Args:
        isotope (str): Element in ZZAAA format (i.e. 92233, 6012).
        temp (str, optional): Temperature in ACE format (i.e. 03c means 300C). Defaults to "03c".

    Returns:
        (tuple): tuple containing:
            path (str): Path to the queried isotope in the ACE directory.
            to_skip (int): Number of lines to skip in the original file until the wanted temperature begins.
            lines (int): Number of total lines covering the queried isotope at a given temperature.

    Raises:
        FileNotFoundError: If a given isotope file does not exists, an error will be raised.
    """
    # Define isotopes for which to assume natural cross sections
    isotope = "6000" if isotope == "6012" else isotope
    line_spaces = 2 if len(isotope) == 4 else 1

    path = Path(os.path.join(ace_dir, isotope + "ENDF7.ace"))
    if not path.is_file():
        raise FileNotFoundError("{} does not exists.".format(path))

    points, indexes = [], []
    ace_file = open(path, "r")
    for index, line in enumerate(ace_file):
        if line.startswith(" "*line_spaces + isotope + "."):
            points.append(line[:10])
            indexes.append(index)

    ace_file.close()
    to_search = " "*line_spaces + isotope + "." + temp
    to_skip = indexes[points.index(to_search)]
    lines = indexes[points.index(to_search) + 1] - to_skip - 12
    return path, to_skip, lines


def get_nxs_jxs_xss(isotope, temp="03c", custom_path=None, reduced=False):
    """Retrieve the NSX, JXS, and XSS tables for a given isotope at a given temperature.

    The JSX DataFrame indicates the indices to the XSS where different pieces of data begin.
    The XSS table contains the actual data needed by many functions of the ACE utilities.

    Args:
        isotope (str): Isotope in ZZAAA format.
        temp (str, optional): Temperature in ace format (i.e. 03c means 300C). Defaults to "03c".
        custom_path (str, optional): File-path to a new ACE file not stored in the configured ACE directory. Defaults
            to None.

    Returns:
        (tuple): tuple containing:
            nxs (DataFrame)
            jxs (DataFrame)
            xss (np.array)
    """
    path, to_skip, lines = get_to_skip_lines(isotope, temp=temp)
    if custom_path is not None:
        to_skip = 0 if reduced else to_skip
        path = custom_path

    nxs = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+6, nrows=2, header=None)
    jxs = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+8, nrows=4, header=None)
    xss = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+12, nrows=lines, header=None).values.flatten()
    return nxs, jxs, xss


def get_nxs_dictionary(nxs_df):
    """Given the extracted NSX DataFrame, this function will transform it into a dictionary.

    The keys are XSS_length, ZZAAA, Num_Energies, "NTR", "NR", "NTRP", "NPCR", "S", "Z", and "A".
    For a definition, check out the ACE formatting documentation.

    Args:
        nxs_df (DataFrame): The NSX DataFrame extracted using the get_nxs_jxs_xss() function.

    Returns:
        dict: dictionary containing the NSX results.
    """
    nxs_dict = {
        "XSS_Length": nxs_df.iloc[0, 0],
        "ZZAAA": nxs_df.iloc[0, 1],
        "Num_Energies": nxs_df.iloc[0, 2],
        "NTR": nxs_df.iloc[0, 3],
        "NR": nxs_df.iloc[0, 4],
        "NTRP": nxs_df.iloc[0, 5],
        "NPCR": nxs_df.iloc[0, 7],
        "S": nxs_df.iloc[1, 0],
        "Z": nxs_df.iloc[1, 1],
        "A": nxs_df.iloc[1, 2],
    }
    return nxs_dict


def get_jxs_dictionary(jxs_df):
    """Given the extracted JXS DataFrame, this function will return a key:value dictionary.

    It will contain the values in the JXS in accordance to the ACE formatting documentation. JXS values
    are mostly indexes in the XSS array to indicate the beggining of a data type.

    Args:
        jxs_df (DataFrame): The JXS DataFrame extracted using the get_nxs_jxs_xss() function.

    Returns:
        dict: Dictionary containing the JXS results.
    """
    jxs_dict = {
        "E_Table": jxs_df.iloc[0, 0],
        "Fis_v_Data": jxs_df.iloc[0, 1],
        "MT_Array": jxs_df.iloc[0, 2],
        "Q_value_Array": jxs_df.iloc[0, 3],
        "Rx_Type_Array": jxs_df.iloc[0, 4],
        "XS_Loc_Table": jxs_df.iloc[0, 5],
        "XS": jxs_df.iloc[0, 6],
        "Ang_Dist_Loc": jxs_df.iloc[0, 7],
        "Ang_Dist": jxs_df.iloc[1, 0],
        "E_Dist_Loc": jxs_df.iloc[1, 1],
        "E_Dist": jxs_df.iloc[1, 2],
        "Photon_Prod_Data": jxs_df.iloc[1, 3],
        "Photon_Prod_MT_Array": jxs_df.iloc[1, 4],
        "Photon_Prod_XS_Loc": jxs_df.iloc[1, 5],
        "Photon_Prod_XS": jxs_df.iloc[1, 6],
        "Photon_Prod_Ang_Dist_Loc": jxs_df.iloc[1, 7],
        "Photon_Prod_Ang_Dist": jxs_df.iloc[2, 0],
        "Photon_Prod_E_Dist_Loc": jxs_df.iloc[2, 1],
        "Photon_Prod_E_Dist": jxs_df.iloc[2, 2],
        "Yield_Mult_Table": jxs_df.iloc[2, 3],
        "Tot_Fis_XS": jxs_df.iloc[2, 4],
        "Last_Word": jxs_df.iloc[2, 5],
        "Probability_Tab": jxs_df.iloc[2, 6],
        "Delayed_v_Data": jxs_df.iloc[2, 7],
        "Basic_Delayed_Neut_Precursor_Data": jxs_df.iloc[3, 0],
        "Delayed_Neut_E_Dist_Loc": jxs_df.iloc[3, 1],
    }
    return jxs_dict


def get_pointers(nxs, jxs):
    """Get general information from NXS and JXS needed to start manipulating cross sections.

    This includes several pointers for Energy, MT Array, LSIG, SIG, and Fission. More specifically, the dictionary
    contains:

    - "nes": Number of Energy points (3),
    - "ntr": Number of reaction types excluding elastic scattering MT2 (4)
    - "energy_pointer": (1) ENERGY TABLE POINTER
    - "mt_pointer": (3) MT ARRAY POINTER
    - "xs_pointers": LSIG (6) TABLE OF XS LOCATORS/POINTERS
    - "xs_table_pointer": SIG (7) CROSS SECTION ARRAY POINTER
    - "mt_18_pointer": FIS (21) FISSION POINTER

    Args:
        nxs (dict): Dictionary obtained using the get_nxs_dictionary().
        jxs (dict): Dictionary obtained using the get_jxs_dictionary().

    Returns:
        dict: Contains several pointers to different data types in the xss array.
    """
    nxs_dict = get_nxs_dictionary(nxs)
    jxs_dict = get_jxs_dictionary(jxs)

    final_dict = {
        "nes": nxs_dict["Num_Energies"],  # NES: Number of Energy points (3),
        "ntr": nxs_dict["NTR"],           # NTR: Number of reaction types excluding elastic scattering MT2 (4)
        "energy_pointer": jxs_dict["E_Table"] - 1,              # ESZ (1) ENERGY TABLE POINTER
        "mt_pointer": jxs_dict["MT_Array"] - 1,                  # MTR (3) MT ARRAY POINTER
        "xs_pointers": jxs_dict["XS_Loc_Table"] - 1,            # LSIG (6) TABLE OF XS LOCATORS/POINTERS
        "xs_table_pointer": jxs_dict["XS"] - 1,                 # SIG (7) CROSS SECTION ARRAY POINTER
        "mt_18_pointer": (jxs_dict["Tot_Fis_XS"] - 1).clip(0)  # FIS (21) FISSION POINTER
    }
    return final_dict


def get_energy_array(xss, pointers_dict):
    """Return the energy grid given the XSS array and the pointers dictionary.

    Args:
        xss (np.array): The XSS array obtained through the get_nxs_jxs_xss() function.
        pointers_dict (dict): Pointers dictionary obtained through the get_pointers() function.

    Returns:
        np.array: Numpy array containing the energy grid values.
    """
    energies = xss[pointers_dict["energy_pointer"]: pointers_dict["energy_pointer"] + pointers_dict["nes"]]
    return energies


def get_mt_array(xss, pointer_dict):
    """Return the avaliable MT reactions as a numpy array given the XSS and pointers dictionary.

    Args:
        xss (np.array): The XSS array obtained through the get_nxs_jxs_xss() function.
        pointers_dict (dict): Pointers dictionary obtained through the get_pointers() function.

    Returns:
        np.array: Numpy array containing the avaliable MT values.
    """
    mt_array = xss[pointer_dict["mt_pointer"]: pointer_dict["mt_pointer"] + pointer_dict["ntr"]]
    return mt_array


def get_mt_xs_pointers_array(xss, pointer_dict):
    """Return the XS pointers for each MT reaction in the MT array in the form of an equal length array.

    Args:
        xss (np.array): The XSS array obtained through the get_nxs_jxs_xss() function.
        pointers_dict (dict): Pointers dictionary obtained through the get_pointers() function.

    Returns:
        np.array: Numpy array containing the XS pointers grid values.
    """
    mt_xs_pointers_array = xss[pointer_dict["xs_pointers"]: pointer_dict["xs_pointers"] + pointer_dict["ntr"]]
    return mt_xs_pointers_array.astype(int)


def get_mt_array_w_pointers(mt_array, mt_xs_pointers_array):
    """Return a dictionary with the avaliable MT reactions as keys and the cross section pointers as values.

    Args:
        mt_array (np.array): Array containing the MT reactions. Usually obtained by the get_mt_array() function.
        mt_xs_pointers_array (np.array): Array containing the XS pointers obtained through the
            get_mt_xs_pointers_array() function.

    Returns:
        dict: Dictionary containing the mt_array:xs_pointer key:value pairs.
    """
    mt_pointer_dict = {}
    for A, B in zip(mt_array, mt_xs_pointers_array):
        mt_pointer_dict[A] = B
    return mt_pointer_dict


def get_basic_mts(xss, pointer_dict):
    """Return a dcitionary containing the cross section values for MT1, MT 101, MT2, and MT3.

    These are not part of the MT array. The Cross Section values start inmediatly after the energy points. The
    corresponding energy grid is the entire energy grid.

    Args:
        xss (np.array): The XSS array obtained through the get_nxs_jxs_xss() function.
        pointers_dict (dict): Pointers dictionary obtained through the get_pointers() function.

    Returns:
        dict: Dicitonary containing the cross section values for the basic reaction types.
    """
    nes = pointer_dict["nes"]
    mt1 = xss[nes:nes*2]
    mt101 = xss[nes*2:nes*3]
    mt2 = xss[nes*3:nes*4]
    mt3 = mt1 - mt2

    mt_data = {"MT_1": mt1, "MT_2": mt2, "MT_3": mt3, "MT_101": mt101}
    return mt_data


def get_xs_for_mt(MT, jxs_df, xss, pointers):
    """Return cross section and energy points and data needed to index the xss array.

    It also contains the indexes corresponding to the starting and end point in the xss array for a given reaction
    channel as a dictionary.

    Args:
        MT (int): MT number for the reaction to extract.
        jxs_df (pandas.DataFrame): DataFrame containing the JXS values.
        xss (numpyp.array): The XSS array obtained through the get_nxs_jxs_xss() function.

    Returns:
        dict: Contains cross section values and metadata for a given reaction channel.
    """
    mt_array = get_mt_array(xss, pointers)
    mt_xs_pointers_array = get_mt_xs_pointers_array(xss, pointers)
    mt_index = np.where(mt_array == MT)[0][0]                               # GET INDEX FOR REACTION TYPE MT
    start_index = pointers["xs_table_pointer"] + mt_xs_pointers_array[mt_index] - 1
    if MT == mt_array[-1]:                                                  # IF REQUESTED MT IS THE LAST ONE ON TABLE
        # TABLE BEGINS + NUMBER OF ITEMS ACCORDING TO LSIG
        end_index = jxs_df.iloc[0, 7] - 1                                    # ENDING INDEX
    else:
        end_index = pointers["xs_table_pointer"] + mt_xs_pointers_array[mt_index + 1] - 1
    mt_data = xss[start_index: end_index]

    # THE FIRST AND SECOND VALUE CORRESPOND TO ENERGY INDEX AND POINTS
    energy_index = int(mt_data[0])   # START INDEX IN ENERGY ARRAY FOR MT REACTION
    energy_points = int(mt_data[1])  # NUMBER OF ENERGY POINTS FROM START INDEX
    energy_array = xss[energy_index - 1: energy_index + energy_points - 1]
    xs_data = mt_data[2:]            # ACTUAL ARRAY BELONGING TO MT REACTION CHANNEL

    xs_info_dict = {
        "xs": xs_data,
        "energy": energy_array,
        "xss_start": start_index,
        "xss_end": end_index,
        "energy_start": energy_index - 1,
        "energy_end": energy_index + energy_points - 1
    }
    return xs_info_dict


def create_mt2_mt3_mt101(rx_grid, mt_array):
    """Help with the creation of MT3 and MT101 which requires them to be the summation of other MT reactions.

    TODO: CHOOSE WHICH ONE TO ADJUST BASED ON EXPERIMENTAL DATAPOINTS (MT3 OR MT102)

    Args:
        rx_grid (DataFrame): DataFrame containing the grid by which to adjust the MT. This is usually the energy grid.
        mt_array (np.array): Array containing the MT values avaliable.

    Returns:
        DataFrame: Resulting dataframe containing the adjusted MT values.
    """
    df = pd.DataFrame(index=rx_grid.index)
    # to_add = pd.DataFrame(index=rx_grid.index)

    df["MT_3"] = 0
    df["MT_101"] = 0

    # CREATE MT3 FROM ML VALUES MT 102, MT 18
    for i in [4, 5, 11, 16, 17, 18, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 41, 42, 44, 45,
              # BELONGS TO MT 101
              102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]:
        # CREATE MT 3 FROM ALL OTHER MTS
        if i in mt_array:
            df["MT_3"] = df["MT_3"].values + rx_grid["MT_" + str(i)].values

    # WE ADJUST MT 102 FOR EXCESS RELATIVE TO MT 3 ML
    # to_add["to_add"] = 0
    # to_add["to_add"] = rx_grid.MT_3 - df["MT_3"]
    # rx_grid["MT_102"] = rx_grid["MT_102"] + to_add["to_add"]
    rx_grid["MT_3"] = df["MT_3"]

    # ADJUSTING MT_102 means readjusting MT_101
    for i in np.linspace(102, 117, 117-102+1).astype(int):
        if i in mt_array:
            df["MT_101"] = df["MT_101"].values + rx_grid["MT_" + str(i)].values

    # ADJUSTING MT_101 means adjusting MT_1
    # to_add["to_add"] = 0
    # to_add["to_add"] = rx_grid.MT_101 - df["MT_101"]
    # rx_grid["MT_1"] = rx_grid["MT_1"] + to_add["to_add"]

    rx_grid["MT_101"] = df["MT_101"]

    # MT2 almost always has no experimental datapoints, so we adjust MT2
    rx_grid["MT_2"] = rx_grid["MT_1"] - rx_grid["MT_3"]

    return rx_grid


def enforce_unitarity(ml_ace_df):
    """Create MT2 XS and adjusts MT1 accordingly.

    Calculates MT2 by substracting MT3 from MT1. If MT2 is negative:
    - Replace negatives with NaN
    - Interpolates from closests values
    - Adds back amount to MT1 to conserve XS.

    TODO: INTRODUCE THIS FUNCTION TO THE get_final_ml_ace_df()

    Args:
        ml_ace_df (DataFram): DataFrame containing the original MT1, MT2, and MT3.

    Returns:
        DataFrame: returns an adjusted dataframe.
    """
    adjusting_mt1 = pd.DataFrame({"MT_1": ml_ace_df.MT_1.values, "MT_3": ml_ace_df.MT_3.values})
    adjusting_mt1["MT_2"] = adjusting_mt1.MT_1.values - adjusting_mt1.MT_3.values

    # neg_ix = adjusting_mt1[adjusting_mt1.MT_2 < 0].index.tolist()
    adjusting_mt1["MT_2"] = adjusting_mt1["MT_2"].apply(lambda x: x if x > 0 else -1)
    adjusting_mt1 = adjusting_mt1.replace(to_replace=-1, value=np.nan)
    adjusting_mt1 = adjusting_mt1.interpolate()
    adjusting_mt1["MT_1_int"] = adjusting_mt1["MT_2"] + adjusting_mt1["MT_3"]
    adjusting_mt1["MT_1_to_add"] = adjusting_mt1["MT_1_int"] - adjusting_mt1["MT_1"]
    adjusting_mt1["MT_1_Final"] = adjusting_mt1["MT_1"] + adjusting_mt1["MT_1_to_add"]

    ml_ace_df["MT_1"] = adjusting_mt1["MT_1_Final"].values
    ml_ace_df["MT_2"] = adjusting_mt1["MT_2"].values

    return ml_ace_df


def modify_xss_w_df(xss, ml_ace_df, jxs_df, pointers):
    """Return a modified XSS array. It substitutes the MT reactions in ml_list in the original ACE xss array.

    The resulting xss can then be used to create a new .ace file for use in monte carlo or deterministic codes.

    Args:
        ml_list (list): List of mt reactions to substitute in the original ACE xss.
        xss (np.array): Original xss .ace array.
        ml_ace_df (DataFrame): DataFrame containing all modified and unmodified MT values.
        nes (int): Number of energy points in the original .ace file.
        mt_array (np.array): Array containing the avaliable mt values in the .ace file.
        mt_xs_pointers_array (np.array): Indexes where the XS starts in the xss array for each MT value.
        xs_table_pointer (int): Index at which the XS values begin.
        jxs_df (DataFrame): JXS DataFrame from the .ACE file.

    Returns:
        np.array: Modified xss array.
    """
    nes = pointers["nes"]
    mt_xs_pointers_array = get_mt_xs_pointers_array(xss, pointers)

    # for i in ml_list:
    for i in ml_ace_df.columns:
        mt_value = int(i.split("_")[1])
        mt_array = get_mt_array(xss, pointers)
        if mt_value == 1:
            xss[nes:nes*2] = ml_ace_df[i].values
        elif mt_value == 2:
            xss[nes*3:nes*4] = ml_ace_df[i].values
        elif mt_value == 101:
            xss[nes*2:nes*3] = ml_ace_df[i].values
        elif mt_value in mt_array:
            xs_info_dict = get_xs_for_mt(mt_value, mt_array, mt_xs_pointers_array, jxs_df, xss, pointers)
            start, end = xs_info_dict["xss_start"], xs_info_dict["xss_end"]
            xss[start + 2: end] = ml_ace_df.reset_index(drop=True)[i].iloc[
                xs_info_dict["energy_start"]: xs_info_dict["energy_end"]].values

    np_nan_needed = 4 - (len(xss)/4 - len(xss)//4)/0.25
    xss = np.append(xss, ([np.nan] * np_nan_needed))
    return xss


def parsing_datatypes(x):
    """Correctly format numbers before writing them to new .ACE files.

    Args:
        x (float): Number to format.

    Returns:
        float: Formatted value.
    """
    if math.isnan(x):
        return ""
    elif x - int(x) == 0:
        return "{:20}".format(int(x))
    else:
        return "{:20.11E}".format(x)


def _format_ml_ace_file_and_data(ace, new_ace, new_data, to_skip, line_count):
    ace_lines = ace.readlines()
    new_lines = new_data.readlines()
    for i, line in enumerate(ace_lines):
        new_ace.write(line) if i < to_skip + 12 else None
    for i, line in enumerate(new_lines):
        new_ace.write(line)
    for i, line in enumerate(ace_lines):
        condition = i > to_skip + line_count + 12 - 1
        new_ace.write(line) if condition else None


def create_new_ace(xss, ZZAAA, saving_dir=""):
    """Generate a new .ACE file ready to be used in transport codes. Everything is encoded in the XSS array.

    The header is to be kept the same since different energy arrays are not supported.

    Args:
        xss (np.array): Modified XSS array.
        ZZAAA (int): ZZAAA formatted isotope for which to create modified .ACE file.
        saving_dir (str): Path-like string on where to save the created .ACE file.

    Returns:
        None
    """
    # ACE FILES ARE IN FOUR COLUMNS SO WE RESHAPE OUR XSS ARRAY
    to_write = pd.DataFrame(xss.reshape((-1, 4)), columns=["column_1", "column_2", "column_3", "column_4"])
    to_write = to_write.astype("object").applymap(parsing_datatypes)

    ace_name = ZZAAA + "ENDF7"
    tmp_file = os.path.join(saving_dir, ace_name + "_TMP.txt")
    tmp_file_2 = os.path.join(saving_dir, ace_name + "_TMP2.txt")
    ml_ace_filename = os.path.join(saving_dir, ace_name + ".ace")

    to_write.to_csv(tmp_file, float_format="%.11E", sep="\t", index=False, header=False)
    with open(tmp_file) as fin, open(tmp_file_2, 'w') as fout:
        for line in fin:
            fout.write(line.replace('\t', ''))

    path, to_skip, line_count = get_to_skip_lines(ZZAAA)

    with open(path, 'r') as ace, open(ml_ace_filename, 'w') as new_ace, open(tmp_file_2, 'r') as new_data:
        _format_ml_ace_file_and_data(ace, new_ace, new_data, to_skip, line_count)

    gen_utils.convert_dos_to_unix(ml_ace_filename)
    gen_utils.remove_files([tmp_file, tmp_file_2])


def create_new_ace_w_df(ZZAAA, path_to_ml_csv, saving_dir=None, ignore_basename=False):
    """Create new ACE file from a given processed ML generated CSV file.

    Args:
        ZZAAA (str): [description]
        path_to_ml_csv (str): Path-like string pointing towards the ML and ACE processed CSV.
        saving_dir (str, optional): Path where the newly generated ACE file will be saved. Defaults to None.
        ignore_basename (bool, optional): TODO. Defaults to False.

    Returns:
        None
    """
    nsx, jxs, xss = get_nxs_jxs_xss(ZZAAA, temp="03c")
    pointers_info = get_pointers(nsx, jxs)
    # nes, ntr, energy_pointer, mt_pointer, xs_pointers, xs_table_pointer, _ = get_pointers(nsx, jxs)
    energies = get_energy_array(xss, pointers_info)
    mt_array = get_mt_array(xss, pointers_info)
    mt_xs_pointers_array = get_mt_xs_pointers_array(xss, pointers_info)  # lsig
    mt_data = get_basic_mts(xss, pointers_info)
    path_to_ml_csv = path_to_ml_csv if isinstance(path_to_ml_csv, list) else [path_to_ml_csv]

    for i in list(path_to_ml_csv):
        if ignore_basename:
            saving_dir_2 = saving_dir
        else:
            saving_dir_2 = os.path.basename(i).split(".")[0]
            saving_dir_2 = os.path.join(saving_dir, saving_dir_2)
            gen_utils.initialize_directories(saving_dir_2, reset=False)

        ml_df = pd.read_csv(i)
        ml_df["Energy"] = ml_df["Energy"] / 1E6

        ml_df_mod = ml_utils.get_hybrid_ml_xs(ml_df, mt_data, mt_array, mt_xs_pointers_array, pointers_info, jxs, xss)
        Energy_Grid = ml_utils.get_final_ml_ace_df(
            energies, mt_array, mt_xs_pointers_array, pointers_info, jxs, xss, ml_df_mod)
        xss = modify_xss_w_df(xss, Energy_Grid, jxs, pointers_info)
        create_new_ace(xss, ZZAAA, saving_dir=saving_dir_2)


def copy_ace_w_name(ZAAA, saving_dir):
    """Copy an ACE file from the configured ace directory.

    Args:
        ZAAA (str): Isotope ace file to copy (i.e. 92233).
        saving_dir (str): Path indicating the directory where the ACE file will be copy to.

    Returns:
        None
    """
    files = filter(lambda x: True if x.startswith(ZAAA) else False, os.listdir(ace_dir))
    for i in files:
        shutil.copyfile(os.path.join(ace_dir, i), os.path.join(saving_dir, i))
        gen_utils.convert_dos_to_unix(os.path.join(saving_dir, i))


def reduce_ace_filesize(directory, keep="03c"):
    """Reduce the size of ACE files.

    Useful to remove all other information other than the section of interest (i.e. .03c).

    Args:
        directory (str): Path to directory where all ACE files will be searched and processed.
        keep (str, optional): Temperature to keep. Must not include the dot. Defaults to "03c".

    Returns:
        None
    """
    all_ace_files = []

    for root, _, files in os.walk(directory):
        filtered_files = filter(lambda x: True if x.endswith(".ace") else False, files)
        for file in filtered_files:
            all_ace_files.append(os.path.abspath(os.path.join(root, file)))

    for i in all_ace_files:
        with open(i) as infile, open(i, "w") as outfile:
            lines = infile.readlines()
            filtered_lines = [line for line in lines if line[:10].endswith(keep)]
            outfile.writelines(filtered_lines)

        gen_utils.convert_dos_to_unix(i)


def generate_sss_xsdata(saving_dir):
    """Copy the .xsdata file to a given directory and configures it to match the paths.

    Args:
        saving_dir (str): Path-like directory where the .xsdata file will be copied to.

    Returns:
        None
    """
    xsdata_filepath = os.path.join(template_path, "sss_endfb7u.xsdata")
    new_file_path = os.path.join(saving_dir, "sss_endfb7u.xsdata")
    to_replace = "to_replace/"
    with open(xsdata_filepath, "rt") as file, open(new_file_path, "wt") as new_file:
        to_insert = os.path.abspath(os.path.join(saving_dir, "acelib/"))
        to_insert = to_insert.replace("C:\\", "/mnt/c/").replace("\\", "/") + "/"

        for line in file:
            new_file.write(line.replace(to_replace, to_insert))

    gen_utils.convert_dos_to_unix(new_file_path)


def get_energies(isotope, temp="03c", ev=False, log=False):
    """Retrieve the energy array from a given isotope ENDF .ACE file.

    Args:
        isotope (str): ZZAAA formatted isotope.
        temp (str, optional): Extension of ACE file to retrieve energies from. Defaults to "03c".
        ev (bool, optional): If True, energies are converted to eV from MeV. Defaults to False.
        log (bool, optional): If True, the log is taken before returning the array. Defaults to False.

    Returns:
        np.array: Energy grid numpy array.
    """
    nxs, jxs, xss = get_nxs_jxs_xss(isotope, temp="03c")
    pointers = get_pointers(nxs, jxs)
    energies = xss[pointers["energy_pointer"]: pointers["energy_pointer"] + pointers["nes"]]   # ENERGY TABLE ARRAY
    if ev:
        energies = energies * 1E6
    if log:
        energies = np.log10(energies)
    return energies
