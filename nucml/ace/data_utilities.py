"""Data manipulation utilities for ACE datasets."""
import math
import os
import shutil
import numpy as np
import pandas as pd

import nucml.general_utilities as gen_utils
import nucml.ace.ml_utilities as ml_utils
import nucml.ace.querying_utils as query_utils

from nucml import configure


empty_df = pd.DataFrame()
config = configure._get_config()
ace_dir = config['DATA_PATHS']['ACE']
template_path = config['BENCHMARKING_TEMPLATE_PATH']


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
    mt_xs_pointers_array = query_utils.get_mt_xs_pointers_array(xss, pointers)

    # for i in ml_list:
    for i in ml_ace_df.columns:
        mt_value = int(i.split("_")[1])
        mt_array = query_utils.get_mt_array(xss, pointers)
        if mt_value == 1:
            xss[nes:nes*2] = ml_ace_df[i].values
        elif mt_value == 2:
            xss[nes*3:nes*4] = ml_ace_df[i].values
        elif mt_value == 101:
            xss[nes*2:nes*3] = ml_ace_df[i].values
        elif mt_value in mt_array:
            xs_info_dict = query_utils.get_xs_for_mt(mt_value, mt_array, mt_xs_pointers_array, jxs_df, xss, pointers)
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

    path, to_skip, line_count = query_utils.get_to_skip_lines(ZZAAA)

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
    nsx, jxs, xss = query_utils.get_nxs_jxs_xss(ZZAAA, temp="03c")
    pointers_info = query_utils.get_pointers(nsx, jxs)
    # nes, ntr, energy_pointer, mt_pointer, xs_pointers, xs_table_pointer, _ = query_utils.get_pointers(nsx, jxs)
    energies = query_utils.get_energy_array(xss, pointers_info)
    mt_array = query_utils.get_mt_array(xss, pointers_info)
    mt_xs_pointers_array = query_utils.get_mt_xs_pointers_array(xss, pointers_info)  # lsig
    mt_data = query_utils.get_basic_mts(xss, pointers_info)
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
