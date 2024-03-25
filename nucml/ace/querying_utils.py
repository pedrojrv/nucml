"""Utilities to query ACE or ACE-related files."""
import os
import pandas as pd
import numpy as np
from pathlib import Path

from nucml import configure

config = configure._get_config()
ace_dir = config['DATA_PATHS']['ACE']


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
        if line.startswith(" " * line_spaces + isotope + "."):
            points.append(line[:10])
            indexes.append(index)

    to_search = " " * line_spaces + isotope + "." + temp
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
    mt_pointer_dict = {MT: pointer for MT, pointer in zip(mt_array, mt_xs_pointers_array)}
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
