import logging
import math
import os
import re
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import find_peaks
import sys

sys.path.append("..")

import nucml.general_utilities as gen_utils 
import nucml.model.model_utilities as model_utils
import nucml.exfor.data_utilities as exfor_utils
import nucml.config as config

empty_df = pd.DataFrame()
ace_dir = config.ace_path
template_path = config.bench_template_path


def get_to_skip_lines(element, temp="03c", ace_dir=ace_dir):
    """This utility returns the path to the corresponding ace file for a given element.
    It also returns the number of lines to skip to get to the data block belonging to 
    the given energy and the number of lines corresponding to that block.

    Beware, the temperature needs to match one in the ACE files.

    Args:
        element (str): element in ZZAAA format.
        temp (str, optional): temperature in ace format (i.e. 03c means 300C). Defaults to "03c".
        ace_dir (str, optional): path-like string indicating directory where ace files are stored. Defaults to ace_dir.

    Returns:
        :rtype: (str, int, int)
    """
    if element == "6012":
        element = "6000"
    if len(element) == 4:
        line_spaces = 2
    else:
        line_spaces = 1
    path = Path(os.path.join(ace_dir, element + "ENDF7.ace"))
    if path.is_file():
        with open(path, "r") as ace_file:
            points = []
            indexes = []
            for index, line in enumerate(ace_file):
                if line.startswith(" "*line_spaces + element + "."): 
                    points.append(line[:10])
                    indexes.append(index)
                    
        to_search = " "*line_spaces + element + "." + temp
        to_skip = indexes[points.index(to_search)]
        lines = indexes[points.index(to_search) + 1] - to_skip - 12

        return path, to_skip, lines
    else:
        raise FileNotFoundError("{} does not exists.".format(path))


def get_nxs_jxs_xss(element, temp="03c", ace_dir=ace_dir, custom_path=None):
    """Retrieves the NSX, JXS, and XSS tables for a given element at a given temperature
    The NSX has 16 integers:
        First: the number of elements in the XSS array.
    The JSX DataFrame indicates the indices to the XSS where different pieces of data begin.
    The XSS table contains the actual data.

    Args:
        element (str): element in ZZAAA format.
        temp (str, optional): temperature in ace format (i.e. 03c means 300C). Defaults to "03c".
        ace_dir (str, optional): path-like string indicating directory where .ace files are stored. Defaults to ace_dir.

    Returns:
        :rtype: (DataFrame, DataFrame, array)
    """    
    path, to_skip, lines = get_to_skip_lines(element, temp=temp, ace_dir=ace_dir)
    if (path != None) and (custom_path != None):
        nxs = pd.read_csv(custom_path, delim_whitespace=True, skiprows=to_skip+6, nrows=2, header=None) # .values.flatten()
        jxs = pd.read_csv(custom_path, delim_whitespace=True, skiprows=to_skip+8, nrows=4, header=None)
        xss = pd.read_csv(custom_path, delim_whitespace=True, skiprows=to_skip+12, nrows=lines, header=None).values.flatten()
        # xss = xss[~np.isnan(xss)] # ALL CROSS SECTIONS, FIRST VALUES BELONG TO MT1, 2, 101
        return nxs, jxs, xss
    elif path != None:
        nxs = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+6, nrows=2, header=None) # .values.flatten()
        jxs = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+8, nrows=4, header=None)
        xss = pd.read_csv(path, delim_whitespace=True, skiprows=to_skip+12, nrows=lines, header=None).values.flatten()
        # xss = xss[~np.isnan(xss)] # ALL CROSS SECTIONS, FIRST VALUES BELONG TO MT1, 2, 101
        return nxs, jxs, xss
    

def get_nxs_dictionary(nxs_df):
    """Given the extracted NSX DataFrame, this function will return a key:value dictionary
    of the values in the NSX in accordance to the documentation of ACE formatting. 

    Args:
        nxs_df (DataFrame): The NSX DataFrame extracted using the get_nxs_jxs_xss() function.

    Returns:
        dict: dictionary containing the NSX results.
    """    
    nxs_dict = {
        "XSS_Length": nxs_df.iloc[0,0],
        "ZZAAA": nxs_df.iloc[0,1],
        "Num_Energies": nxs_df.iloc[0,2],
        "NTR": nxs_df.iloc[0,3],
        "NR": nxs_df.iloc[0,4],
        "NTRP": nxs_df.iloc[0,5],
        "NPCR": nxs_df.iloc[0,7],
        "S": nxs_df.iloc[1,0],
        "Z": nxs_df.iloc[1,1],
        "A": nxs_df.iloc[1,2],
    }
    return nxs_dict

def get_jxs_dictionary(jxs_df):
    """Given the extracted JXS DataFrame, this function will return a key:value dictionary
    of the values in the JXS in accordance to the documentation of ACE formatting.

    JXS are mostly indexes in the XSS array to indicate the beggining of a data type. 

    Args:
        jxs_df (DataFrame): The JXS DataFrame extracted using the get_nxs_jxs_xss() function.

    Returns:
        dict: dictionary containing the JXS results.
    """    
    jxs_dict = {
        "E_Table": jxs_df.iloc[0,0],
        "Fis_v_Data": jxs_df.iloc[0,1],
        "MT_Array": jxs_df.iloc[0,2],
        "Q_value_Array": jxs_df.iloc[0,3],
        "Rx_Type_Array": jxs_df.iloc[0,4],
        "XS_Loc_Table": jxs_df.iloc[0,5],
        "XS": jxs_df.iloc[0,6],
        "Ang_Dist_Loc": jxs_df.iloc[0,7],
        "Ang_Dist": jxs_df.iloc[1,0],
        "E_Dist_Loc": jxs_df.iloc[1,1],
        "E_Dist": jxs_df.iloc[1,2],
        "Photon_Prod_Data": jxs_df.iloc[1,3],
        "Photon_Prod_MT_Array": jxs_df.iloc[1,4],
        "Photon_Prod_XS_Loc": jxs_df.iloc[1,5],
        "Photon_Prod_XS": jxs_df.iloc[1,6],
        "Photon_Prod_Ang_Dist_Loc": jxs_df.iloc[1,7],
        "Photon_Prod_Ang_Dist": jxs_df.iloc[2,0],
        "Photon_Prod_E_Dist_Loc": jxs_df.iloc[2,1],
        "Photon_Prod_E_Dist": jxs_df.iloc[2,2],
        "Yield_Mult_Table": jxs_df.iloc[2,3],
        "Tot_Fis_XS": jxs_df.iloc[2,4],
        "Last_Word": jxs_df.iloc[2,5],
        "Probability_Tab": jxs_df.iloc[2,6],
        "Delayed_v_Data": jxs_df.iloc[2,7],
        "Basic_Delayed_Neut_Precursor_Data": jxs_df.iloc[3,0],
        "Delayed_Neut_E_Dist_Loc": jxs_df.iloc[3,1],
    }
    return jxs_dict

def get_pointers(nxs, jxs):
    """Gets generall information from NXS and JXS needed to start manipulating 
    cross sections. This includes several pointers for Energy, MT Array, LSIG, 
    SIG, and Fission. 

    The MT Array is an array containing all reactions avaliable in the XSS Array.
    The LSIG contains the same number of items as the MT Array. The LSIG points
    correspond to the index on the XSS array belonging to the corresponding MT Array.

    Args:
        nxs (dict): dictionary obtained using the get_nxs_dictionary()
        jxs (dict): dictionary obtained using the get_jxs_dictionary()

    Returns:
        :rtype: (int, int, int, int, int, int, int)
    """    
    nxs_dict = get_nxs_dictionary(nxs)
    jxs_dict = get_jxs_dictionary(jxs)    

    final_dict = {
        "nes":nxs_dict["Num_Energies"], # NES: Number of Energy points (3),
        "ntr":nxs_dict["NTR"],          # NTR: Number of reaction types excluding elastic scattering MT2 (4)
        "energy_pointer":jxs_dict["E_Table"] - 1,              # ESZ (1) ENERGY TABLE POINTER
        "mt_pointer":jxs_dict["MT_Array"] - 1,                  # MTR (3) MT ARRAY POINTER
        "xs_pointers":jxs_dict["XS_Loc_Table"] - 1,            # LSIG (6) TABLE OF XS LOCATORS/POINTERS
        "xs_table_pointer":jxs_dict["XS"] - 1,                 # SIG (7) CROSS SECTION ARRAY POINTER
        "mt_18_pointer":(jxs_dict["Tot_Fis_XS"] - 1).clip(0)  # FIS (21) FISSION POINTER
    }
    return final_dict

def get_energy_array(xss, pointers_dict):
    """Returns the energy grid given the XSS and needed pointers.

    Args:
        xss (np.array): an array containing all information including energy points.
        energy_pointer (int): integer indicating the index at which the energy grid begins.
        nes (int): the number of energy points expected.

    Returns:
        np.array: numpy array containing the energy grid values.
    """    
    energies = xss[pointers_dict["energy_pointer"] : pointers_dict["energy_pointer"] + pointers_dict["nes"]]   # ENERGY TABLE ARRAY
    return energies

def get_mt_array(xss, pointer_dict):
    """Returns the avaliable MT reactions given the XSS and needed pointers.

    Args:
        xss (np.array): an array containing all information including the mt array.
        mt_pointer (int): integer indicating the index at which the mt array begins.
        ntr (int): the number of mt reactions excluding MT2.

    Returns:
        np.array: numpy array containing the MT values values.
    """    
    mt_array = xss[pointer_dict["mt_pointer"] : pointer_dict["mt_pointer"] + pointer_dict["ntr"]]           # MT TABLE ARRAY
    return mt_array

def get_mt_xs_pointers_array(xss, pointer_dict):
    """Returns the XS pointers for each MT reaction in the MT array.

    Args:
        xss (np.array): an array containing all information including the XS pointers.
        xs_pointers (int): integer indicating the index at which the XS pointer begins.
        ntr (int): the number of mt reactions excluding MT2 wich corresponds directly
            to the number of XS pointers avaliable.

    Returns:
        np.array: numpy array containing the XS pointers grid values.
    """    
    mt_xs_pointers_array = xss[pointer_dict["xs_pointers"] : pointer_dict["xs_pointers"] + pointer_dict["ntr"]].astype(int) # CROSS SECTION TABLE ARRAY
    return mt_xs_pointers_array

def get_mt_array_w_pointers(mt_array, mt_xs_pointers_array):
    """Returns a dictionary with the avaliable MT reactions as keys and the 
    cross section pointers as values. 

    Args:
        mt_array (np.array): array containing the MT reactions.
        mt_xs_pointers_array (np.array): array containing the XS pointers.

    Returns:
        dict: dictionary containing the mt_array:xs_pointer key:value pairs. 
    """    
    mt_pointer_dict = {}
    for A, B in zip(mt_array, mt_xs_pointers_array):
        mt_pointer_dict[A] = B
    return mt_pointer_dict
    
def get_basic_mts(xss, pointer_dict):
    """The MT1, MT101, MT2, and MT3 are not part of the MT array. The XS values
    start inmediatly after the energy points. Since these reactions energy grid 
    are the entire "Energy Grid" then there as as many XS values as there are 
    Energy values which is given by the NES (number of energy points).

    Args:
        xss (np.array): array containing all the information including the energy grid.
        nes (int): the number of energy points avaliable in the XSS array.

    Returns:
        :rtype: (np.array, np.array, np.array, np.array)
    """    
    nes = pointer_dict["nes"]
    mt1 = xss[nes:nes*2]
    mt101 = xss[nes*2:nes*3]
    mt2 = xss[nes*3:nes*4]
    mt3 = mt1 - mt2

    mt_data = {"MT_1":mt1, "MT_2":mt2, "MT_3":mt3, "MT_101":mt101}
    return mt_data


def get_xs_for_mt(MT, mt_array, mt_xs_pointers_array, jxs_df, xss, pointers, verbose=True):
    """Returns cross section values, the corresponding energy points, and the indexes corresponding
    the starting and end point in the xss array.

    When extracting the array belonging to an MT reaction the first and second value

    Args:
        MT (int): MT number for which to query the cross section values.
        mt_array (np.array): the np.array of the mt reactions avaliable in the ace file.
        mt_xs_pointers_array (np.array): array indicating the indexes where the MT reactions begin in the xss array.
        xs_table_pointer (int): index at which the XS values start in the xss array.
        jxs (DataFrame): dataframe containing the JXS values
        xss (np.array): array containing all information directly from the .ace file
        verbose (bool, optional): to or not to print statements througout the process. Defaults to True.

    Returns:
        :rtype: (np.array, np.array, int, int)
    """
    mt_index = np.where(mt_array==MT)[0][0]                                 # GET INDEX FOR REACTION TYPE MT
    if MT == mt_array[-1]:                                                  # IF REQUESTED MT IS THE LAST ONE ON TABLE 
        start_index = pointers["xs_table_pointer"] + mt_xs_pointers_array[mt_index] - 1 # TABLE BEGINS + NUMBER OF ITEMS ACCORDING TO LSIG 
        end_index = jxs_df.iloc[0,7] - 1                                    # ENDING INDEX
        mt_data = xss[start_index: end_index]               
    else:
        start_index = pointers["xs_table_pointer"] + mt_xs_pointers_array[mt_index] - 1 
        end_index = pointers["xs_table_pointer"] + mt_xs_pointers_array[mt_index + 1] - 1
        mt_data = xss[start_index: end_index]
    
    # THE FIRST AND SECOND VALUE CORRESPOND TO ENERGY INDEX AND POINTS
    energy_index = int(mt_data[0])   # START INDEX IN ENERGY ARRAY FOR MT REACTION
    energy_points = int(mt_data[1])  # NUMBER OF ENERGY POINTS FROM START INDEX 
    energy_array = xss[energy_index - 1 : energy_index + energy_points - 1] 
    xs_data = mt_data[2:]            # ACTUAL ARRAY BELONGING TO MT REACTION CHANNEL 
    if verbose:
        logging.info("{} Energy and {} Cross Section Points Avaliable for MT{}.".format(energy_points, len(xs_data), str(int(MT))))

    xs_info_dict = {
        "xs":xs_data,
        "energy":energy_array, 
        "xss_start":start_index,
        "xss_end":end_index,
        "energy_start":energy_index - 1,
        "energy_end":energy_index + energy_points - 1
    }
    return xs_info_dict


def fill_ml_xs(MT, ml_xs, ace_xs, use_peaks=True):
    """Utility function used by the get_merged() function to fill in the head and tail
    of a set of cross section values. 

    Args:
        MT (int): endf mt reaction value for which to adjust the xs. 
        ml_xs (DataFrame): the dataframe containing the MT reaction to adjust.
        ace_xs (np.array): array containing the ace cross sections to fill in the ml_xs dataframe.

    Returns:
        DataFrame: adjusted ML cross sections dataframe.
    """
    if use_peaks:
        fallback = False
    if use_peaks:
        peaks, properties = find_peaks(ace_xs, prominence=1, width=5)
        if len(peaks) == 0:
            fallback = True
            pass
        else:
            properties["prominences"], properties["widths"]
            to_append = ace_xs[:peaks[0]]
            new_xs = np.concatenate((to_append, ml_xs[MT][peaks[0]:].values), axis=0)
            ml_xs[MT] = new_xs
    if fallback or not use_peaks:
        # FILLS VALUES IN ML DERIVED XS WHERE ALGORITHM IS UNABLE TO PERFORM (1/V AND TAIL)
        # FOR ALL VALUES THE SAME AS THE FIRST AND LAST ONE SUBSTITUTE FOR EQUIVALENT VALUE IN ENERGY IN ACE XS
        ml_xs.loc[0:ml_xs[ml_xs[MT] == ml_xs[MT].values[0]].shape[0], MT] = ace_xs[
            0:ml_xs[ml_xs[MT] == ml_xs[MT].values[0]].shape[0]+1]
        ml_xs.iloc[-ml_xs[ml_xs[MT] == ml_xs[MT].values[-1]].shape[0]:, ml_xs.columns.get_loc(MT)] = ace_xs[
            -ml_xs[ml_xs[MT] == ml_xs[MT].values[-1]].shape[0] - 1:-1]
    return ml_xs

def get_hybrid_ml_xs(ml_df, basic_mt_dict, mt_array, mt_xs_pointers_array, pointers, jxs_df, xss, use_peaks=True):
    """Substitutes the ACE MT values in a machine learning generate dataframe containing a set of
    reaction channels. For MT1, MT2, and MT3 we fix the 1/v and tail region of each cross section.
    ML models like KNN and DT are coarse and sometimes unable to accuratly keep predicting a given 
    trend. For this we:

    Previously get_merged_df()

    Args:
        ml_df (DataFrame): contains the ML generated predictions under each columned named "MT_MT".
        basic_mt_dict (DICT): a dicitonary containing MT1, MT2, MT3, and MT101 ({"mt1":values, "mt2":...})
        mt_array (np.array): contains the mt reactions avaliable in the .ace file.
        mt_xs_pointers_array (np.array): contains the mt XS pointers in the xss array.
        xs_table_pointer (int): index at which XS's start in the xss array.
        jxs_df (DataFrame): the dataframe obtained using the get_nxs_jxs_xss() function.
        xss (np.array): array containing all information in the .ace file.

    Returns:
        DataFrame: merged dataframe containing all ML and ACE reaction values
    """    
    for i in list(ml_df.columns):
        if i == "Energy":
            continue
        elif i == "MT_1":
            ml_df = fill_ml_xs(i, ml_df, basic_mt_dict["MT_1"], use_peaks=use_peaks)
        elif i == "MT_2":
            ml_df = fill_ml_xs(i, ml_df, basic_mt_dict["MT_2"], use_peaks=use_peaks)
        elif i == "MT_3":
            ml_df = fill_ml_xs(i, ml_df, basic_mt_dict["MT_3"], use_peaks=use_peaks)
        elif i == "MT_101":
            ml_df = fill_ml_xs(i, ml_df, basic_mt_dict["MT_101"], use_peaks=use_peaks)
        else:
            if (i in ["MT_18", "MT_102"]):
                MT = i.split("_")[1]
                mt_info = get_xs_for_mt(int(MT), mt_array, mt_xs_pointers_array, jxs_df, xss, pointers)
                ml_df = fill_ml_xs(i, ml_df, mt_info["xs"], use_peaks=use_peaks)
            else:
                MT = i.split("_")[1]
                mt_info = get_xs_for_mt(int(MT), mt_array, mt_xs_pointers_array, jxs_df, xss, pointers)
                new_xs = np.concatenate((np.zeros(mt_info["energy_start"]), mt_info["xs"]), axis=0)
                ml_df[i] = new_xs
    return ml_df

def create_mt2_mt3_mt101(rx_grid, mt_array):
    """Helps with the creation of MT3 and MT101 which requires them to be the summation of other MT reactions.

    TODO: CHOOSE WHICH ONE TO ADJUST BASED ON EXPERIMENTAL DATAPOINTS (MT3 OR MT102)
    Args:
        rx_grid (DataFrame): dataframe containing the grid by which to adjust the mt. This is usually the energy grid.
        MT (int): mt reaction for which to create the adjusted MT values
        mt_array (np.array): array containing the MT values avaliable. Not all MT values will be avaliable.

    Returns:
        DataFrame: resulting dataframe containing the adjusted MT values.
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
    """Creates MT2 XS and adjusts MT1 accordingly.

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
    adjusting_mt1 = pd.DataFrame({"MT_1":ml_ace_df.MT_1.values, "MT_3":ml_ace_df.MT_3.values})
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
    
def get_final_ml_ace_df(energies, mt_array, mt_xs_pointers_array, pointers, jxs_df, xss, ml_df, ml_list=["MT_1", "MT_2", "MT_3", "MT_18", "MT_101", "MT_102"]):
    """Given a set of ML generated XS (adjusted), this function allows to fill in other reaction channels not included
    by ML. This is useful since for some calculations some MT reactions are not required but are still present
    in the ace files. This allows to preserve the ACE file values. 

    For MT3 the ML generated cross sections are adjusted so that conservation rules are not broken. Same for MT101. 
    All values will be at the energy grid specified by the energies array.  

    Notice that MT2 is not calculated here. You need to use the adjust_mt1_mt2() function.

    TODO: DEAL WITH MT3 BETTER IN CASE IT IS GENERATED BY ML, WHAT ELSE TO ADJUST

    Args:
        energies (np.array): array containing the energy values at which the ML generated values are created.
        mt_array (np.array): array containing all mt reactions avaliable in ACE.
        mt_xs_pointers_array (np.array): pointers for every reaction in the MT array in the XSS array.
        xs_table_pointer (int): index of xxs at which the XS values start.
        jxs_df (DataFrame): the dataframe containing the JXS table. 
        xss (np.array): array containing all info for a specific ace file. 
        ml_df (DataFrame): dataframe containing the ML generated cross sections. 
        ml_list (list): list containing the ML generated column names that should not be modified at any point.

    Returns:
        DataFrame: the dataframe containing the resulting cross sections from both ML and ACE.
    """
    Energy_Grid = pd.DataFrame({"Energy": energies})
    Energy_Grid = Energy_Grid.set_index("Energy")

    for i in mt_array:
        # we get the ace cross sections and add them to our main dataframe some are not going to have the 
        # same energy grid as mt1 so we fill missing values with 0
        mt_info = get_xs_for_mt(i, mt_array, mt_xs_pointers_array, jxs_df, xss, pointers, verbose=False)
        to_add = pd.DataFrame({"Energy": mt_info["energy"], "MT_" + str(int(i)): mt_info["xs"]})
        to_add = to_add.set_index("Energy")
        Energy_Grid = pd.merge(Energy_Grid, to_add, left_index=True, right_index=True, how="outer") 
    Energy_Grid = Energy_Grid.fillna(value=0)
    
    for i in ml_list:
        # Once the ace XS are in a dataframe we can substitute the ace xs by those of ml 
        Energy_Grid[i] = ml_df[i].values
    
    # MT3 and MT101 are dependant on a bunch of other MTs, we need to adjust them in order for 
    # the conservation rules to be mantained.
    # Energy_Grid["MT_3"] = create_mt(Energy_Grid, "MT_3", mt_array)
    # Energy_Grid["MT_101"] = create_mt(Energy_Grid, "MT_101", mt_array)
    Energy_Grid = create_mt2_mt3_mt101(Energy_Grid, mt_array)
    Energy_Grid = enforce_unitarity(Energy_Grid)
    return Energy_Grid



def modify_xss_w_df(xss, ml_ace_df, mt_array, mt_xs_pointers_array, jxs_df, pointers):
    """Returns a modified XSS array. It substitutes the MT reactions in ml_list in the original ACE xss array.
    The resulting xss can then be used to create a new .ace file for use in monte carlo or deterministic codes.

    Args:
        ml_list (list): list of mt reactions to substitute in the original ACE xss.
        xss (np.array): original xss .ace array.
        ml_ace_df (DataFrame): the dataframe containing all modified and unmodified MT values.
        nes (int): number of energy points in the original .ace file.
        mt_array (np.array): array containing the avaliable mt values in the .ace file.
        mt_xs_pointers_array (np.array): indexes where the XS starts in the xss array for each MT value.
        xs_table_pointer (int): index at which the XS values begin.
        jxs_df (DataFrame): the JXS dataframe from the .ACE file.

    Returns:
        np.array: modified xss array.
    """    
    nes = pointers["nes"]

    # for i in ml_list:
    for i in ml_ace_df.columns:
        mt_value = int(i.split("_")[1])
        if mt_value == 1:
            xss[nes:nes*2] = ml_ace_df[i].values
        elif mt_value == 2:
            xss[nes*3:nes*4] = ml_ace_df[i].values
        elif mt_value == 101:
            xss[nes*2:nes*3] = ml_ace_df[i].values
        else:
            if mt_value in mt_array:
                xs_info_dict = get_xs_for_mt(mt_value, mt_array, mt_xs_pointers_array, jxs_df, xss, pointers)
                start = xs_info_dict["xss_start"]
                end = xs_info_dict["xss_end"]
                xss[start + 2 : end] = ml_ace_df.reset_index(drop=True)[i].iloc[xs_info_dict["energy_start"]:xs_info_dict["energy_end"]].values
            
    if (len(xss)/4 - len(xss)//4)/0.25 == 0:
        logging.info("It is wokring")
    elif 4 - (len(xss)/4 - len(xss)//4)/0.25 == 3:
        xss = np.append(xss, (np.nan, np.nan, np.nan))
    elif 4 - (len(xss)/4 - len(xss)//4)/0.25 == 2:
        xss = np.append(xss, (np.nan, np.nan))
    elif 4 - (len(xss)/4 - len(xss)//4)/0.25 == 1:
        xss = np.append(xss, (np.nan))
        
    return xss

def parsing_datatypes(x):
    """Utility function to correctly format numbers before writing them to new .ACE file.

    Args:
        x (float): number to format.

    Returns:
        float: formatted value.
    """    
    if math.isnan(x):
        return ""
    elif x - int(x) == 0:
        return "{:20}".format(int(x))
    else:
        return "{:20.11E}".format(x)

def create_new_ace(xss, ZZAAA, saving_dir=""):
    """Generates a new .ACE file ready to be used in transport codes. Everything is
    encoded in the XSS array. The header is to be kept the same since different energy 
    arrays are not supported. 

    Args:
        xss (np.array): modified xss array.
        ZZAAA (int): ZZAAA formatted element for which to create modified .ACE file.
        saving_dir (str): path-like string on where to save the created .ACE file.

    Returns:
        None
    """    
    # ACE FILES ARE IN FOUR COLUMNS SO WE RESHAPE OUR XSS ARRAY
    to_write = pd.DataFrame(xss.reshape((-1,4)))
    to_write.columns = ["column_1", "column_2", "column_3", "column_4"]
    to_write = to_write.astype("object")
    to_write = to_write.applymap(parsing_datatypes)
    
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
        ace_lines = ace.readlines()
        new_lines = new_data.readlines()
        for i, line in enumerate(ace_lines):
            if (i < to_skip + 12):
                new_ace.write(line)
        for i, line in enumerate(new_lines):
            new_ace.write(line)
        for i, line in enumerate(ace_lines):
            if (i > to_skip + line_count + 12 - 1):
                new_ace.write(line)
    
    convert_dos_to_unix(ml_ace_filename)

    os.remove(tmp_file)
    os.remove(tmp_file_2)
    
    return None

def create_new_ace_w_df(ZZAAA, path_to_ml_csv, saving_dir=None, ignore_basename=False):
    nsx, jxs, xss = get_nxs_jxs_xss(ZZAAA, temp="03c")
    pointers_info = get_pointers(nsx, jxs)
    # nes, ntr, energy_pointer, mt_pointer, xs_pointers, xs_table_pointer, _ = get_pointers(nsx, jxs)
    energies = get_energy_array(xss, pointers_info)
    mt_array = get_mt_array(xss, pointers_info)
    mt_xs_pointers_array = get_mt_xs_pointers_array(xss, pointers_info) # lsig
    mt_data = get_basic_mts(xss, pointers_info)

    if type(path_to_ml_csv) is not list:
        path_to_ml_csv = [path_to_ml_csv]
    for i in list(path_to_ml_csv):
        if ignore_basename:
            saving_dir_2 = saving_dir
        else:
            saving_dir_2 = os.path.basename(i).split(".")[0]
            saving_dir_2 = os.path.join(saving_dir, saving_dir_2)
            gen_utils.initialize_directories(saving_dir_2, reset=False)

        ml_df = pd.read_csv(i)
        ml_df["Energy"] = ml_df["Energy"] / 1E6
        
        ml_df_mod = get_hybrid_ml_xs(ml_df, mt_data, mt_array, mt_xs_pointers_array, pointers_info, jxs, xss)
        
        Energy_Grid = get_final_ml_ace_df(energies, mt_array, mt_xs_pointers_array, pointers_info, jxs, xss, ml_df_mod)   
        
        xss = modify_xss_w_df(xss, Energy_Grid, mt_array, mt_xs_pointers_array, jxs, pointers_info)

        create_new_ace(xss, ZZAAA, saving_dir=saving_dir_2)

    return None


def convert_dos_to_unix(file_path):
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    # relative or absolute file path, e.g.:

    with open(file_path, 'rb') as open_file:
        content = open_file.read()

    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

    with open(file_path, 'wb') as open_file:
        open_file.write(content)

    return None

################################################################################################
################################################################################################

################################################################################################
################################################################################################

def generate_bench_ml_xs(df, models_df, bench_name, to_scale, raw_saving_dir, reset=False, template_dir=template_path, comp_threshold=0.10, reduce_ace_size=True):
    results_df = models_df.copy()
    bench_composition = pd.read_csv(os.path.join(template_dir, os.path.join(bench_name, "composition.csv")))
    bench_composition_nonml = bench_composition[bench_composition.Fraction < comp_threshold]
    bench_composition_ml = bench_composition[bench_composition.Fraction > comp_threshold]

    results_df["run_name"] = results_df.model_path.apply(lambda x: os.path.basename(os.path.dirname(x)))
    # 3. We iterate over the rows to create data for each run
    for _, row in results_df.iterrows():
        run_name = row.run_name
        
        # 3a. We create a directory for each model but before we check if it has already been created in the inventory
        bench_saving_dir = os.path.abspath(os.path.join(raw_saving_dir, run_name + "/" + bench_name + "/"))
        ml_xs_saving_dir = os.path.join(bench_saving_dir, "ml_xs_csv")
        acelib_saving_dir = os.path.join(bench_saving_dir, "acelib")
        if (os.path.isdir(bench_saving_dir)) and not reset:
            continue
        if (os.path.isdir(bench_saving_dir)) and reset:
            gen_utils.initialize_directories(bench_saving_dir, reset=True)
            gen_utils.initialize_directories(ml_xs_saving_dir, reset=True)
            gen_utils.initialize_directories(acelib_saving_dir, reset=True)
        else:
            gen_utils.initialize_directories(ml_xs_saving_dir, reset=False)
            gen_utils.initialize_directories(acelib_saving_dir, reset=False)

        model, scaler = model_utils.load_model_and_scaler({"model_path":row.model_path, "scaler_path":row.scaler_path}, df=False)

        for _, comp_row in bench_composition_ml.iterrows():
            Z = int(comp_row.Z)
            A = int(comp_row.A)
            filename = "{}{}_ml.csv".format(Z, A)
            path_to_ml_csv = os.path.join(ml_xs_saving_dir, filename)
            if not os.path.isfile(path_to_ml_csv):
                _ = exfor_utils.get_csv_for_ace(
                    df, Z, A, model, scaler, to_scale, saving_dir=ml_xs_saving_dir, saving_filename=filename)

            create_new_ace_w_df(str(Z) + str(A).zfill(3), path_to_ml_csv, saving_dir=acelib_saving_dir, ignore_basename=True)

        bench_composition_nonml["ZA"] = bench_composition_nonml.Z.astype(str) + bench_composition_nonml.A.astype(str)
        for _, comp_row in bench_composition_nonml.iterrows():

            copy_ace_w_name(comp_row.ZA, acelib_saving_dir)

        generate_sss_xsdata(bench_saving_dir)
        copy_benchmark_files(bench_name, bench_saving_dir)

        if reduce_ace_size:
            reduce_ace_filesize(bench_saving_dir)

            
    return None


def copy_ace_w_name(ZAAA, saving_dir, ace_dir=ace_dir, ignore=None):
    files = os.listdir(ace_dir) 
    for i in files:
        if i.startswith(ZAAA):
            shutil.copyfile(os.path.join(ace_dir, i), os.path.join(saving_dir, i))
            convert_dos_to_unix(os.path.join(saving_dir, i))
        else:
            continue
    return None


def reduce_ace_filesize(directory, keep=".03c"):
    all_ace_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".ace"):
                all_ace_files.append(os.path.abspath(os.path.join(root, file)))

    for i in all_ace_files:
        tags = []
        final_tags = []
        new_file_lines = []

        with open(i) as infile:
            for line in infile:
                if line[9] == "c":
                    tags.append(line[:10])

        for t in tags:
            if t.endswith("03c"):
                continue
            else:
                final_tags.append(t)

        with open(i, 'r') as infile:
            flag = False
            for line in infile:
                if line[7:10] == "03c":
                    flag = True
                if flag:
                    new_file_lines.append(line)
                if line in final_tags:
                    flag = False
        
        with open(i, "w") as outfile:
            outfile.writelines(new_file_lines) 

        convert_dos_to_unix(i)
    return None


def generate_sss_xsdata(saving_dir, template_path=template_path):
    xsdata_filepath = os.path.join(template_path, "sss_endfb7u.xsdata")
    new_file_path = os.path.join(saving_dir, "sss_endfb7u.xsdata")

    file = open(xsdata_filepath, "rt")
    new_file = open(new_file_path, "wt")

    to_replace = "to_replace/"
    to_insert = os.path.abspath(os.path.join(saving_dir, "acelib/")).replace("C:\\", "/mnt/c/").replace("\\", "/") + "/"

    for line in file:
        new_file.write(line.replace(to_replace, to_insert))
    
    file.close()
    new_file.close()

    convert_dos_to_unix(new_file_path)

def copy_benchmark_files(benchmark_name, saving_dir, template_path=template_path):
    to_replace = "to_replace"
    new_file_path = os.path.join(saving_dir, "sss_endfb7u.xsdata")
    to_insert = os.path.abspath(new_file_path).replace("C:\\", "/mnt/c/").replace("\\", "/")

    benchmark_path = os.path.join(template_path, benchmark_name + "/input")
    new_benchmark_path = os.path.join(saving_dir, "input")

    benchmark_file = open(benchmark_path, "rt")
    new_benchmark_file = open(new_benchmark_path, "wt")

    for line in benchmark_file:
        new_benchmark_file.write(line.replace(to_replace, to_insert))

    benchmark_file.close()
    new_benchmark_file.close()

    convert_dos_to_unix(new_benchmark_path)

    shutil.copyfile(os.path.join(template_path, "converter.m"), os.path.join(saving_dir, "converter.m"))

    return None


def generate_serpent_bash(searching_directory):
    all_serpent_files = []
    all_serpent_files_linux = []

    for root, _, files in os.walk(searching_directory):
        for file in files:
            if file.endswith("input"):
                all_serpent_files.append(os.path.abspath(os.path.join(root, file)))
                    
    for i in all_serpent_files:
        new = i.replace("C:\\", "/mnt/c/").replace("\\", "/")
        if "template" in new:
            continue
        else:
            all_serpent_files_linux.append("cd {}".format(os.path.dirname(new)) + "/")
            all_serpent_files_linux.append("sss2 -omp 10 " + os.path.basename(new))
            all_serpent_files_linux.append("/mnt/c/Program\ Files/MATLAB/R2019a/bin/matlab.exe -nodisplay -nosplash -nodesktop -r \"run('converter.m');exit;\" ")  # pylint: disable=anomalous-backslash-in-string  
        
    script_path = os.path.join(searching_directory, 'serpent_script.sh')

    with open(script_path, 'w') as f:
        for item in all_serpent_files_linux:
            f.write("%s\n" % item)

    convert_dos_to_unix(script_path)

    return None


def gather_benchmark_results(searching_directory):
    all_results = []
    names = []
    benchmark_names = []

    for root, _, files in os.walk(searching_directory):
        for file in files:
            if file.endswith(".mat"):
                name_to_append = os.path.basename(Path(root).parents[0])
                names.append(name_to_append)
                all_results.append(os.path.abspath(os.path.join(root, file)))
                benchmark_names.append(os.path.basename(os.path.dirname(os.path.abspath(os.path.join(root, file)))))

    k_results_ana = []
    k_unc_ana = []

    k_results_imp = []
    k_unc_imp = []

    for i in all_results:
        mat = scipy.io.loadmat(i)
        k_results_ana.append(mat["ANA_KEFF"][0][0])
        k_unc_ana.append(mat["ANA_KEFF"][0][1])
        k_results_imp.append(mat["IMP_KEFF"][0][0])
        k_unc_imp.append(mat["IMP_KEFF"][0][1])

    results_df = pd.DataFrame({"Model":names, "Benchmark":benchmark_names,"K_eff_ana":k_results_ana, "Unc_ana":k_unc_ana, "K_eff_imp":k_results_imp, "Unc_imp":k_unc_imp})
    results_df["Deviation_Ana"] = results_df.K_eff_ana.apply(lambda k: abs((k-1)/1))
    results_df["Deviation_Imp"] = results_df.K_eff_imp.apply(lambda k: abs((k-1)/1))

    return results_df


def get_energies(element, temp="03c", ev=False, log=False):
    """Retrieves the energy array from the ENDF .ACE files.

    Args:
        element (str): ZZAAA formatted element.
        temp (str, optional): extension of ACE file to retrieve energies from. Defaults to "03c".
        ev (bool, optional): if True, energies are converted to eV from MeV. Defaults to False.
        log (bool, optional): if True, the log is taken before returning the array. Defaults to False.

    Returns:
        [type]: [description]
    """    
    nxs, jxs, xss = get_nxs_jxs_xss(element, temp="03c", ace_dir=ace_dir)
    if nxs is not None:
        pointers = get_pointers(nxs, jxs)
        energies = xss[pointers["energy_pointer"] : pointers["energy_pointer"] + pointers["nes"]]   # ENERGY TABLE ARRAY
        if ev:
            energies = energies * 1E6
        if log:
            energies = np.log10(energies)
        return energies
    else:
        return empty_df

# # testing = remove_unused_models("../ML_EXFOR_neutrons/2_DT/DT_B1/dt_results.csv", "acedata_ml/U233/DT_B1/")
# def remove_unused_models(model_results_path, acedate_directory):
#     model_results_df = pd.read_csv(model_results_path)
#     model_results_df["Model"] = model_results_df.model_path.apply(lambda x: os.path.basename(os.path.dirname(x)))
#     model_results_df["main_directory"] = model_results_df.model_path.apply(lambda x: os.path.dirname(x) + "\\")
#     model_results_df = model_results_df[["Model", "train_mae", "val_mae", "test_mae", "main_directory"]]
    
#     benchmark_results = gather_benchmark_results(acedate_directory)
#     model_results_df = model_results_df.merge(benchmark_results, on="Model")
    
#     # KEEP BEST TRAIN VAL TEST
#     # KEEP TOP 3 SORTED DEVIATION ANA
#     to_keep = []
#     to_keep.extend(list(model_results_df.iloc[model_results_df.sort_values(by="Deviation_Ana").head().index].Model.values))
#     to_keep.extend(list(model_results_df.iloc[model_utils.get_best_models_df(model_results_df).index].Model.values))
#     model_results_df["filtering"] = model_results_df.Model.apply(lambda name: True if name not in to_keep else False)
#     to_remove = model_results_df[model_results_df.filtering == True]
    
# #     for i in to_remove.main_directory.values:
# #         shutil.rmtree(i)
    
#     return None

def generate_ml_xs(df, Z, A, results, to_scale, raw_saving_dir, reset=False):
    # 2. We extract the run name from the model path
    results_df = results.copy()
    results_df["run_name"] = results_df.model_path.apply(lambda x: os.path.basename(os.path.dirname(x)))
    # 3. We iterate over the rows to create data for each run
    for _, row in results_df.iterrows():
        run_name = row.run_name
        filename = "ml_xs.csv"
        
        # 3a. We create a directory for each model but before we check if it has already been created in the inventory
        model_ace_saving_dir = os.path.abspath(os.path.join(raw_saving_dir, run_name + "/"))
        if os.path.isdir(model_ace_saving_dir) and not reset:
            continue
        # 3b. If it has not been created, the model and scaler is loaded and a csv is created needed to generate acelib.
        else:
            gen_utils.initialize_directories(model_ace_saving_dir, reset=False)
            model, scaler = model_utils.load_model_and_scaler({"model_path":row.model_path, "scaler_path":row.scaler_path}, df=False)
            _ = exfor_utils.get_csv_for_ace(
                df, Z, A, model, scaler, to_scale, saving_dir=model_ace_saving_dir, saving_filename=filename)
    return None

def generate_acelib(inventory_path, ZZAAA, generate_xsdata=True, reset=False):
    inventory = pd.read_csv(inventory_path)

    for index, row in inventory.iterrows():
        run_dir = row.directory
        acelib_status = row.acelib_generated

        if acelib_status == "yes" and not reset:
            continue
        else:
            path_to_ml_csv = os.path.join(run_dir, "ml_xs.csv")
            path_to_acelib = os.path.join(run_dir, "acelib/")
            gen_utils.initialize_directories(path_to_acelib, reset=False)
            create_new_ace_w_df(ZZAAA, path_to_ml_csv, saving_dir=path_to_acelib, ignore_basename=True)
            inventory.loc[index, 'acelib_generated'] = "yes"

            if generate_xsdata:
                generate_sss_xsdata(run_dir)
    
    inventory.to_csv(inventory_path, index=False)
    return None

def copy_ace_files(ZZ, saving_dir, ace_dir=ace_dir, ignore=None):
    files = os.listdir(ace_dir) 
    for i in files:

        acelib_filename = i.split("ENDF")[0]
        if len(acelib_filename) == 4:
            to_compare = acelib_filename[0]
        else:
            to_compare = acelib_filename[0:2]

        if len(to_compare) == len(ZZ):
            if to_compare.startswith(ZZ):
                if ignore is not None:
                    if acelib_filename in ignore:
                        continue
                    else:
                        shutil.copyfile(os.path.join(ace_dir, i), os.path.join(saving_dir, i))
                        convert_dos_to_unix(os.path.join(saving_dir, i))
                else:
                    shutil.copyfile(os.path.join(ace_dir, i), os.path.join(saving_dir, i))
                    convert_dos_to_unix(os.path.join(saving_dir, i))
        else:
            continue
    return None