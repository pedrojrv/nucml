import os
import shutil
import logging

import numpy as np
import pandas as pd
from natsort import natsorted
import sys

sys.path.append("..")

from nucml import general_utilities # pylint: disable=import-error
import nucml.datasets as nuc_data # pylint: disable=import-error

def get_ripl_dat_paths(dat_directory):
    """Searches directory for RIPL .dat files and returns a list of relative paths for each one.

    Args:
        dat_directory (str): path to the RIPL directory containing all dat files.

    Returns:
        list: contains relative path to each .dat file found.
    """
    logging.info("RIPL: Searching {} directory for .dat files...".format(dat_directory))
    names = general_utilities.get_files_w_extension(dat_directory, "*.dat")
    logging.info("RIPL: Finished. Found {} .dat files.".format(len(names)))
    return names

def get_headers(dat_list, saving_directory):
    """Retrieves the avaliable raw header for all .dat files.

    Args:
        dat_list (list): iterable containing the paths to all .dat files to be processed.
        saving_directory (str): path to directory for saving the header file.

    Returns:
        None
    """
    logging.info("ENSDF: Extracting headers ...")
    raw_header_file = os.path.join(saving_directory, "all_ensdf_headers.txt")
    for i in dat_list:
        with open(i) as infile, open(raw_header_file, 'a') as outfile:
            for line in infile:
                for z in nuc_data.exfor_elements: 
                    if z in line.split():
                        outfile.write(line)

    header_file = os.path.join(saving_directory, 'all_ensdf_headers_formatted.csv')
    with open(raw_header_file) as infile, open(header_file, 'w') as outfile:
        for line in infile:
            if line.strip():
                string = list(line)
                for i, j in enumerate([5, 10, 15, 20, 25, 30, 35, 47]):
                    string.insert(i + j, '|')
                outfile.write("".join(string))
    os.remove(raw_header_file)
    logging.info("ENSDF: Finished. Saved to {}".format(header_file))
    return None



def generate_elemental_ensdf(dat_list, header_directory, saving_directory):
    """Generates a file for each element rather than for each proton number. 
    Three directories are created: one for raw element with header, another one 
    without header and a third one which is a formatted version of the second one.

    Notice that running this script will delete all existing header files and create them again.

    Args:
        dat_list (list): list containing paths to the ENSDF .dat files.
        header_directory (str): path to where the all_ensdf_headers_formatted.csv file is located.
        saving_directory (str): path where the new directories and ENSDF files will be stored.

    Returns:
        None
    """
    csv_file = os.path.join(header_directory, "all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["SYMB", "A", "Z", "Nol", "Nog", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    ensdf_index["Text_Filenames"] = ensdf_index["SYMB"].apply(lambda x: x.strip())
    element_list_endf = ensdf_index.SYMB.tolist() # string that files start with
    element_list_names = ensdf_index.Text_Filenames.tolist() # same strings but stripped

    ensdf_v1_path = os.path.join(saving_directory, "Elemental_ENSDF_v1/")
    general_utilities.initialize_directories(ensdf_v1_path, reset=True)
    logging.info("ENSDF Elemental: Extracting ENSDF data per element with header...")
    for e in element_list_endf:
        for i in dat_list:
            with open(i, "r") as infile, open(os.path.join(ensdf_v1_path, str(e).strip() + '.txt'), 'a') as outfile:
                lines = infile.readlines()
                for z, line in enumerate(lines):
                    if line.startswith(str(e)):
                        for y in range(0, 1 + ensdf_index[ensdf_index["SYMB"] == e][["Nol"]].values[0][0] + ensdf_index[ensdf_index["SYMB"] == e][["Nog"]].values[0][0]):
                            outfile.write(lines[z + y])

    ensdf_v2_path = os.path.join(saving_directory, "Elemental_ENSDF_no_Header/")
    general_utilities.initialize_directories(ensdf_v2_path, reset=True)
    logging.info("ENSDF Elemental: Removing header from ENSDF elemental files...")
    for e in element_list_endf:
        for i in dat_list:
            with open(i, "r") as infile, open(os.path.join(ensdf_v2_path, str(e).strip() + '.txt'), 'a') as outfile:
                lines = infile.readlines()
                for z, line in enumerate(lines):
                    if line.startswith(str(e)):
                        for y in range(1, 1 + ensdf_index[ensdf_index["SYMB"] == e][["Nol"]].values[0][0] + ensdf_index[ensdf_index["SYMB"] == e][["Nog"]].values[0][0]):
                            outfile.write(lines[z + y])

    ensdf_v3_path = os.path.join(saving_directory, "Elemental_ENSDF_no_Header_F/")
    general_utilities.initialize_directories(ensdf_v3_path, reset=True)
    logging.info("ENSDF Elemental: Formatting files...")
    for i in element_list_names:
        with open(os.path.join(ensdf_v2_path, i + ".txt")) as infile, open(os.path.join(ensdf_v3_path, i + ".txt"), 'w') as outfile:
            for line in infile:
                if line.strip():
                    string = list(line)
                    for i, j in enumerate([4, 15, 20, 23, 34, 37, 39, 43, 54, 65, 66]):
                        string.insert(i + j, '|')
                    outfile.write("".join(string))
    logging.info("ENSDF Elemental: Finished formating data.")
    return None


def get_stable_states(dat_list, header_directory, saving_directory=None):
    """Generates a CSV containing only information for the stable state for each isotope.

    Args:
        dat_list (list): list containing the path to each .dat file
        header_directory (str): path where the all_ensdf_headers_formatted.csv file is located.
        saving_directory (str): path where to save the resulting CSV file. If None, files will be 
            stored in the same directory as the header_directory.

    Returns:
        None: None
    """
    if saving_directory == None:
        saving_directory = header_directory
    csv_file = os.path.join(header_directory, "all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["SYMB", "A", "Z", "Nol", "Nog", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    element_list_endf = ensdf_index.SYMB.tolist() # string that files start with

    logging.info("STABLE STATES: Extracting stable states from .dat files...")
    for e in element_list_endf:
        for i in dat_list:
            with open(i, "r") as infile, open(os.path.join(header_directory, "ensdf_stable_state.txt"), 'a') as outfile:
                lines = infile.readlines()
                for z, line in enumerate(lines):
                    if line.startswith(str(e)):
                        outfile.write(e + lines[1 + z])

    logging.info("STABLE STATES: Formatting text file...")
    with open(os.path.join(header_directory, "ensdf_stable_state.txt")) as infile, open(
        os.path.join(saving_directory, 'ensdf_stable_state_formatted.csv'), 'w') as outfile:
        for line in infile:
            if line.strip():
                string = list(line)
                for i, j in enumerate([5, 10, 19, 25, 28, 39, 42, 44, 68, 71, 74]):
                    string.insert(i + j, '|')
                outfile.write("".join(string))
    logging.info("STABLE STATES: Finished.")
    os.remove(os.path.join(header_directory, "ensdf_stable_state.txt"))
    return None


def generate_ensdf_csv(header_directory, elemental_directory, saving_directory=None):
    """Generates a single csv file containing the information from all isotopes.

    Args:
        header_directory (str): path to where the all_ensdf_headers_formatted.csv file is located.
        elemental_directory (str): path to directory where the Elemental_ENSDF_no_Header_F directory is located.
        saving_directory (str): path where the new ENSDF .csv file will be stored. If None, 
            it will be saved in the header directory.

    Returns:
        None
    """
    if saving_directory == None:
        saving_directory = header_directory
    csv_file = os.path.join(header_directory, "all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["SYMB", "A", "Z", "Nol", "Nog", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    ensdf_index["Text_Filenames"] = ensdf_index["SYMB"].apply(lambda x: x.strip())
    element_list_names = ensdf_index.Text_Filenames.tolist() # same strings but stripped

    logging.info("ENSDF CSV: Creatign DataFrame with Basic ENSDF data ...")
    appended_data = []
    for e in element_list_names:
        element_ensdf = nuc_data.load_ensdf_isotopic(e)
        element_ensdf["Element_w_A"] = e
        appended_data.append(element_ensdf)
    logging.info("ENSDF CSV: Finished creating list of dataframes.")

    appended_data = pd.concat(appended_data)
    appended_data.to_csv(os.path.join(saving_directory, "ensdf.csv"), index=False)
    return None

def get_level_parameters(level_params_directory, saving_directory=None):
    """Converts the levels-param.data file from RIPL into a CSV file.

    Args:
        level_params_directory (str): path where the directory where levels-param.data file is located.
        saving_directory (str): path to directory where the resulting CSV will be stored. If None, 
            it will be stored in the same directory as the levels-param.data file is located.

    Returns:
        None: None
    """
    if saving_directory == None:
        saving_directory = level_params_directory
    data_file = os.path.join(level_params_directory, "levels-param.data")
    # Using the document with all data we insert commas following the EXFOR format
    logging.info("ENSDF RIPL: Parsing and formatting level parameters...")
    save_file = os.path.join(saving_directory, 'ripl_cut_off_energies.csv')
    with open(data_file) as infile, open(save_file, 'w') as outfile:
        for line in infile:
            if line.strip():
                string = list(line)
                for i, j in enumerate([4, 8, 11, 21, 31, 41, 51, 55, 59, 63, 67, 76, 85, 96, 98, 100, 104, 116]):
                    string.insert(i + j, ';')
                outfile.write("".join(string))
    logging.info("ENSDF RIPL: Finished formating data. Converting to CSV...")

    cut_off_cols = ["Z", "A", "Element", "Temperature_MeV", "Temperature_U", "Black_Shift", 
            "Black_Shift_U", "N_Lev_ENSDF", "N_Max_Lev_Complete", "Min_Lev_Complete", 
            "Num_Lev_Unique_Spin", "E_Max_N_Max", "E_Num_Lev_U_Spin", "Chi", "Fit", 
            "Flag", "Nox", "Xm_Ex", "Sigma"]
    cut_off = pd.read_csv(save_file, names=cut_off_cols, skiprows=4, sep=";")
    cut_off["Element"] = cut_off["Element"].apply(lambda x: x.strip())
    cut_off["Element_w_A"] = cut_off["A"].astype(str) + cut_off["Element"]
    cut_off = cut_off[~cut_off.Element.str.contains(r'\d')]
    cut_off.to_csv(save_file, index=False)
    logging.info("ENSDF RIPL: Finished.")
    return None


def generate_cutoff_ensdf(ensdf_directory, elemental_directory, ripl_directory=None, saving_directory=None):
    """This function uses the RIPL level parameters cut-off information to generate a new ENSDF .csv file.
    It removes all nuclear level information above the given parameters for each isotope. It uses
    the CSV file created using the .get_level_parameters() function from the parsing utilities. 

    Args:
        ensdf_directory (str): path to the directory where the ensdf.csv file is stored
        elemental_directory (str): path to the Elemental_ENSDF_no_Header_F directory.
        ripl_directory (str, optional): path to the directory where the ripl_cut_off_energies.csv file is 
            stored. If None, it is assumed that the file is located in the ensdf_directory. Defaults to None.
        saving_directory (str, optional): the resulting file will be saved in this directory. If None,
            the saving_directory will be set to the ensdf_directory. Defaults to None.

    Returns:
        None: 
    """    
    if saving_directory == None:
        saving_directory = ensdf_directory
    if ripl_directory == None:
        ripl_directory = ensdf_directory

    ensdf_path = os.path.join(ensdf_directory, "ensdf.csv")
    cut_off_path = os.path.join(ripl_directory, "ripl_cut_off_energies.csv")

    logging.info("ENSDF CutOff: Loading ENSDF and RIPL parameters...")
    ensdf = pd.read_csv(ensdf_path)
    cut_off = pd.read_csv(cut_off_path)

    str_cols = ["Spin", "Parity", "Element_w_A"]
    ensdf[str_cols] = ensdf[str_cols].astype('category')
    ensdf["Level_Number"] = ensdf["Level_Number"].astype(int)

    element_list_names = ensdf.Element_w_A.unique()

    appended_data = []
    logging.info("ENSDF CutOff: Cutting off ENSDF...")
    for e in element_list_names:
        element_ensdf = nuc_data.load_ensdf_isotopic(e)
        element_ensdf["Element_w_A"] = e
        x = cut_off[cut_off.Element_w_A == e].N_Max_Lev_Complete.values[0]
        if x == 0:
            element_ensdf = element_ensdf.iloc[0:1]
        else:
            element_ensdf = element_ensdf.iloc[0:x]    
        appended_data.append(element_ensdf)

    appended_data = pd.concat(appended_data)
    appended_data = appended_data.to_csv(os.path.join(saving_directory, "ensdf_cutoff.csv"), index=False)
    logging.info("ENSDF CutOff: Finished.")
    return None
