"""Parsing utilities for ENSDF database files."""
import os
import logging
import pandas as pd
import itertools

from nucml import general_utilities
import nucml.datasets as nuc_data


logger = logging.getLogger(__name__)


def get_ripl_dat_paths(dat_directory):
    """Search directory for RIPL .dat files and returns a list of relative paths for each one.

    Args:
        dat_directory (str): Path to the RIPL directory containing all dat files.

    Returns:
        list: Contains relative path to each .dat file found.
    """
    logger.info("RIPL: Searching {} directory for .dat files...".format(dat_directory))
    names = general_utilities.get_files_w_extension(dat_directory, "*.dat")
    logger.info("RIPL: Finished. Found {} .dat files.".format(len(names)))
    return names


def _filter_lines_with_exfor_elements(infile, outfile):
    for line in infile:
        for z in nuc_data.exfor_elements:
            if z in line.split():
                outfile.write(line)


def get_headers(dat_list, saving_directory):
    """Retrieve the avaliable raw headers for all .dat files.

    Args:
        dat_list (list): List containing the paths to all .dat files to be processed. Usually generated using the
            get_ripl_dat_paths() function.
        saving_directory (str): Path-like string to directory for saving the header file.

    Returns:
        None
    """
    raw_header_file = os.path.join(saving_directory, "all_ensdf_headers.txt")
    for dat in dat_list:
        with open(dat) as infile, open(raw_header_file, 'a') as outfile:
            _filter_lines_with_exfor_elements(infile, outfile)

    header_file = os.path.join(saving_directory, 'all_ensdf_headers_formatted.csv')
    _write_file_with_separators(raw_header_file, header_file, [5, 10, 15, 20, 25, 30, 35, 47])
    os.remove(raw_header_file)


def _strip_and_store(df, column="SYMB", storing_column="Text_Filenames"):
    df[storing_column] = df[column].apply(lambda x: x.strip())
    return df


def _read_header_file(header_directory):
    csv_file = os.path.join(header_directory, "all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["SYMB", "A", "Z", "Nol", "Nog", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    ensdf_index = _strip_and_store(ensdf_index)


def _write_file_with_separators(open_path, output_path, separator_index):
    with open(open_path) as infile, open(output_path, 'w') as outfile:
        string = _insert_separator(infile, separator_index)
        outfile.write("".join(string))


def generate_elemental_ensdf(dat_list, header_directory, saving_directory):
    """Generate a new RIPL/ENSDF file for each element.

    The original files are organized by proton number. This function is useful to organize by element instead. Three
    directories are created: one for raw elemental files with headers, another one without headers, and a third one
    which is a formatted version of the second one.

    Note: This script will delete all existing header files and create them again if run by a second time using the
    same directories.

    Args:
        dat_list (list): List containing the paths to all .dat files to be processed. Usually generated using the
            get_ripl_dat_paths() function.
        header_directory (str): Path-like string where the all_ensdf_headers_formatted.csv file is located. This file is
            generated using the get_headers() function.
        saving_directory (str): Path-like string where the new directories and ENSDF files will be stored.

    Returns:
        None
    """
    ensdf_index = _read_header_file(header_directory)
    element_list_endf = ensdf_index.SYMB.tolist()  # string that files start with
    element_list_names = ensdf_index.Text_Filenames.tolist()  # same strings but stripped

    ensdf_v1_path = os.path.join(saving_directory, "Elemental_ENSDF_v1/")
    ensdf_v2_path = os.path.join(saving_directory, "Elemental_ENSDF_no_Header/")
    general_utilities.initialize_directories(ensdf_v1_path, reset=True)
    general_utilities.initialize_directories(ensdf_v2_path, reset=True)

    element_dat_it = itertools.product(element_list_endf, dat_list)
    for e, i in element_dat_it:
        # for e in element_list_endf:
        # for i in dat_list:
        elem_path_v1 = os.path.join(ensdf_v1_path, str(e).strip() + '.txt')
        elem_path_v2 = os.path.join(ensdf_v2_path, str(e).strip() + '.txt')
        infile = open(i, "r")
        outfile1 = open(elem_path_v1, 'a')
        outfile2 = open(elem_path_v2, 'a')
        lines = infile.readlines()
        for z in [z for z, line in enumerate(lines) if line.startswith(e)]:
            # for z, line in enumerate(lines):
            # if line.startswith(str(e)):
            value = ensdf_index[ensdf_index["SYMB"] == e]
            for y in range(0, 1 + value[["Nol"]].values[0][0] + value[["Nog"]].values[0][0]):
                to_write = lines[z + y]
                outfile1.write(to_write)
                outfile2.write(to_write) if not y else None

        general_utilities.close_open_files([infile, outfile1, outfile1])

    ensdf_v3_path = os.path.join(saving_directory, "Elemental_ENSDF_no_Header_F/")
    general_utilities.initialize_directories(ensdf_v3_path, reset=True)
    for i in element_list_names:
        _write_file_with_separators(
            os.path.join(ensdf_v2_path, i + ".txt"),
            os.path.join(ensdf_v3_path, i + ".txt"),
            [4, 15, 20, 23, 34, 37, 39, 43, 54, 65, 66]
        )


def _insert_separator(infile, separation_points, separator="|"):
    for line in infile:
        if line.strip():
            string = list(line)
            for i, j in enumerate(separation_points):
                string.insert(i + j, separator)
    return string


def get_stable_states(dat_list, header_directory, saving_directory=None):
    """Generate a CSV file containing only stable state information for each isotope.

    Args:
        dat_list (list): List containing the paths to all .dat files to be processed. Usually generated using the
            get_ripl_dat_paths() function.
        header_directory (str): Path-like string where the all_ensdf_headers_formatted.csv file is located. This file is
            generated using the get_headers() function.
        saving_directory (str): Path-like string where the resulting CSV file will be saved. If None, files will be
            stored in the same directory as the header_directory.

    Returns:
        None
    """
    if saving_directory is None:
        saving_directory = header_directory
    csv_file = os.path.join(header_directory, "all_ensdf_headers_formatted.csv")
    ensdf_index_col = ["SYMB", "A", "Z", "Nol", "Nog", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(csv_file, names=ensdf_index_col, sep="|")
    element_list_endf = ensdf_index.SYMB.tolist()  # string that files start with

    element_dat_it = itertools.product(element_list_endf, dat_list)
    for e, i in element_dat_it:
        with open(i, "r") as infile, open(os.path.join(header_directory, "ensdf_stable_state.txt"), 'a') as outfile:
            lines = infile.readlines()
            for z in [z for z, line in enumerate(lines) if line.startswith(e)]:
                outfile.write(e + lines[1 + z])

    logger.info("STABLE STATES: Formatting text file...")
    _write_file_with_separators(
        os.path.join(header_directory, "ensdf_stable_state.txt"),
        os.path.join(saving_directory, 'ensdf_stable_state_formatted.csv'),
        [5, 10, 19, 25, 28, 39, 42, 44, 68, 71, 74]
    )
    logger.info("STABLE STATES: Finished.")
    os.remove(os.path.join(header_directory, "ensdf_stable_state.txt"))
    return None


def generate_ensdf_csv(header_directory, elemental_directory, saving_directory=None):
    """Generate a single CSV file containing information from all isotopes.

    Args:
        header_directory (str): Path-like string where the all_ensdf_headers_formatted.csv file is located. This file is
            generated using the get_headers() function.
        elemental_directory (str): Path-like string to directory where the Elemental_ENSDF_no_Header_F directory is
            located. This directory is first created using the generate_elemental_ensdf() function.
        saving_directory (str): Path-like string where the new ENSDF CSV file will be stored. If None,
            it will be saved in the header_directory.

    Returns:
        None
    """
    if saving_directory is None:
        saving_directory = header_directory
    ensdf_index = _read_header_file(header_directory)
    element_list_names = ensdf_index.Text_Filenames.tolist()  # same strings but stripped

    logger.info("ENSDF CSV: Creatign DataFrame with Basic ENSDF data ...")
    appended_data = []
    for e in element_list_names:
        element_ensdf = nuc_data.load_ensdf_isotopic(e)
        element_ensdf["Element_w_A"] = e
        appended_data.append(element_ensdf)
    logger.info("ENSDF CSV: Finished creating list of dataframes.")

    appended_data = pd.concat(appended_data)
    appended_data.to_csv(os.path.join(saving_directory, "ensdf.csv"), index=False)
    return None


def get_level_parameters(level_params_directory, saving_directory=None):
    """Convert the levels-param.data file from RIPL into a usable CSV file.

    Args:
        level_params_directory (str): Path-like string pointing towards the directory where the levels-param.data file
            is located.
        saving_directory (str): Path-like string pointing towards the directory where the resulting CSV will be stored.
            If None, it will be stored in the level_params_directory.

    Returns:
        None
    """
    if saving_directory is None:
        saving_directory = level_params_directory
    data_file = os.path.join(level_params_directory, "levels-param.data")
    save_file = os.path.join(saving_directory, 'ripl_cut_off_energies.csv')
    # Using the document with all data we insert commas following the EXFOR format
    with open(data_file) as infile, open(save_file, 'w') as outfile:
        separation_points = [4, 8, 11, 21, 31, 41, 51, 55, 59, 63, 67, 76, 85, 96, 98, 100, 104, 116]
        string = _insert_separator(infile, separation_points, separator=';')
        outfile.write("".join(string))

    cut_off_cols = [
        "Z", "A", "Element", "Temperature_MeV", "Temperature_U", "Black_Shift",
        "Black_Shift_U", "N_Lev_ENSDF", "N_Max_Lev_Complete", "Min_Lev_Complete",
        "Num_Lev_Unique_Spin", "E_Max_N_Max", "E_Num_Lev_U_Spin", "Chi", "Fit",
        "Flag", "Nox", "Xm_Ex", "Sigma"]
    cut_off = pd.read_csv(save_file, names=cut_off_cols, skiprows=4, sep=";")
    cut_off = _strip_and_store(cut_off, 'Element', 'Element')
    cut_off["Element_w_A"] = cut_off["A"].astype(str) + cut_off["Element"]
    cut_off = cut_off[~cut_off.Element.str.contains(r'\d')]
    cut_off.to_csv(save_file, index=False)


def generate_cutoff_ensdf(ensdf_directory, elemental_directory, ripl_directory=None, saving_directory=None):
    """Use the RIPL level parameters cut-off information to generate a new ENSDF CSV file.

    It removes all nuclear level information above the given parameters for each isotope by using
    the CSV file created using the get_level_parameters() function.

    Args:
        ensdf_directory (str): Path-like string indicating the directory where the ensdf.csv file is stored.
        elemental_directory (str): Path-like string indicating the the Elemental_ENSDF_no_Header_F directory.
        ripl_directory (str, optional): Path-like string indicating the directory where the ripl_cut_off_energies.csv
            file is stored. If None, it is assumed that the file is located in the ensdf_directory. Defaults to None.
        saving_directory (str, optional): The resulting file will be saved in this directory. If None,
            the saving_directory will be set to the ensdf_directory. Defaults to None.

    Returns:
        None
    """
    if saving_directory is None:
        saving_directory = ensdf_directory
    if ripl_directory is None:
        ripl_directory = ensdf_directory

    ensdf_path = os.path.join(ensdf_directory, "ensdf.csv")
    cut_off_path = os.path.join(ripl_directory, "ripl_cut_off_energies.csv")

    logger.info("ENSDF CutOff: Loading ENSDF and RIPL parameters...")
    ensdf = pd.read_csv(ensdf_path)
    cut_off = pd.read_csv(cut_off_path)

    str_cols = ["Spin", "Parity", "Element_w_A"]
    ensdf[str_cols] = ensdf[str_cols].astype('category')
    ensdf["Level_Number"] = ensdf["Level_Number"].astype(int)

    element_list_names = ensdf.Element_w_A.unique()

    appended_data = []
    logger.info("ENSDF CutOff: Cutting off ENSDF...")
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
    logger.info("ENSDF CutOff: Finished.")
    return None
