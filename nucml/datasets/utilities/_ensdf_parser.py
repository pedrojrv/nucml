"""Parsing utilities for ENSDF database files."""
import logging
import itertools
import pandas as pd

from typing import List
from pathlib import Path
from io import TextIOWrapper
from natsort import realsorted

from nucml.datasets import ensdf
from nucml import general_utilities


logger = logging.getLogger(__name__)


def _get_ripl_dat_paths(dat_directory: Path) -> List[Path]:
    """Search directory for RIPL .dat files and returns a list of relative paths for each one.

    Args:
        dat_directory (str): Path to the RIPL directory containing all dat files.

    Returns:
        list: Contains relative path to each .dat file found.
    """
    return general_utilities.get_files_w_extension(dat_directory, "*.dat")


def _filter_lines_with_elements(infile: TextIOWrapper, outfile: TextIOWrapper) -> None:
    """Only retrieve lines with EXFOR elements.

    Args:
        infile (TextIOWrapper): Raw file.
        outfile (TextIOWrapper): Wrapper used to write the filtered file.
    """
    for line in infile:
        splitted = line.split()
        if not splitted:
            continue
        elif not splitted[0].isnumeric():
            outfile.write(line)


def _strip_and_store(df: pd.DataFrame, column: str, storing_column: str) -> pd.DataFrame:
    """Strip strings in an entire DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame containing column to strip.
        column (str): Column name to strip.
        storing_column (str): Column on which to save the stripped text.

    Returns:
        pd.DataFrame: DataFrame after stripping text.
    """
    df[storing_column] = df[column].apply(lambda x: x.strip())
    return df


def _generate_headers(dat_list: List[Path], saving_directory: Path) -> None:
    """Retrieve the avaliable raw headers for all .dat files.

    Args:
        dat_list (list): List containing the paths to all .dat files to be processed. Usually generated using the
            get_ripl_dat_paths() function.
        saving_directory (str): Path-like string to directory for saving the header file.
    """
    raw_header_file = saving_directory / "all_ensdf_headers.txt"
    # TODO: Add multiprocessing to optimize operation
    for dat in dat_list:
        with open(dat) as infile, open(raw_header_file, 'a') as outfile:
            _filter_lines_with_elements(infile, outfile)

    header_file = saving_directory / 'ensdf_headers.csv'
    general_utilities._write_file_with_separators(raw_header_file, header_file, [5, 10, 15, 20, 25, 30, 35, 47])
    raw_header_file.unlink()

    ensdf_index_col = ["Element_w_A", "A", "Z", "NoL", "NoG", "Nmax", "Nc", "Sn", "Sp"]
    ensdf_index = pd.read_csv(header_file, names=ensdf_index_col, sep="|")
    ensdf_index = _strip_and_store(ensdf_index, 'Element_w_A', "Element_w_A")
    ensdf_index.to_csv(header_file, index=False)


def _get_elem_idx(element: str, lines: List[int]) -> int or None:
    """Get the index of the first line starting with the given element. Returns None if not found.

    Args:
        element (str): Element to search for.
        lines (List[int]): List of lines to search in.

    Returns:
        int or None: Index if found, None otherwise.
    """
    for z, line in enumerate(lines):
        splitted = line.split()
        if splitted and splitted[0] == element:
            return z


def _generate_elemental_ensdf(dat_list: List[Path], saving_directory: Path) -> None:
    """Generate a new RIPL/ENSDF file for each element.

    The original files are organized by proton number. This function is useful to organize by element instead. Three
    directories are created: one for raw elemental files with headers, another one without headers, and a third one
    which is a formatted version of the second one.

    Note: This script will delete all existing header files and create them again if run by a second time using the
    same directories.

    Args:
        dat_list (list): List containing the paths to all .dat files to be processed. Usually generated using the
            get_ripl_dat_paths() function.
        saving_directory (str): Path-like string where the new directories and ENSDF files will be stored.
    """
    ensdf_index = pd.read_csv(saving_directory / "ensdf_headers.csv")
    element_list_names = ensdf_index.Element_w_A.tolist()

    ensdf_path = saving_directory / "Elemental_ENSDF/"
    general_utilities.initialize_directories(ensdf_path, reset=True)

    # TODO: Parallelize limiting each thread to a given element.
    element_dat_iter = itertools.product(element_list_names, dat_list)
    for elem, dat in element_dat_iter:
        # elem = "233U"
        # dat = "data/ensdf/levels/z092.dat"
        infile = open(dat, "r")
        lines = infile.readlines()
        z = _get_elem_idx(elem, lines)
        if z is not None:
            levels_out = open(ensdf_path / f'{elem}.txt', 'a')
            gammas_out = open(ensdf_path / f'{elem}_g.txt', 'a')
            value = ensdf_index[ensdf_index["Element_w_A"] == elem]
            for y in range(1, 1 + value[["NoL"]].values[0][0] + value[["NoG"]].values[0][0]):
                to_write = lines[z + y]
                level = to_write[:3].strip().isnumeric()
                if level:
                    last_level = to_write[:3].strip()
                    to_write = general_utilities._get_str_with_separators(
                        to_write, [4, 15, 20, 23, 34, 37, 39, 43, 54, 65, 66])
                    levels_out.write(to_write)
                else:
                    to_write = last_level + "|" + "|".join(to_write.split()) + "\n"
                    gammas_out.write(to_write)
            general_utilities.close_open_files([levels_out, gammas_out])

        general_utilities.close_open_files([infile])


def _get_stable_states(saving_directory: Path) -> None:
    """Generate a CSV file containing only stable state information for each isotope.

    Args:
        saving_directory (str): Path-like string where the resulting CSV file will be saved. If None, files will be
            stored in the same directory as the header_directory.
    """
    elemental_dir = saving_directory / 'Elemental_ENSDF'
    stable_out = open(saving_directory / 'ensdf_stable_states.csv', 'w')
    files = [str(file) for file in elemental_dir.glob("*")]
    for file in realsorted(files):
        if str(file).endswith("_g.txt"):
            continue
        element = Path(file).name.split('.')[0]
        infile = open(file, 'r')
        to_write = infile.readline()
        stable_out.write(element + "|" + to_write)
        infile.close()
    stable_out.close()


def _generate_ensdf_csv(saving_directory: Path) -> None:
    """Generate a single CSV file containing information from all isotopes.

    Args:
        saving_directory (str): Path-like string where the new ENSDF CSV file will be stored. If None,
            it will be saved in the header_directory.
    """
    ensdf_headers_file = saving_directory / "ensdf_headers.csv"
    ensdf_index = pd.read_csv(ensdf_headers_file)
    element_list_names = ensdf_index.Element_w_A.tolist()

    appended_data = []
    for element in element_list_names:
        element_ensdf = ensdf.load_ensdf_isotopic(element)
        element_ensdf["Element_w_A"] = element
        appended_data.append(element_ensdf)

    appended_data = pd.concat(appended_data)
    appended_data.to_csv(saving_directory / "ensdf.csv", index=False)


def _get_level_parameters(saving_directory: Path) -> None:
    """Convert the levels-param.data file from RIPL into a usable CSV file.

    Args:
        level_params_directory (str): Path-like string pointing towards the directory where the levels-param.data file
            is located.
        saving_directory (str): Path-like string pointing towards the directory where the resulting CSV will be stored.
            If None, it will be stored in the level_params_directory.

    Returns:
        None
    """
    data_file = saving_directory / "levels/level-params.data"
    save_file = saving_directory / 'ripl_cut_off_energies.csv'
    # Using the document with all data we insert commas following the EXFOR format
    indexes = [4, 8, 11, 21, 31, 41, 51, 55, 59, 63, 67, 76, 85, 96, 98, 100, 104, 116]
    general_utilities._write_file_with_separators(data_file, save_file, indexes, ",")

    cut_off_cols = [
        "Z", "A", "Element", "Temperature_MeV", "Temperature_U", "Black_Shift",
        "Black_Shift_U", "N_Lev_ENSDF", "N_Max_Lev_Complete", "Min_Lev_Complete",
        "Num_Lev_Unique_Spin", "E_Max_N_Max", "E_Num_Lev_U_Spin", "Chi", "Fit",
        "Flag", "Nox", "Xm_Ex", "Sigma"
    ]

    cut_off = pd.read_csv(save_file, names=cut_off_cols, skiprows=4)
    cut_off = _strip_and_store(cut_off, 'Element', 'Element')
    cut_off["Element_w_A"] = cut_off["A"].astype(str) + cut_off["Element"]
    cut_off = cut_off[~cut_off.Element.str.contains(r'\d')]
    cut_off.to_csv(save_file, index=False)


def _generate_cutoff_ensdf(saving_dir: Path) -> None:
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
    ensdf_df = pd.read_csv(saving_dir / "ensdf.csv")
    cut_off = pd.read_csv(saving_dir / "ripl_cut_off_energies.csv")

    str_cols = ["Spin", "Parity", "Element_w_A"]
    ensdf_df[str_cols] = ensdf_df[str_cols].astype('category')
    ensdf_df["Level_Number"] = ensdf_df["Level_Number"].astype(int)

    element_list_names = ensdf_df.Element_w_A.unique()

    appended_data = []
    for e in element_list_names:
        element_ensdf = ensdf.load_ensdf_isotopic(e)
        element_ensdf["Element_w_A"] = e
        x = cut_off[cut_off.Element_w_A == e].N_Max_Lev_Complete.values[0]
        if x == 0:
            element_ensdf = element_ensdf.iloc[0:1]
        else:
            element_ensdf = element_ensdf.iloc[0:x]
        appended_data.append(element_ensdf)

    appended_data = pd.concat(appended_data)
    appended_data = appended_data.to_csv(saving_dir / "ensdf_cutoff.csv", index=False)
