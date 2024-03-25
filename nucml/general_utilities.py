"""Contains general utilities related to file querying and manipulation, parsing, and saving and loading objects."""

import os
import logging
import glob
import shutil
import pickle
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, List, Union
from sqlite3 import NotSupportedError
from natsort import natsorted


def get_files_w_extension(directory: Path, extension: str) -> List[Path]:
    """Get a list of relative paths to files that match the given extension in the given directory.

    Args:
        directory (str): Path-like string to the directory where the search will be conducted.
        extension (str): The extension for which to search files in the directory and all subdirectories (i.e. ".csv").

    Returns:
        list: Contains relative path to each encountered file containing the given extension.
    """
    extension = "*" + extension
    logging.info("GEN_UTILS: Searching for {} files...".format(extension))
    files = glob.glob(os.path.join(directory, extension))
    files = natsorted(files)
    files = [Path(f) for f in files]
    return files


def initialize_directories(directory: Path, reset: bool = False) -> None:
    """Create and/or reset the given directory path.

    Args:
        directory (str): Path-like string to directory to create and/or reset.
        reset (bool, optional): If True, the directory will be deleted and created again.

    Returns:
        None
    """
    if not isinstance(directory, list):
        directory = [directory]

    for dir in directory:
        if os.path.isdir(dir) and reset:
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


def remove_file(file: Path) -> None:
    """Remove file if it exists."""
    if os.path.exists(file):
        os.remove(file)


def remove_files(files: List[Path]) -> None:
    """Remove all files in list if they exist."""
    for file in files:
        remove_file(file)


def check_if_files_exist(files_list: List[Path]) -> bool:
    """Check if all files in a list of filepaths exists.

    Args:
        files_list (list): List of relative or absolute path-like strings to check for existence.

    Returns:
        bool: True if all exists, False if more than one does not exist.
    """
    if all([os.path.isfile(f) for f in files_list]):
        return True
    else:
        return False


def func(x: Optional[Union[int, float]], c: Optional[Union[int, float]], d: Optional[Union[int, float]]) -> float:
    """Line equation function. Used to interpolate AME features.

    Args:
        x (int or float): Input parameter.
        c (int or float): Intercept parameter.
        d (int or float): Weight parameter.

    Returns:
        float: Linear equation result.
    """
    return c * x + d


def save_obj(obj: object, saving_dir: Path, name: str) -> None:
    """Save a python object with pickle in the `saving_dir` directory using `name`.

    Useful to quickly store objects such as lists or numpy arrays. Do not include the extension in the name. The
    function automatically adds the `.pkl` extension to all saved files.

    Args:
        obj (object): Object to save. Can be a list, np.array, pd.DataFrame, etc.
        saving_dir (str): Path-like string where the object will be saved.
        name (str): Name of the object without extension.

    Returns:
        None
    """
    with open(os.path.join(saving_dir, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path: Path) -> object:
    """Load a saved pickle python object.

    Args:
        file_path (str): Path-like string to the object to be loaded.

    Returns:
        object
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def parse_mt(mt_number, mt_for="ENDF", one_hot=False):
    """Universal ENDF reaction code parser.

    This internal function is used to parse and format the reaction integer code for internal functions used by NucML.

    Args:
        mt_number (int): Reaction channel code as defined by ENDF/EXFOR.
        mt_for (str, optional): What loader object is requesting the parsing. Options include "EXFOR" and "ENDF".
            Defaults to "ENDF".
        one_hot (bool, optional): If mt_for="EXFOR", then this argument specifies if the MT code should be formated for
            one-hot encoded dataframe. Defaults to False.

    Returns:
        str or int: The formatted reaction channel code.
    """
    mt_number = str(int(mt_number))
    if mt_for.upper() == "ENDF":
        if len(mt_number) != 3:
            mt_number = mt_number.zfill(3)
        return "MT" + mt_number
    elif mt_for.upper() in ["EXFOR", "ACE"]:
        return "MT_" + mt_number if one_hot else int(mt_number)
    else:
        raise NotSupportedError("Only ENDF, EXFOR, or ACE are supported as mt_for.")


def parse_isotope(isotope, parse_for="ENDF"):
    """Transform element tags (i,e, U235) into formats appropiate for other internal functions.

    Args:
        isotope (str): Isotope to format (i.e. U235, 35cl).
        parse_for (str, optional): What loader object is requesting the parsing. Options include "EXFOR" and "ENDF".
            Defaults to "ENDF".

    Returns:
        str: Formatted isotope identifier.
    """
    element, mass = re.findall(r'[A-Za-z]+|\d+', isotope)
    if element.isdigit():
        mass, element = re.findall(r'[A-Za-z]+|\d+', isotope)
    element = element.capitalize()
    if parse_for.upper() == "ENDF":
        if len(mass) != 3:
            mass = mass.zfill(3)
        return element + mass
    elif parse_for.upper() == "ENSDF":
        return mass + element


def close_open_files(files):
    """Close open files."""
    for file in files:
        file.close()


def _get_str_with_separators(text, separator_index, separator="|"):
    string = list(text)
    for i, j in enumerate(separator_index):
        string.insert(i + j, separator)
    return "".join(string)


def _write_file_with_separators(open_path, output_path, separator_index, separator="|"):
    with open(open_path) as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if not line.strip():
                continue
            string = list(line)
            for i, j in enumerate(separator_index):
                string.insert(i + j, separator)

            outfile.write("".join(string))


def convert_dos_to_unix(file_path):
    """Convert a given file from DOS to UNIX.

    Args:
        file_path (str): Path to file to convert.

    Returns:
        None
    """
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    # relative or absolute file path, e.g.:
    with open(file_path, 'rb') as open_file:
        content = open_file.read()

    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

    with open(file_path, 'wb') as open_file:
        open_file.write(content)


def parse_zzzaaa(ZZZAAA):
    z = int(ZZZAAA[:3])
    a = int(ZZZAAA[3:])
    return z, a


def _download_url_file(url, saving_path):
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'NucML')
    opener.retrieve(url, saving_path)
    return


def _download_and_extract_zip_file(url: str, saving_path: Path):
    _download_url_file(url, saving_path)
    with zipfile.ZipFile(saving_path, 'r') as zip_ref:
        zip_ref.extractall(saving_path.parent)
    saving_path.unlink()
    return
