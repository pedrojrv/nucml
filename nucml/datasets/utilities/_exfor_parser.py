"""Parsing utilities for the EXFOR database."""
from io import TextIOWrapper
from pathlib import Path
from typing import List
from natsort import natsorted

from nucml import general_utilities
from nucml._constants import EXFOR_DATASET_URL
from nucml import configure

config = configure._get_config()
ame_dir_path = config['DATA_PATHS']['AME']


def _download_exfor_c4(saving_dir: Path) -> None:
    """Download and unzip the EXFOR files from the IAEA.

    Args:
        saving_dir (Path): Path on which to download and unzip the EXFOR file.

    Raises:
        FileExistsError: If the directory already has an EXFOR dataset.
    """
    zip_dir = saving_dir / 'exfor.zip'
    unzip_dir = saving_dir / 'exfor/'
    if unzip_dir.exists():
        raise FileExistsError(f"Unable to download and generate EXFOR dataset. Directory {unzip_dir} already exists.")

    # Download latest EXFOR data from the IAEA
    general_utilities._download_and_extract_zip_file(EXFOR_DATASET_URL, zip_dir)


def get_c4_names(c4_directory: Path) -> List[str]:
    """Search given directory for EXFOR-generated C4 files.  It returns a list of relative paths for each found file.

    Args:
        c4_directory (str): Path to the directory containing all .c4 files.

    Returns:
        list: Contains relative paths to each encountered .c4 file.

    Raises:
        FileNotFoundError: If no C4 files are found an error will be raised.
    """
    names = natsorted(general_utilities.get_files_w_extension(c4_directory, ".c4"))
    if len(names) == 0:
        raise FileNotFoundError("No .C4 files found. Check your provided path.")
    return names


def _extract_basic_data_from_c4(c4_file: str, tmp_path: Path) -> None:
    # Extract experimental data, authors, years, institutes, and dates
    with open(c4_file) as infile, \
            open(tmp_path / "all_cross_sections.txt", 'a') as num_data, \
            open(tmp_path / 'authors.txt', 'a') as authors, \
            open(tmp_path / 'years.txt', 'a') as years, \
            open(tmp_path / 'institutes.txt', 'a') as institute, \
            open(tmp_path / 'entry.txt', 'a') as entry, \
            open(tmp_path / 'refcode.txt', 'a') as refcode, \
            open(tmp_path / 'dataset_num.txt', 'a') as dataset_num, \
            open(tmp_path / 'dates.txt', 'a') as date:
        tag_writers = {'#AUTHOR1': authors, '#YEAR': years, '#ENTRY': entry, '#REF-CODE': refcode}
        copy = False
        for line in infile:
            matched = [match for match in tag_writers.keys() if line.startswith(match)]
            if matched:
                copy = False
                tag_writers[matched[0]].write(line)
            elif line.startswith(r'#DATASET') and len(line) > 16:
                copy = False
                dataset_num.write(line)
            elif line.startswith(r"#INSTITUTE"):
                copy = False
                institute.write(line)
            elif line.startswith(r"#DATE"):
                copy = False
                date.write(line)
            elif line.startswith(r"#---><---->o<-><-->ooo<-------><-------><-------><-------><-------><-------><-------><-------><-><-----------------------><---><->o"):  # noqa
                copy = True
            elif line.startswith(r"#/DATA"):
                copy = False
            elif copy:
                num_data.write(line)


def _write_complex_data(outfile: TextIOWrapper, lines: List[str], idx: int) -> None:
    line = lines[idx]
    detection_point = r"#+"
    if not lines[idx + 2].startswith(detection_point):
        outfile.write(line)
        return

    to_write = str(line.strip('\n')) + " " + str(lines[idx+2].strip('#+').strip())
    if lines[idx + 4].startswith(detection_point):
        to_write += " " + str(lines[idx+4].strip('#+').strip())
        if lines[idx + 6].startswith(detection_point):
            to_write += " " + str(lines[idx+6].strip('#+').strip())

    to_write += "\n"
    outfile.write(to_write)


def _extract_complex_data_from_c4(c4_file: str, tmp_path: Path) -> None:
    with open(c4_file, "r") as infile, \
            open(tmp_path / 'titles.txt', 'a') as titles, \
            open(tmp_path / 'references.txt', 'a') as references, \
            open(tmp_path / 'data_points_per_experiment_refined.txt', 'a') as data_points, \
            open(tmp_path / 'reaction_notations.txt', 'a') as reactions:
        lines = infile.readlines()
        # The space after #DATA is important to distinguish it from other similar words
        writers = {'#TITLE': titles, '#REFERENCE': references, '#DATA ': data_points, '#REACTION': reactions}
        for idx, line in enumerate(lines):
            matched = [match for match in writers.keys() if line.startswith(match)]
            if matched:
                _write_complex_data(writers[matched[0]], lines, idx)
        reactions.write(line)


def get_all(c4_files: List[str], saving_dir: Path, tmp_path: Path, mode: str) -> None:
    """Retrieve all avaliable information from all .c4 files.

    This function combines the proccesses defined on:

    - get_c4_names()
    - get_raw_datapoints()
    - get_authors()
    - get_years()
    - get_institutes()
    - get_dates()
    - get_titles()
    - get_references()
    - get_reaction_notation()
    - get_datapoints_per_experiment()

    It is optimized to run faster than running the individual functions.

    Args:
        c4_files (list): List containing paths to all .c4 files.
        saving_dir (str): Path to directory where heavy files are to be saved.
        tmp_path (str): Path to directory where temporary files are to be saved.
        mode (str): The reaction projectile of the provided C4 files.
    """
    tmp_path = tmp_path / f"Extracted_Text_{mode}"
    saving_dir = saving_dir / mode
    # general_utilities.initialize_directories([tmp_path, saving_dir], reset=True)

    # for c4_file in c4_files:
    #     _extract_basic_data_from_c4(c4_file, tmp_path)

    # Extract titles, references, and number of data points per experiment...")
    # for c4_file in c4_files:
    #     _extract_complex_data_from_c4(c4_file, tmp_path)

    # Format experimental data
    tmp_xs_file = tmp_path / "all_cross_sections.txt"
    output_path = saving_dir / "all_cross_sections.txt"
    indexex = [5, 11, 12, 15, 19, 20, 21, 22, 31, 40, 49, 58, 67, 76, 85, 94, 97, 122, 127, 130]
    general_utilities._write_file_with_separators(tmp_xs_file, output_path, indexex, ";")
    tmp_xs_file.unlink()
