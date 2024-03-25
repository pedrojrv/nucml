"""Utilities to generate data/datasets."""
import argparse
import shutil
import glob
import logging
from pathlib import Path

from nucml import general_utilities
from nucml.datasets.utilities import _ame_parser, _ensdf_parser, _exfor_c4_parser, _exfor_parser, _exfor_csv_creator
from nucml.configure import set_data_paths
from nucml.general_utilities import _download_and_extract_zip_file
from nucml._constants import EXFOR_MODES, EVALUATION_DATASET_URL, RIPL_DATASET_URL

logger = logging.getLogger(__name__)


def generate_ame_dataset(saving_dir: Path) -> None:
    """Create 5 CSV files: Proccesed (1) mass, (2) rct1, and (3) rct2 files.

    It then creates a (4) single CSV merging the first three CSV files. It then creates (5) a proccesed CSV file
    containing isotpic and natural element data with NaN values. If wanted a (6) copy of the fifth
    CSV file is saved with imputed NaN values.

    Args:
        saving_dir (str): Path to directory where the resulting formatted csv file will be saved.
    """
    _ame_parser._get_ame_originals(saving_dir)
    _ame_parser._parse_mass(saving_dir)
    _ame_parser._parse_rct(saving_dir, rct_file=1)
    _ame_parser._parse_rct(saving_dir, rct_file=2)
    _ame_parser._merge_mass_rct(saving_dir, add_qvalues=True, create_imputed=True)
    _ame_parser._create_natural_element_data(saving_dir, fillna=True, fill_value=0)
    _ame_parser._clean_up_originals(saving_dir)


def generate_evaluation_dataset(saving_dir: Path) -> None:
    """Download the evaluation directory containing neutron and proton data.

    Args:
        saving_dir (str or pathlib.Path): Path to directory on which to save the generated files.
    """
    zip_dir = saving_dir / 'evaluations.zip'
    _download_and_extract_zip_file(EVALUATION_DATASET_URL, zip_dir)


def generate_ensdf_dataset(saving_dir: Path) -> None:
    """Download and parse the RIPL dataset.

    Args:
        saving_dir (Path): Path to directory on which to save the generated files.
    """
    zip_dir = saving_dir / 'levels.zip'
    _download_and_extract_zip_file(RIPL_DATASET_URL, zip_dir)

    dat_files = _ensdf_parser._get_ripl_dat_paths(saving_dir / 'levels/')
    _ensdf_parser._generate_headers(dat_files, saving_dir)
    _ensdf_parser._generate_elemental_ensdf(dat_files, saving_dir)
    _ensdf_parser._get_stable_states(saving_dir)
    _ensdf_parser._generate_ensdf_csv(saving_dir)
    _ensdf_parser._get_level_parameters(saving_dir)
    _ensdf_parser._generate_cutoff_ensdf(saving_dir)

    shutil.rmtree(zip_dir.with_suffix(''))


def generate_exfor_dataset(saving_dir: Path, modes=EXFOR_MODES) -> None:
    """Generate all EXFOR datasets for neutron-, proton-, alpha-, deuterons-, gammas-, and helion-induce reactions.

    Beware, NucML configuration needs to be performed first. See nucml.configure. The `modes` argument can be modified
    for the function to generate only user-defined datasets.

    Args:
        saving_dir (str): path-like string where all information including the datasets will be stored.
        modes (list, optional): Type of projectile for which to generate the datasets.
            Defaults to ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"].

    Returns:
        None
    """
    _exfor_parser._download_exfor_c4(saving_dir)
    xc4_files = glob.glob(str(saving_dir / '*.xc4'))
    if len(xc4_files) != 1:
        raise FileExistsError(f'There should be (only) one .xc4 file. Found {len(xc4_files)} files.')

    xc4_file = Path(xc4_files[0])
    c4_dir = saving_dir / 'c4_files/'
    c4_dir.mkdir(exist_ok=True)
    _exfor_c4_parser.parse_and_sort_c4_files(xc4_file, c4_dir)

    tmp_dir = saving_dir / "tmp/"
    for mode in modes:
        exfor_directory = c4_dir / f"{mode}"
        c4_files = _exfor_parser.get_c4_names(exfor_directory)
        _exfor_parser.get_all(c4_files, saving_dir, tmp_dir, mode)
        _exfor_csv_creator.csv_creator(saving_dir, tmp_dir, mode, append_ame=True)
        _exfor_csv_creator.impute_original_exfor(saving_dir, tmp_dir, mode)

    shutil.rmtree(saving_dir / 'tmp/')


def generate_all_datasets(user_path):
    """Generate the AME, Evaluation, ENSDF, and EXFOR datasets used by NucML.

    Raises:
        FileNotFoundError: If the provided saving directory is not valid.
    """
    ame_saving_dir = user_path / 'ame/'
    evaluation_saving_dir = user_path / 'evaluations/'
    ensdf_saving_dir = user_path / 'ensdf/'
    exfor_saving_dir = user_path / 'exfor/'

    data_paths = {
        'AME': str(ame_saving_dir.resolve()),
        'EVALUATION': str(evaluation_saving_dir.resolve()),
        'ENSDF': str(ensdf_saving_dir.resolve()),
        'EXFOR': str(exfor_saving_dir.resolve()),
    }

    set_data_paths(data_paths)

    # Generate AME datasets
    general_utilities.initialize_directories(ame_saving_dir, reset=True)
    generate_ame_dataset(ame_saving_dir)

    # Download evaluation dataset from GCP
    generate_evaluation_dataset(user_path)

    # Download RIPL and generate ENSDF dataset
    general_utilities.initialize_directories(ensdf_saving_dir, reset=True)
    generate_ensdf_dataset(ensdf_saving_dir)

    general_utilities.initialize_directories(exfor_saving_dir, reset=True)
    generate_exfor_dataset(exfor_saving_dir)


def main():
    """See documentation for generate_all_datasets()."""
    parser = argparse.ArgumentParser(description='Download and generate datasets used by NucML.')
    parser.add_argument(
        "-s", "--saving-dir", action='store', dest='user_path',
        help='Path to directory on which to generate and store all the data.')
    args = parser.parse_args()

    user_path = Path(args.user_path)
    if not user_path.exists():
        raise FileNotFoundError(f"Directory {user_path} was not found. Make sure it is a valid directory.")
    generate_all_datasets(user_path)


if __name__ == "__main__":
    main()
