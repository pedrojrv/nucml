"""Utilities to generate data/datasets."""
import os

import nucml.exfor.parsing as exfor_parsing
import nucml.exfor.csv_creator as csv_creator
import nucml.config as config

exfor_path = config.exfor_path


def generate_exfor_dataset(user_path, modes=["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"]):
    """Generate all EXFOR datasets for neutron-, proton-, alpha-, deuterons-, gammas-, and helion-induce reactions.

    Beware, NucML configuration needs to be performed first. See nucml.configure. The `modes` argument can be modified
    for the function to generate only user-defined datasets.

    Args:
        user_path (str): path-like string where all information including the datasets will be stored.
        modes (list, optional): Type of projectile for which to generate the datasets.
            Defaults to ["neutrons", "protons", "alphas", "deuterons", "gammas", "helions"].

    Returns:
        None
    """
    user_abs_path = os.path.abspath(user_path)
    tmp_dir = os.path.join(user_abs_path, "EXFOR/tmp/")
    heavy_dir = os.path.join(user_abs_path, "EXFOR/CSV_Files/")
    for mode in modes:
        tmp_dir = os.path.join(user_abs_path, "EXFOR/tmp/")
        heavy_dir = os.path.join(user_abs_path, "EXFOR/CSV_Files/")
        exfor_directory = os.path.join(user_abs_path, "EXFOR/C4_Files/{}".format(mode))

        exfor_parsing.get_all(exfor_parsing.get_c4_names(exfor_directory), heavy_dir, tmp_dir, mode=mode)
        csv_creator.csv_creator(heavy_dir, tmp_dir, mode, append_ame=True)
        csv_creator.impute_original_exfor(heavy_dir, tmp_dir, mode)
