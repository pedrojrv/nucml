"""Utilities to generate data/datasets."""
import os
import pandas as pd

import nucml.exfor.parsing as exfor_parsing
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
        exfor_parsing.csv_creator(heavy_dir, tmp_dir, mode, append_ame=True)
        exfor_parsing.impute_original_exfor(heavy_dir, tmp_dir, mode)
    return None


def generate_bigquery_csv():
    """Create a single EXFOR data file to update Google BigQuery database.

    Returns:
        None
    """
    alphas = pd.read_csv(os.path.join(exfor_path, "EXFOR_alphas/EXFOR_alphas_ORIGINAL.csv"))
    deuterons = pd.read_csv(os.path.join(exfor_path, "EXFOR_deuterons/EXFOR_deuterons_ORIGINAL.csv"))
    gammas = pd.read_csv(os.path.join(exfor_path, "EXFOR_gammas/EXFOR_gammas_ORIGINAL.csv"))
    helions = pd.read_csv(os.path.join(exfor_path, "EXFOR_helions/EXFOR_helions_ORIGINAL.csv"))
    neutrons = pd.read_csv(os.path.join(exfor_path, "EXFOR_neutrons/EXFOR_neutrons_ORIGINAL.csv"))
    protons = pd.read_csv(os.path.join(exfor_path, "EXFOR_protons/EXFOR_protons_ORIGINAL.csv"))

    final = alphas.append(deuterons).append(gammas).append(helions).append(neutrons).append(protons)

    NEW_NAMES = {"Cos/LO": "Cos_LO", "dCos/LO": "dCos_LO", "ELV/HL": "ELV_HL", "dELV/HL": "dELV_HL"}
    final = final.rename(NEW_NAMES, axis=1)

    final.to_csv(os.path.join(exfor_path, "EXFOR_original.csv"), index=False)
    return None
