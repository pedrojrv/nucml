"""Data loading functions.

Contains the main utility functions to load different datasets including EXFOR, AME, ENDF, ENSDF, and more.
"""

import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd


from nucml import configure
import nucml.general_utilities as gen_utils

logging.basicConfig(level=logging.INFO)


config = configure._get_config()
ame_dir_path = config['DATA_PATHS']['AME']
evaluations_path = config['DATA_PATHS']['EVALUATION']
ensdf_path = config['DATA_PATHS']['ENSDF']
exfor_path = config['DATA_PATHS']['EXFOR']


dtype_exfor = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/EXFOR_AME_dtypes.pkl'))
exfor_elements = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/exfor_elements_list.pkl'))
elements_dict = gen_utils.load_obj(os.path.join(os.path.dirname(__file__), 'objects/Element_AAA.pkl'))


def load_evaluation(isotope: str, MT, mode="neutrons", library="endfb8.0", log=True, drop_u=True):
    """Read an evaluation file for a specific isotope, reaction channel, and evaluated library.

    It is important to inspect the returned data since it queries a local database of an external source which
    extracted data from ENDF using an extraction script. It has been found that some particular reactions are not
    included. These can be added manually for future loading.

    Args:
        isotope (str): Isotope to query (i.e. U233, Cl35).
        MT (int): Reaction channel ENDF code. Must be an integer (i.e. 1, 2, 3)
        mode (str): Type of projectile. Only "neutrons" and "protons" are supported for now.
        library (str): Evaluation library to query. Allowed options include endfb8.0, jendl4.0, jeff3.3, and tendl.2019.
        mev_to_ev (bool): If True, it converts the energy from MeV to eV.
        mb_to_b (bool): If True, it converts the cross sections from millibarns to barns.
        log (bool): If True, it applies the log10 to both the Energy and the Cross Section.
        drop_u (bool): Sometimes, evaluation files contain uncertainty values. If True, these features are removed.

    Returns:
        evaluation (DataFrame): pandas DataFrame containing the ENDF datapoints.
    """
    MT = gen_utils.parse_mt(MT)
    isotope = gen_utils.parse_isotope(isotope, parse_for='endf')
    projectile_dict = {'protons': 'p', 'neutrons': 'n'}
    projectile = projectile_dict[mode]

    path = os.path.join(evaluations_path, f'{mode}/{isotope}/{library}/tables/xs/{projectile}-{isotope}-{MT}.{library}')

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError('Evaluation file does not exists at {}'.format(path))
    logging.debug("EVALUATION: Extracting data from {}".format(path))
    evaluation = pd.read_csv(
        path, skiprows=5, header=None, names=["Energy", "Data", "dDataLow", "dDataUpp"],
        delim_whitespace=True)

    # Convert energy into eV rather than Mev
    evaluation["Energy"] = evaluation["Energy"]*1E6
    evaluation["Data"] = evaluation["Data"]*0.001
    if log:
        evaluation["Energy"] = np.log10(evaluation["Energy"])
        evaluation["Data"] = np.log10(evaluation["Data"])
        evaluation["dDataLow"] = np.log10(evaluation["dDataLow"])
        evaluation["dDataUpp"] = np.log10(evaluation["dDataUpp"])
    if drop_u:
        evaluation = evaluation.drop(columns=["dDataLow", "dDataUpp", "dDataLow", "dDataUpp"], errors='ignore')
    logging.info("EVALUATION: Finished. ENDF data contains {} datapoints.".format(evaluation.shape[0]))
    return evaluation
