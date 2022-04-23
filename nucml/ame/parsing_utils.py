"""Utility functions for the parsing utilities."""
import logging
import os
import pandas as pd

from nucml.general_utilities import check_if_files_exist
from nucml.processing import impute_values

logger = logging.getLogger(__name__)


def add_extra_features(df):
    """Add reaction Q-values.

    ["Q(g,p)"] = -1 * ["S(p)"]
    ["Q(g,n)"] = -1 * ["S(n)"]
    ["Q(g,pn)"] = ["Q(d,a)"] - 26071.0939
    ["Q(g,d)"] = ["Q(d,a)"] - 23846.5279
    ["Q(g,t)"] = ["Q(p,a)"] - 19813.8649
    ["Q(g,He3)"] = ["Q(n,a)"] - 20577.6194
    ["Q(g,2p)"] = -1 * ["S(2p)"]
    ["Q(g,2n)"] = -1 * ["S(2n)"]
    ["Q(g,a)"] = ["Q(a)"]
    ["Q(p,n)"] = ["B_Decay_Energy"] - 782.3465
    ["Q(p,2p)"] = -1 * ["S(p)"]
    ["Q(p,pn)"] = -1 * ["S(n)"]
    ["Q(p,d)"] = -1 * ["S(n)"] + 2224.5660
    ["Q(p,2n)"] = ["Q(B-n)"] - 782.3465
    ["Q(p,t)"] = -1 * ["S(2n)"] + 8481.7949
    ["Q(p,3He)"] = ["Q(d,a)"] - 18353.0535
    ["Q(n,2p)"] = ["Q(ep)"] + 782.3465
    ["Q(n,np)"] = -1 * ["S(p)"]
    ["Q(n,d)"] = -1 * ["S(p)"] + 2224.5660
    ["Q(n,2n)"] = -1 * ["S(n)"]
    ["Q(n,t)"] = ["Q(d,a)"] - 17589.2989
    ["Q(n,3He)"] = -1 * ["S(2p)"] + 7718.0404
    ["Q(d,t)"] = -1 * ["S(n)"] + 6257.2290
    ["Q(d,3He)"] = -1 * ["S(p)"] + 5493.4744
    ["Q(3He,t)"] = ["B_Decay_Energy"] - 18.5920
    ["Q(3He,a)"] = -1 * ["S(n)"] + 20577.6194
    ["Q(t,a)"] = -1 * ["S(p)"] + 19813.8649

    Args:
        df (pandas.DataFrame):

    Returns:
        pandas.DataFrame:
    """
    df["Q(g,p)"] = -1 * df["S(p)"]
    df["Q(g,n)"] = -1 * df["S(n)"]
    df["Q(g,pn)"] = df["Q(d,a)"] - 26071.0939
    df["Q(g,d)"] = df["Q(d,a)"] - 23846.5279
    df["Q(g,t)"] = df["Q(p,a)"] - 19813.8649
    df["Q(g,He3)"] = df["Q(n,a)"] - 20577.6194
    df["Q(g,2p)"] = -1 * df["S(2p)"]
    df["Q(g,2n)"] = -1 * df["S(2n)"]
    df["Q(g,a)"] = df["Q(a)"]
    df["Q(p,n)"] = df["B_Decay_Energy"] - 782.3465
    df["Q(p,2p)"] = -1 * df["S(p)"]
    df["Q(p,pn)"] = -1 * df["S(n)"]
    df["Q(p,d)"] = -1 * df["S(n)"] + 2224.5660
    df["Q(p,2n)"] = df["Q(B-n)"] - 782.3465
    df["Q(p,t)"] = -1 * df["S(2n)"] + 8481.7949
    df["Q(p,3He)"] = df["Q(d,a)"] - 18353.0535
    df["Q(n,2p)"] = df["Q(ep)"] + 782.3465
    df["Q(n,np)"] = -1 * df["S(p)"]
    df["Q(n,d)"] = -1 * df["S(p)"] + 2224.5660
    df["Q(n,2n)"] = -1 * df["S(n)"]
    df["Q(n,t)"] = df["Q(d,a)"] - 17589.2989
    df["Q(n,3He)"] = -1 * df["S(2p)"] + 7718.0404
    df["Q(d,t)"] = -1 * df["S(n)"] + 6257.2290
    df["Q(d,3He)"] = -1 * df["S(p)"] + 5493.4744
    df["Q(3He,t)"] = df["B_Decay_Energy"] - 18.5920
    df["Q(3He,a)"] = -1 * df["S(n)"] + 20577.6194
    df["Q(t,a)"] = -1 * df["S(p)"] + 19813.8649
    return df


def create_natural_element_data(originals_directory, saving_directory, fillna=True, mode="elemental", fill_value=0):
    """Create natural element data by averaging isotopic data.

    Additionally it adds a flag to indicate rows which correspond to isotopic or natural data.

    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the
            periodic_table csv file is located.
        saving_directory (str): Path to directory where the resulting formatted
            csv file will be saved including the AME_all_merged.csv file.
        fillna (bool): If True, missing values are filled. For the remaining NaN values not filled by the
            used `mode`, a value of 0 will be inserted unless specified otherwise.
        mode (str): The supported modes are:
            elemental: missing values are filled using linear interpolation element-wise.
        fill_value (float): Value to fill remaining missing values with after imputation is finished
            with selected `mode`. Defaults to 0.

    Returns:
        None
    """
    directory = saving_directory
    logger.info("FEAT ENG: Initializing. Checking documents...")
    filename = os.path.join(directory, "AME_all_merged.csv")
    periodic_filename = os.path.join(originals_directory, "periodic_table.csv")
    if not check_if_files_exist([filename, periodic_filename]):
        raise FileNotFoundError("One file does not exist.")

    logger.info("FEAT ENG: Reading data from {}".format(filename))
    ame = pd.read_csv(filename)
    ame = ame.replace(to_replace=-0, value=0)  # FORMATTING

    logger.info("FEAT ENG: Reading data from {}".format(periodic_filename))
    masses_natural = pd.read_csv(periodic_filename).rename(
        # Renaming columns for consistency with EXFOR:
        columns={
            'NumberofNeutrons': 'Neutrons', 'NumberofProtons': 'Protons',
            'AtomicMass': 'Atomic_Mass_Micro', 'Symbol': 'EL'})

    logger.info("FEAT ENG: Beginning data creation...")
    masses_natural["Mass_Number"] = masses_natural["Neutrons"] + masses_natural["Protons"]
    # We don't need other columns in the periodic table csv file
    masses_natural = masses_natural[["Neutrons", "Protons", "Mass_Number", "EL", "Atomic_Mass_Micro"]]

    # In EXFOR natural data is represented with a negative neutron value so we create this here:
    masses_natural["N"] = masses_natural["Neutrons"] * 0
    masses_natural["A"] = masses_natural["Mass_Number"] * 0
    masses_natural.columns = ["N", "Z", "A", "EL", "Atomic_Mass_Micro", "Neutrons", "Mass_Number"]
    masses_natural["Neutrons"] = masses_natural["Mass_Number"] - masses_natural["Z"]

    # AME datasets deal with atomic mass in micro units:
    masses_natural["Atomic_Mass_Micro"] = masses_natural["Atomic_Mass_Micro"] * 1E6

    # We need to distinguish natural form isotopic. To accomplish this we introduce a flag:
    masses_natural["Flag"] = "N"

    logger.info("FEAT ENG: Finished creating natural data. Merging with AME...")
    result = ame.append(masses_natural, sort=False)

    # Due to the merging process many NaN values are introduced. Here we fix this:
    result["Neutrons"] = result.Neutrons.fillna(result.N).astype(int)  # Fill the Neutrons column with the N column
    result["Mass_Number"] = result.Mass_Number.fillna(result.A).astype(int)  # same for Mass Number and A
    result.Flag.fillna("I", inplace=True)  # We already have our natural tags we now that all NaNs are isotopic now.
    result["O"].fillna(value="Other", inplace=True)  # ASSUMPTION: We assume natural data was derive with Other

    logger.info("FEAT ENG: Finishing up...")
    result = result.drop(columns=["Element_w_A"])  # We don't need this
    result = result.sort_values(by="Z")

    csv_name = os.path.join(saving_directory, "AME_Natural_Properties_w_NaN.csv")
    logger.info("FEAT ENG: Saving file to {}".format(csv_name))
    result.to_csv(csv_name, index=False)

    if fillna:
        logger.info("FEAT ENG: Filling missing values using {} mode".format(mode.upper()))

        # The imputation methods change the column data data types, we save them
        # and transfer them after the imputation is perform.
        types = result.iloc[0:2]
        if mode.upper() == "ELEMENTAL":
            # we fill the nans by taking the average of all isotopes, same for all other parameters.
            result = impute_values(result)

        logger.info("FEAT ENG: Filling remaining NaN values with 0...")
        result = result.fillna(fill_value)

        logger.info("FEAT ENG: Returning features to original data types...")
        for x in result.columns:
            result[x] = result[x].astype(types[x].dtypes.name)

        csv_name = os.path.join(saving_directory, "AME_Natural_Properties_no_NaN.csv")
        logger.info("FEAT ENG: Saving imputed file to {}".format(csv_name))
        result.to_csv(csv_name, index=False)
