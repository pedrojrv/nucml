"""Parsing utilities for the AME database."""
import logging
import os
import sys
import numpy as np
import pandas as pd
import requests
import warnings

from nucml.general_utilities import check_if_files_exist
from nucml.processing import impute_values

pd.options.mode.chained_assignment = None  # default='warn'


def get_ame_originals(originals_directory):
    """Request and store the three AME original files for further processing from the IAEA website.

    Args:
        originals_directory (str): Path-like string where the text files will be stored.

    Returns:
        None
    """
    logging.info("AME: Requesting data text files.")
    # periodic_table = requests.get(
    # 'https://raw.githubusercontent.com/pedrojrv/ML_Nuclear_Data/master/AME/Originals/periodic_table.csv').content
    mass16_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/mass16.txt').content
    rct1_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/rct1-16.txt').content
    rct2_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/rct2-16.txt').content

    logging.info('AME: Saving text files in provided directory.')
    # with open(os.path.join(originals_directory, 'periodic_table.csv'), 'wb') as f:
    #     f.write(periodic_table)
    with open(os.path.join(originals_directory, 'mass16.txt'), 'wb') as f:
        f.write(mass16_txt)
    with open(os.path.join(originals_directory, 'rct1-16.txt'), 'wb') as f:
        f.write(rct1_txt)
    with open(os.path.join(originals_directory, 'rct2-16.txt'), 'wb') as f:
        f.write(rct2_txt)
    return None


def read_mass16(originals_directory, saving_directory):
    """Read the mass16.txt file and creates a formatted CSV file.

    The Mass 16 file contains a variety of features including atomic mass, mass excess, binding energy, beta decay
    energy, and more. For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5


    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the mass16_toparse.txt file is
            located.
        saving_directory (str): Path to save resulting formatted csv file.

    Returns:
        None
    """
    # Formating rules based on fortran formatting given by AME
    formatting = ((0, 1), (1, 5), (5, 9), (9, 14), (14, 19), (19, 20), (20, 23), (23, 27), (27, 28),
                  (28, 41), (41, 52), (52, 63), (63, 72), (72, 73), (73, 75), (75, 86), (86, 95), (95, 96),
                  (96, 112), (112, 123))

    # Column names as given by the AME documentation
    column_names = [
        "Page_Feed", "NZ", "N", "Z", "A", "Other", "EL", "O", "Other2",
        "Mass_Excess", "dMass_Excess", "Binding_Energy", "dBinding_Energy", "Other3",
        "Beta_Type", "B_Decay_Energy", "dB_Decay_Energy", "Other4",
        "Atomic_Mass_Micro", "dAtomic_Mass_Micro"]

    filename = os.path.join(originals_directory, "mass16.txt")
    logging.info("MASS16: Reading data from {}".format(filename))
    data = pd.read_fwf(filename, colspecs=formatting, header=None, skiprows=39, names=column_names)

    logging.info("MASS16: Beginning formatting sequences...")
    data["O"].fillna(value="Other", inplace=True)
    data = data.replace(value=np.nan, to_replace="*")

    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].apply(lambda x: str(x).replace("#", ""))

    mass16_dtypes = [
        'float64', 'int64', 'int64', 'int64', 'int64', 'float64', 'object', 'object', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'object', 'object', 'float64', 'float64', 'object', 'float64']

    for col, types in zip(data.columns, mass16_dtypes):
        data[col] = data[col].astype(types)

    for col in ["Atomic_Mass_Micro"]:
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip("\"")
        data[col] = data[col].str.replace(" ", "")
        data[col] = data[col].str.strip()

    data["Atomic_Mass_Micro"] = data["Atomic_Mass_Micro"].astype(float)
    data["B_Decay_Energy"] = data["B_Decay_Energy"].astype(float)

    data.drop(columns=["Page_Feed", "Other", "Other2", "Other3", "Beta_Type", "Other4", "NZ"], inplace=True)

    data["Element_w_A"] = data["A"].astype(str) + data["EL"]

    csv_name = os.path.join(saving_directory, "AME_mass16.csv")
    logging.info("MASS16: Formatting done. Saving file to {}".format(csv_name))
    data.to_csv(csv_name, index=False)
    logging.info("MASS16: Succesfully formated mass16.txt file.")
    return None


def read_rct1(originals_directory, saving_directory):
    """Read the rct1-16.txt file and creates a formatted CSV file.

    The rct1-16 file contains a variety of
    features including neutron and proton separation energies and q-values for a variety of reactions.
    For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,1x,a3,i3,1x,6(f10.2,f8.2)

    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the rct1-16.txt file is located.
        saving_directory (str): Path to save resulting formatted csv file.

    Returns:
        None
    """
    formatting = (
        (0, 1), (1, 4), (4, 5), (5, 8), (8, 11), (11, 12), (12, 22), (22, 30), (30, 40),
        (40, 48), (48, 58), (58, 66), (66, 76), (76, 84), (84, 94), (94, 102), (102, 112), (112, 120))

    column_names = [
        "Page_Feed", "A", "Other", "EL", "Z", "Other2", "S(2n)", "dS(2n)", "S(2p)", "dS(2p)",
        "Q(a)", "dQ(a)", "Q(2B-)", "dQ(2B-)", "Q(ep)", "dQ(ep)", "Q(B-n)", "dQ(B-n)"]

    filename = os.path.join(originals_directory, "rct1-16.txt")

    logging.info("RCT1: Reading data from {}".format(filename))
    data = pd.read_fwf(filename, colspecs=formatting, header=None, skiprows=39, names=column_names)

    logging.info("RCT1: Beginning formatting sequences...")
    data = data.replace(to_replace="*", value=np.nan)
    data.drop(columns=["Other", "Other2"], inplace=True)

    for col in list(data.columns):
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip("\"")
        data[col] = data[col].str.strip()
        data[col] = data[col].str.replace("#", ".")

    for col in list(data.columns):
        if col == "EL":
            pass
        else:
            data[col] = data[col].astype(float)

    data[["A", "Z"]] = data[["A", "Z"]].astype(int)
    data["N"] = data["A"] - data["Z"]
    data["Element_w_A"] = data["A"].astype(str) + data["EL"]
    data.drop(columns=["Page_Feed", "A", "EL", "Z", "N"], inplace=True)

    csv_name = os.path.join(saving_directory, "AME_rct1.csv")
    logging.info("RCT1: Formatting done. Saving file to {}".format(csv_name))
    data.to_csv(csv_name, index=False)
    logging.info("RCT1: Succesfully formated rct1-16.txt file.")
    return None


def read_rct2(originals_directory, saving_directory):
    """Read the rct2-16.txt file and creates a formatted CSV file.

    The rct2-16 file contains a variety of
    features including neutron and proton separation energies and q-values for a variety of reactions.
    For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,1x,a3,i3,1x,6(f10.2,f8.2)

    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the rct2-16.txt file is located.
        saving_directory (str): Path to save resulting formatted csv file.

    Returns:
        None
    """
    formatting = (
        (0, 1), (1, 4), (4, 5), (5, 8), (8, 11), (11, 12), (12, 22), (22, 30), (30, 40),
        (40, 48), (48, 58), (58, 66), (66, 76), (76, 84), (84, 94), (94, 102), (102, 112), (112, 120))

    column_names = [
        "Page_Feed", "A", "Other", "EL", "Z", "Other2", "S(n)", "dS(n)", "S(p)", "dS(p)", "Q(4B-)",
        "dQ(4B-)", "Q(d,a)", "dQ(d,a)", "Q(p,a)", "dQ(p,a)", "Q(n,a)", "dQ(n,a)"]

    filename = os.path.join(originals_directory, "rct2-16.txt")

    logging.info("RCT2: Reading data from {}".format(filename))
    data = pd.read_fwf(filename, colspecs=formatting, header=None, skiprows=39, names=column_names)

    logging.info("RCT2: Beginning formatting sequences...")
    data = data.replace(to_replace="*", value=np.nan)
    data.drop(columns=["Other", "Other2"], inplace=True)

    for col in list(data.columns):
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip("\"")
        data[col] = data[col].str.strip()
        data[col] = data[col].str.replace("#", ".")

    for col in list(data.columns):
        if col == "EL":
            pass
        else:
            data[col] = data[col].astype(float)

    data[["A", "Z"]] = data[["A", "Z"]].astype(int)
    data["N"] = data["A"] - data["Z"]
    data["Element_w_A"] = data["A"].astype(str) + data["EL"]
    data.drop(columns=["Page_Feed", "A", "EL", "Z", "N"], inplace=True)

    csv_name = os.path.join(saving_directory, "AME_rct2.csv")
    logging.info("RCT2: Formatting done. Saving file to {}".format(csv_name))
    data.to_csv(csv_name, index=False)
    logging.info("RCT2: Succesfully formated rct2-16.txt file.")
    return None


def merge_mass_rct(directory, create_imputed=True, add_qvalues=True):
    """Read the proccessed mass16, rct1, and rct2 files and merges them while adding other reaction Q-values if needed.

    It creates one main CSV file when finished. This assumes the three files
    were created using the read_mass(), read_rct1(), and read_rct2() functions. For more information
    visit the IAEA webiste: https://www-nds.iaea.org/amdc/. It also creates a new CSV file where
    missing values are filled via linear imputation paramenter- and element-wise.

    Args:
        directory (str): Path to the Atomic Mass Evaluation directory where
            the processed mass16, rct2, and rct2 files are saved. The length of all three
            files must be the same. The resulting file will be stored in the same directory.
        create_imputed (bool): If True, missing values will be imputed.
        add_qvalues (bool): If true it will add the following reaction Q-values:
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
    Returns:
        None
    """
    saving_directory = directory
    logging.info("MERGE: Initializing. Checking documents...")
    mass16_path = os.path.join(directory, "AME_mass16.csv")
    rct1_path = os.path.join(directory, "AME_rct1.csv")
    rct2_path = os.path.join(directory, "AME_rct2.csv")

    if check_if_files_exist([mass16_path, rct1_path, rct2_path]):
        logging.info("MERGE: Files exists. Reading data into dataframes...")
        data = pd.read_csv(mass16_path)
        rct1 = pd.read_csv(rct1_path)
        rct2 = pd.read_csv(rct2_path)

        df_final = pd.merge(data, rct1, on='Element_w_A')
        df_final = pd.merge(df_final, rct2, on='Element_w_A')

        if add_qvalues:
            logging.info("MERGE: Q-value Calculation: enabled. Calculating additional reaction energies...")
            df_final["Q(g,p)"] = -1 * df_final["S(p)"]
            df_final["Q(g,n)"] = -1 * df_final["S(n)"]
            df_final["Q(g,pn)"] = df_final["Q(d,a)"] - 26071.0939
            df_final["Q(g,d)"] = df_final["Q(d,a)"] - 23846.5279
            df_final["Q(g,t)"] = df_final["Q(p,a)"] - 19813.8649
            df_final["Q(g,He3)"] = df_final["Q(n,a)"] - 20577.6194
            df_final["Q(g,2p)"] = -1 * df_final["S(2p)"]
            df_final["Q(g,2n)"] = -1 * df_final["S(2n)"]
            df_final["Q(g,a)"] = df_final["Q(a)"]
            df_final["Q(p,n)"] = df_final["B_Decay_Energy"] - 782.3465
            df_final["Q(p,2p)"] = -1 * df_final["S(p)"]
            df_final["Q(p,pn)"] = -1 * df_final["S(n)"]
            df_final["Q(p,d)"] = -1 * df_final["S(n)"] + 2224.5660
            df_final["Q(p,2n)"] = df_final["Q(B-n)"] - 782.3465
            df_final["Q(p,t)"] = -1 * df_final["S(2n)"] + 8481.7949
            df_final["Q(p,3He)"] = df_final["Q(d,a)"] - 18353.0535
            df_final["Q(n,2p)"] = df_final["Q(ep)"] + 782.3465
            df_final["Q(n,np)"] = -1 * df_final["S(p)"]
            df_final["Q(n,d)"] = -1 * df_final["S(p)"] + 2224.5660
            df_final["Q(n,2n)"] = -1 * df_final["S(n)"]
            df_final["Q(n,t)"] = df_final["Q(d,a)"] - 17589.2989
            df_final["Q(n,3He)"] = -1 * df_final["S(2p)"] + 7718.0404
            df_final["Q(d,t)"] = -1 * df_final["S(n)"] + 6257.2290
            df_final["Q(d,3He)"] = -1 * df_final["S(p)"] + 5493.4744
            df_final["Q(3He,t)"] = df_final["B_Decay_Energy"] - 18.5920
            df_final["Q(3He,a)"] = -1 * df_final["S(n)"] + 20577.6194
            df_final["Q(t,a)"] = -1 * df_final["S(p)"] + 19813.8649

        csv_name = os.path.join(saving_directory, "AME_all_merged.csv")
        logging.info("MERGE: Formatting done. Saving file to {}".format(csv_name))
        df_final.to_csv(csv_name, index=False)
        logging.info("MERGE: Succesfully merged files.")

        if impute_values:
            logging.info("MERGE: Imputing enabled. Interpolating...")
            csv_name = os.path.join(saving_directory, "AME_all_merged_no_NaN.csv")

            warnings.filterwarnings('ignore')
            df_final = impute_values(df_final)
            df_final = df_final.interpolate(method='spline', order=1, limit=10, limit_direction='both')
            df_final = df_final.interpolate()
            warnings.filterwarnings('default')
            df_final.to_csv(csv_name, index=False)
            logging.info("MERGE: Succesfully merged files. Imputing missing values...")

    return None


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
    logging.info("FEAT ENG: Initializing. Checking documents...")
    filename = os.path.join(directory, "AME_all_merged.csv")
    periodic_filename = os.path.join(originals_directory, "periodic_table.csv")
    if check_if_files_exist([filename, periodic_filename]):
        logging.info("FEAT ENG: Reading data from {}".format(filename))
        ame = pd.read_csv(filename)
        ame = ame.replace(to_replace=-0, value=0)  # FORMATTING

        logging.info("FEAT ENG: Reading data from {}".format(periodic_filename))
        masses_natural = pd.read_csv(periodic_filename).rename(
            # Renaming columns for consistency with EXFOR:
            columns={
                'NumberofNeutrons': 'Neutrons', 'NumberofProtons': 'Protons',
                'AtomicMass': 'Atomic_Mass_Micro', 'Symbol': 'EL'})

        logging.info("FEAT ENG: Beginning data creation...")
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

        logging.info("FEAT ENG: Finished creating natural data. Merging with AME...")
        result = ame.append(masses_natural, sort=False)

        # Due to the merging process many NaN values are introduced. Here we fix this:
        result["Neutrons"] = result.Neutrons.fillna(result.N).astype(int)  # Fill the Neutrons column with the N column
        result["Mass_Number"] = result.Mass_Number.fillna(result.A).astype(int)  # same for Mass Number and A
        result.Flag.fillna("I", inplace=True)  # We already have our natural tags we now that all NaNs are isotopic now.
        result["O"].fillna(value="Other", inplace=True)  # ASSUMPTION: We assume natural data was derive with Other

        logging.info("FEAT ENG: Finishing up...")
        result = result.drop(columns=["Element_w_A"])  # We don't need this
        result = result.sort_values(by="Z")

        csv_name = os.path.join(saving_directory, "AME_Natural_Properties_w_NaN.csv")
        logging.info("FEAT ENG: Saving file to {}".format(csv_name))
        result.to_csv(csv_name, index=False)

        if fillna:
            warnings.filterwarnings('ignore')
            logging.info("FEAT ENG: Filling missing values using {} mode".format(mode.upper()))

            # The imputation methods change the column data data types, we save them
            # and transfer them after the imputation is perform.
            types = result.iloc[0:2]
            if mode.upper() == "ELEMENTAL":
                # we fill the nans by taking the average of all isotopes, same for all other parameters.
                result = impute_values(result)

            logging.info("FEAT ENG: Filling remaining NaN values with 0...")
            result = result.fillna(fill_value)

            logging.info("FEAT ENG: Returning features to original data types...")
            for x in result.columns:
                result[x] = result[x].astype(types[x].dtypes.name)
            warnings.filterwarnings('default')

            csv_name = os.path.join(saving_directory, "AME_Natural_Properties_no_NaN.csv")
            logging.info("FEAT ENG: Saving imputed file to {}".format(csv_name))
            result.to_csv(csv_name, index=False)

            logging.info("FEAT ENG: Sucessfully created natural data. Nan values were imputed.")
        else:
            logging.info("FEAT ENG: Succesfully created natural data. NaN values were not imputed.")
    else:
        logging.error("FEAT ENG: Merged file does not exists. Check your path and files.")
        sys.exit()
    return None


def get_all(originals_directory, saving_directory, fillna=True, fill_value=0, create_imputed=True, add_qvalues=True):
    """Create 5 CSV files: Proccesed (1) mass16, (2) rct1, and (3) rct2 files.

    It then creates a (4) single CSV merging the first three CSV files. It then creates (5) a proccesed CSV file
    containing isotpic and natural element data with NaN values. If wanted a (6) copy of the fifth
    CSV file is saved with imputed NaN values.

    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the
            periodic_table csv file is located.
        saving_directory (str): Path to directory where the resulting formatted
            csv file will be saved.
        fillna (bool): If True, it fills the missing values. For NaN values not filled by the
            used "mode", then the filling method is just the mean of the entire dataset.
        fill_value (int, float): Value to fill remaining missing values with after imputation is finished
            with selected `mode`. Defaults to 0.
        create_imputed (bool): If True, missing values will be imputed.
        add_qvalues (bool): If true it will add the following reaction Q-values:
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
    Returns:
        None
    """
    get_ame_originals(originals_directory)
    read_mass16(originals_directory, saving_directory)
    read_rct1(originals_directory, saving_directory)
    read_rct2(originals_directory, saving_directory)
    merge_mass_rct(saving_directory, add_qvalues=add_qvalues, create_imputed=create_imputed)
    create_natural_element_data(originals_directory, saving_directory, fillna=fillna, fill_value=fill_value)
    return None
