"""Parsing utilities for the AME database."""
import logging
import os
import numpy as np
import pandas as pd
import requests

import nucml.ame.parsing_utils as parse_utils
from nucml.general_utilities import check_if_files_exist
from nucml.processing import impute_values


logger = logging.getLogger(__name__)


def get_ame_originals(originals_directory):
    """Request and store the three AME original files for further processing from the IAEA website.

    Args:
        originals_directory (str): Path-like string where the text files will be stored.

    Returns:
        None
    """
    # periodic_table = requests.get(
    # 'https://raw.githubusercontent.com/pedrojrv/ML_Nuclear_Data/master/AME/Originals/periodic_table.csv').content
    mass16_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/mass16.txt').content
    rct1_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/rct1-16.txt').content
    rct2_txt = requests.get('https://www-nds.iaea.org/amdc/ame2016/rct2-16.txt').content

    # with open(os.path.join(originals_directory, 'periodic_table.csv'), 'wb') as f:
    #     f.write(periodic_table)
    with open(os.path.join(originals_directory, 'mass16.txt'), 'wb') as f:
        f.write(mass16_txt)
    with open(os.path.join(originals_directory, 'rct1-16.txt'), 'wb') as f:
        f.write(rct1_txt)
    with open(os.path.join(originals_directory, 'rct2-16.txt'), 'wb') as f:
        f.write(rct2_txt)


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
    data = pd.read_fwf(filename, colspecs=formatting, header=None, skiprows=39, names=column_names)

    data["O"].fillna(value="Other", inplace=True).replace(value=np.nan, to_replace="*")

    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].apply(lambda x: str(x).replace("#", ""))

    mass16_dtypes = [
        'float64', 'int64', 'int64', 'int64', 'int64', 'float64', 'object', 'object', 'float64', 'float64',
        'float64', 'float64', 'float64', 'float64', 'object', 'object', 'float64', 'float64', 'object', 'float64']

    for col, types in zip(data.columns, mass16_dtypes):
        data[col] = data[col].astype(types)

    for col in ["Atomic_Mass_Micro"]:
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip("\"").replace(" ", "").strip()

    data["Atomic_Mass_Micro"] = data["Atomic_Mass_Micro"].astype(float)
    data["B_Decay_Energy"] = data["B_Decay_Energy"].astype(float)

    data.drop(columns=["Page_Feed", "Other", "Other2", "Other3", "Beta_Type", "Other4", "NZ"], inplace=True)

    data["Element_w_A"] = data["A"].astype(str) + data["EL"]

    csv_name = os.path.join(saving_directory, "AME_mass16.csv")
    data.to_csv(csv_name, index=False)


def read_rct(originals_directory, saving_directory, rct_file=1):
    """Read the rct1-16.txt file and creates a formatted CSV file.

    The rct1-16 file contains a variety of
    features including neutron and proton separation energies and q-values for a variety of reactions.
    For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,1x,a3,i3,1x,6(f10.2,f8.2)

    Args:
        originals_directory (str): Path to the Atomic Mass Evaluation directory where the rct1-16.txt file is located.
        saving_directory (str): Path to save resulting formatted csv file.
        rct_file (int): The rct file to  process. Options include 1 and 2.

    Returns:
        None
    """
    formatting = (
        (0, 1), (1, 4), (4, 5), (5, 8), (8, 11), (11, 12), (12, 22), (22, 30), (30, 40), (40, 48), (48, 58), (58, 66),
        (66, 76), (76, 84), (84, 94), (94, 102), (102, 112), (112, 120)
    )

    column_names = ["Page_Feed", "A", "Other", "EL", "Z", "Other2"]
    if rct_file == 1:
        column_names.extend([
            "S(2n)", "dS(2n)", "S(2p)", "dS(2p)", "Q(a)", "dQ(a)", "Q(2B-)", "dQ(2B-)", "Q(ep)", "dQ(ep)", "Q(B-n)",
            "dQ(B-n)"])
    elif rct_file == 2:
        column_names.extend([
            "S(n)", "dS(n)", "S(p)", "dS(p)", "Q(4B-)", "dQ(4B-)", "Q(d,a)", "dQ(d,a)", "Q(p,a)", "dQ(p,a)", "Q(n,a)",
            "dQ(n,a)"])
    else:
        raise ValueError("rct_file argument not valid. It can only be 1 or 2.")

    filename = os.path.join(originals_directory, f"rct{rct_file}-16.txt")
    data = pd.read_fwf(filename, colspecs=formatting, header=None, skiprows=39, names=column_names)
    data = data.replace(to_replace="*", value=np.nan)
    data.drop(columns=["Other", "Other2"], inplace=True)

    for col in list(data.columns):
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip("\"").strip().replace("#", ".")

    for col in list(data.columns):
        if col != "EL":
            data[col] = data[col].astype(float)

    data[["A", "Z"]] = data[["A", "Z"]].astype(int)
    data["N"] = data["A"] - data["Z"]
    data["Element_w_A"] = data["A"].astype(str) + data["EL"]
    data.drop(columns=["Page_Feed", "A", "EL", "Z", "N"], inplace=True)

    csv_name = os.path.join(saving_directory, f"AME_rct{rct_file}.csv")
    data.to_csv(csv_name, index=False)


def merge_mass_rct(directory, create_imputed=True, add_qvalues=True):
    """Read the proccessed mass16, rct1, and rct2 files and merges them while adding other reaction Q-values if needed.

    It creates one main CSV file when finished. This assumes the three files
    were created using the read_mass() and the read_rct() functions. For more information
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
    logger.info("MERGE: Initializing. Checking documents...")
    mass16_path = os.path.join(directory, "AME_mass16.csv")
    rct1_path = os.path.join(directory, "AME_rct1.csv")
    rct2_path = os.path.join(directory, "AME_rct2.csv")

    if check_if_files_exist([mass16_path, rct1_path, rct2_path]):
        logger.info("MERGE: Files exists. Reading data into dataframes...")
        data = pd.read_csv(mass16_path)
        rct1 = pd.read_csv(rct1_path)
        rct2 = pd.read_csv(rct2_path)

        df_final = pd.merge(data, rct1, on='Element_w_A')
        df_final = pd.merge(df_final, rct2, on='Element_w_A')

        if add_qvalues:
            df_final = parse_utils.add_extra_features(df_final)

        csv_name = os.path.join(saving_directory, "AME_all_merged.csv")
        df_final.to_csv(csv_name, index=False)

        if impute_values:
            csv_name = os.path.join(saving_directory, "AME_all_merged_no_NaN.csv")
            df_final = impute_values(df_final)
            df_final = df_final.interpolate(method='spline', order=1, limit=10, limit_direction='both')
            df_final = df_final.interpolate()
            df_final.to_csv(csv_name, index=False)


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
    read_rct(originals_directory, saving_directory, rct_file=1)
    read_rct(originals_directory, saving_directory, rct_file=2)
    merge_mass_rct(saving_directory, add_qvalues=add_qvalues, create_imputed=create_imputed)
    parse_utils.create_natural_element_data(originals_directory, saving_directory, fillna=fillna, fill_value=fill_value)
