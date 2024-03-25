"""Parsing utilities for the AME database."""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from nucml.processing import impute_values
from nucml._constants import AMEDatasetURLs
from nucml.general_utilities import check_if_files_exist, _download_url_file, remove_files

logger = logging.getLogger(__name__)


def _get_ame_originals(saving_dir: str | Path) -> None:
    """Request and store the three AME original files for further processing from the IAEA website.

    Args:
        saving_dir (str): Path-like string where the text files will be stored.
    """
    _download_url_file(
        'https://raw.githubusercontent.com/pedrojrv/ML_Nuclear_Data/master/AME/Originals/periodic_table.csv',
        saving_dir / 'periodic_table.csv'
    )
    _download_url_file(AMEDatasetURLs.MASS, saving_dir / 'mass.txt')
    _download_url_file(AMEDatasetURLs.RCT1, saving_dir / 'rct1.txt')
    _download_url_file(AMEDatasetURLs.RCT2, saving_dir / 'rct2.txt')


def _clean_up_originals(saving_dir: Path) -> None:
    """Remove the original AME files after they have been processed.

    Args:
        saving_dir (Path): Path to the directory where the original files are stored.
    """
    to_remove = [
        saving_dir / 'periodic_table.csv',
        saving_dir / 'mass.txt',
        saving_dir / 'rct1.txt',
        saving_dir / 'rct2.txt',
    ]
    remove_files(to_remove)


def _preprocess_ame_df(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the AME dataframe.

    This includese replacing the '*' character with NaN valuese and replacing '#' with ''. It also drops the Page Feed
    and any other column that starts with the `ignore` tag.

    Args:
        data (pd.DataFrame): AME dataframe.

    Returns:
        pd.DataFrame: Preprocessed AME dataframe.
    """
    # * represents not calculable quantity, we change this for nan
    data.replace(value=np.nan, to_replace="*", inplace=True)
    # hashtag in place of decimal point for estimated quantities (theoretical)
    for col in data.select_dtypes(include=['object']):
        data[col] = data[col].apply(lambda x: str(x).replace("#", ""))

    data.drop(columns=["Page_Feed"], inplace=True)
    for col in data.columns:
        if col.startswith('ignore'):
            data.drop(columns=[col], inplace=True)
    return data


def _parse_mass(saving_dir: Path) -> None:
    """Read the mass.txt file and creates a formatted CSV file.

    The Mass 16 file contains a variety of features including atomic mass, mass excess, binding energy, beta decay
    energy, and more. For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,i5,i5,i5,1x,a3,a4,1x,f13.5,f11.5,f11.3,f9.3,1x,a2,f11.3,f9.3,1x,i3,1x,f12.5,f11.5

    Args:
        saving_dir (str): Path to save resulting formatted csv file.
    """
    # Formating rules based on fortran formatting given by AME
    column_formatting = {
        "Page_Feed": (0, 1),
        "NZ": (1, 4),
        "N": (4, 9),
        "Z": (9, 14),
        "A": (14, 19),
        "ignore": (19, 20),
        "EL": (20, 23),
        "O": (23, 27),
        "ignore1": (27, 28),
        "Mass_Excess": (28, 42),
        "dMass_Excess": (42, 54),
        "Binding_Energy": (54, 67),
        'ignore2': (67, 68),
        "dBinding_Energy": (68, 78),
        "ignore3": (78, 79),
        "Beta_Type": (79, 81),
        "B_Decay_Energy": (81, 94),
        "dB_Decay_Energy": (94, 105),
        "ignore4": (105, 106),
        "Atomic_Mass_Micro": (106, 109),
        'ignore5': (109, 110),
        'Atomic_Mass_Micro2': (110, 123),
        "dAtomic_Mass_Micro": (123, 135),
    }

    column_dtypes = {
        "NZ": 'int64',
        "N": 'int64',
        "Z": 'int64',
        "A": 'int64',
        "EL": 'category',
        "O": 'category',
        "Mass_Excess": 'float64',
        "dMass_Excess": 'float64',
        "Binding_Energy": 'float64',
        "dBinding_Energy": 'float64',
        "Beta_Type": 'category',
        "B_Decay_Energy": 'float64',
        "dB_Decay_Energy": 'float64',
        "Atomic_Mass_Micro": 'float64',
        "dAtomic_Mass_Micro": 'float64',
    }

    filename = saving_dir / "mass.txt"
    data = pd.read_fwf(
        filename, colspecs=tuple(column_formatting.values()), header=None, skiprows=36, names=column_formatting.keys())

    data["O"].fillna(value="Other", inplace=True)

    data = _preprocess_ame_df(data)
    data['Atomic_Mass_Micro'] = data['Atomic_Mass_Micro'].astype('str') + data['Atomic_Mass_Micro2']
    data.drop(columns=['Atomic_Mass_Micro2'], inplace=True)

    for col, types in column_dtypes.items():
        data[col] = data[col].astype(types)

    data["Element_w_A"] = (data["A"].astype(str) + data["EL"].astype(str)).astype('category')
    csv_name = saving_dir / "mass.csv"
    data.to_csv(csv_name, index=False)


def _parse_rct(saving_dir: Path, rct_file: int = 1) -> None:
    """Read the rct1-16.txt file and creates a formatted CSV file.

    The rct1-16 file contains a variety of
    features including neutron and proton separation energies and q-values for a variety of reactions.
    For more information visit the IAEA webiste: https://www-nds.iaea.org/amdc/

    It  is parse base on the Fortran formatting:
    a1,i3,1x,a3,i3,1x,6(f10.2,f8.2)

    Args:
        saving_dir (str): Path to save resulting formatted csv file.
        rct_file (int): The rct file to  process. Options include 1 and 2.
    """
    column_formatting = {
        "Page_Feed": (0, 1),
        "A": (1, 4),
        'ignore': (4, 5),
        'EL': (5, 8),
        'Z': (8, 11),
        'ignore1': (11, 13),
    }

    column_dtypes = {
        "A": 'int64',
        'EL': 'category',
        'Z': 'int64',
    }

    if rct_file == 1:
        column_formatting.update({
            'S(2n)': (13, 25),
            'dS(2n)': (25, 35),
            "S(2p)": (35, 47),
            "dS(2p)": (47, 57),
            "Q(a)": (57, 69),
            "dQ(a)": (69, 79),
            "Q(2B-)": (79, 91),
            "dQ(2B-)": (91, 101),
            "Q(ep)": (101, 113),
            "dQ(ep)": (113, 123),
            "Q(B-n)": (123, 135),
            "dQ(B-n)": (135, 145),
        })
        skip_footer = 0
        skip_rows = 35
    elif rct_file == 2:
        column_formatting.update({
            'S(n)': (13, 25),
            'dS(n)': (25, 35),
            "S(p)": (35, 47),
            "dS(p)": (47, 57),
            "Q(4B-)": (57, 69),
            "dQ(d4B-)": (69, 79),
            "Q(d,a)": (79, 91),
            "dQ(d,a)": (91, 101),
            "Q(p,a)": (101, 113),
            "dQ(p,a)": (113, 123),
            "Q(n,a)": (123, 135),
            "dQ(n,a)": (135, 145),
        })
        skip_footer = 17
        skip_rows = 37
    else:
        raise ValueError("rct_file argument not valid. It can only be 1 or 2.")

    filename = saving_dir / f"rct{rct_file}.txt"
    data = pd.read_fwf(
        filename, colspecs=tuple(column_formatting.values()), header=None, skiprows=skip_rows,
        names=column_formatting.keys(), skipfooter=skip_footer)

    data = _preprocess_ame_df(data)

    for col in list(data.columns):
        dtype = column_dtypes.get(col, 'float64')
        data[col] = data[col].astype(dtype)

    data["Element_w_A"] = (data["A"].astype(str) + data["EL"].astype(str)).astype('category')
    data.drop(columns=["A", "EL", "Z"], inplace=True)

    csv_name = saving_dir / f"rct{rct_file}.csv"
    data.to_csv(csv_name, index=False)


def _add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
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
        pd.DataFrame: Original dataframe with new features appended.
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


def _merge_mass_rct(saving_dir: Path, create_imputed: bool = True, add_qvalues: bool = True) -> None:
    """Read the proccessed mass, rct1, and rct2 files and merges them while adding other reaction Q-values if needed.

    It creates one main CSV file when finished. This assumes the three files
    were created using the read_mass() and the read_rct() functions. For more information
    visit the IAEA webiste: https://www-nds.iaea.org/amdc/. It also creates a new CSV file where
    missing values are filled via linear imputation paramenter- and element-wise.

    Args:
        saving_dir (str): Path to the Atomic Mass Evaluation directory where
            the processed mass, rct2, and rct2 files are saved. The length of all three
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
    """
    logger.info("MERGE: Initializing. Checking documents...")
    mass_path = saving_dir / "mass.csv"
    rct1_path = saving_dir / "rct1.csv"
    rct2_path = saving_dir / "rct2.csv"

    if not check_if_files_exist([mass_path, rct1_path, rct2_path]):
        raise FileNotFoundError("One ore more needed files do not exist (mass.csv, rct1.csv, or rct2.csv).")

    data = pd.read_csv(mass_path)
    rct1 = pd.read_csv(rct1_path)
    rct2 = pd.read_csv(rct2_path)

    df_final = pd.merge(data, rct1, on='Element_w_A')
    df_final = pd.merge(df_final, rct2, on='Element_w_A')

    if add_qvalues:
        df_final = _add_extra_features(df_final)

    csv_name = saving_dir / "all_merged.csv"
    df_final.to_csv(csv_name, index=False)

    if create_imputed:
        csv_name = saving_dir / "all_merged_imputed.csv"
        df_final = impute_values(df_final)
        df_final = df_final.interpolate(method='spline', order=1, limit=10, limit_direction='both')
        df_final = df_final.interpolate()
        df_final.to_csv(csv_name, index=False)


def _create_natural_element_data(
        saving_dir: Path, fillna: bool = True, mode: str = "elemental", fill_value: int = 0) -> None:
    """Create natural element data by averaging isotopic data.

    Additionally it adds a flag to indicate rows which correspond to isotopic or natural data.

    Args:
        saving_dir (str): Path to directory where the resulting formatted
            csv file will be saved including the all_merged.csv file.
        fillna (bool): If True, missing values are filled. For the remaining NaN values not filled by the
            used `mode`, a value of 0 will be inserted unless specified otherwise.
        mode (str): The supported modes are:
            elemental: missing values are filled using linear interpolation element-wise.
        fill_value (float): Value to fill remaining missing values with after imputation is finished
            with selected `mode`. Defaults to 0.
    """
    filename = saving_dir / "all_merged.csv"
    periodic_filename = saving_dir / "periodic_table.csv"
    if not check_if_files_exist([filename, periodic_filename]):
        raise FileNotFoundError("One file does not exist.")

    ame = pd.read_csv(filename)

    masses_natural = pd.read_csv(periodic_filename).rename(
        # Renaming columns for consistency with EXFOR:
        columns={
            'NumberofNeutrons': 'Neutrons', 'NumberofProtons': 'Protons',
            'AtomicMass': 'Atomic_Mass_Micro', 'Symbol': 'EL'
        })

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

    result = ame.append(masses_natural, sort=False)

    # Due to the merging process many NaN values are introduced. Here we fix this:
    result["Neutrons"] = result.Neutrons.fillna(result.N).astype(int)  # Fill the Neutrons column with the N column
    result["Mass_Number"] = result.Mass_Number.fillna(result.A).astype(int)  # same for Mass Number and A
    result.Flag.fillna("I", inplace=True)  # We already have our natural tags we now that all NaNs are isotopic now.
    result["O"].fillna(value="Other", inplace=True)  # ASSUMPTION: We assume natural data was derive with Other

    result = result.drop(columns=["Element_w_A"])  # We don't need this
    result = result.sort_values(by="Z")
    result.to_csv(saving_dir / "all_merged_w_natural.csv", index=False)

    if fillna:
        # The imputation methods change the column data data types, we save them
        # and transfer them after the imputation is perform.
        types = result.iloc[0:2]
        if mode.upper() == "ELEMENTAL":
            # we fill the nans by taking the average of all isotopes, same for all other parameters.
            result = impute_values(result)

        result = result.fillna(fill_value)

        for x in result.columns:
            result[x] = result[x].astype(types[x].dtypes.name)

        result.to_csv(saving_dir / "all_merged_w_natural_imputed.csv", index=False)
