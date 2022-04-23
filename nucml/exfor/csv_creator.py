"""Hosts main function needed to generate a CSV file from the parsed EXFOR C4 files."""
import os
import logging
import numbers
import numpy as np
import pandas as pd

import nucml.objects.objects as objects
import nucml.config as config

ame_dir_path = config.ame_dir_path


def _get_additional_features(tmp_path):
    logging.info("EXFOR CSV: Reading .txt files from {} into DataFrames...".format(tmp_path))
    # Reading experiments reaction notation
    df1 = pd.read_csv(
        os.path.join(tmp_path, "reaction_notations.txt"), delim_whitespace=True, header=None,
        names=["Reaction", "Reaction_Notation"])

    # Reading Experiment Titles
    df2 = pd.read_csv(
        os.path.join(tmp_path, "titles.txt"), sep="#TITLE      ", header=None, engine="python",
        names=["Keyword", "Title"])

    # Reading Data Points per Experiment
    df3 = pd.read_csv(
        os.path.join(tmp_path, "data_points_per_experiment_refined.txt"), delim_whitespace=True, header=None,
        names=["Data", "Multiple"])

    # Reading Experiment Year
    df4 = pd.read_csv(
        os.path.join(tmp_path, "years.txt"), delim_whitespace=True, header=None, names=["Keyword", "Year"])

    # Reading Experiment Date
    df5 = pd.read_csv(
        os.path.join(tmp_path, "authors.txt"), sep="    ", header=None, engine="python", names=["Keyword", "Author"])

    # Reading Experiment Institute
    df6 = pd.read_csv(
        os.path.join(tmp_path, "institutes.txt"), sep="  ", header=None, engine="python",
        names=["Keyword", "Institute"])

    # Reading Experiment Year
    df7 = pd.read_csv(
        os.path.join(tmp_path, "dates.txt"), delim_whitespace=True, header=None, names=["Keyword", "Date"])

    # Reading Experiment Refere
    df8 = pd.read_csv(
        os.path.join(tmp_path, "references.txt"), sep="#REFERENCE  ", header=None, engine="python",
        names=["Keyword", "Reference"])

    # Reading Dataset Number
    df9 = pd.read_csv(
        os.path.join(tmp_path, "dataset_num.txt"), sep="#DATASET    ", header=None, engine="python",
        names=["Keyword", "Dataset_Number"])

    # Reading EXFOR entry number
    df10 = pd.read_csv(
        os.path.join(tmp_path, "entry.txt"), sep="#ENTRY      ", header=None, engine="python",
        names=["Keyword", "EXFOR_Entry"])

    # Reading reference code
    df11 = pd.read_csv(
        os.path.join(tmp_path, "refcode.txt"), sep="#REF-CODE   ", header=None, engine="python",
        names=["Keyword", "Reference_Code"])

    # Merging Datapoints, notation and titles and expanding based on datapoints
    logging.info("EXFOR CSV: Expanding information based on the number of datapoints per experimental campaign...")
    pre_final = pd.concat([df3, df1, df2, df4, df5, df6, df7, df8, df9, df10, df11], axis=1)
    final = pre_final.reindex(pre_final.index.repeat(pre_final.Multiple))
    final['position'] = final.groupby(level=0).cumcount() + 1

    # Indexing only required information and saving file
    final = final[[
        "Reaction_Notation", "Title", "Year", "Author", "Institute", "Date", "Reference",
        "Dataset_Number", "EXFOR_Entry", "Reference_Code"]]
    final = final.reset_index(drop=True)
    return final


def _format_projectiles(df):
    df = df.replace({
        'Projectile': {1: "neutron", 1001: "proton", 2003: "helion", 0: "gamma", 1002: "deuteron", 2004: "alpha"}})

    if df.Projectile.unique()[0] == "neutron":
        Projectile_Z, Projectile_A, Projectile_N = 0, 1, 1
    elif df.Projectile.unique()[0] == "proton":
        Projectile_Z, Projectile_A, Projectile_N = 1, 1, 0
    elif df.Projectile.unique()[0] == "helion":
        Projectile_Z, Projectile_A, Projectile_N = 2, 3, 1
    elif df.Projectile.unique()[0] == "gamma":
        Projectile_Z, Projectile_A, Projectile_N = 0, 0, 0
    elif df.Projectile.unique()[0] == "deuteron":
        Projectile_Z, Projectile_A, Projectile_N = 1, 2, 1
    elif df.Projectile.unique()[0] == "alpha":
        Projectile_Z, Projectile_A, Projectile_N = 2, 4, 2
    df["Projectile_Z"] = Projectile_Z
    df["Projectile_A"] = Projectile_A
    df["Projectile_N"] = Projectile_N
    return df


def _append_ame(df_workxs, heavy_path, mode):
    logging.info("EXFOR CSV: Reading AME file...")
    # df_workxs = df.copy()
    masses = pd.read_csv(os.path.join(ame_dir_path, "AME_Natural_Properties_w_NaN.csv")).rename(
        columns={'N': 'Neutrons', 'A': 'Mass_Number', 'Neutrons': 'N', 'Mass_Number': 'A', 'Flag': 'Element_Flag'})
    df_workxs = df_workxs.reset_index(drop=True)
    masses = masses.reset_index(drop=True)
    logging.info("EXFOR CSV: Appending AME data to EXFOR File...")
    df = df_workxs.merge(masses, on=['N', 'Z'], how='left')
    df = df.drop(columns=["A_x", "A_y", "N", "EL"]).rename(columns={'Neutrons': 'N', 'Mass_Number': 'A'})
    df = df[~df['N'].isnull()]
    df[["N", "A"]] = df[["N", "A"]].astype(int)
    csv_name = os.path.join(heavy_path, "EXFOR_" + mode + "_ORIGINAL_w_AME.csv")
    logging.info("EXFOR CSV: Saving EXFOR CSV file to {}...".format(csv_name))
    df.to_csv(csv_name, index=False)


def csv_creator(heavy_path, tmp_path, mode, append_ame=True):
    """Create various CSV files from the information extracted using the get_all() function.

    This function is usually called after the get_all() function. The following CSV files will be created:

    - EXFOR_mode_ORIGINAL.csv: Contains the EXFOR database in it's original state.
    - EXFOR_mode_ORIGINAL_w_AME.csv: Contains the same features as in the original CSV
        file plus AME data appended to each row. The appended AME is the original data therefore
        not containing natural element data.

    Note: mode refers to the type of projectile selected in the 'mode' argument.

    Args:
        heavy_path (str): Path to directory where heavy files generated by the get_all() function are stored. This
            directory will also be used to store the resulting CSV files.
        tmp_path (str): Path to directory where temporary files are to be saved.
        mode (str): The reaction projectile of the originally provided C4 files.
        append_ame (bool): If True, the AME data will be appended to the final EXFOR CSV files. It is recommended this
            is always set to True. The features can be afterwards eliminated if wanted. Defaults to True.

    Returns:
        None
    """
    heavy_path = os.path.join(heavy_path, "EXFOR_{}".format(mode))
    tmp_path = os.path.join(tmp_path, "Extracted_Text_{}".format(mode))

    colnames = [
        "Projectile", "Target_ZA", "Target_Metastable_State", "MF", "MT", "Product_Metastable_State",
        "EXFOR_Status", "Center_of_Mass_Flag", "Energy", "dEnergy", "Data", "dData", "Cos/LO", "dCos/LO",
        "ELV/HL", "dELV/HL", "I78", "Short_Reference", "EXFOR_Accession_Number", "EXFOR_SubAccession_Number",
        "EXFOR_Pointer"]
    df = pd.read_csv(
        os.path.join(heavy_path, "all_cross_sections_v1.txt"), names=colnames, header=None, index_col=False, sep=";")

    # #######################################################################################################
    # ########################## FORMATTING CATEGORICAL AND STRING DATA #####################################
    # #######################################################################################################

    logging.info("EXFOR CSV: Formatting data (this may take a couple minutes)...")
    # make string version of original column
    df['Target_ZA'] = df['Target_ZA'].astype(str)

    # Making Sure all rows have the same number of values
    max_length = 5
    df.Target_ZA = df.Target_ZA.apply(lambda x: '0'*(max_length - len(x)) + x)

    # Target feature is formated as ZZAAA
    df['Z'] = df['Target_ZA'].str[0:2].astype(int).fillna(0)
    df['A'] = df['Target_ZA'].str[2:5].astype(int).fillna(0)

    # Calculating number of neutrons = mass number - protons
    df['N'] = df['A'] - df["Z"]

    metastate_dict = {
        " ": "All_or_Total", "G": "Ground", "1": "M1", "2": "M2", "3": "M3", "4": "M4", "5": "M5", "?": "Unknown",
        "+": "More_than_1", "T": "All_or_Total"}
    exfor_status_dict = {
        "U": "Un_normalized", "A": "Approved_by_Author", "C": "Correlated", "D": "Dependent", "O": "Outdated",
        "P": "Preliminary", "R": "Re_normalized", "S": "Superseded", " ": "Other"}
    df = df.replace({
        "Target_Metastable_State": metastate_dict, "Product_Metastable_State": metastate_dict,
        "EXFOR_Status": exfor_status_dict, "Center_of_Mass_Flag": {"C": "Center_of_Mass", " ": "Lab"}})

    # #######################################################################################################
    # ################################## FORMATTING NUMERICAL DATA ##########################################
    # #######################################################################################################
    # Defining Numerical Columns to Fix and casting them as strings
    cols = ["Energy", "dEnergy", "Data", "dData", "Cos/LO", "dCos/LO", "ELV/HL", "dELV/HL"]
    df[cols] = df[cols].astype(str)
    # df[cols] = df[cols].replace(to_replace="         ", value="0.0000000")
    df[cols] = df[cols].replace(to_replace="         ", value=np.nan)

    # We now strip values that may contain quatation marks and starting and trailing spaces
    for col in cols:
        df[col] = df[col].str.strip("\"").strip()

    # df[cols] = df[cols].replace(to_replace="", value="0.0000000")
    df[cols] = df[cols].replace(to_replace="", value=np.nan)

    # For the numerical values we know per formatting that each of them should be 9 characters in length
    max_length = 9

    for col in cols:
        df[col] = df[col].apply(lambda x: x if pd.isnull(x) else ' '*(max_length - len(x)) + x)

    # Add appropiate formating for python to recognize it as numerical
    for col in cols:
        new_col = []
        values = df[col].values
        for x in values:
            if pd.isnull(x):
                new_col.append(x)
            elif "+" == x[7] or "-" == x[7]:
                y = x[0:7]
                z = x[7:]
                new_col.append(y + "E" + z)
            elif "+" == x[6] or "-" == x[6]:
                y = x[0:6]
                z = x[6:]
                new_col.append(y + "E" + z)
            else:
                new_col.append(x)
        df[col] = new_col

    # We now convert the columns to numerical
    for col in cols:
        df[col] = df[col].astype(float)
        logging.info("EXFOR CSV: Finished converting {} to float.".format(col))

    cat_cols = ["Target_Metastable_State", "MF", "MT", "I78", "Product_Metastable_State", "Center_of_Mass_Flag"]

    # Convering all columns to strings and stripping whitespace
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.strip("\"").strip()

    # Replace empty values in I78 for L representing Low
    df = df.replace({"I78": {
        "E2": "Secondary_Energy", "LVL": "Level", "HL": "Half_Life", "DLV": "Level_Range",
        "EXC": "Excitation", "DE2": "Secondary_Energy_Range", "MIN": "Minimum_Energy",
        "MAX": "Maximum_Energy", "": "Other"}})

    df.drop(columns=['Target_ZA'], inplace=True)

    # #######################################################################################################
    # ################################ APPENDING OTHER INFORMATION ##########################################
    # #######################################################################################################
    final = _get_additional_features(tmp_path)

    # Reset Indexes to make copying faster
    df = df.reset_index(drop=True)

    logging.info("EXFOR CSV: Appending information to main DataFrame...")
    # Assign newly extracted data to main dataframe
    fillna_defaults = {
        'Title': "No Title Found. Check EXFOR.",
        'Reference': "No Reference Found. Check EXFOR.",
        'Short_Reference': "No Reference Found. Check EXFOR.",
        'Author': "No Author Found. Check EXFOR.",
        'EXFOR_Pointer': "No Pointer"
    }
    for name in [
            'Reaction_Notation', 'Title', 'Year', 'Author', 'Institute', 'Date', 'Reference', 'Dataset_Number',
            'EXFOR_Entry', 'Reference_Code']:
        df[name] = final[name].fillna(fillna_defaults[name])

    df.EXFOR_Pointer = df.EXFOR_Pointer.apply(lambda x: str(int(x)) if isinstance(x, numbers.Number) else x)
    df.Date = df.Date.apply(lambda x: str(x)[:4] + "/" + str(x)[4:6] + "/" + str(x)[6:])
    df.EXFOR_SubAccession_Number = df.EXFOR_SubAccession_Number.astype(int)
    df.Institute = df.Institute.apply(lambda x: x.replace("(", "").replace(")", ""))

    df = _format_projectiles(df)

    element_w_a = objects.load_zan()
    element_w_a = pd.DataFrame.from_dict(element_w_a, orient='index')
    # There are no molecular
    element_w_a.loc['12019'] = ['Heavy Water', 19, 1, 20, "Heavy Water"]

    df = df.merge(element_w_a, on=['N', 'Z', 'A'], how='left')

    df[["EXFOR_Accession_Number", "Dataset_Number", "EXFOR_Entry"]] = df[[
        "EXFOR_Accession_Number", "Dataset_Number", "EXFOR_Entry"]].astype(str)
    csv_name = os.path.join(heavy_path, "EXFOR_" + mode + "_ORIGINAL.csv")
    logging.info("EXFOR CSV: Saving EXFOR CSV file to {}...".format(csv_name))
    df.to_csv(csv_name, index=False)

    if append_ame:
        _append_ame(df, heavy_path, mode)
