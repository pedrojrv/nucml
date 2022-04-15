"""Parsing utilities for the EXFOR database."""
import os
import logging
import numbers
import numpy as np
import pandas as pd
from natsort import natsorted

from nucml import general_utilities
import nucml.objects.objects as objects
import nucml.config as config


ame_dir_path = config.ame_dir_path


def get_c4_names(c4_directory):
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
    else:
        logging.info("C4: Finished. Found {} .c4 files.".format(len(names)))
        return names


def _extract_basic_data_from_c4(c4_file, tmp_path, heavy_path):
    # Extract experimental data, authors, years, institutes, and dates
    with open(c4_file) as infile, \
            open(os.path.join(heavy_path, "all_cross_sections.txt"), 'a') as num_data, \
            open(os.path.join(tmp_path, 'authors.txt'), 'a') as authors, \
            open(os.path.join(tmp_path, 'years.txt'), 'a') as years, \
            open(os.path.join(tmp_path, 'institutes.txt'), 'a') as institute, \
            open(os.path.join(tmp_path, 'entry.txt'), 'a') as entry, \
            open(os.path.join(tmp_path, 'refcode.txt'), 'a') as refcode, \
            open(os.path.join(tmp_path, 'dataset_num.txt'), 'a') as dataset_num, \
            open(os.path.join(tmp_path, 'dates.txt'), 'a') as date:
        copy = False
        for line in infile:
            if line.startswith(r"#AUTHOR1"):
                copy = False
                authors.write(line)
            elif line.startswith(r"#YEAR"):
                copy = False
                years.write(line)
            elif line.startswith(r'#ENTRY'):
                copy = False
                entry.write(line)
            elif line.startswith(r'#REF-CODE'):
                copy = False
                refcode.write(line)
            elif line.startswith(r'#DATASET'):
                if len(line) > 16:
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
                continue
            elif line.startswith(r"#/DATA"):
                copy = False
                continue
            elif copy:
                num_data.write(line)


def _write_complex_data(outfile, lines, idx, line):
    if lines[idx + 2].startswith(r"#+"):
        if lines[idx + 4].startswith(r"#+"):
            if lines[idx + 6].startswith(r"#+"):
                outfile.write(
                    str(line.strip('\n')) + " " + str(lines[idx+2].strip('#+').strip()) + " "
                    + str(lines[idx+4].strip('#+').strip()) + " "
                    + str(lines[idx+6].strip('#+').strip()) + "\n")
            else:
                outfile.write(
                    str(line.strip('\n')) + " " + str(lines[idx+2].strip('#+').strip()) + " "
                    + str(lines[idx+4].strip('#+').strip()) + "\n")
        else:
            outfile.write(str(line.strip('\n')) + " " + str(lines[idx+2].strip('#+').strip()) + "\n")
    else:
        outfile.write(line)


def _extract_complex_data_from_c4(c4_file, tmp_path):
    with open(c4_file, "r") as infile, \
            open(os.path.join(tmp_path, 'titles.txt'), 'a') as titles, \
            open(os.path.join(tmp_path, 'references.txt'), 'a') as references, \
            open(os.path.join(tmp_path, 'data_points_per_experiment_refined.txt'), 'a') as data_points, \
            open(os.path.join(tmp_path, 'reaction_notations.txt'), 'a') as reactions:
        lines = infile.readlines()
        for idx, line in enumerate(lines):
            if line.startswith(r"#TITLE"):
                _write_complex_data(titles, lines, idx, line)
            elif line.startswith(r"#REFERENCE"):
                _write_complex_data(references, lines, idx, line)
            elif line.startswith(r"#DATA "):
                _write_complex_data(data_points, lines, idx, line)
            elif line.startswith(r"#REACTION"):
                _write_complex_data(reactions, lines, idx, line)
        reactions.write(line)


def get_all(c4_list, heavy_path, tmp_path, mode="neutrons"):
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
        c4_list (list): List containing paths to all .c4 files.
        heavy_path (str): Path to directory where heavy files are to be saved.
        tmp_path (str): Path to directory where temporary files are to be saved.
        mode (str): The reaction projectile of the provided C4 files.

    Returns:
        None

    Raises:
        FileNotFoundError: If no .c4 files are in the provided list, then an error is raised.
    """
    if len(c4_list) == 0:
        raise FileNotFoundError("No .c4 files found.")

    # This will be appended to the previous directories
    tmp_path = os.path.join(tmp_path, "Extracted_Text_" + mode + "/")
    heavy_path = os.path.join(heavy_path, "EXFOR_" + mode + "/")
    general_utilities.initialize_directories([tmp_path, heavy_path], reset=True)

    cross_section_file = os.path.join(heavy_path, "all_cross_sections.txt")
    if os.path.exists(cross_section_file):
        os.remove(cross_section_file)

    for c4_file in c4_list:
        _extract_basic_data_from_c4(c4_file, tmp_path, heavy_path)

    # Extract titles, references, and number of data points per experiment...")
    for c4_file in c4_list:
        _extract_complex_data_from_c4(c4_file, tmp_path)

    # Format experimental data
    with open(cross_section_file) as infile, open(
            os.path.join(heavy_path, "all_cross_sections_v1.txt"), 'w') as outfile:
        for line in infile:
            if line.strip():
                string = list(line)
                values = [5, 11, 12, 15, 19, 20, 21, 22, 31, 40, 49, 58, 67, 76, 85, 94, 97, 122, 127, 130]
                for i, j in enumerate(values):
                    string.insert(i + j, ';')
                outfile.write("".join(string))
    os.remove(cross_section_file)
    logging.info("EXFOR: Finished.")


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

    logging.info(f"EXFOR CSV: Reading data points from {heavy_path}/all_cross_sections_v1.txt file into a DataFrame...")
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
        " ": "All_or_Total", "G": "Ground", "1": "M1", "2": "M2", "3": "M3", "4": "M4",
        "5": "M5", "?": "Unknown", "+": "More_than_1", "T": "All_or_Total"}
    df = df.replace({"Target_Metastable_State": metastate_dict, "Product_Metastable_State": metastate_dict})

    exfor_status_dict = {
        "U": "Un_normalized", "A": "Approved_by_Author", "C": "Correlated", "D": "Dependent",
        "O": "Outdated", "P": "Preliminary", "R": "Re_normalized", "S": "Superseded", " ": "Other"}
    df = df.replace({"EXFOR_Status": exfor_status_dict})

    df = df.replace({"Center_of_Mass_Flag": {"C": "Center_of_Mass", " ": "Lab"}})

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
        df[col] = df[col].str.strip("\"")
        df[col] = df[col].str.strip()

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
            elif "+" == x[7]:
                y = x[0:7]
                z = x[7:]
                new_col.append(y + "E" + z)
            elif "+" == x[6]:
                y = x[0:6]
                z = x[6:]
                new_col.append(y + "E" + z)
            elif "-" == x[7]:
                y = x[0:7]
                z = x[7:]
                new_col.append(y + "E" + z)
            elif "-" == x[6]:
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
        df[col] = df[col].str.strip("\"")
        df[col] = df[col].str.strip()

    # Replace empty values in I78 for L representing Low
    df = df.replace({"I78": {
        "E2": "Secondary_Energy", "LVL": "Level", "HL": "Half_Life", "DLV": "Level_Range",
        "EXC": "Excitation", "DE2": "Secondary_Energy_Range", "MIN": "Minimum_Energy",
        "MAX": "Maximum_Energy", "": "Other"}})

    df.drop(columns=['Target_ZA'], inplace=True)

    # #######################################################################################################
    # ################################ APPENDING OTHER INFORMATION ##########################################
    # #######################################################################################################

    logging.info("EXFOR CSV: Reading .txt files from {} into DataFrames...".format(tmp_path))
    # Reading experiments reaction notation
    df1 = pd.read_csv(os.path.join(tmp_path, "reaction_notations.txt"), delim_whitespace=True, header=None)
    df1.columns = ["Reaction", "Reaction_Notation"]

    # Reading Experiment Titles
    df2 = pd.read_csv(os.path.join(tmp_path, "titles.txt"), sep="#TITLE      ", header=None, engine="python")
    df2.columns = ["Keyword", "Title"]

    # Reading Data Points per Experiment
    df3 = pd.read_csv(
        os.path.join(tmp_path, "data_points_per_experiment_refined.txt"), delim_whitespace=True, header=None)
    df3.columns = ["Data", "Multiple"]

    # Reading Experiment Year
    df4 = pd.read_csv(os.path.join(tmp_path, "years.txt"), delim_whitespace=True, header=None)
    df4.columns = ["Keyword", "Year"]

    # Reading Experiment Date
    df5 = pd.read_csv(os.path.join(tmp_path, "authors.txt"), sep="    ", header=None, engine="python")
    df5.columns = ["Keyword", "Author"]

    # Reading Experiment Institute
    df6 = pd.read_csv(os.path.join(tmp_path, "institutes.txt"), sep="  ", header=None, engine="python")
    df6.columns = ["Keyword", "Institute"]

    # Reading Experiment Year
    df7 = pd.read_csv(os.path.join(tmp_path, "dates.txt"), delim_whitespace=True, header=None)
    df7.columns = ["Keyword", "Date"]

    # Reading Experiment Refere
    df8 = pd.read_csv(os.path.join(tmp_path, "references.txt"), sep="#REFERENCE  ", header=None, engine="python")
    df8.columns = ["Keyword", "Reference"]

    # Reading Dataset Number
    df9 = pd.read_csv(os.path.join(tmp_path, "dataset_num.txt"), sep="#DATASET    ", header=None, engine="python")
    df9.columns = ["Keyword", "Dataset_Number"]

    # Reading EXFOR entry number
    df10 = pd.read_csv(os.path.join(tmp_path, "entry.txt"), sep="#ENTRY      ", header=None, engine="python")
    df10.columns = ["Keyword", "EXFOR_Entry"]

    # Reading reference code
    df11 = pd.read_csv(os.path.join(tmp_path, "refcode.txt"), sep="#REF-CODE   ", header=None, engine="python")
    df11.columns = ["Keyword", "Reference_Code"]

    # Merging Datapoints, notation and titles and expanding based on datapoints
    logging.info("EXFOR CSV: Expanding information based on the number of datapoints per experimental campaign...")
    pre_final = pd.concat([df3, df1, df2, df4, df5, df6, df7, df8, df9, df10, df11], axis=1)
    final = pre_final.reindex(pre_final.index.repeat(pre_final.Multiple))
    final['position'] = final.groupby(level=0).cumcount() + 1

    # Indexing only required information and saving file
    final = final[[
        "Reaction_Notation", "Title", "Year", "Author", "Institute", "Date", "Reference",
        "Dataset_Number", "EXFOR_Entry", "Reference_Code"]]

    # Reset Indexes to make copying faster
    df = df.reset_index(drop=True)
    final = final.reset_index(drop=True)

    logging.info("EXFOR CSV: Appending information to main DataFrame...")
    # Assign newly extracted data to main dataframe
    df["Reaction_Notation"] = final["Reaction_Notation"]
    df["Title"] = final["Title"]
    df["Year"] = final["Year"]
    df["Author"] = final["Author"]
    df["Institute"] = final["Institute"]
    df["Date"] = final["Date"]
    df["Reference"] = final["Reference"]
    df["Dataset_Number"] = final["Dataset_Number"]
    df["EXFOR_Entry"] = final["EXFOR_Entry"]
    df["Reference_Code"] = final["Reference_Code"]

    df.Title = df.Title.fillna("No Title Found. Check EXFOR.")
    df.Reference = df.Reference.fillna("No Reference Found. Check EXFOR.")
    df.Short_Reference = df.Short_Reference.fillna("No Reference Found. Check EXFOR.")
    df.Reference_Code = df.Reference_Code.fillna("No Reference Code Found. Check EXFOR.")
    df.Author = df.Author.fillna("No Author Found. Check EXFOR.")
    df.EXFOR_Pointer = df.EXFOR_Pointer.fillna("No Pointer")

    df.EXFOR_Pointer = df.EXFOR_Pointer.apply(lambda x: str(int(x)) if isinstance(x, numbers.Number) else x)
    df.Date = df.Date.apply(lambda x: str(x)[:4] + "/" + str(x)[4:6] + "/" + str(x)[6:])
    df.EXFOR_SubAccession_Number = df.EXFOR_SubAccession_Number.astype(int)
    df.Institute = df.Institute.apply(lambda x: x.replace("(", "").replace(")", ""))

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
        logging.info("EXFOR CSV: Reading AME file...")
        df_workxs = df.copy()
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
    return None


def impute_original_exfor(heavy_path, tmp_path, mode, append_ame=True, MF_number="3"):
    """Impute missing values in the CSV files created using the csv_creator() function.

    It generates a new CSV files with filled missing values. The main features with considerable
    missing values are the Uncertainties in both Energy and Cross Section. It also limits the
    dataset to a particular type (MF ENDF code).

    - EXFOR_mode_MF3_AME_no_RawNaN: is a dataset created for personal use. It contains no
        missing values in both the EXFOR and the AME data entries. Furthermore, it is limited to
        reaction data rows (MF=3).

    Args:
        heavy_path (str): Path to directory where heavy files generated by the csv_creator() function are stored.
            This directory will also be used to store the resulting CSV files.
        tmp_path (str): Path to directory where temporary files are to be saved.
        mode (str): The reaction projectile of the originally provided C4 files.
        append_ame (bool): If True, the AME data will be appended to the final EXFOR CSV files. It is recommended this
            is always set to True. The features can be afterwards eliminated if wanted. Defaults to True.
        MF_number (str, optional): The MT ENDF code of data to retrieve and impute. Defaults to "3".

    Returns:
        None
    """
    heavy_path = os.path.join(heavy_path, "EXFOR_{}".format(mode))
    tmp_path = os.path.join(tmp_path, "Extracted_Text_{}".format(mode))

    csv_name = os.path.join(heavy_path, "EXFOR_{}_ORIGINAL.csv".format(mode))
    df = pd.read_csv(csv_name)

    if append_ame:
        logging.info("EXFOR CSV: Reading AME file...")
        df_workxs = df.copy()
        masses = pd.read_csv(os.path.join(ame_dir_path, "AME_Natural_Properties_no_NaN.csv")).rename(
            columns={'N': 'Neutrons', 'A': 'Mass_Number', 'Neutrons': 'N', 'Mass_Number': 'A', 'Flag': 'Element_Flag'})

        df_workxs = df_workxs.reset_index(drop=True)
        masses = masses.reset_index(drop=True)
        logging.info("EXFOR CSV: Appending AME data to EXFOR File...")
        df = df_workxs.merge(masses, on=['N', 'Z'], how='left')
        df = df.drop(columns=["A_x", "A_y", "N", "EL"]).rename(columns={'Neutrons': 'N', 'Mass_Number': 'A'})
        df = df[~df['N'].isnull()]
        df[["N", "A"]] = df[["N", "A"]].astype(int)
        df["O"].fillna(value="Other", inplace=True)

    logging.info("EXFOR CSV: Creating new CSV file with only MF=3 data...")
    df.MF = df.MF.astype(str)
    df.MT = df.MT.astype(str)
    df = df[df["MF"] == MF_number]

    # We get rid of heavy water measurments
    if MF_number == "3":
        logging.info("...")

    columns_drop = ["MF", "Cos/LO", "dCos/LO"]
    df = df.drop(columns=columns_drop)

    logging.info("EXFOR CSV: Filling dEnergy, dData, and dELV by reaction channel...")
    df["Uncertainty_E"] = df["dEnergy"]/df["Energy"]
    df["Uncertainty_D"] = df["dData"]/df["Data"]
    df["Uncertainty_ELV"] = df["dELV/HL"]/df["ELV/HL"]

    df["Uncertainty_E"] = df[["MT", "Uncertainty_E"]].groupby("MT").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_D"] = df[["MT", "Uncertainty_D"]].groupby("MT").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_ELV"] = df[["MT", "Uncertainty_ELV"]].groupby("MT").transform(lambda x: x.fillna(x.mean()))

    logging.info("EXFOR CSV: Filling dEnergy, dData, and dELV by Institute...")
    df["Uncertainty_E"] = df[["Institute", "Uncertainty_E"]].groupby(
        "Institute").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_D"] = df[["Institute", "Uncertainty_D"]].groupby(
        "Institute").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_ELV"] = df[["Institute", "Uncertainty_ELV"]].groupby(
        "Institute").transform(lambda x: x.fillna(x.mean()))

    logging.info("EXFOR CSV: Filling dEnergy, dData, and dELV by Isotope...")
    df["Uncertainty_E"] = df[["Isotope", "Uncertainty_E"]].groupby("Isotope").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_D"] = df[["Isotope", "Uncertainty_D"]].groupby("Isotope").transform(lambda x: x.fillna(x.mean()))
    df["Uncertainty_ELV"] = df[["Isotope", "Uncertainty_ELV"]].groupby(
        "Isotope").transform(lambda x: x.fillna(x.mean()))

    df["Uncertainty_ELV"] = df[["I78", "Uncertainty_ELV"]].groupby("I78").transform(lambda x: x.fillna(x.mean()))

    df.dEnergy = df.dEnergy.fillna(df.Energy * df.Uncertainty_E)
    df.dData = df.dData.fillna(df.Data * df.Uncertainty_D)
    df["dELV/HL"] = df["dELV/HL"].fillna(df["ELV/HL"] * df["Uncertainty_ELV"])

    df.Uncertainty_D = df.Uncertainty_D.replace(to_replace=np.inf, value=0)
    df.dData = df.dData.replace(to_replace=np.nan, value=0)
    df["dELV/HL"] = df["dELV/HL"].replace(to_replace=np.nan, value=0)
    df["ELV/HL"] = df["ELV/HL"].replace(to_replace=np.nan, value=0)

    df.fillna(value=0, inplace=True)

    df["Nucleus_Radius"] = 1.25 * np.power(df["A"], 1/3)
    df["Neutron_Nucleus_Radius_Ratio"] = 0.8 / df["Nucleus_Radius"]

    # Use this for ordering
    new_order = list(df.columns)[:35]
    new_order_2 = list(df.columns)[-6:]
    new_order.extend(new_order_2)
    nuclear_data_target = list(df.columns)[35:-6]
    new_order.extend(nuclear_data_target)

    df = df[new_order]

    df = df.drop(columns=["Uncertainty_D", "Uncertainty_E", "Uncertainty_ELV"])

    logging.info("EXFOR CSV: Dropping RAW experimental datapoints...")
    df = df[~df.Reaction_Notation.str.contains("RAW")]

    df = df[~(df.Data < 0)]

    logging.info("EXFOR CSV: Saving MF3 NaN Imputed RAW Free EXFOR CSV...")
    df.to_csv(os.path.join(heavy_path, "EXFOR_" + mode + "_MF3_AME_no_RawNaN.csv"), index=False)
    logging.info("Finished")
    return None
