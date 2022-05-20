"""Parsing utilities for the EXFOR database."""
import os
import logging
import numpy as np
import pandas as pd
from natsort import natsorted

from nucml import general_utilities
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
        tag_writers = {'#AUTHOR1': authors, '#YEAR': years, '#ENTRY': entry, '#REF-CODE': refcode}
        copy = False
        for line in infile:
            matched = [match for match in tag_writers.keys() if line.startswith(match)]
            if matched:
                copy = False
                tag_writers[matched[0]].write(line)
            elif line.startswith(r'#DATASET') and len(line) > 16:
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
            elif line.startswith(r"#/DATA"):
                copy = False
            elif copy:
                num_data.write(line)


def _write_complex_data(outfile, lines, idx, line):
    detection_point = r"#+"
    if not lines[idx + 2].startswith(detection_point):
        outfile.write(line)
        return

    to_write = str(line.strip('\n')) + " " + str(lines[idx+2].strip('#+').strip())
    if lines[idx + 4].startswith(detection_point):
        to_write += " " + str(lines[idx+4].strip('#+').strip())
        if lines[idx + 6].startswith(detection_point):
            to_write += " " + str(lines[idx+6].strip('#+').strip())

    to_write += "\n"
    outfile.write(to_write)


def _extract_complex_data_from_c4(c4_file, tmp_path):
    with open(c4_file, "r") as infile, \
            open(os.path.join(tmp_path, 'titles.txt'), 'a') as titles, \
            open(os.path.join(tmp_path, 'references.txt'), 'a') as references, \
            open(os.path.join(tmp_path, 'data_points_per_experiment_refined.txt'), 'a') as data_points, \
            open(os.path.join(tmp_path, 'reaction_notations.txt'), 'a') as reactions:
        lines = infile.readlines()
        writers = {'#TITLE': titles, '#REFERENCE': references, '#DATA': data_points, '#REACTION': reactions}
        for idx, line in enumerate(lines):
            matched = [match for match in writers.keys() if line.startswith(match)]
            if matched:
                _write_complex_data(matched[0], lines, idx, line)
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
    general_utilities.remove_file(cross_section_file)

    for c4_file in c4_list:
        _extract_basic_data_from_c4(c4_file, tmp_path, heavy_path)

    # Extract titles, references, and number of data points per experiment...")
    for c4_file in c4_list:
        _extract_complex_data_from_c4(c4_file, tmp_path)

    # Format experimental data
    output_path = os.path.join(heavy_path, "all_cross_sections_v1.txt")
    indexex = [5, 11, 12, 15, 19, 20, 21, 22, 31, 40, 49, 58, 67, 76, 85, 94, 97, 122, 127, 130]
    general_utilities._write_file_with_separators(cross_section_file, output_path, indexex, ";")
    os.remove(cross_section_file)


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

    df = df.drop(columns=["MF", "Cos/LO", "dCos/LO"])

    logging.info("EXFOR CSV: Filling dEnergy, dData, and dELV by reaction channel...")
    df["Uncertainty_E"] = df["dEnergy"]/df["Energy"]
    df["Uncertainty_D"] = df["dData"]/df["Data"]
    df["Uncertainty_ELV"] = df["dELV/HL"]/df["ELV/HL"]

    for uncertainty_col in ['Uncertainty_E', 'Uncertainty_D', 'Uncertainty_ELV']:
        for interpolation_col in ['MT', 'Institute', 'Isotope']:
            df[uncertainty_col] = df[[interpolation_col, uncertainty_col]].groupby(
                interpolation_col).transform(lambda x: x.fillna(x.mean()))

    df["Uncertainty_ELV"] = df[["I78", "Uncertainty_ELV"]].groupby("I78").transform(lambda x: x.fillna(x.mean()))

    df.dEnergy = df.dEnergy.fillna(df.Energy * df.Uncertainty_E)
    df.dData = df.dData.fillna(df.Data * df.Uncertainty_D)
    df["dELV/HL"] = df["dELV/HL"].fillna(df["ELV/HL"] * df["Uncertainty_ELV"])

    df.Uncertainty_D = df.Uncertainty_D.replace(to_replace=np.inf, value=0)
    df.dData = df.dData.replace(to_replace=np.nan, value=0)
    df[["dELV/HL", 'ELV']] = df[["dELV/HL", 'ELV']].replace(to_replace=np.nan, value=0)

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
    df = df[~df.Reaction_Notation.str.contains("RAW")]
    df = df[~(df.Data < 0)]
    df.to_csv(os.path.join(heavy_path, "EXFOR_" + mode + "_MF3_AME_no_RawNaN.csv"), index=False)
