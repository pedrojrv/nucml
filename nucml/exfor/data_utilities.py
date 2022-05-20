"""Data manipulation utilities for the EXFOR dataset."""
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nucml.ace.querying_utils as ace_query_utils
import nucml.evaluation.data_utilities as endf_utils
import nucml.datasets as nuc_data
import nucml.general_utilities as gen_utils
import nucml.exfor.plot as exfor_plot_utils
import nucml.exfor.querying_utils as query_utils
import nucml.exfor.ml_utilities as ml_utils
from nucml.exfor import error_metrics
import nucml.config as config

from nucml.data_utils import copy_data_from_df_to_df


ame_dir_path = config.ame_dir_path
elements_dict = nuc_data.elements_dict

empty_df = pd.DataFrame()


def _copy_over_data_and_scale(main_df, new_data, scaler=None):
    new_data = copy_data_from_df_to_df(main_df, new_data, ignore_cols=['Energy', 'Data'])
    if "dData" in list(new_data.columns):
        new_data.drop(columns="dData", inplace=True)
    if "dEnergy" in list(new_data.columns):
        new_data.drop(columns="dEnergy", inplace=True)
    if scaler:
        new_data = scaler.transform(new_data)


def append_energy(e_array, df, Z, A, MT, nat_iso="I", one_hot=False, log=False, scaler=None, ignore_MT=False):
    """Append the given energy array to the passed DataFrame and feature values are coppied to these new rows.

    Args:
        e_array (np.array): Numpy array with the additional energy values to append.
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        log (bool, optional): If True, the log of both the Energy and Data features will be taken.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].
        ignore_MT (bool, optional): If True, the reaction channel is ignored and data for the total reaction is taken.
            Defaults to False.

    Returns:
        DataFrame
    """
    new_data = pd.DataFrame({"Energy": e_array})
    if log:
        new_data["Energy"] = np.log10(new_data["Energy"])
    if ignore_MT:
        isotope_exfor = query_utils.load_samples(df, Z, A, 1, nat_iso=nat_iso, one_hot=one_hot, mt_for="ACE")
        isotope_exfor.MT_1 = 0
        MT = gen_utils.parse_mt(MT, mt_for="ACE", one_hot=one_hot)
        isotope_exfor[MT] = 1
    else:
        isotope_exfor = query_utils.load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)
    new_data = _copy_over_data_and_scale(isotope_exfor, new_data, scaler)
    return new_data


def expanding_dataset_energy(data, e_min_max, log, N, e_array=None):
    """Expand a given DataFrames energy points by a given number of energy points between E_min and E_max.

    Args:
        data (DataFrame): DataFrame for which the Energy will be expanded.
        E_min (int): Starting point for energy expansion.
        E_max (float): Ending point for energy expansion.
        log (bool): If True, it assumes the Energy in the passed data is already in log form.
        N (int): Number of datapoints between E_min and E_max to create.
        e_array (np.array, optional): If e_array is provided, this gets appended overriding any other parameters.
            Defaults to None.

    Returns:
        DataFrame
    """
    E_min, E_max = e_min_max
    e_array_avaliable = True if not None else False
    if e_array_avaliable:
        energy_to_add = pd.DataFrame({"Energy": e_array})
    else:
        if log:
            energy_range = np.linspace(E_min, E_max, N)
        else:
            energy_range = np.power(10, np.linspace(np.log10(E_min), np.log10(E_max), N))
        energy_to_add = pd.DataFrame({"Energy": energy_range})
    energy_to_add = copy_data_from_df_to_df(data, energy_to_add, start=2)
    data = data.append(energy_to_add, ignore_index=True).sort_values(by='Energy', ascending=True)
    return data


def plot_exfor_w_references(df, Z, A, MT, nat_iso="I", new_data=empty_df, endf=empty_df, error=False, get_endf=True,
                            reverse_log=False, legend_size=21, save=False, interpolate=False, legend=False, alpha=0.7,
                            one_hot=False, log_plot=False, path='', ref=False, new_data_label="Additional Data",
                            dpi=300, figure_size=(14, 10)):
    """Plot Cross Section for a particular Isotope with or without references.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        new_data (DataFrame, optional): New data for which to make predictions, get errors, and plot. Assumes it has
            all needed information. Defaults to empty_df.
        endf (DataFrame, optional): DataFrame containing the appropiate ENDF data to plot against. Defaults to empty_df.
        error (bool, optional): If True, error between the EXFOR and ENDF datapoints are calculated. Defaults to False.
        get_endf (bool, optional): If True, the endf file will be extracted to calculate errors and create plots.
            Defaults to False.
        reverse_log (bool, optional): If True, the log in Energy and Data is first removed from the passed dataframe.
            Defaults to False.
        legend_size (int, optional): Legend size in plots. Useful when there are many experimental campaigns. Defaults
            to 21.
        save (bool, optional): If True, the plot will be saved. Defaults to False.
        interpolate (bool, optional): If True, the EXFOR will be ploted as a line rather than scatter points. Defaults
            to False.
        legend (bool, optional): If True, a legend will appear in the image. Defaults to False.
        alpha (float, optional): Level of transparency of ENDF and EXFOR plots. Defaults to 0.7.
        one_hot (bool, optional): If True, the passed dataframe is assumed to be preprocessed. Defaults to False.
        log_plot (bool, optional): If True, log scales will be applied. Defaults to False.
        path (str, optional): Path-like string on which to save the rendered plots. Defaults to "".
        ref (bool, optional): If True, EXFOR will be ploted per experimental campaign (one color for each). Defaults to
            False.
        new_data_label (str, optional): If new data is provided, this sets the label in the legend. Defaults to
            "Additional Data".

    Returns:
        dict: All information requested including original data and errors are contained in a python dictionary.
    """
    if reverse_log:
        df["Energy"] = 10**df["Energy"].values
        df["Data"] = 10**df["Data"].values
    if get_endf:
        endf = endf_utils.get_for_exfor(Z, A, MT, log=False)
    # Extracting dataframe to make predictions and creating copy for evaluation
    exfor_sample = query_utils.load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)

    # Initializing Figure and Plotting
    plt.figure(figsize=figure_size)
    ax = plt.subplot(111)
    if ref:
        groups = exfor_sample[["Energy", "Data", "Short_Reference"]].groupby("Short_Reference")
        for name, group in groups:
            ax.plot(group["Energy"], group["Data"], marker="o", linestyle="", label=name, alpha=0.9)
    else:
        ax.scatter(exfor_sample["Energy"], exfor_sample["Data"], alpha=alpha, label="EXFOR", marker="o")
    if new_data.shape[0] != 0:
        ax.plot(new_data.Energy, new_data.Data, marker="o", linestyle="", label=new_data_label, alpha=0.9)
    if endf.shape[0] != 0:
        ax.plot(endf.Energy, endf.Data, label="ENDF/B-VIII.0", alpha=1, color='tab:orange')  # alpha previously 0.8
    if interpolate:
        ax.plot(exfor_sample["Energy"], exfor_sample["Data"], alpha=alpha*0.5, label="Interpolation", ci=None)
    if log_plot:
        plt.xscale('log')
        plt.yscale('log')
    if legend:
        ax.legend(fontsize=legend_size)

    # Setting Figure Limits
    exfor_plot_utils.plot_limits_ref(exfor_sample, endf, new_data)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Cross Section (b)')
    plt.tick_params(bottom=True, left=True, width=3, direction='out')

    all_dict = {"exfor": exfor_sample}

    if save:
        plt.savefig(
            path + "EXFOR_{}_{}_XS.png".format(exfor_sample.Isotope.values[0], MT), bbox_inches='tight', dpi=dpi)
    if error:
        if endf.shape[0] != 0:
            exfor_endf, error_endf = error_metrics.get_error_endf_exfor(endf=endf, exfor_sample=exfor_sample)
            all_dict.update({"endf": endf, "exfor_endf": exfor_endf, "error_metrics": error_endf})
            if new_data.shape[0] != 0:
                exfor_endf_new_data, error_endf_new = error_metrics.get_error_endf_exfor(
                    endf, new_data, filter_energy=False)
                error_df = error_endf.append(error_endf_new)
                all_dict.update({"exfor_endf_new": exfor_endf_new_data, "error_metrics": error_df})

    return all_dict


def _get_isotope_df_cols(df, Z, A, scaler, to_scale):
    kwargs = {"nat_iso": "I", "one_hot": True, "scaler": scaler, "to_scale": to_scale}
    exfor_isotope = query_utils.load_isotope(df, Z, A, **kwargs)
    exfor_isotope_cols = exfor_isotope.loc[:, (exfor_isotope != 0).any(axis=0)][:1]
    return exfor_isotope_cols


def get_csv_for_ace(df, Z, A, model, scaler, to_scale, model_type=None):
    """Create a CSV with the model predictions for a particular isotope in the appropiate format for the ACE utilities.

    The function returns a DataFrame which can then be saved as a CSV. The saving_dir argument provides a direct method
    by which to save the CSV file in the process.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        model (object): Trained model object.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.
        model_type (str): Type of model. Options include None meaning scikit-learn models, "tf", for tensorflow, and
            "xgb" for gradient boosting machines.
        saving_dir (str, optional): Path-like string on where to save the CSV file. If given, the CSV file will be
            saved. Defaults to None.
        saving_filename (str, optional): Name for the CSV file to be saved. Defaults to None.

    Returns:
        DataFrame
    """
    ace_array = ace_query_utils.get_energies('{:<02d}'.format(Z) + str(A).zfill(3), ev=True, log=True)
    data_ace = pd.DataFrame({"Energy": ace_array})

    exfor_isotope_cols = _get_isotope_df_cols(df, Z, A, scaler, to_scale)
    for col in exfor_isotope_cols.columns:
        if "MT" not in col or col in ["MT_9000"]:
            continue
        mt_num = col.split("_")[1]
        predictions = ml_utils.make_predictions_w_energy(
            ace_array, df, Z, A, mt_num, model,
            model_type, scaler, to_scale, log=False, show=False)
        data_ace[col] = predictions

    data_ace = 10**data_ace
    return data_ace


def add_compound_nucleus_info(df, drop_q=False):
    """Add compound nucleus data to the original EXFOR DataFrame.

    This is performed by just appending the AME data by shifting the number of neutrons by 1.

    Args:
        df (DataFrame): Original EXFOR dataframe.
        drop_q (bool, optional): If True, it will drop all Q-value information. Defaults to False.

    Returns:
        DataFrame
    """
    if drop_q:
        logging.info("EXFOR CSV: Dropping Q-Values...")
        q_value = [col for col in df.columns if 'Q' in col]
        df = df.drop(columns=q_value)

    logging.info("EXFOR CSV: Adding information for Compound Nucleus...")
    df["Compound_Neutrons"] = df.N + 1
    df["Compound_Mass_Number"] = df.A + 1
    df["Compound_Protons"] = df.Z

    masses = pd.read_csv(os.path.join(ame_dir_path, "AME_Natural_Properties_no_NaN.csv"))
    masses = masses[masses.Flag == "I"]
    masses = masses.drop(columns=["Neutrons", "Mass_Number", "Flag"])
    masses = masses.rename(columns={'N': 'Neutrons', 'A': 'Mass_Number', "Z": "Protons", "O": "Origin"})

    nuclear_data_compound = list(masses.columns)
    nuclear_data_compound_cols = ["Compound_" + s for s in nuclear_data_compound]
    masses.columns = nuclear_data_compound_cols

    df = df.reset_index(drop=True)
    masses = masses.reset_index(drop=True)

    df = df.merge(masses, on=['Compound_Neutrons', 'Compound_Protons'], how='left')

    df = df.drop(columns=["Compound_Mass_Number_y"])
    df = df.rename(columns={'Compound_Mass_Number_x': 'Compound_Mass_Number'})
    return df
