"""Data manipulation utilities for the EXFOR dataset."""
import os
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import nucml.ace.data_utilities as ace_utils
import nucml.evaluation.data_utilities as endf_utils
import nucml.datasets as nuc_data
import nucml.model.utilities as model_utils
import nucml.general_utilities as gen_utils
import nucml.exfor.plot as exfor_plot_utils
import nucml.config as config

from nucml.data_utils import copy_data_from_df_to_df


ame_dir_path = config.ame_dir_path
elements_dict = nuc_data.elements_dict

empty_df = pd.DataFrame()


def _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, MT=None, scaler=None, to_scale=[]):
    if one_hot:
        sample = df[(df["Z"] == Z) & (df["A"] == A) & (df["Element_Flag_" + nat_iso] == 1)]
    else:
        sample = df[(df["Z"] == Z) & (df["A"] == A) & (df["Element_Flag"] == nat_iso)]

    if MT is not None:
        sample = sample[sample[MT] == 1] if one_hot else sample[sample["MT"] == MT]
    if scaler:
        sample[to_scale] = scaler.transform(sample[to_scale])
    return sample.sort_values(by='Energy', ascending=True)


def load_samples(df, Z, A, MT, nat_iso="I", one_hot=False, scaler=None, to_scale=[], mt_for="EXFOR"):
    """Extract datapoints belonging to a particular isotope-reaction channel pair.

    Args:
        df (DataFrame): DataFrame containing all avaliable datapoints from where to extract the information.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (ENDF-coded).
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    MT = gen_utils.parse_mt(MT, mt_for=mt_for, one_hot=one_hot)
    return _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, MT=MT, scaler=scaler, to_scale=to_scale)


def load_isotope(df, Z, A, nat_iso="I", one_hot=False, scaler=None, to_scale=[]):
    """Load all datapoints avaliable for a particular isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    return _filter_and_scale_by_ZA_MT(df, one_hot, Z, A, nat_iso, scaler=scaler, to_scale=to_scale)


def load_element(df, Z, nat_iso="I", one_hot=False, scale=False, scaler=None, to_scale=[]):
    """Load all datapoints avaliable for a particular element.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        scale (bool, optional): If True, the scaler object passed will be use to normalize the extracted information.
            Defaults to False.
        scaler (object, optional): Fitted scaler object. Defaults to None.
        to_scale (list, optional): List of feature names that are to be scaled. Defaults to [].

    Returns:
        DataFrame
    """
    if one_hot:
        sample = df[(df["Z"] == Z) & (df["Element_Flag_" + nat_iso] == 1)].sort_values(by='Energy', ascending=True)
    else:
        sample = df[(df["Z"] == Z) & (df["Element_Flag"] == nat_iso)].sort_values(by='Energy', ascending=True)
    if scale:
        sample[to_scale] = scaler.transform(sample[to_scale])
    return sample


def _copy_over_data_and_scale(main_df, new_data, scaler=None, to_scale=None):
    new_data = copy_data_from_df_to_df(main_df, new_data, ignore_cols=['Energy', 'Data'])
    if "dData" in list(new_data.columns):
        new_data.drop(columns="dData", inplace=True)
    if "dEnergy" in list(new_data.columns):
        new_data.drop(columns="dEnergy", inplace=True)
    if scaler is not None and to_scale is not None:
        new_data[to_scale] = scaler.transform(new_data[to_scale])


def load_newdata(datapath, df, Z, A, MT, nat_iso="I", one_hot=False, log=False, scaler=None, to_scale=[]):
    """Load new measurments and appends the appropiate EXFOR isotopic data.

    Assumes new data only have two columns: Energy and Data.

    Args:
        datapath (str): Path-like string of the new data file.
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

    Returns:
        DataFrame
    """
    new_data = pd.read_csv(datapath)
    if log:
        new_data["Energy"] = np.log10(new_data["Energy"])
        new_data["Data"] = np.log10(new_data["Data"])
    isotope_exfor = load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)
    new_data = _copy_over_data_and_scale(isotope_exfor, new_data, scaler, to_scale)
    logging.info("EXFOR extracted DataFrame has shape: {}".format(new_data.shape))
    return new_data


def append_energy(e_array, df, Z, A, MT, nat_iso="I", one_hot=False, log=False, scaler=None, to_scale=[],
                  ignore_MT=False):
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
        isotope_exfor = load_samples(df, Z, A, 1, nat_iso=nat_iso, one_hot=one_hot, mt_for="ACE")
        isotope_exfor.MT_1 = 0
        MT = gen_utils.parse_mt(MT, mt_for="ACE", one_hot=one_hot)
        isotope_exfor[MT] = 1
    else:
        isotope_exfor = load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)
    new_data = _copy_over_data_and_scale(isotope_exfor, new_data, scaler, to_scale)
    return new_data


def expanding_dataset_energy(data, E_min, E_max, log, N, e_array=None):
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


def _plot_with_prediction(df, infer_df, y_hat):
    plt.plot(df.Energy, df.Data, alpha=0.5, c="g")
    plt.plot(infer_df.Energy, y_hat)


def make_predictions_w_energy(e_array, df, Z, A, MT, model, model_type, scaler, to_scale, one_hot=True, log=False,
                              show=False):
    """Return predictions using a model at the given energy grid for a given isotope.

    Args:
        e_array (np.array): Numpy array representing energy points at which inferences will be made.
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        model (object): Trained model object.
        model_type (str): Type of model. Options include None meaning scikit-learn models, "tf", for tensorflow, and
            "xgb" for gradient boosting machines.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.
        one_hot (bool, optional): If True, the script assumes that the reaction channel is one-hot encoded. Defaults to
            False.
        log (bool, optional): If True, the log of both the Energy and Data features will be taken.
        show (bool, optional): If True, a plot of the predictions will be rendered. Defaults to False.

    Returns:
        np.array
    """
    data_kwargs = {
        "Z": Z, "A": A, "MT": MT, "log": log, "scaler": scaler, "to_scale": to_scale, "one_hot": True,
        "ignore_MT": True}
    to_infer = append_energy(e_array, df, **data_kwargs)
    exfor = load_samples(df, Z, A, MT, one_hot=one_hot, mt_for="ACE")
    # Make Predictions
    y_hat = model_utils.make_predictions(to_infer.values, model, model_type)
    _plot_with_prediction(exfor, to_infer, y_hat) if show else None
    return y_hat


def make_predictions_from_df(df, Z, A, MT, model, model_type, scaler, to_scale, log=False, show=False):
    """Return predictions for all avaliable datapoints for a particular isotope-reaction channel pair.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        model (object): Trained model object.
        model_type (str): Type of model. Options include None meaning scikit-learn models, "tf", for tensorflow, and
            "xgb" for gradient boosting machines.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.
        log (bool, optional): If True, the log of both the Energy and Data features will be taken.
        show (bool, optional): If True, a plot of the predictions will be rendered. Defaults to False.

    Returns:
        np.array
    """
    kwargs = {"nat_iso": "I", "one_hot": True, "scaler": scaler, "to_scale": to_scale}
    exfor = load_samples(df, Z, A, MT, **kwargs)
    # Make Predictions
    y_hat = model_utils.make_predictions(exfor.drop(columns=["Data"]).values, model, model_type)
    _plot_with_prediction(exfor, exfor, y_hat) if show else None
    return y_hat


def _plot_save_predictions(plotter, all_dict, order_dict, save, path, show, log, save_both):
    if plotter == "plotly":
        exfor_plot_utils.ml_results(all_dict, save=save, save_dir=path, order_dict=order_dict, show=show)
    elif plotter == "plt":
        exfor_plot_utils.ml_results(
            all_dict, save=save, save_dir=path, order_dict=order_dict, show=show, log=log, plot_type="sns")
    if save_both:
        if plotter == "plotly":
            if len(order_dict) != 0:
                order_dict = {k: int(v) for k, v in order_dict.items()}
            exfor_plot_utils.ml_results(
                all_dict, save=save, save_dir=path, order_dict=order_dict, show=False, log=log, plot_type="sns")
        elif plotter == "plt":
            if len(order_dict) != 0:
                order_dict = {str(v): k for k, v in order_dict.items()}
            exfor_plot_utils.ml_results(all_dict, save=save, save_dir=path, order_dict=order_dict, show=False)
    return


def predicting_nuclear_xs_v2(df, Z, A, MT, model, to_scale=None, scaler=None, e_array="ace", log=False,
                             model_type=None, new_data=empty_df, nat_iso="I", get_endf=False, inv_trans=False,
                             show=False, plotter="plotly", save=False, path="", save_both=True, order_dict={}):
    """Predict values for a given isotope-reaction channel pair.

    This all-in-one function allows to not only get predictions
    but also calculate the errors relative to the EXFOR and ENDF datapoints (if avaliable). In addition, the plotting
    capabilities allow the user to inspect the predictions in a typical cross section plot. In addition to predicting
    values at the original exfor datapoint energies, the .ACE energy grid is used for further comparison.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        MT (int): Reaction channel (endf-coded).
        model (object): Trained model object.
        to_scale (list): List of feature names that are to be scaled.
        scaler (object): Fitted scaler object.
        e_array (str, optional): If "ace", the energy grid from the appropiate ACE file is appended. An
            alternative is to provide a specific energy array. Defaults to "ace".
        log (bool, optional): If True, it assumes the Energy is already in a log form. Defaults to False.
        model_type (str): Type of model. Options include None meaning scikit-learn models, "tf", for tensorflow, and
            "xgb" for gradient boosting machines.
        html (bool, optional): If True, the plot will be rendered in an interactive browser tab. Defaults to False.
        new_data (DataFrame, optional): New data for which to make predictions, get errors, and plot. Assumes it has
            all needed information. Defaults to empty_df.
        save (bool, optional): If True, the plot will be saved. Defaults to False.
        show (bool, optional): If True, a plot of the predictions will be rendered. Defaults to False.
        path (str, optional): Path-like string on which to save the rendered plots. Defaults to "".
        nat_iso (str, optional): "I" means isotopic while "N" means natural experimental campaigns. Defaults to "I".
        order_dict (dict, optional): Order in which to plot the different lines. See plotly_ml_results() for more info.
            Defaults to {}.
        get_endf (bool, optional): If True, the endf file will be extracted to calculate errors and create plots.
            Defaults to False.
        inv_trans (bool, optional): If True, the returned data will be in its original form (not scaled). Defaults to
            False.

    Returns:
        dict: contains a variety of information including predictions, errors, and more.
    """
    endf = empty_df
    if get_endf:
        endf = endf_utils.get_for_exfor(Z, A, MT, log=log)
    if e_array == "ace":
        # e_array = ace_utils.get_energies('{:<02d}'.format(Z) + str(A).zfill(3), ev=True, log=log)
        e_array = ace_utils.get_energies(str(Z) + str(A).zfill(3), ev=True, log=log)

    new_data_avaliable = True if new_data.shape[0] != 0 else False
    endf_avaliable = True if endf.shape[0] != 0 else False
    e_array_avaliable = True if e_array.shape[0] != 0 else False

    kwargs = {"nat_iso": nat_iso, "one_hot": True}
    to_infer = load_samples(df, Z, A, MT, **kwargs)
    to_plot = load_samples(df, Z, A, MT, scaler=scaler, to_scale=to_scale, **kwargs)
    to_infer = to_infer.drop(columns=["Data"])

    if e_array_avaliable:
        to_infer = expanding_dataset_energy(to_infer, 0, 0, log, 0, e_array=e_array)
    else:
        to_infer = expanding_dataset_energy(to_infer, -5.00, 7.30, log, 500)

    if to_scale is not None:
        to_infer[to_scale] = scaler.transform(to_infer[to_scale])

    pred_exfor_expanded = model_utils.make_predictions(to_infer.values, model, model_type)
    pred_exfor_original = model_utils.make_predictions(to_plot.drop(columns=["Data"]).values, model, model_type)

    if inv_trans:
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
        to_plot[to_scale] = scaler.inverse_transform(to_plot[to_scale])

    all_dict = {"exfor_ml_expanded": {"df": to_infer, "predictions": pred_exfor_expanded},
                "exfor_ml_original": {"df": to_plot, "predictions": pred_exfor_original}}

    exfor_ml_error = model_utils.regression_error_metrics(to_plot["Data"], pred_exfor_original)
    error_df = model_utils.create_error_df("EXFOR VS ML", exfor_ml_error)
    all_dict.update({"error_metrics": error_df})

    if new_data_avaliable:
        pred_exfor_new = model_utils.make_predictions(new_data.drop(columns=["Data"]).values, model, model_type)
        all_dict.update({"exfor_ml_new": {"df": new_data, "predictions": pred_exfor_new}})

        exfor_ml_new_error = model_utils.regression_error_metrics(new_data["Data"], pred_exfor_new)
        error_new_df = model_utils.create_error_df("EXFOR VS ML (NEW DATA)", exfor_ml_new_error)
        error_df = error_df.append(error_new_df)
        all_dict.update({"error_metrics": error_df})

    if endf_avaliable:
        # Gets interpolated endf data with anchor exfor
        exfor_endf, error_endf = get_error_endf_exfor(endf, to_plot)
        error_df = error_df.append(error_endf)
        all_dict.update({"exfor_endf_original": exfor_endf, "error_metrics": error_df, "endf": endf})
        if new_data_avaliable:
            # Gets interpolated endf data with anchor new exfor
            exfor_endf_new_data, error_endf_new = get_error_endf_exfor(endf, new_data, filter_energy=False)
            error_df = error_df.append(error_endf_new)
            all_dict.update({"exfor_endf_new": exfor_endf_new_data, "error_metrics": error_df})
    _plot_save_predictions(plotter, all_dict, order_dict, save, path, show, log, save_both)
    return all_dict


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
    exfor_sample = load_samples(df, Z, A, MT, nat_iso=nat_iso, one_hot=one_hot)

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
            exfor_endf, error_endf = get_error_endf_exfor(endf=endf, exfor_sample=exfor_sample)
            all_dict.update({"endf": endf, "exfor_endf": exfor_endf, "error_metrics": error_endf})
            if new_data.shape[0] != 0:
                exfor_endf_new_data, error_endf_new = get_error_endf_exfor(endf, new_data, filter_energy=False)
                error_df = error_endf.append(error_endf_new)
                all_dict.update({"exfor_endf_new": exfor_endf_new_data, "error_metrics": error_df})

    return all_dict


def get_error_endf_exfor(endf, df_sample, filter_energy=True):
    """Calculate the error between a given dataframe of experimental datapoints to ENDF.

    Args:
        endf (DataFrame): DataFrame containing the ENDF datapoints.
        df_sample (DataFrame): DataFrame containing the new experimental data points.

    Returns:
        DataFrame, DataFrame: first dataframe contains original values while the second one
            the calculated errors.
    """
    endf_copy = endf.copy()
    df = df_sample.copy()
    if filter_energy:
        df = df[df.Energy > endf_copy.Energy.min()]
    indexes = np.arange(len(endf), len(endf) + len(df))  # start our index numbering after len(endf)
    df.index = indexes  # This will return a dataframe with non zero index
    energy_interest = df[["Energy"]]  # energy_interest will carry previous indexes
    energy_interest["Data"] = np.nan
    endf_copy = endf_copy.append(energy_interest, ignore_index=False).sort_values(by=['Energy'])
    endf_copy["Data"] = endf_copy["Data"].interpolate(limit_direction="forward")

    # Measuring metrics on predictions.
    error_endf_exfor = model_utils.regression_error_metrics(df["Data"], endf_copy[["Data"]].loc[indexes])
    error_endf_exfor_df = model_utils.create_error_df("EXFOR VS ENDF", error_endf_exfor)

    exfor_endf = pd.DataFrame({
        "Energy": df.Energy.values,
        "EXFOR": df["Data"].values, "ENDF": endf_copy["Data"].loc[indexes].values})
    return exfor_endf, error_endf_exfor_df


def _get_isotope_df_cols(df, Z, A, scaler, to_scale):
    kwargs = {"nat_iso": "I", "one_hot": True, "scaler": scaler, "to_scale": to_scale}
    exfor_isotope = load_isotope(df, Z, A, **kwargs)
    exfor_isotope_cols = exfor_isotope.loc[:, (exfor_isotope != 0).any(axis=0)][:1]
    return exfor_isotope_cols


def get_mt_errors_exfor_ml(df, Z, A, scaler, to_scale, model):
    """Calculate the error between EXFOR and ML predictions for a given isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.
        model (object): Machine learning object.

    Returns:
        DataFrame
    """
    exfor_isotope_cols = _get_isotope_df_cols(df, Z, A, scaler, to_scale)
    error_results = pd.DataFrame(columns=['MT', 'MAE', 'MSE', 'EVS', 'MAE_M', 'R2'])
    for col in exfor_isotope_cols.columns:
        if "MT" in col:
            exfor_sample = load_samples(df, Z, A, col, nat_iso="I", one_hot=True, scaler=scaler, to_scale=to_scale)
            error_dict = model_utils.regression_error_metrics(model.predict(exfor_sample.drop(columns=["Data"])),
                                                              exfor_sample.Data)
            error_results = error_results.append(pd.DataFrame({
                "MT": [col], "MAE": [error_dict["mae"]],
                "MSE": [error_dict["mse"]], "EVS": [error_dict["evs"]],
                "MAE_M": [error_dict["mae_m"]], "R2": [error_dict["r2"]]}))
    return error_results


def get_mt_error_exfor_endf(df, Z, A, scaler, to_scale):
    """Calculate the error between EXFOR and ENDF for a given isotope.

    Args:
        df (DataFrame): All avaliable experimental datapoints.
        Z (int): Number of protons.
        A (int): Atomic mass number.
        scaler (object): Fitted scaler object.
        to_scale (list): List of feature names that are to be scaled.

    Returns:
        DataFrame
    """
    # TODO: FIND OUT IF WE NEED SCALER OR TO SCALE
    # We don't care if its scaled or not since we only care abuot the Data feature, but we still need it?
    exfor_isotope_cols = _get_isotope_df_cols(df, Z, A, scaler, to_scale)
    error_results = pd.DataFrame(columns=['id', 'mae', 'mse', 'evs', 'mae_m', 'r2', 'MT'])
    for col in exfor_isotope_cols.columns:
        if "MT" in col:
            if col in ["MT_101", "MT_9000"]:
                continue
            else:
                exfor_sample = load_samples(df, Z, A, col, nat_iso="I", one_hot=True, scaler=scaler, to_scale=to_scale)
                endf_data = endf_utils.get_for_exfor(Z, A, col)
                _, error_exfor_endf = get_error_endf_exfor(endf_data, exfor_sample)
                error_exfor_endf["MT"] = col
                error_results = error_results.append(error_exfor_endf)
    return error_results


def get_csv_for_ace(df, Z, A, model, scaler, to_scale, model_type=None, saving_dir=None, saving_filename=None):
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
    ace_array = ace_utils.get_energies('{:<02d}'.format(Z) + str(A).zfill(3), ev=True, log=True)
    data_ace = pd.DataFrame({"Energy": ace_array})

    exfor_isotope_cols = _get_isotope_df_cols(df, Z, A, scaler, to_scale)
    for col in exfor_isotope_cols.columns:
        if "MT" in col:
            if col in ["MT_9000"]:
                continue
            else:
                mt_num = col.split("_")[1]
                logging.info(col)
                predictions = make_predictions_w_energy(
                    ace_array, df, Z, A, mt_num, model,
                    model_type, scaler, to_scale, log=False, show=False)
                data_ace[col] = predictions

    data_ace = 10**data_ace
    if saving_dir is not None:
        data_ace.to_csv(os.path.join(saving_dir, saving_filename), index=False)
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
