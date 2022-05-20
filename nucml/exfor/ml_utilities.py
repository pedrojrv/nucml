"""EXFOR-related ML utilities."""
import pandas as pd
from functools import partial
import matplotlib.pyplot as plt
from nucml.exfor import error_metrics

import nucml.exfor.plot as exfor_plot_utils
import nucml.evaluation.data_utilities as endf_utils
import nucml.model.utilities as model_utils
import nucml.exfor.querying_utils as query_utils
import nucml.exfor.data_utilities as data_utils

empty_df = pd.DataFrame()


def _plot_save_predictions(plotter, all_dict, order_dict, save, show):  # , path, show, log, save_both):
    plotly_plot = partial(
        exfor_plot_utils.ml_results_plotly, results_dict=all_dict, order_dict=order_dict, save=save, show=show)
    sns_plot = partial(
        exfor_plot_utils.ml_results_plotly, results_dict=all_dict, order_dict=order_dict, save=save, show=show)
    plotter = plotly_plot if plotter == "plotly" else sns_plot
    plotter()
    # if save_both:
    #     if plotter == "plotly":
    #         if len(order_dict) != 0:
    #             order_dict = {k: int(v) for k, v in order_dict.items()}
    #         exfor_plot_utils.ml_results(
    #             all_dict, save=save, save_dir=path, order_dict=order_dict, show=False, log=log, plot_type="sns")
    #     elif plotter == "plt":
    #         if len(order_dict) != 0:
    #             order_dict = {str(v): k for k, v in order_dict.items()}
    #         exfor_plot_utils.ml_results(all_dict, save=save, save_dir=path, order_dict=order_dict, show=False)


def predicting_nuclear_xs_v2(df, Z, A, MT, model, scaler=None, e_array="ace", log=False,
                             model_type=None, new_data=empty_df, nat_iso="I", get_endf=False, inv_trans=False,
                             show=False, plotter="plotly", save=False, order_dict={}):
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
        # e_array = query_utils.get_energies('{:<02d}'.format(Z) + str(A).zfill(3), ev=True, log=log)
        e_array = query_utils.get_energies(str(Z) + str(A).zfill(3), ev=True, log=log)

    new_data_avaliable = True if new_data.shape[0] != 0 else False
    endf_avaliable = True if endf.shape[0] != 0 else False
    e_array_avaliable = True if e_array.shape[0] != 0 else False

    kwargs = {"nat_iso": nat_iso, "one_hot": True}
    to_infer = query_utils.load_samples(df, Z, A, MT, **kwargs)
    to_plot = query_utils.load_samples(df, Z, A, MT, scaler=scaler, **kwargs)
    to_infer = to_infer.drop(columns=["Data"])

    if e_array_avaliable:
        to_infer = data_utils.expanding_dataset_energy(to_infer, (0, 0), log, 0, e_array=e_array)
    else:
        to_infer = data_utils.expanding_dataset_energy(to_infer, (-5.00, 7.30), log, 500)

    if scaler:
        to_infer = scaler.transform(to_infer)

    pred_exfor_expanded = model_utils.make_predictions(to_infer.values, model, model_type)
    pred_exfor_original = model_utils.make_predictions(to_plot.drop(columns=["Data"]).values, model, model_type)

    if inv_trans:
        to_infer = scaler.inverse_transform(to_infer)
        to_plot = scaler.inverse_transform(to_plot)

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
        exfor_endf, error_endf = error_metrics.get_error_endf_exfor(endf, to_plot)
        error_df = error_df.append(error_endf)
        all_dict.update({"exfor_endf_original": exfor_endf, "error_metrics": error_df, "endf": endf})
        if new_data_avaliable:
            # Gets interpolated endf data with anchor new exfor
            exfor_endf_new_data, error_endf_new = error_metrics.get_error_endf_exfor(
                endf, new_data, filter_energy=False)
            error_df = error_df.append(error_endf_new)
            all_dict.update({"exfor_endf_new": exfor_endf_new_data, "error_metrics": error_df})

    _plot_save_predictions(plotter, all_dict, order_dict, save, show)
    return all_dict


def _plot_with_prediction(df, infer_df, y_hat):
    plt.plot(df.Energy, df.Data, alpha=0.5, c="g")
    plt.plot(infer_df.Energy, y_hat)


def make_predictions_w_energy(e_array, df, Z, A, MT, model, model_type, scaler, one_hot=True, log=False,
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
        "Z": Z, "A": A, "MT": MT, "log": log, "scaler": scaler, "one_hot": True,
        "ignore_MT": True}
    to_infer = data_utils.append_energy(e_array, df, **data_kwargs)
    exfor = query_utils.load_samples(df, Z, A, MT, one_hot=one_hot, mt_for="ACE")
    # Make Predictions
    y_hat = model_utils.make_predictions(to_infer.values, model, model_type)
    _plot_with_prediction(exfor, to_infer, y_hat) if show else None
    return y_hat


def make_predictions_from_df(df, Z, A, MT, model, model_type, scaler):
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
    kwargs = {"nat_iso": "I", "one_hot": True, "scaler": scaler}
    exfor = query_utils.load_samples(df, Z, A, MT, **kwargs)
    # Make Predictions
    y_hat = model_utils.make_predictions(exfor.drop(columns=["Data"]).values, model, model_type)
    # _plot_with_prediction(exfor, exfor, y_hat) if show else None
    return y_hat
