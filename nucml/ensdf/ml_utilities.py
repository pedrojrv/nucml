"""ML-related data utilities."""
import nucml.model.utilities as model_utils
import nucml.ensdf.plot as ensdf_plot
import nucml.ensdf.data_utilities as ensdf_utils


def make_predictions_w_levels(df, Z, A, num_levels, model, model_type, scaler, inv_transform=False,
                              log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    """Return a set of ML predictions up to num_levels.

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        num_levels (int): Upper level number for which to predict the level energy.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges.
            Defaults to False.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied.
            Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied.
            Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to
            False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """
    ensdf = ensdf_utils.load_ensdf_samples(df, Z, A)
    data_kwargs = {"Z": Z, "A": A, "log": log, "scaler": scaler}
    to_infer = ensdf_utils.append_ensdf_levels(num_levels, df, **data_kwargs)
    to_infer["Energy"] = model_utils.make_predictions(to_infer.values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        to_infer = _invert_data_w_scaler(to_infer, scaler, log)
    return to_infer


def make_predictions_w_levels_nodata(df, num_levels, model, model_type, scaler, inv_transform=False,
                                     log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    """Return a set of ML predictions up to num_levels for isotopes with no known nuclear structure data.

    Args:
        df (DataFrame): DataFrame containing all needed information for a given Z, A. Just one row is sufficient.
        num_levels (int): Upper level number for which to predict the level energy.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges. Defaults to
            False.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied.
            Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied.
            Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to
            False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """
    data_kwargs = {"log": log, "scaler": scaler}
    to_infer = ensdf_utils.append_ensdf_levels_nodata(num_levels, df, **data_kwargs)
    to_infer["Energy"] = model_utils.make_predictions(to_infer.values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(
            to_infer.copy(), to_infer.copy(), log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        to_infer = _invert_data_w_scaler(to_infer, scaler, log)
    return to_infer


def _invert_data_w_scaler(to_infer, to_scale, scaler, log):
    if log:
        to_infer["Level_Number"] = 10**to_infer.Level_Number.values
        to_infer["Energy"] = 10**to_infer.Energy.values
    to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
    return to_infer


def make_predictions_from_df(df, Z, A, model, model_type, scaler, log_sqrt=False, log=False, plot=False,
                             save=False, save_dir=""):
    """Return a set of ML predictions at all known levels within the passed DataFrame.

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied.
            Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied.
            Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to
            False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """
    ensdf = ensdf_utils.load_ensdf_samples(df, Z, A)
    to_infer = ensdf_utils.load_ensdf_samples(df, Z, A, scaler=scaler)
    to_infer["Energy"] = model_utils.make_predictions(
        to_infer.drop(columns=["Energy"]).values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    return ensdf, to_infer


def predicting_nuclear_xs_v2(df, Z, A, model, scaler, num_levels=100, log_sqrt=False, model_type=None,
                             save=False, plot=False, save_dir="", inv_trans=False):
    """Plot and return a set of ML predictions at all known levels and up to num_levels.

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        model (object): Trained machine learning model.
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        num_levels (int): Upper level number for which to predict the level energy. Defaults to 100.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied.
            Defaults to False.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit"/None. Defaults to None.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to
            False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges. Defaults to
            False.

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """
    expand_levels = True if num_levels != 0 else False

    to_plot = ensdf_utils.load_ensdf_samples(df, Z, A, scaler=scaler)

    if expand_levels:
        data_kwargs = {"Z": Z, "A": A, "log": log_sqrt, "scaler": scaler}
        to_infer = ensdf_utils.append_ensdf_levels(num_levels, df, **data_kwargs)
    else:
        to_infer = to_plot.drop(columns=["Energy"])

    # Making Predictions
    pred_expanded = model_utils.make_predictions(to_infer.values, model, model_type)
    pred_original = model_utils.make_predictions(to_plot.drop(columns=["Energy"]).values, model, model_type)

    if inv_trans:
        # De-Transforming Scaled Data
        to_infer = scaler.inverse_transform(to_infer)
        to_plot = scaler.inverse_transform(to_plot)

    all_dict = {"expanded": {"df": to_infer, "predictions": pred_expanded},
                "original": {"df": to_plot, "predictions": pred_original}}

    ml_error_metrics = model_utils.regression_error_metrics(to_plot["Energy"], pred_original)
    error_df = model_utils.create_error_df("ENSDF VS ML", ml_error_metrics)
    all_dict.update({"error_metrics": error_df})
    if plot:
        to_infer["Energy"] = pred_expanded
        ensdf_plot.level_density_ml(to_plot, to_infer, log_sqrt=log_sqrt, save=save, save_dir=save_dir)
    return all_dict
