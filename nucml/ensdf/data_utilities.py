import pandas as pd 
import numpy as np 
import logging
from sklearn import linear_model
import os
from joblib import dump
import sys

sys.path.append("..")
sys.path.append("../..")

import nucml.model.utilities as model_utils   # pylint: disable=import-error
import nucml.ensdf.plot as ensdf_plot         # pylint: disable=import-error
import nucml.general_utilities as gen_utils   # pylint: disable=import-error


def load_ensdf_samples(df, Z, A, scale=False, scaler=None, to_scale=[]):
    """Loads ENSDF data for a particular isotope (Z, A).

    Args:
        df (DataFrame): DataFrame containing all necessary information for Z, A. 
        Z (int): Number of protons.
        A (int): Mass Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted isotope sample.
    """
    logging.info("Extracting samples from dataframe.")
    sample = df[(df["Z"] == Z) & (df["A"] == A)].sort_values(by='Level_Number', ascending=True)
    if scale:
        logging.info("Scaling dataset...")
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def load_ensdf_element(df, Z, scale=False, scaler=None, to_scale=[]):
    """Loads ENSDF data for a given element (Z).

    Args:
        df (DataFrame): DataFrame containing all necessary information for Z, A. 
        Z (int): Number of protons.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    logging.info("Extracting samples from dataframe.")
    sample = df[(df["Z"] == Z)]
    if scale:
        logging.info("Scaling dataset...")
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def append_ensdf_levels(tot_num_levels, df, Z, A, log=False, scale=False, scaler=None, to_scale=[]):
    """Expands the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample.

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample. 
        Z (int): Number of protons.
        A (int): Mass Number.
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number":np.arange(1,tot_num_levels + 1)})
    isotope_exfor = load_ensdf_samples(df, Z, A)
    for i in list(isotope_exfor.columns)[2:]:
        new_data[i] = isotope_exfor[i].values[0] # changed to 0 from 1
    logging.info("Expanded Dataset has shape: {}".format(new_data.shape))
    if scale:
        logging.info("Scaling dataset...")
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data

def append_ensdf_levels_nodata(tot_num_levels, df, log=False, scale=False, scaler=None, to_scale=[]):
    """Expands the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample if no 
    Level Energy data is avaliable.

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample. 
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number":np.arange(1,tot_num_levels + 1)})
    isotope_exfor = df.copy()
    if "Energy" in isotope_exfor.columns:
        isotope_exfor =  isotope_exfor.drop(columns="Energy")
    for i in list(isotope_exfor.columns)[1:]:
        new_data[i] = isotope_exfor[i].values[0] # changed to 0 from 1
    logging.info("Expanded Dataset has shape: {}".format(new_data.shape))
    if scale:
        logging.info("Scaling dataset...")
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data

def append_ensdf_levels_range(tot_num_levels, df, Z, A, steps=1, log=False, scale=False, scaler=None, to_scale=[]):
    """Expands the energy levels up to "tot_num_levels" for the given ENSDF isotopic sample using
    a range with n steps rather than linear. 

    Args:
        tot_num_levels (int): Total number of levels to include (i.e 50 will include levels 1-50).
        df (DataFrame): DataFrame containing an already extracted isotopic sample. 
        Z (int): Number of protons.
        A (int): Mass Number.
        steps (int): Number of intermediate steps between 1 and tot_num_levels to create. 
        log (bool, optional): If True, the logarithm will be applied to the Level Number.
        scale (bool, optional): If True, the data will be tranform using the provided scaler. Defaults to False.
        scaler (object, optional): Scikit-Learn trained transformer. Defaults to None.
        to_scale (list, optional): List of features to be scaled. Defaults to [].

    Returns:
        DataFrame: Extracted element sample.
    """
    new_data = pd.DataFrame({"Level_Number":np.arange(1,tot_num_levels + 1, steps)})
    isotope_exfor = load_ensdf_samples(df, Z, A)
    for i in list(isotope_exfor.columns)[2:]:
        new_data[i] = isotope_exfor[i].values[1]
    logging.info("Expanded Dataset has shape: {}".format(new_data.shape))
    if scale:
        logging.info("Scaling dataset...")
        new_data[to_scale] = scaler.transform(new_data[to_scale])
    if log:
        new_data["Level_Number"] = np.log10(new_data["Level_Number"])
    return new_data

def generate_level_density_csv(df, Z, A, nodata=False, upper_energy_mev=20, get_upper=False, tot_num_levels=0, it_limit=500, plot=False, save=False, saving_dir=""):
    """Fits a linear model to the isotopic sample provided, and saves a CSV file with the linear model values for each 
    energy level avaliable. If get_upper is True, then a new level number will be appended until the linear model predicts
    a value above the upper_energy_mev value. 

    Args:
        df (DataFrame): DataFrame containing the needed data for isotope Z, A. 
        Z (int): Number of protons.
        A (int): Mass number.
        nodata (bool, optional): If True, it assumes there is no avaliable data for the queried isotope. Defaults to False.
        upper_energy_mev (int, optional): If get_upper is True, the algorithm will iterate until this energy is reached. Defaults to 20 MeV.
        get_upper (bool, optional): If True, more levels will be added until the level energy in the level density reaches the 20 MeV mark. Defaults to False.
        tot_num_levels (int, optional): If any value other than 0 is given, it will append the remaining energy levels until reaching tot_num_levels. Defaults to 0.
        it_limit (int, optional): Sets the iteration limits for the linear model to reach the upper_energy_mev value. Defaults to 500.
        plot (bool, optional): If True, a plot of the linear model along the experimental levels will be rendered. Defaults to False.
        save (bool, optional): If True, the resulting DataFrame using the linear model will be saved. Defaults to False.
        saving_dir (str, optional): Path-like string pointing towars the directory where the DataFrame will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with Level Number and Level Energy as predicted by the linear model. 
    """
    if nodata:
        original = df.copy()
    else:
        original = load_ensdf_samples(df, Z, A)
    
    element = original.Element_w_A.values[0]
    logging.info("Generating level density for {}".format(element))

    if tot_num_levels != 0:
        if nodata:
            simple = append_ensdf_levels_nodata(tot_num_levels, df.copy(), log=True, scale=False)
        else:
            simple = append_ensdf_levels(tot_num_levels, df.copy(), Z, A, log=True, scale=False)
    else:
        simple = original.copy()
    

    original = original[["Level_Number", "Energy"]]
    simple = simple[["Level_Number"]]
    
    reg = linear_model.LinearRegression()
    reg.fit(original.drop("Energy", 1), original.Energy)
    
    pred = pd.DataFrame()
    pred["Level_Number"] = simple.Level_Number
    pred["Energy"] = reg.predict(pred)

    if get_upper:
        logging.info("Initalizing starting variables for NLD extrapolation...")
        last_energy = pred.Energy.values[-1]
        number_levels = tot_num_levels
        upper_limit = np.log10(upper_energy_mev)
        x = 0
        
        while last_energy < upper_limit:
                number_levels = number_levels + 100
                if nodata:
                    simple = append_ensdf_levels_nodata(number_levels, df.copy(), log=True, scale=False)
                else:
                    simple = append_ensdf_levels(number_levels, df.copy(), Z, A, log=True, scale=False)
                pred = pd.DataFrame()
                pred["Level_Number"] = simple.Level_Number
                pred["Energy"] = reg.predict(pred)
                last_energy = pred.Energy.values[-1]
                x = x + 1
                if x == it_limit:
                    logging.info("Iteration limit reached. Target energy not reached.")
                    break
    if plot:
        ensdf_plot.level_density_ml(original, pred, log_sqrt=False, log=True)
    if save:
        pred["A"] = A
        pred["Z"] = Z
        pred["Element_w_A"] = element
        pred["Level_Number"] = 10**pred.Level_Number.values
        pred["Energy"] = 10**pred.Energy.values
        pred["Level_Number"] = pred.Level_Number.astype(int)
        gen_utils.initialize_directories(saving_dir)
        pred.to_csv(os.path.join(saving_dir, "{}_Level_Density.csv".format(element)), index=False)
        dump(reg, os.path.join(saving_dir, '{}_NLD_linear_model.joblib'.format(element))) 
    return pred

def get_level_density(energy_mev, df):
    """Given an energy density DataFrame, the level density at a wanted energy is returned.

    Args:
        energy_mev (float): Energy point at which the level density is to be returned.
        df (DataFrame): Level Density DataFrame to interpolate at energy_mev. 

    Returns:
        float: Level density at "energy_mev".
    """    
    to_append = pd.DataFrame({"Level_Number":[np.nan], "Energy": [np.log10(energy_mev)], "N":[np.nan]})
    to_interpolate = df.append(to_append, ignore_index=True)
    to_interpolate = to_interpolate.sort_values(by="Energy")
    new_index = len(to_interpolate) - 1
    to_interpolate = to_interpolate.interpolate()
    level_density = to_interpolate.loc[new_index]["N"]
    return level_density

def make_predictions_w_levels(df, Z, A, num_levels, model, model_type, scaler, to_scale, inv_transform=False, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    """Returns a set of ML predictions up to num_levels. 

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        num_levels (int): Upper level number for which to predict the level energy.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges. Defaults to False.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied. Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied. Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """    
    ensdf = load_ensdf_samples(df, Z, A)
    data_kwargs = {"Z":Z, "A":A, "log":log, "scale":True, "scaler":scaler, "to_scale":to_scale}
    to_infer = append_ensdf_levels(num_levels, df, **data_kwargs)
    to_infer["Energy"] = model_utils.make_predictions(to_infer.values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        if log:
            to_infer["Level_Number"] = 10**to_infer.Level_Number.values
            to_infer["Energy"] = 10**to_infer.Energy.values
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
    return to_infer

def make_predictions_w_levels_nodata(df, num_levels, model, model_type, scaler, to_scale, inv_transform=False, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    """Returns a set of ML predictions up to num_levels for isotopes with no known nuclear structure data.

    Args:
        df (DataFrame): DataFrame containing all needed information for a given Z, A. Just one row is sufficient.
        num_levels (int): Upper level number for which to predict the level energy.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges. Defaults to False.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied. Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied. Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """    
    data_kwargs = {"log":log, "scale":True, "scaler":scaler, "to_scale":to_scale}
    to_infer = append_ensdf_levels_nodata(num_levels, df, **data_kwargs)
    to_infer["Energy"] = model_utils.make_predictions(to_infer.values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(to_infer.copy(), to_infer.copy(), log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        if log:
            to_infer["Level_Number"] = 10**to_infer.Level_Number.values
            to_infer["Energy"] = 10**to_infer.Energy.values
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
    return to_infer

def make_predictions_from_df(df, Z, A, model, model_type, scaler, to_scale, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    """Returns a set of ML predictions at all known levels within the passed DataFrame. 

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        model (object): Trained machine learning model.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit".
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied. Defaults to False.
        log (bool, optional): If True, it assumes the models where trained on Level Energy data with LOG applied. Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to False.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".

    Returns:
        DataFrame: New DataFrame with ML predictions.
    """    
    kwargs = {"scale": True, "scaler": scaler, "to_scale": to_scale}
    ensdf = load_ensdf_samples(df, Z, A)
    to_infer = load_ensdf_samples(df, Z, A, **kwargs)
    to_infer["Energy"] = model_utils.make_predictions(
        to_infer.drop(columns=["Energy"]).values, model, model_type)
    if plot:
        ensdf_plot.level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    return ensdf, to_infer


def predicting_nuclear_xs_v2(df, Z, A, model, scaler, to_scale, num_levels=100, log_sqrt=False, model_type=None,
    save=False, plot=False, save_dir="", inv_trans=False):
    """Plots and returns a set of ML predictions at all known levels and up to num_levels in cases where the known levels are fewer. 

    Args:
        df (DataFrame): DataFrame containing all needed information for Z, A.
        Z (INT): Number of protons.
        A (int): Mass Number.
        model (object): Trained machine learning model.
        scaler (object): Trained scikit-learn normalizer/transformer.
        to_scale (list): List of features that are to be subject to transformation by the scaler.
        num_levels (int): Upper level number for which to predict the level energy. Defaults to 100.
        log_sqrt (bool, optional): If True, it assumes the models where trained on Level Energy data with SQRT applied. Defaults to False.
        model_type (str): Type of ML model. Options include "tf", "xgb", or "scikit"/None. Defaults to None.
        save (bool, optional): If True, the rendered figure will be saved. Defaults to False.
        plot (bool, optional): If True, the ML predictions will be plotted along the true known values. Defaults to False.
        save_dir (str, optional): Path-like string indicating directory where the figure will be saved. Defaults to "".
        inv_transform (bool, optional): If True, the returned DataFrame will be in its original ranges. Defaults to False.
        
    Returns:
        DataFrame: New DataFrame with ML predictions.
    """   

    expand_levels = True if num_levels != 0 else False

    to_plot = load_ensdf_samples(df, Z, A, scale=True, scaler=scaler, to_scale=to_scale) 

    if expand_levels: 
        data_kwargs = {"Z":Z, "A":A, "log":log_sqrt, "scale":True, "scaler":scaler, "to_scale":to_scale}
        to_infer = append_ensdf_levels(num_levels, df, **data_kwargs)
    else:
        to_infer = to_plot.drop(columns=["Energy"])  
    
    # Making Predictions
    pred_expanded = model_utils.make_predictions(to_infer.values, model, model_type)
    pred_original = model_utils.make_predictions(to_plot.drop(columns=["Energy"]).values, model, model_type)

    if inv_trans:
        # De-Transforming Scaled Data
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
        to_plot[to_scale] = scaler.inverse_transform(to_plot[to_scale])

    all_dict = {"expanded":{"df":to_infer, "predictions":pred_expanded}, 
                "original":{"df":to_plot, "predictions":pred_original}}

    ml_error_metrics = model_utils.regression_error_metrics(to_plot["Energy"], pred_original)
    error_df = model_utils.create_error_df("ENSDF VS ML", ml_error_metrics)
    all_dict.update({"error_metrics":error_df})
    if plot:
        to_infer["Energy"] = pred_expanded
        ensdf_plot.level_density_ml(to_plot, to_infer, log_sqrt=log_sqrt, save=save, save_dir=save_dir)
    return all_dict