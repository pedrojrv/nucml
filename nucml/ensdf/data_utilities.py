import pandas as pd 
import numpy as np 
import tensorflow as tf 
import xgboost as xgb 
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

import nucml.model.model_utilities as model_utils   # pylint: disable=import-error
import nucml.ensdf.plotting_utilities as ensdf_plot    # pylint: disable=import-error
import nucml.general_utilities as gen_utils # pylint: disable=import-error

import logging

from sklearn import linear_model

import os

from joblib import dump, load

def load_ensdf_samples(df, Z, A, scale=False, scaler=None, to_scale=[]):
    """
    Loads ENSDF data for a particular Isotope
    """
    logging.info("Extracting samples from dataframe.")
    sample = df[(df["Protons"] == Z) & (df["Mass_Number"] == A)].sort_values(by='Level_Number', ascending=True)
    if scale:
        logging.info("Scaling dataset...")
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def load_ensdf_element(df, Z, scale=False, scaler=None, to_scale=[]):
    """
    Loads ENSDF data for a particular element (includes all isotopes)
    """
    logging.info("Extracting samples from dataframe.")
    sample = df[(df["Protons"] == Z)]
    if scale:
        logging.info("Scaling dataset...")
        sample[to_scale] = scaler.transform(sample[to_scale])
    logging.info("ENSDF extracted DataFrame has shape: {}".format(sample.shape))
    return sample


def append_ensdf_levels(tot_num_levels, df, Z, A, log=False, scale=False, scaler=None, to_scale=[]):
    """
    Loads New Measurments and appends ENSDF isotopic data to it. 
    Assumes new data only has an Energy and Data column
    It does not depend on df/
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
    """
    Loads New Measurments and appends ENSDF isotopic data to it. 
    Assumes new data only has an Energy and Data column
    It does not depend on df/
    """
    new_data = pd.DataFrame({"Level_Number":np.arange(1,tot_num_levels + 1)})
    isotope_exfor = df.copy()
    if "Level_Energy" in isotope_exfor.columns:
        isotope_exfor =  isotope_exfor.drop(columns="Level_Energy")
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
    """
    Loads New Measurments and appends ENSDF isotopic data to it. 
    Assumes new data only has an Energy and Data column
    It does not depend on df/
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
    if nodata:
        original = df.copy()
    else:
        original = load_ensdf_samples(df, Z, A)
    
    element = original.Target_Element_w_A.values[0]
    logging.info("Generating level density for {}".format(element))

    if tot_num_levels != 0:
        if nodata:
            simple = append_ensdf_levels_nodata(tot_num_levels, df.copy(), log=True, scale=False)
        else:
            simple = append_ensdf_levels(tot_num_levels, df.copy(), Z, A, log=True, scale=False)
    else:
        simple = original.copy()
    

    original = original[["Level_Number", "Level_Energy"]]
    simple = simple[["Level_Number"]]
    
    reg = linear_model.LinearRegression()
    reg.fit(original.drop("Level_Energy", 1), original.Level_Energy)
    
    pred = pd.DataFrame()
    pred["Level_Number"] = simple.Level_Number
    pred["Level_Energy"] = reg.predict(pred)

    if get_upper:
        logging.info("Initalizing starting variables for NLD extrapolation...")
        last_energy = pred.Level_Energy.values[-1]
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
                pred["Level_Energy"] = reg.predict(pred)
                last_energy = pred.Level_Energy.values[-1]
                x = x + 1
                if x == it_limit:
                    logging.info("Iteration limit reached. Target energy not reached.")
                    break
    if plot:
        ensdf_plot.plot_level_density_ml(original, pred, log_sqrt=False, log=True)
    if save:
        pred["A"] = A
        pred["Z"] = Z
        pred["Target_Element_w_A"] = element
        pred["Level_Number"] = 10**pred.Level_Number.values
        pred["Level_Energy"] = 10**pred.Level_Energy.values
        pred["Level_Number"] = pred.Level_Number.astype(int)
        gen_utils.create_directories(saving_dir)
        pred.to_csv(os.path.join(saving_dir, "{}_Level_Density.csv".format(element)), index=False)
        dump(reg, os.path.join(saving_dir, '{}_NLD_linear_model.joblib'.format(element))) 
    return pred

def get_level_density(energy_mev, pred_df):
    to_append = pd.DataFrame({"Level_Number":[np.nan], "Level_Energy": [np.log10(energy_mev)], "N":[np.nan]})
    to_interpolate = pred_df.append(to_append, ignore_index=True)
    to_interpolate = to_interpolate.sort_values(by="Level_Energy")
    new_index = len(to_interpolate) - 1
    to_interpolate = to_interpolate.interpolate()
    level_density = to_interpolate.loc[new_index]["N"]
    return level_density

def make_predictions_w_levels(df, Z, A, num_levels, clf, clf_type, scaler, to_scale, inv_transform=False, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    ensdf = load_ensdf_samples(df, Z, A)
    data_kwargs = {"Z":Z, "A":A, "log":log, "scale":True, "scaler":scaler, "to_scale":to_scale}
    to_infer = append_ensdf_levels(num_levels, df, **data_kwargs)
    to_infer["Level_Energy"] = model_utils.make_predictions(to_infer.values, clf, clf_type)
    if plot:
        ensdf_plot.plot_level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        if log:
            to_infer["Level_Number"] = 10**to_infer.Level_Number.values
            to_infer["Level_Energy"] = 10**to_infer.Level_Energy.values
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
    return to_infer

def make_predictions_w_levels_nodata(df, num_levels, clf, clf_type, scaler, to_scale, inv_transform=False, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    data_kwargs = {"log":log, "scale":True, "scaler":scaler, "to_scale":to_scale}
    to_infer = append_ensdf_levels_nodata(num_levels, df, **data_kwargs)
    to_infer["Level_Energy"] = model_utils.make_predictions(to_infer.values, clf, clf_type)
    if plot:
        ensdf_plot.plot_level_density_ml(to_infer.copy(), to_infer.copy(), log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    if inv_transform:
        if log:
            to_infer["Level_Number"] = 10**to_infer.Level_Number.values
            to_infer["Level_Energy"] = 10**to_infer.Level_Energy.values
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
    return to_infer

def make_predictions_from_df(df, Z, A, clf, clf_type, scaler, to_scale, log_sqrt=False, log=False, plot=False, save=False, save_dir=""):
    kwargs = {"scale": True, "scaler": scaler, "to_scale": to_scale}
    ensdf = load_ensdf_samples(df, Z, A)
    to_infer = load_ensdf_samples(df, Z, A, **kwargs)
    to_infer["Level_Energy"] = model_utils.make_predictions(
        to_infer.drop(columns=["Level_Energy"]).values, clf, clf_type)
    if plot:
        ensdf_plot.plot_level_density_ml(ensdf, to_infer, log_sqrt=log_sqrt, log=log, save=save, save_dir=save_dir)
    return ensdf, to_infer


def predicting_nuclear_xs_v2(df, Z, A, clf, scaler, to_scale, num_levels=100, log_sqrt=False, clf_type=None,
    save=False, plot=False, save_dir="", inv_trans=False):
    '''
    endf=empty_df, 
    Used to plot predictions of the clf model for specific isotope (Z, A) and runs.
    MT is the reaction type (e.g 1 is total cross section)
    E_min and E_max are the energy region in which to make additional inferences.

    pred_expanded: expanded or non expanded to infer data
    pred_original: original ENSDF data points.
    pred_exfor_new: new data predictions (if avaliable)

    '''
    expand_levels = True if num_levels != 0 else False

    to_plot = load_ensdf_samples(df, Z, A, scale=True, scaler=scaler, to_scale=to_scale) 

    if expand_levels: 
        data_kwargs = {"Z":Z, "A":A, "log":log_sqrt, "scale":True, "scaler":scaler, "to_scale":to_scale}
        to_infer = append_ensdf_levels(num_levels, df, **data_kwargs)
    else:
        to_infer = to_plot.drop(columns=["Level_Energy"])  
    
    # Making Predictions
    pred_expanded = model_utils.make_predictions(to_infer.values, clf, clf_type)
    pred_original = model_utils.make_predictions(to_plot.drop(columns=["Level_Energy"]).values, clf, clf_type)

    if inv_trans:
        # De-Transforming Scaled Data
        to_infer[to_scale] = scaler.inverse_transform(to_infer[to_scale])
        to_plot[to_scale] = scaler.inverse_transform(to_plot[to_scale])

    all_dict = {"expanded":{"df":to_infer, "predictions":pred_expanded}, 
                "original":{"df":to_plot, "predictions":pred_original}}

    ml_error_metrics = model_utils.regression_error_metrics(to_plot["Level_Energy"], pred_original)
    error_df = model_utils.create_error_df("ENSDF VS ML", ml_error_metrics)
    all_dict.update({"error_metrics":error_df})
    if plot:
        to_infer["Level_Energy"] = pred_expanded
        ensdf_plot.plot_level_density_ml(to_plot, to_infer, log_sqrt=log_sqrt, save=save, save_dir=save_dir)
    return all_dict

    

# def load_ensdf_ml(log=True, log_sqrt=False, cutoff=False, append_ame=False, basic=-1, num=False, frac=0.3, scaling_type="pt", scaler_dir=None):
#     """Loads the Evalauted Nuclear Structure Data File generated using NucML. This allows the user to load
#     the raw file or preprocessed the dataset for ML applications. See options below.

#     The basic feature allows you to load only some basic features if needed. The AME dataset contains
#     many features including q-reactions and separation energies. Some of these may not be needed. The
#     basic argument allows to quickly remove extra features. 
#         basic = 0: "Level_Number", "Level_Energy", "Protons", "Neutrons", "Mass_Number", "Spin", "Parity", "Atomic_Mass_Micro"
#         basic = 1: "Level_Number", "Level_Energy", "Protons", "Neutrons", "Mass_Number", "Spin", "Parity", 
#             "Atomic_Mass_Micro", 'Mass_Excess', 'Binding_Energy', 'B_Decay_Energy', 'S(2n)', 'S(n)', 'S(p)'
#         Any other number will default to loading the entire set of features.

#     Args:
#         cutoff (bool, optional): If True, the RIPL cutoff ENSDF file is loaded. Defaults to False.
#         log (bool, optional): If True, the log10 is applied to the Level Number feature. It also applies 
#             the square root to the Level Energy feature. Defaults to False.
#         append_ame (bool, optional): if True, it appends the AME database. Defaults to False.
#         basic (int, optional): This allows to retrieve only basic features. 
#             Only meaningful when append_ame is True. Defaults to -1.
#         num (bool, optional): [description]. Defaults to False.
#         frac (float, optional): [description]. Defaults to 0.3.
#         scaling_type (str, optional): [description]. Defaults to "pt".
#         scaler_dir ([type], optional): [description]. Defaults to None.

#     Returns:
#         DataFrame: if num=True, the function returns 6 variables. 
#     """    
#     if cutoff:
#         datapath = "../../ENSDF/CSV_Files/ensdf_cutoff.csv"
#     else:
#         datapath = "../../ENSDF/CSV_Files/ensdf.csv"

#     logging.info("Reading data from {}".format(datapath))
#     df = pd.read_csv(datapath)
#     df["Level_Number"] = df["Level_Number"].astype(int)


#     if log_sqrt and not log:
#         df["Level_Energy"] = np.sqrt(df["Level_Energy"])
#         df["Level_Number"] = np.log10(df["Level_Number"])
#     if log and not log_sqrt:
#         df = df[(df["Level_Energy"] != 0)]
#         df["Level_Energy"] = np.log10(df["Level_Energy"])
#         df["Level_Number"] = np.log10(df["Level_Number"])
#     if append_ame:
#         ame = load_ame(imputed_nan=True)
#         df = pd.merge(df, ame, on='Element_w_A')

#     if basic == 0:
#         basic_cols = ["Level_Number", "Energy", "Z", "N", "A", "Spin", "Parity", "Atomic_Mass_Micro"]
#         df = df[basic_cols]
#     elif basic == 1:
#         basic_cols = ["Level_Number", "Energy", "Z", "N", "A", "Spin", "Parity", "Atomic_Mass_Micro",
#                     'Mass_Excess', 'Binding_Energy', 'B_Decay_Energy', 'S(n)', 'S(p)']
#         df = df[basic_cols] 

#     if num:
#         logging.info("Dropping unnecessary features and one-hot encoding categorical columns...")
#         if basic == 0 or basic == 1:
#             cat_cols = ["Parity"]
#         else:
#             columns_drop = ["Target_Element_w_A", "EL", "O", "Decay_Info", "ENSDF_Spin"]
#             cat_cols = ["Parity", "Flag"]
#             df = df.drop(columns=columns_drop)
#         # We need to keep track of columns to normalize excluding categorical data.
#         norm_columns = len(df.columns) - len(cat_cols) - 1
#         df = pd.concat([df, pd.get_dummies(df[cat_cols])], axis=1).drop(columns=cat_cols)
#         df = df.fillna(value=0)
#         logging.info("Splitting dataset into training and testing...")
#         x_train, x_test, y_train, y_test = train_test_split(df.drop(["Level_Energy"], axis=1), df["Level_Energy"], test_size=frac)
#         logging.info("Normalizing dataset...")
#         to_scale = list(x_train.columns)[:norm_columns]
#         if log_sqrt or log:
#             to_scale.remove("Level_Number")
#         if scaler_dir is not None:
#             logging.info("Using previously saved scaler.")
#             scaler = load(open(scaler_dir, 'rb'))
#         else:
#             logging.info("Fitting new scaler.")
#             if scaling_type == "pt":
#                 scaler = preprocessing.PowerTransformer().fit(x_train[to_scale])
#             elif scaling_type == "std":
#                 scaler = preprocessing.StandardScaler().fit(x_train[to_scale])
#             elif scaling_type == "minmax":
#                 scaler = preprocessing.MinMaxScaler().fit(x_train[to_scale])
#         x_train[to_scale] = scaler.transform(x_train[to_scale])
#         x_test[to_scale] = scaler.transform(x_test[to_scale])
#         logging.info("Finished. Resulting dataset has shape {}, Training and Testing dataset shapes are {} and {} respesctively.".format(df.shape, x_train.shape, x_test.shape))
#         return df, x_train, x_test, y_train, y_test, to_scale, scaler
#     else:
#         logging.info("Finished. Resulting dataset has shape {}".format(df.shape))
#         return df