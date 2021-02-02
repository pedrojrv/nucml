import pandas as pd
import os 
import shutil
from joblib import load
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, median_absolute_error, r2_score
import sys

sys.path.append("..")
sys.path.append("../..")

import nucml.ace.data_utilities as ace_utils

def regression_error_metrics(v1, v2):
    """Calculates the MAE, MSE, EVS, MAEM, and R2 between two vectors. 

    Args:
        v1 (np.array): First array.
        v2 (np.array): Second array.

    Returns:
        dict: Dictionary containing all 5 error metrics in key:value pairs.
    """    
    error_metrics = {}
    error_metrics["mae"] = mean_absolute_error(v1, v2)
    error_metrics["mse"] = mean_squared_error(v1, v2)
    error_metrics["evs"] = explained_variance_score(v1, v2)
    error_metrics["mae_m"] = median_absolute_error(v1, v2)
    error_metrics["r2"] = r2_score(v1, v2)
    return error_metrics

def create_error_df(identifier, error_metrics_dict):
    """Creates a simple dataframe from the performance metrics dictionary yielded by the regression_error_metrics() function.

    Args:
        identifier (str or int or float): String or number used for identifying the created dataframe row.
        error_metrics_dict (dict): Dictionary containing the performance metrics.

    Returns:
        DataFrame
    """    
    error_metrics_df = pd.DataFrame({"id":[identifier], 
                              "mae":[error_metrics_dict["mae"]], 
                              "mse":[error_metrics_dict["mse"]], 
                              "evs":[error_metrics_dict["evs"]], 
                              "mae_m":[error_metrics_dict["mae_m"]], 
                              "r2":[error_metrics_dict["r2"]]})
    return error_metrics_df

def create_train_test_error_df(identifier, train_error_metrics, test_error_metrics, val_error_metrics=None):
    """Creates a pandas DataFrame containing the error metrics provided by both the train and test dictionaries generated
        by the regression_error_metrics() function. A validation error metrics dictionary can also be provided.

    Args:
        identifier (str, int): Label use for identification of the row.
        train_error_metrics (dict): Dictionary containing the error metrics for the train set.
        test_error_metrics (dict): Dictionary containing the error metrics for the test set.
        val_error_metrics (dict, optional): Dictionary containing the error metrics for the val set. Defaults to None.

    Returns:
        DataFrame
    """    
    if val_error_metrics is not None:
        error_metrics_df = pd.DataFrame({"id":[identifier], 
                                "train_mae":[train_error_metrics["mae"]], 
                                "train_mse":[train_error_metrics["mse"]], 
                                "train_evs":[train_error_metrics["evs"]], 
                                "train_mae_m":[train_error_metrics["mae_m"]], 
                                "train_r2":[train_error_metrics["r2"]], 
                                "val_mae":[val_error_metrics["mae"]], 
                                "val_mse":[val_error_metrics["mse"]], 
                                "val_evs":[val_error_metrics["evs"]], 
                                "val_mae_m":[val_error_metrics["mae_m"]], 
                                "val_r2":[val_error_metrics["r2"]],
                                "test_mae":[test_error_metrics["mae"]], 
                                "test_mse":[test_error_metrics["mse"]], 
                                "test_evs":[test_error_metrics["evs"]], 
                                "test_mae_m":[test_error_metrics["mae_m"]], 
                                "test_r2":[test_error_metrics["r2"]]
                                })
    else:
        error_metrics_df = pd.DataFrame({"id":[identifier], 
                                "train_mae":[train_error_metrics["mae"]], 
                                "train_mse":[train_error_metrics["mse"]], 
                                "train_evs":[train_error_metrics["evs"]], 
                                "train_mae_m":[train_error_metrics["mae_m"]], 
                                "train_r2":[train_error_metrics["r2"]], 
                                "test_mae":[test_error_metrics["mae"]], 
                                "test_mse":[test_error_metrics["mse"]], 
                                "test_evs":[test_error_metrics["evs"]], 
                                "test_mae_m":[test_error_metrics["mae_m"]], 
                                "test_r2":[test_error_metrics["r2"]]})
    return error_metrics_df


def make_predictions(data, model, model_type):
    """Makes prediction using a trained model. Currently handles tensorflow, xgboost, and scikit-learn models

    Args:
        data (np.array): Numpy matrix needed for model predictions. The data will be prepared
            using tf.data.Dataset for TensorFlow, xgb.DMatrix for xgboost, and passed as is 
            for sklearn models. 
        model (object): Trained machine learning model.
        model_type (str): Type of model being provided. Options include "tf" for TensorFlow, "xgb" for 
            XGBoost, and "sk" for sklearn models.  


    Returns:
        object: object containing the model predictions. Type will be dependent on model type.
    """
    if model_type == "tf":
        tf_dataset = tf.data.Dataset.from_tensor_slices((data)).batch(len(data))
        pred_vector = model.predict(tf_dataset)
    elif model_type == "xgb":
        xg_dataset = xgb.DMatrix(data)
        pred_vector = model.predict(xg_dataset)
    else:
        pred_vector = model.predict(data)
    return pred_vector



def get_best_models_df(results_df, keep_first=False):
    """Returns a three row minimum dataframe with the best models based on training, validation, and testing 
    performance metrics. The results_df argument is based on the file generated by the python training scripts 
    including knn.py, dt.py, and xgb.py which includes results for all training iterations along with stored 
    model and scaler paths.  

    Args:
        results_df (DataFrame): Results dataframe created by the model training scripts. 
        keep_first (bool, optional): In some cases there might be duplicates. If True, this will keep the 
        the first instance of a duplicate value. Defaults to False.

    Returns:
        DataFrame
    """    
    best_train = results_df[results_df.train_mae == results_df.train_mae.min()].drop_duplicates(keep="last")
    best_train["tag"] = "Train"

    best_val = results_df[results_df.val_mae == results_df.val_mae.min()].drop_duplicates(keep="last")
    best_val["tag"] = "Val"

    best_test = results_df[results_df.test_mae == results_df.test_mae.min()].drop_duplicates(keep="last")
    best_test['tag'] = "Test"

    if keep_first:
        best_train = best_train.head(1)
        best_val = best_val.head(1)
        best_test = best_test.head(1)

    best_train = best_train.append(best_val)
    best_models = best_train.append(best_test)
    return best_models

def load_model_and_scaler(model_scaler_info, df=True):
    """Loads both the model and scaler given a dataframe with path's specified.

    Args:
        model_scaler_info (DataFrame): Must contain a "model_path" and a "scaler_path" feature if a DataFrame
            is passed. Else, it must contain the "model_path" and "scaler_path" as keys in a dictionary.
        df (bool, optional): If True, the model_scaler_info variable must be a DataFrame. If False, it must be
            a python dictionary.

    Returns:
        object, object: returns the loaded model and scaler.
    """    
    if df:
        path_to_model = model_scaler_info["model_path"].values[0]
        path_to_scaler = model_scaler_info["scaler_path"].values[0]
    else:
        path_to_model = model_scaler_info["model_path"]
        path_to_scaler = model_scaler_info["scaler_path"]
    model = load(path_to_model) 
    scaler = load(path_to_scaler)
    return model, scaler



def cleanup_model_dir(results_df, model_dir, keep_best=True, keep_first=False):
    """Deletes unwanted models and scalers. Keeps best models based on training, validation, and testing perfromance if wanted.

    Args:
        results_df (DataFrame): The loaded results data file created by the various training scripts.
        model_dir (str): Path-like string where all model directories are stored.
        keep_best (bool, optional): If True, it will keep three or more models based on performance. Defaults to True.
        keep_first (bool, optional): If True, it will keep the first appearance in case of a duplicate rows. Defaults to False.
    
    Returns:
        None
    """    
    not_to_delete = []
    if keep_best:
        best_models = get_best_models_df(results_df, keep_first=keep_first)
        for i in best_models.model_path.values:
            not_to_delete.extend([os.path.basename(os.path.dirname(i))])

    for i in os.listdir(model_dir):
        if i not in not_to_delete:
            if os.path.isdir(os.path.join(model_dir,i)):
                shutil.rmtree(os.path.join(model_dir,i))
    return None


# # testing = remove_unused_models("../ML_EXFOR_neutrons/2_DT/DT_B1/dt_results.csv", "acedata_ml/U233/DT_B1/")
def remove_unused_models(model_results_path, acedate_directory):
    """Finds best models in terms of train, validation and testing sets and deletes all others. It
    also keeps the best models in terms of multiplication factor.

    WARNING: Once deleted, other models will not be accessible. 

    Args:
        model_results_path (str): Filepath to model training results CSV file generated using the model training scripts.
        acedate_directory (str): Path to the relevant directory were all models for a given algorithm are stored. 

    Returns:
        None
    """    
    model_results_df = pd.read_csv(model_results_path)
    model_results_df["Model"] = model_results_df.model_path.apply(lambda x: os.path.basename(os.path.dirname(x)))
    model_results_df["main_directory"] = model_results_df.model_path.apply(lambda x: os.path.dirname(x) + "\\")
    model_results_df = model_results_df[["Model", "train_mae", "val_mae", "test_mae", "main_directory"]]
    
    benchmark_results = ace_utils.gather_benchmark_results(acedate_directory)
    model_results_df = model_results_df.merge(benchmark_results, on="Model")
    
    # KEEP BEST TRAIN VAL TEST
    # KEEP TOP 3 SORTED DEVIATION ANA
    to_keep = []
    to_keep.extend(list(model_results_df.iloc[model_results_df.sort_values(by="Deviation_Ana").head().index].Model.values))
    to_keep.extend(list(model_results_df.iloc[get_best_models_df(model_results_df).index].Model.values))
    model_results_df["filtering"] = model_results_df.Model.apply(lambda name: True if name not in to_keep else False)
    to_remove = model_results_df[model_results_df.filtering == True]
    
    for i in to_remove.main_directory.values:
        shutil.rmtree(i)
    
    return None