"""Training functions for several ML models."""
import itertools
import os
import time
import pandas as pd
import xgboost as xgb
from joblib import dump
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor

import nucml.model.utilities as model_utils


def train_knn(x_train, y_train, x_test, y_test, k_list, save_models=False, save_dir="."):
    """Train multiple KNN models given a list of k-values.

    Useful for quick experimentation. For a more efficient and advanced KNN training method, see the knn.py script.

    Args:
        x_train (DataFrame or np.array): Training data.
        y_train (DataFrame or np.array): Training labels.
        x_test (DataFrame or np.array): Testing data.
        y_test (DataFrame or np.array): Testing labels.
        k_list (list): List of k-values to iterate through.
        save_models (bool, optional): If True, the trained models will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the trained models will be saved. Defaults to ".".

    Returns:
        DataFrame: Contains performance metrics for all trained models.
    """
    # Creating DataFrame to store performance regression metrics
    model_error_metrics = pd.DataFrame(columns=[
        "id", "train_mae", "train_mse", "train_evs", "train_mae_m", "train_r2",
        "test_mae", "test_mse", "test_evs", "test_mae_m", "test_r2",
        "model_path", "training_time", "scaler_path", "run_name"])

    for i in k_list:
        start_training_time = time.time()
        print("Training kNN with k = {}".format(i))
        neigh_model = KNeighborsRegressor(
            n_neighbors=i, weights='distance', algorithm='auto', leaf_size=30, p=2,
            metric='minkowski', metric_params=None, n_jobs=-1)
        neigh_model.fit(x_train, y_train)

        # neigh_model = load(os.path.join(save_dir, "neigh_model_k{}.joblib".format(str(int(i)))))

        stop_training_time = time.time()

        print("Calculating error metrics...")
        y_hat_train = neigh_model.predict(x_train)
        train_error_metrics = model_utils.regression_error_metrics(y_hat_train, y_train)

        y_hat_test = neigh_model.predict(x_test)
        test_error_metrics = model_utils.regression_error_metrics(y_hat_test, y_test)

        to_append = model_utils.create_train_test_error_df(i, train_error_metrics, test_error_metrics)
        to_append["training_time"] = stop_training_time - start_training_time

        if save_models:
            print("Saving Model...")
            saving_dir = os.path.join(save_dir, "neigh_model_k{}.joblib".format(str(int(i))))
            dump(neigh_model, saving_dir)
            to_append["model_path"] = os.path.abspath(saving_dir)

        model_error_metrics = model_error_metrics.append(to_append)

        del neigh_model

    print("Done")
    csv_path = os.path.join(save_dir, "knn_results.csv")
    if os.path.exists(csv_path):
        previous = pd.read_csv(csv_path)
        new = previous.append(model_error_metrics)
        new.to_csv(csv_path, index=False)
    else:
        model_error_metrics.to_csv(csv_path, index=False)
    return model_error_metrics


def train_dt(x_train, y_train, x_test, y_test, parameters_dict, save_models=False, save_dir="."):
    """Train multiple DT models according to the parameters given in the parameters_dict.

    Useful for quick experimentation. For a more efficient and advanced DT training method, see the dt.py script.

    Args:
        x_train (DataFrame or np.array): Training data.
        y_train (DataFrame or np.array): Training labels.
        x_test (DataFrame or np.array): Testing data.
        y_test (DataFrame or np.array): Testing labels.
        parameters_dict (dict): Dictionary object. Keys must be "max_depth_list", "min_split_list", and
            "min_leaf_split". Values must be lists of parameteres to test.
        save_models (bool, optional): If True, the trained models will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the trained models will be saved. Defaults to ".".

    Returns:
        DataFrame: contains the performance metrics for all trained models.
    """
    # Creating DataFrame to store performance regression metrics
    model_error_metrics = pd.DataFrame(columns=[
        "id", "max_depth", "mss", "msl",
        "train_mae", "train_mse", "train_evs", "train_mae_m", "train_r2",
        "test_mae", "test_mse", "test_evs", "test_mae_m", "test_r2",
        "model_path", "training_time", "scaler_path"])

    a = [
        list(parameters_dict["max_depth_list"]),
        list(parameters_dict["min_split_list"]),
        list(parameters_dict["min_leaf_list"])]
    total_num_iterations = len(list(itertools.product(*a)))

    # if tuning:
    loop_number = 0  # ORIGINAL INDENT
    for i in parameters_dict["max_depth_list"]:
        for mss in parameters_dict["min_split_list"]:
            for msl in parameters_dict["min_leaf_list"]:
                if mss <= msl:
                    print("Iteration {}/{}".format(loop_number, total_num_iterations))
                    print("MSS smaller than MSL. Skipping Iteration")
                    loop_number = loop_number + 1
                else:
                    print("Iteration {}/{}".format(loop_number, total_num_iterations))
                    start_training_time = time.time()
                    print("Training Decision Tree Regressor")
                    print("Max Depth = {}; Min Samples Split = {}; Min Samples Leaf = {}".format(i, mss, msl))
                    dt_model = tree.DecisionTreeRegressor(max_depth=i, min_samples_split=mss, min_samples_leaf=msl)
                    dt_model.fit(x_train, y_train)
                    stop_training_time = time.time()

                    print("    Calculating error metrics...")
                    y_hat_train = dt_model.predict(x_train)
                    train_error_metrics = model_utils.regression_error_metrics(y_hat_train, y_train)

                    y_hat_test = dt_model.predict(x_test)
                    test_error_metrics = model_utils.regression_error_metrics(y_hat_test, y_test)

                    to_append = model_utils.create_train_test_error_df(
                        loop_number, train_error_metrics, test_error_metrics)
                    to_append["max_depth"] = dt_model.get_depth()
                    to_append["mss"] = mss
                    to_append["msl"] = msl
                    to_append["training_time"] = stop_training_time - start_training_time

                    if save_models:
                        print("    Saving Model...")
                        saving_dir = os.path.join(save_dir, "dt_model_md{}_mss{}_msl{}.joblib".format(i, mss, msl))
                        dump(dt_model, saving_dir)
                        to_append["model_path"] = os.path.abspath(saving_dir)

                    model_error_metrics = model_error_metrics.append(to_append)
                    loop_number = loop_number + 1

                    del dt_model
    print("Done")
    csv_path = os.path.join(save_dir, "dt_results.csv")
    if os.path.exists(csv_path):
        previous = pd.read_csv(csv_path)
        new = previous.append(model_error_metrics)
        new.to_csv(csv_path, index=False)
    else:
        model_error_metrics.to_csv(csv_path, index=False)
    return model_error_metrics


def train_xgb(x_train, y_train, x_test, y_test, parameters_dict, save_models=False, save_dir="./"):
    """Train multiple DT models according to the parameters given in the parameters_dict argument.

    Useful for quick experimentation. For a more efficient and advanced DT training method, see the dt.py script.

    Args:
        x_train (DataFrame or np.array): Training data.
        y_train (DataFrame or np.array): Training labels.
        x_test (DataFrame or np.array): Testing data.
        y_test (DataFrame or np.array): Testing labels.
        parameters_dict (dict): Dictionary object. Keys must be "max_depth_list", "num_estimator_list", and
            "learning_rate_list". Values must be lists of parameteres to test.
        save_models (bool, optional): If True, the trained models will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the trained models will be saved. Defaults to ".".

    Returns:
        DataFrame: contains the performance metrics for all trained models.
    """
    # Creating DataFrame to store performance regression metrics
    if save_models:
        model_error_metrics = pd.DataFrame(columns=[
            "id", "max_depth", "num_estimator", "lr",
            "train_mae", "train_mse", "train_evs", "train_mae_m", "train_r2", "test_mae", "test_mse",
            "test_evs", "test_mae_m", "test_r2", "model_path", "training_time"])
    else:
        model_error_metrics = pd.DataFrame(columns=[
            "id", "max_depth", "num_estimator", "lr",
            "train_mae", "train_mse", "train_evs", "train_mae_m", "train_r2", "test_mae", "test_mse",
            "test_evs", "test_mae_m", "test_r2", "training_time"])

    a = [
        list(parameters_dict["max_depth_list"]),
        list(parameters_dict["num_estimator_list"]),
        list(parameters_dict["learning_rate_list"])]
    total_num_iterations = len(list(itertools.product(*a)))

    all_dict = {}
    dtrain = xgb.DMatrix(x_train.values, y_train.values)

    evallist = [(dtrain, 'train')]

    loop_number = 1
    for i in parameters_dict["max_depth_list"]:
        for num_estimator in parameters_dict["num_estimator_list"]:
            for lr in parameters_dict["learning_rate_list"]:
                print("Iteration {}/{}".format(loop_number, total_num_iterations))
                start_training_time = time.time()
                print("Training Gradient Boosting Trees")
                print("Max Depth = {}; Number of Estimators = {}; Learning Rate = {}".format(i, num_estimator, lr))

                # specify parameters via map
                param = {"booster": "gbtree", "verbosity": 0, 'eta': lr, "gamma": 0, 'max_depth': i, "lambda": 1,
                         "tree_method": "hist", 'objective': 'reg:squarederror'}
                progress = dict()
                xgb_model = xgb.train(
                    param, dtrain, num_estimator, evallist, evals_result=progress,
                    verbose_eval=False, early_stopping_rounds=10)

                stop_training_time = time.time()

                print("    Calculating error metrics...")
                y_hat_train = xgb_model.predict(xgb.DMatrix(x_train.values))
                train_error_metrics = model_utils.regression_error_metrics(y_hat_train, y_train)

                y_hat_test = xgb_model.predict(xgb.DMatrix(x_test.values))
                test_error_metrics = model_utils.regression_error_metrics(y_hat_test, y_test)

                to_append = model_utils.create_train_test_error_df(loop_number, train_error_metrics, test_error_metrics)
                to_append["max_depth"] = i
                to_append["num_estimator"] = num_estimator
                to_append["lr"] = lr
                to_append["training_time"] = stop_training_time - start_training_time

                model_name = "xgb_md{}_ne{}_lr{}.joblib".format(i, num_estimator, lr)
                if save_models:
                    print("    Saving Model...")
                    saving_dir = os.path.join(save_dir, model_name)
                    dump(xgb_model, saving_dir)
                    to_append["model_path"] = os.path.abspath(saving_dir)

                all_dict.update({model_name: pd.DataFrame(progress["train"])})
                model_error_metrics = model_error_metrics.append(to_append)
                loop_number = loop_number + 1
                xgb_model.__del__()
                del xgb_model

    print("Done")
    csv_path = os.path.join(save_dir, "xgb_results.csv")
    if os.path.exists(csv_path):
        previous = pd.read_csv(csv_path)
        new = previous.append(model_error_metrics)
        new.to_csv(csv_path, index=False)
    else:
        model_error_metrics.to_csv(csv_path, index=False)

    return model_error_metrics, all_dict
