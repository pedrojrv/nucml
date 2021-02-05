import argparse
import numpy as np

CLI=argparse.ArgumentParser()

CLI.add_argument(
  "--version",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="?",  # 0 or more values expected => creates a list
  type=str,
  default="v1",  # default if nothing is provided
)

CLI.add_argument(
  "--range",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[60, 410, 10],  # default if nothing is provided
)

CLI.add_argument(
  "--dataset",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="?",  # 0 or more values expected => creates a list
  type=str,
  default="B0"  # default if nothing is provided
)

CLI.add_argument(
  "--normalizer",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="?",  # 0 or more values expected => creates a list
  type=str,
  default="standard"  # default if nothing is provided
)

args = CLI.parse_args()


DATASET = args.dataset
VERSION = '_' + args.version
DATASET_DICT = {"B0":0, "B1":1, "B2":2, "B3":3}
NORMALIZER_TYPE = args.normalizer
MT_STRATEGY = "one_hot"
TRAIN_FRACTION = 0.9
MAX_DEPTH_LIST = np.arange(args.range[0], args.range[1], args.range[2])
MIN_SPLIT_LIST = [2, 5, 10, 15]
MIN_LEAF_LIST = [1, 3, 5, 7, 10]


##############################################################################################
################################### IMPORTING MODULES ########################################
##############################################################################################

import pandas as pd
import os
from joblib import dump
import time
from sklearn import tree
import sys
import itertools
from sklearn.model_selection import train_test_split

# This allows us to import the nucml utilities
sys.path.append("../..")

import nucml.datasets as nuc_data # pylint: disable=import-error
import nucml.model.model_utilities as model_utils # pylint: disable=import-error



parameters_dict = {"max_depth_list":MAX_DEPTH_LIST, "min_split_list":MIN_SPLIT_LIST, "min_leaf_list":MIN_LEAF_LIST}

df, x_train, x_test, y_train, y_test, to_scale, scaler = nuc_data.load_exfor(
    pedro=True, basic=DATASET_DICT[DATASET], frac=1-TRAIN_FRACTION, scaling_type=NORMALIZER_TYPE, mt_coding=MT_STRATEGY)

x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)


a = [list(MAX_DEPTH_LIST), list(MIN_SPLIT_LIST), list(MIN_LEAF_LIST)]
total_num_iterations = len(list(itertools.product(*a)))
loop_number = 1 # ORIGINAL INDENT
for i in MAX_DEPTH_LIST:
    for mss in MIN_SPLIT_LIST:
        for msl in MIN_LEAF_LIST:
            if mss <= msl:
                print("Iteration {}/{}".format(loop_number, total_num_iterations))
                print("MSS smaller than MSL. Skipping Iteration")
                loop_number = loop_number + 1
            else:
                print("Iteration {}/{}".format(loop_number, total_num_iterations))

                main_storage = "E:ML_Models_EXFOR/DT_{}/".format(DATASET)
                if not os.path.isdir(main_storage):
                    os.makedirs(main_storage)

                # ############################# TRAINING ##############################
                print("Max Depth = {}; Min Samples Split = {}; Min Samples Leaf = {}".format(i, mss, msl))
                start_training_time = time.time()
                dt_model = tree.DecisionTreeRegressor(max_depth=i, min_samples_split=mss, min_samples_leaf=msl)
                dt_model.fit(x_train, y_train)
                stop_training_time = time.time()

                y_hat_train = dt_model.predict(x_train)
                y_hat_val = dt_model.predict(x_val)
                y_hat_test = dt_model.predict(x_test)
                
                train_error_metrics = model_utils.regression_error_metrics(y_hat_train, y_train)
                val_error_metrics = model_utils.regression_error_metrics(y_hat_val, y_val)
                test_error_metrics = model_utils.regression_error_metrics(y_hat_test, y_test)

                ################# MODEL AND SCALER SAVING ###########################
                RUN_NAME = 'DT{}_MSS{}_MSL{}_{}_{}_{}{}'.format(int(dt_model.get_depth()), mss, msl, NORMALIZER_TYPE, MT_STRATEGY, DATASET, VERSION) 

                if RUN_NAME in os.listdir("E:ML_Models_EXFOR/DT_{}/".format(DATASET)):
                    print("Duplicate Training. Skipping")
                    loop_number = loop_number + 1
                    continue
                
                model_saving_directory = os.path.join("E:/ML_Models_EXFOR/DT_{}/".format(DATASET), RUN_NAME + "/")
                os.makedirs(model_saving_directory)
                model_saving_path = os.path.join(model_saving_directory, RUN_NAME + ".joblib")
                scaler_saving_path = os.path.join(model_saving_directory, 'scaler.pkl')
                dump(dt_model, model_saving_path)  # dump it on wandb.run.dir
                dump(scaler, open(scaler_saving_path, 'wb'))

                #################### GATHERING RESULTS ##############################
                model_error_metrics = pd.DataFrame(columns=[
                    "id", "max_depth", "mss", "msl", 'mt_strategy', 'normalizer', 
                    "train_mae", "train_mse", "train_evs", "train_mae_m", "train_r2",
                    "val_mae", "val_mse", "val_evs",  "val_mae_m", "val_r2", 
                    "test_mae", "test_mse", "test_evs",  "test_mae_m", "test_r2",
                    "model_path", "training_time", "scaler_path"])
                to_append = model_utils.create_train_test_error_df(loop_number, train_error_metrics, test_error_metrics, val_error_metrics=val_error_metrics)
                to_append['mt_strategy'] = MT_STRATEGY
                to_append["normalizer"] = NORMALIZER_TYPE
                to_append["max_depth"] = dt_model.get_depth()
                to_append["mss"] = mss
                to_append["msl"] = msl
                to_append["training_time"] = stop_training_time - start_training_time
                to_append["model_path"] = os.path.abspath(model_saving_path)
                to_append["scaler_path"] = os.path.abspath(scaler_saving_path)
                model_error_metrics = model_error_metrics.append(to_append)

                csv_path = os.path.join("", "dt_results{}.csv".format(DATASET))
                if os.path.exists(csv_path):
                    previous = pd.read_csv(csv_path)
                    new = previous.append(model_error_metrics)
                    new.to_csv(csv_path, index=False)
                else:
                    model_error_metrics.to_csv(csv_path, index=False)

                loop_number = loop_number + 1
