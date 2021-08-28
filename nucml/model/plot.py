"""Plotting utilities for model training and evaluation."""
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.ticker import MaxNLocator
import pandas as pd


def xgb_training(dictionary, save=False, show=True, title="", save_dir=''):
    """Plot the Loss vs Number of Estimators resulting from an XGBoost training process.

    Args:
        dictionary (dict): dictionary generated from the XGBoost training process.
        save (bool, optional): If True, the image is saved. Defaults to False.
        show (bool, optional): If True, the image is rendered. Defaults to True.
        title (str, optional): Title to render above the plot. Defaults to "".
        path (str, optional): Path-like string where the figure will be saved. Defaults to "".

    Returns:
        None
    """
    plt.figure(figsize=(14, 8))
    plt.plot(dictionary["eval"]["rmse"], label="Evaluation")
    plt.plot(dictionary["train"]["rmse"], label="Training")
    plt.legend()
    plt.xlabel("Number of Estimators")
    plt.ylabel("RMSE")
    plt.title(title)
    if save:
        plt.savefig(os.path.join(save_dir, "xgb_training.png"), bbox_inches="tight", dpi=600)
    if not show:
        plt.close()
    return None


def train_test(df, x_feature, train_metric, test_metric, save=False, save_dir='', render_browser=False, paper=False):
    """Plot both the train and test loss as a function of a second feature (i.e. training steps).

    Args:
        df (pd.DataFrame): Pandas DataFrame containing the train and test metric information.
        x_feature (str): Feature containing the x-axis information. Can contain information such as the training steps
            or parameters such as k-number, number of estimators, etc.
        train_metric (str): Name of the feature containing the train performance metric.
        test_metric (str): Name of the feature containing the test performance metric.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the resulting figure will be saved. Defaults to ''.
        render_browser (bool, optional): If True, the figure will be rendered in a new browser tab. Defaults to False.
        paper (bool, optional): If True, the figure will be resized to fit into two-column documents. Defaults to False.

    Returns:
        object: Plotly figure object.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df[x_feature], y=df[train_metric], name=train_metric), secondary_y=False)
    fig.add_trace(go.Scatter(x=df[x_feature], y=df[test_metric], name=test_metric), secondary_y=True)

    fig.update_xaxes(title_text=x_feature)
    fig.update_yaxes(title_text="<b>{}</b>".format(train_metric), secondary_y=False)
    fig.update_yaxes(title_text="<b>{}</b>".format(test_metric), secondary_y=True)

    fig.update_layout(template="simple_white")

    if paper:
        fig.update_layout(height=600, width=700)
        fig.update_layout(legend=dict(x=0.7, y=1))
    if render_browser:
        fig.show(renderer="browser")
    else:
        fig.show()
    if save:
        fig.write_image(os.path.join(save_dir, "model_performance_metric.svg"))
        fig.write_html(os.path.join(save_dir, "model_performance_metric.html"))
    return fig


def knn_training(results_df, x_feature="id", train_metric="train_mae", val_metric="val_mae", test_metric="test_mae",
                 save=False, save_path='', show=True):
    """Plot both the train, val, and test loss as a function of a given parameter (i.e. number of neighbors).

    Args:
        results_df (pd.DataFrame): Pandas DataFrame containing the train, val, and test metric information.
        x_feature (str): Feature containing the x-axis information. Can contain information such as the training steps
            or parameters such as k-number.
        train_metric (str): Name of the feature containing the train performance metric.
        val_metric (str): Name of the feature containing the validation performance metric.
        test_metric (str): Name of the feature containing the test performance metric.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the resulting figure will be saved. Defaults to ''.
        show (bool, optional): If True, the image is rendered. Defaults to True.

    Returns:
        object: Plotly figure object.
    """
    fig, ax1 = plt.subplots(figsize=(18, 8))

    color = 'tab:orange'
    lns1 = ax1.plot(results_df[x_feature], results_df[train_metric], color=color, marker="o", label="Train MAE")
    ax1.set_xlabel('Number of Neighbors (K)')
    ax1.set_ylabel('Train Metric', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    # ax1.legend()

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    lns2 = ax2.plot(results_df[x_feature], results_df[val_metric], color=color, marker="o", label="Val MAE")
    lns3 = ax2.plot(
        results_df[x_feature], results_df[test_metric], color=color, marker="x", markersize=10, label="Test MAE")
    ax2.set_ylabel('Test and Validation Metric', color=color)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.legend()

    # added these three lines
    lns = lns1+lns2+lns3
    labs = [label.get_label() for label in lns]
    ax1.legend(lns, labs, loc=0)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if save:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    if not show:
        plt.close()
    return None


def dt_training(results_df, param_1="max_depth", param_2="msl", train_metric="train_mae", test_metric="test_mae",
                save=False, save_dir='', show=True):
    """Plot both the train and test loss as a function of a second feature (i.e. training steps).

    Args:
        results_df (pd.DataFrame): Pandas DataFrame containing the train and test metric information.
        param_1 (str): Feature containing the information for a given parameter to plot.
        param_2 (str): Feature containing the information for a second parameter to plot.
        train_metric (str): Name of the feature containing the train performance metric.
        test_metric (str): Name of the feature containing the test performance metric.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        save_dir (str, optional): Path-like string where the resulting figure will be saved. Defaults to ''.
        show (bool, optional): If True, the image is rendered. Defaults to True.

    Returns:
        object: Plotly figure object.
    """
    fig, (ax1, ax3) = plt.subplots(2, figsize=(14, 18))

    color = 'tab:orange'
    ax1.set_xlabel('Train MAE (b)')
    ax1.set_ylabel('Max Depth', color=color)
    ax1.scatter(results_df[train_metric], results_df[param_1], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Minimum Samples per Leaf (MSL)', color=color)  # we already handled the x-label with ax1
    ax2.scatter(results_df[train_metric], results_df[param_2], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:orange'
    ax3.set_xlabel('Test MAE (b)')
    ax3.set_ylabel('Max Depth', color=color)
    ax3.scatter(results_df[test_metric], results_df[param_1], color=color, marker="o")
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4.set_ylabel('Minimum Samples per Leaf (MSL)', color=color)  # we already handled the x-label with ax1
    ax4.scatter(results_df[test_metric], results_df[param_2], color=color, marker="o")
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(save_dir, "dt_training.png"), bbox_inches="tight", dpi=600)
    if not show:
        plt.close()
    return None


# def plot_dt_3d_hyperparams(df):
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     fig.set_size_inches(20, 10.5, forward=True)
#     surf = ax.plot_trisurf(df.max_depth, df.mss, df.train_mae, linewidth=0.2, antialiased=True, cmap=cm.viridis)
#     ax.view_init(20, -30)
#     ax.set_ylabel("MSS", labelpad=15)
#     ax.set_xlabel("Max Depth", labelpad=10)
#     ax.set_zlabel("Train MAE", labelpad=15)

#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     plt.savefig("figures/dt_3d_hyperparams.png", bbox_inches='tight', dpi=300)


def xgb_training_w_path(path_to_csv, save=False, saving_path="xgb_training.png"):
    """Plot XGB retraining given the path to the results."""
    training_progress = pd.read_csv(path_to_csv)
    plt.figure(figsize=(18, 8))
    plt.plot(training_progress.mae_train, label="Train MAE", marker="x", markersize="20")
    plt.plot(training_progress.mae_test, label="Validation MAE", marker="o", markersize="5")
    plt.xlabel('Number of Trees (Rounds)')
    plt.ylabel('MAE')
    plt.legend()
    if save:
        plt.savefig(saving_path, bbox_inches='tight', dpi=300)
