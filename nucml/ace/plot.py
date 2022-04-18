"""Plotting utilities for ACE Data."""
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator


def dt_dual_keff_plot(dt_df, train_mae, val_mae, hyperparameter, keff_metric, save=False, saving_dir=""):
    """Create a two figure plot comparing k-eff, MAE (train/val), and model hyperparameters for DT models.

    Contains both the performance metric (i.e. MAE) and the criticality calculation (i.e K-eff) vs two model
    hyperparameters. These decision tree hyperparameteres are usually the Max Depth and the Minimum Numbers for Split.

    Args:
        dt_df (DataFrame): DataFrame containing all trained models performance in criticality benchmarks.
        train_mae (str): Feature name for the performance metric to plot.
        val_mae (str): Feature name for the validation performance metric..
        hyperparameter (str): Feature name for the hyperparameter to plot.
        keff_metric (str): Feature name for the multiplication factor to plot.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        saving_dir (str, optional): Directory where the figure will be saved. Defaults to "".

    Returns:
        None
    """
    fig, (ax1, ax3) = plt.subplots(2, figsize=(14, 18))

    if keff_metric == "Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else:
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Train MAE (b)')
    ax1.set_ylabel('Max Depth', color=color)
    ax1.scatter(dt_df[train_mae], dt_df[hyperparameter], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.scatter(dt_df[train_mae], dt_df[keff_metric], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)

    color = 'tab:orange'
    ax3.set_xlabel('Validation MAE (b)')
    ax3.set_ylabel('Max Depth', color=color)
    ax3.scatter(dt_df[val_mae], dt_df[hyperparameter], color=color, marker="o")
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax4.scatter(dt_df[val_mae], dt_df[keff_metric], color=color, marker="o")
    ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "dt_mae_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()
    return None


def dt_keff_plot(dt_df, mae, hyperparameter, keff_metric, save=False):
    """Create a figure comparing k-eff and MAE model and a single hyperparameter for DT models.

    The decision tree hyperparameteres are usually the Max Depth and the Minimum Numbers for Split.

    Args:
        dt_df (DataFrame): DataFrame containing all trained models performance in criticality benchmarks.
        mae (str): Feature name for the performance metric to plot (i.e. Train MAE, Val MAE).
        hyperparameter (str): Feature name for the main hyperparameter to plot.
        keff_metric (str): Feature name for the multiplication factor metric.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        saving_dir (str, optional): Directory where the figure will be saved. Defaults to "".

    Returns:
        None
    """
    fig, ax1 = plt.subplots(figsize=(18, 10))
    label = "Multiplication Factor (K-eff)"
    if keff_metric == "Deviation_Ana":
        label = label + " Error"

    color = 'tab:orange'
    ax1.set_xlabel('Validation MAE (b)')
    ax1.set_ylabel('Max Depth', color=color)
    ax1.scatter(dt_df[mae], dt_df[hyperparameter], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.scatter(dt_df[mae], dt_df[keff_metric], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # dt_single_mae_keff.png
    plt.savefig(save, bbox_inches="tight", dpi=600)
    plt.show()


def knn_dual_plot(knn_df, hyperparameter, train_mae, val_mae, keff_metric, save=False, saving_dir=""):
    """Create a two figure plot comparing k-eff and MAE model performance for KNN models.

    Creates a two figure plot containing both performance metrics (i.e. Train MAE and Val MAE), the
    criticality calculation (i.e K-eff) and a model hyperparameter.

    Args:
        knn_df (DataFrame): DataFrame containing all trained models performance in criticality benchmarks.
        hyperparameter (str): Feature name for the hyperparameter to plot.
        train_mae (str): Feature name for the performance metric to plot.
        val_mae (str): Feature name for the validation performance metric.
        keff_metric (str): Feature name for the multiplication factor to plot.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        saving_dir (str, optional): Directory where the figure will be saved. Defaults to "".

    Returns:
        None
    """
    fig, (ax1, ax3) = plt.subplots(2, figsize=(14, 18))
    # fig, (ax1, ax3) = plt.subplots(1,2, figsize=(30,13))
    if keff_metric == "Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else:
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Train MAE (b)', color=color)
    ax1.plot(knn_df[hyperparameter], knn_df[train_mae], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Validation MAE (b)', color=color)  # we already handled the x-label with ax1
    ax2.plot(knn_df[hyperparameter], knn_df[val_mae], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    color = 'tab:orange'
    ax3.set_xlabel('Number of Neighbors (k)')
    ax3.set_ylabel('Validation MAE (b)', color=color)
    ax3.plot(knn_df[hyperparameter], knn_df[val_mae], color=color, marker="o")
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax4.plot(knn_df[hyperparameter], knn_df[keff_metric], color=color, marker="o")
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "knn_train_val_mae_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()
    return None


def knn_keff_plot(knn_df, hyperparameter, mae, keff_metric, save=False, saving_dir=""):
    """Compare a hyperparameter and a performance metric for KNN models.

    Args:
        knn_df (DataFrame): DataFrame containing all trained models performance in criticality benchmarks.
        hyperparameter (str): Feature name for the main hyperparameter to plot.
        mae (str): Feature name for the performance metric to plot (i.e. Train MAE, Val MAE).
        keff_metric (str): Feature name for the multiplication factor metric.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        saving_dir (str, optional): Directory where the figure will be saved. Defaults to "".

    Returns:
        None
    """
    fig, ax1 = plt.subplots(figsize=(18, 10))

    if keff_metric == "Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else:
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Validation MAE (b)', color=color)
    ax1.plot(knn_df[hyperparameter], knn_df[mae], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.plot(knn_df[hyperparameter], knn_df[keff_metric], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "knn_val_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()
    return None
