import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator



def dt_dual_keff_plot(dt_df, train_mae, main_metric2, metric1, metric2, save=False, saving_dir=""):
    fig, (ax1, ax3) = plt.subplots(2, figsize=(14,18))
    # fig, (ax1, ax3) = plt.subplots(1,2, figsize=(30,13))

    if metric2 =="Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else: 
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Train MAE (b)')
    ax1.set_ylabel('Max Depth', color=color)
    ax1.scatter(dt_df[train_mae], dt_df[metric1], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.scatter(dt_df[train_mae], dt_df[metric2], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)


    color = 'tab:orange'
    ax3.set_xlabel('Validation MAE (b)')
    ax3.set_ylabel('Max Depth', color=color)
    ax3.scatter(dt_df[main_metric2], dt_df[metric1], color=color, marker="o")
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax4.scatter(dt_df[main_metric2], dt_df[metric2], color=color, marker="o")
    ax4.tick_params(axis='y', labelcolor=color)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "dt_mae_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()

def dt_keff_plot(dt_df, val_mae, hyperparameter, keff_metric, save=False, saving_dir=""):
    fig, ax1 = plt.subplots(figsize=(18,10))

    if keff_metric =="Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else: 
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Validation MAE (b)')
    ax1.set_ylabel('Max Depth', color=color)
    ax1.scatter(dt_df[val_mae], dt_df[hyperparameter], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.scatter(dt_df[val_mae], dt_df[keff_metric], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "dt_single_mae_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()

def knn_dual_plot(dt_df, main_metric, metric0, metric1, keff_metric, save=False, saving_dir=""):
    fig, (ax1, ax3) = plt.subplots(2, figsize=(14,18))
    # fig, (ax1, ax3) = plt.subplots(1,2, figsize=(30,13))
    if keff_metric =="Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else: 
        label = "Multiplication Factor (K-eff)"

    color = 'tab:orange'
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Train MAE (b)', color=color)
    ax1.plot(dt_df[main_metric], dt_df[metric0], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Validation MAE (b)', color=color)  # we already handled the x-label with ax1
    ax2.plot(dt_df[main_metric], dt_df[metric1], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


    color = 'tab:orange'
    ax3.set_xlabel('Number of Neighbors (k)')
    ax3.set_ylabel('Validation MAE (b)', color=color)
    ax3.plot(dt_df[main_metric], dt_df[metric1], color=color, marker="o")
    ax3.tick_params(axis='y', labelcolor=color)

    ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax4.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax4.plot(dt_df[main_metric], dt_df[keff_metric], color=color, marker="o")
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "knn_train_val_mae_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()

def knn_keff_plot(knn_df, k, val_mae, keff_metric, save=False, saving_dir=""):
    fig, ax1 = plt.subplots(figsize=(18,10))

    if keff_metric =="Deviation_Ana":
        label = "Multiplication Factor (K-eff) Error"
    else: 
        label = "Multiplication Factor (K-eff)"


    color = 'tab:orange'
    ax1.set_xlabel('Number of Neighbors (k)')
    ax1.set_ylabel('Validation MAE (b)', color=color)
    ax1.plot(knn_df[k], knn_df[val_mae], color=color, marker="o")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(label, color=color)  # we already handled the x-label with ax1
    ax2.plot(knn_df[k], knn_df[keff_metric], color=color, marker="o")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save:
        plt.savefig(os.path.join(saving_dir, "knn_val_keff.png"), bbox_inches="tight", dpi=600)
    plt.show()




