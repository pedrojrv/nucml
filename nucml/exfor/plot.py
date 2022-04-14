"""Plotting utilities for the EXFOR dataset."""
import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

import nucml.exfor.data_utilities as exfor_utils
import nucml.datasets as nuc_data
import nucml.ace.data_utilities as ace_utils

sns.set(style="white", font_scale=2.5)


def ml_results(results_dict, order_dict={}, save_dir='', save=False, render_browser=False, show=False, paper=False,
               log=True, plot_type="plotly"):
    """Plot the machine learning predictions from the dictionary generated by the.

    Args:
        results_dict (dict): Generated dictionary from the .
        order_dict (dict, optional): Order of plots. Defaults to {}.
        save_dir (str, optional): Path-like string where the figure will be saved. Defaults to ''.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        render_browser (bool, optional): If True, the plotly plot will be render in a new tab. Defaults to False.
        show (bool, optional): If True, the plot will be render. Defaults to False.
        paper (bool, optional): If True, the plot will be resized to 600 by 700 pixels. Defaults to False.

    Returns:
        object: Plotly graph object.
    """
    if plot_type == "plotly":
        fig = go.Figure()

        if len(order_dict) == 0:
            order_dict = {"1": "endf", "4": "exfor_ml_original", "3": "exfor_ml", "2": "exfor_new"}

        exfor_original_trace = go.Scattergl(
            x=results_dict["exfor_ml_original"]['df']["Energy"],
            y=results_dict["exfor_ml_original"]['df']["Data"],
            mode='markers', name='EXFOR')

        exfor_tag = "exfor_ml_expanded" if "exfor_ml_expanded" in results_dict.keys() else "exfor_ml_original"
        exfor_ml_trace = go.Scattergl(
            x=results_dict[exfor_tag]['df']["Energy"],
            y=results_dict[exfor_tag]['predictions'].flatten(),
            mode='lines', name='ML')

        if "endf" in results_dict.keys():
            endf_trace = go.Scattergl(
                x=results_dict["endf"].Energy,
                y=results_dict["endf"].Data,
                mode='lines', name='ENDF')

        if "exfor_ml_new" in results_dict.keys():
            exfor_ml_new_trace = go.Scattergl(x=results_dict["exfor_ml_new"]["df"].Energy,
                                              y=results_dict["exfor_ml_new"]["df"].Data,
                                              mode='markers', name='EXFOR New')

        fig.update_layout(
            xaxis_title="Energy (eV)",
            yaxis_title="Cross Section (b)",
        )
        for i in np.arange(1, len(order_dict) + 1):
            for plot_order, name in order_dict.items():
                if plot_order == str(i):
                    if name == "endf":
                        if "endf" in results_dict.keys():
                            fig.add_trace(endf_trace)
                    elif name == "exfor_ml_original":
                        fig.add_trace(exfor_original_trace)
                    elif name == "exfor_ml":
                        fig.add_trace(exfor_ml_trace)
                    elif name == "exfor_new":
                        if "exfor_ml_new" in results_dict.keys():
                            fig.add_trace(exfor_ml_new_trace)
        fig.update_layout(template="simple_white")
        if paper:
            fig.update_layout(height=600, width=700)
            fig.update_layout(legend=dict(x=0.8, y=1))
        if render_browser:
            fig.show(renderer="browser")
        elif show:
            fig.show()
        if save:
            fig.write_html(os.path.splitext(save_dir)[0] + '.html')
        return fig
    elif plot_type == "sns":
        plt.figure(figsize=(18, 8))
        if len(order_dict) == 0:
            order_dict = {"endf": 1, "exfor_ml_original": 4, "exfor_ml": 3, "exfor_new": 2}

        plt.scatter(10**results_dict["exfor_ml_original"]['df']["Energy"],
                    10**results_dict["exfor_ml_original"]['df']["Data"],
                    label='EXFOR', zorder=order_dict["exfor_ml_original"],
                    s=15, c="tab:green")

        exfor_tag = "exfor_ml_expanded" if "exfor_ml_expanded" in results_dict.keys() else "exfor_ml_original"
        plt.plot(
            10**results_dict[exfor_tag]['df']["Energy"],
            10**results_dict[exfor_tag]['predictions'].flatten(),
            label='ML', zorder=order_dict["exfor_ml"], c="tab:orange")

        if "endf" in results_dict.keys():
            plt.plot(
                10**results_dict["endf"].Energy, 10**results_dict["endf"].Data,
                label='ENDF', zorder=order_dict["endf"])
        if "exfor_ml_new" in results_dict.keys():
            plt.scatter(10**results_dict["exfor_ml_new"]['df'].Energy,
                        10**results_dict["exfor_ml_new"]['df'].Data,
                        label='EXFOR New', zorder=order_dict["exfor_new"],
                        s=15, c="tab:red")

        plt.xlabel('Energy (eV)')
        plt.ylabel('Cross Section (b)')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()

        if save:
            plt.savefig(save_dir, bbox_inches='tight', dpi=300)
        if not show:
            plt.close()

        return None


def plot_limits(data, endf, new_data, y_hat, y_hat2, y_hat3):
    """Set new plot limits based on plotted data.

    It is a known issue that matplotlib "looses" the limits when thousand of scatter points are plotted. In these cases
    the limits need to be
    set manually. These occurs often for highly researched isotopes like U-235. ML-predictions can
    sometimes be outsied the default plot limits.

    Note: Internal Function.

    Args:
        data (DataFrame): DataFrame containing the EXFOR or Predicted data.
        endf (DataFrame): DataFrame containing the ENDF data.
        new_data (DataFrame): DataFrame containing new external data if applicable.
        y_hat (np.array): Numpy array containing the ML-predictions for EXFOR energy points.
        y_hat2 (np.array): Numpy array containing the ML-predictions for ENDF energy points.
        y_hat3 (np.array): Numpy array containing the ML-predicitons for the new data points.

    Returns:
        None
    """
    endf_avaliable = True if endf.shape[0] != 0 else False
    new_data_avaliable = True if new_data.shape[0] != 0 else False
    if (new_data_avaliable and endf_avaliable):  # if both
        plt.legend()
        all_y = np.concatenate(
            (data["Data"].values, y_hat.flatten(), endf["Data"].values, new_data["Data"].values))
        minimum_y = all_y.min() - all_y.min() * 0.05
        maximum_y = all_y.max() + all_y.max() * 0.05
        plt.ylim(minimum_y, maximum_y)
    elif not new_data_avaliable and endf_avaliable:  # if ENDF only
        # plt.legend((endf_eval, true, pred), ('ENDF', 'EXFOR', "EXFOR Pred"), loc='upper left')
        plt.legend()
        all_y = np.concatenate((data["Data"].values, y_hat[0].flatten(), y_hat2[0].flatten(), endf["Data"].values))
        minimum_y = all_y.min() - all_y.min() * 0.05
        maximum_y = all_y.max() + all_y.max() * 0.05
        plt.ylim(minimum_y, maximum_y)
    elif new_data_avaliable and not endf_avaliable:  # if ADDITIONAL only
        # plt.legend((true, unseen, pred, pred_unseen),
        #            ('EXFOR', "New Measurments", "EXFOR Pred", "New Pred"), loc='upper left')
        plt.legend()
        all_y = np.concatenate((data["Data"].values, y_hat, y_hat2, new_data["Data"].values))
        minimum_y = all_y.min() - all_y.min() * 0.05
        maximum_y = all_y.max() + all_y.max() * 0.05
        plt.ylim(minimum_y, maximum_y)
    else:  # if no ENDF and Additional
        # plt.legend((true, pred), ('EXFOR', "EXFOR Pred"), loc='upper left')
        plt.legend()
        all_y = np.concatenate((data["Data"].values.flatten(), y_hat, y_hat2))
        minimum_y = all_y.min() - all_y.min() * 0.05
        maximum_y = all_y.max() + all_y.max() * 0.05
        plt.ylim(minimum_y, maximum_y)
    return None


def plot_limits_ref(exfor, endf, new_data):
    """Set new plot limits based on plotted data.

    It is a known issue that matplotlib "looses" the limits when thousand of scatter points are plotted. In these cases
    the limits need to be set manually. These occurs often for highly researched isotopes like U-235.

    Note: Internal Function.

    Args:
        exfor_sample (pd.DataFrame): Contains the EXFOR datapoints being plotted.
        endf (pd.DataFrame): Contains the ENDF datapoints being plotted.
        new_data (pd.DataFrame): Contains the New Datapoints being plotted if applicable.

    Returns:
        None
    """
    # Setting Figure Limits
    num_samples = new_data.shape[0]
    endf_samples = endf.shape[0]
    if (num_samples != 0 and endf_samples != 0):  # if both
        all_y = np.concatenate((exfor["Data"], endf["Data"], new_data["Data"]))
    elif num_samples == 0 and endf_samples != 0:  # if ENDF only
        all_y = np.concatenate((exfor["Data"], endf["Data"]))
    elif num_samples != 0 and endf_samples == 0:  # if ADDITIONAL only
        all_y = np.concatenate((exfor["Data"].values, new_data["Data"].values))
    else:  # if no ENDF and Additional
        all_y = exfor["Data"].values

    minimum_y = all_y[all_y > 0].min() - all_y[all_y > 0].min() * 0.05
    maximum_y = all_y.max() + all_y.max() * 0.05
    plt.ylim(minimum_y, maximum_y)


def make_chlorine_paper_figure(df, dt_model, dt_scaler, knn_model, knn_scaler, to_scale, save=False, saving_dir=""):
    """Personal function used to create the Chlorine figure used in a conference summary.

    Args:
        df (pd.DataFrame): Dataframe containing all the relevant chlorine datapoints.
        dt_model (object): Trained scikit-learn Decision Tree regressor model.
        dt_scaler (object): Trained scikit-learn scaler object used to scale data for the dt_model.
        knn_model (object): Trained scikit-learn KNN regressor model.
        knn_scaler (object): Trained scikit-learn scaler object used to scale data fro the knn_model.
        to_scale (list): Features subject to transformation/normalization by the applicable scalers.

    Returns:
        None
    """
    kwargs = {"nat_iso": "I", "one_hot": True, "to_scale": to_scale}
    chlorine_35_np_knn = exfor_utils.load_samples(df, 17, 35, "MT_103", scaler=knn_scaler, **kwargs)
    chlorine_35_np_dt = exfor_utils.load_samples(df, 17, 35, "MT_103", scaler=dt_scaler, **kwargs)

    new_cl_data_kwargs = {
        "Z": 17, "A": 35, "MT": "MT_103", "log": True, "to_scale": to_scale, "one_hot": True}
    new_cl_data_knn = exfor_utils.load_newdata(
        "../EXFOR/New_Data/Chlorine_Data/new_cl_np.csv", df, scaler=knn_scaler, **new_cl_data_kwargs)
    new_cl_data_dt = exfor_utils.load_newdata(
        "../EXFOR/New_Data/Chlorine_Data/new_cl_np.csv", df, scaler=dt_scaler, **new_cl_data_kwargs)

    endf_cl = nuc_data.load_evaluation("Cl035", 103, log=True)
    ace_cl = ace_utils.get_energies("17035", ev=True, log=True)

    _, (ax1, ax2) = plt.subplots(2, figsize=(30, 20))

    for np_data, model, plot_axis, label, new_data in zip(
            [chlorine_35_np_dt, chlorine_35_np_knn], [dt_model, knn_model], [ax1, ax2], ['DT', 'KNN'],
            [new_cl_data_dt, new_cl_data_knn]):
        chlorine_data_ext = exfor_utils.expanding_dataset_energy(np_data, 0, 0, False, 0, e_array=ace_cl)
        chlorine_data_ext = chlorine_data_ext[chlorine_data_ext.Energy > chlorine_35_np_dt.Energy.min()]

        plot_axis.plot(
            10**(chlorine_data_ext.Energy),
            10**(model.predict(chlorine_data_ext.drop(columns=["Data"]))),
            label=label, linestyle="dashed", c="firebrick", linewidth=3)
        plot_axis.scatter(10**(np_data.Energy), 10**(np_data.Data), alpha=0.5, c='#1f77b4', label="EXFOR")
        plot_axis.scatter(
            10**(new_data.Energy),
            10**(new_data.Data), alpha=1, c='#ff7f0e', s=250, marker="x", label="J.C.Batchelder (2019)")
        plot_axis.plot(10**(endf_cl.Energy), 10**(endf_cl.Data), alpha=0.5, c="orange", label="ENDF")
        plot_axis.legend(loc=3)

    for plot_axis in [ax1, ax2]:
        plot_axis.set(ylabel='Cross Section (b)')
        plot_axis.set(xlabel='Energy (eV)')
        plot_axis.set_xscale('log')
        plot_axis.set_yscale('log')
        plot_axis.set_xlim(10**-2, 10**7.5)

    if save:
        plt.savefig(os.path.join(saving_dir, "ML_Cl.png"), dpi=600, bbox_inches="tight")
    return None
