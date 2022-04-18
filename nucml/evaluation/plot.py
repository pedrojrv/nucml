"""Plotting utilities for evaluation (i.e. ENDF, JENDL) datasets."""
import matplotlib.pyplot as plt
import seaborn as sns
import os


import nucml.datasets as nuc_data

sns.set(font_scale=2)
sns.set_style('white')


z_order_dict = {"endf": 1, "new_data": 2, "exfor": 3, "tendl": 4, "jendl": 5, "jeff": 6}


def _load_all_evaluations(isotope, MT, mode, mev_to_ev, mb_to_b):
    endf = nuc_data.load_evaluation(
        isotope, MT, mode=mode, library="endfb8.0", mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=False)
    tendl = nuc_data.load_evaluation(
        isotope, MT, mode=mode, library="tendl.2019", mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=False)
    jendl = nuc_data.load_evaluation(
        isotope, MT, mode=mode, library="jendl4.0", mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=False)
    jeff = nuc_data.load_evaluation(
        isotope, MT, mode=mode, library="jeff3.3", mev_to_ev=mev_to_ev, mb_to_b=mb_to_b, log=False)
    return endf, tendl, jendl, jeff


def plot(isotope, MT, exfor=None, exclude=[], new_data=None, new_data_label="", save=False,
         z_order_dict=z_order_dict, mode="neutrons", mev_to_ev=True, mb_to_b=True):
    """Plot all evaluations for a specific reaction and a given isotope.

    It is possible to exclude some evaluations if needed. New data can also be added. The avaliable evaluations
    include endfb8.0, jendl4.0, jeff3.3, and tendl.2019

    Args:
        isotope (str): Isotope to query (i.e. Cl35, u235).
        MT ([type]): Reaction channel to extract as an integer (ENDF MT codes).
        exfor (DataFrame, optional): EXFOR DataFrame to plot along the evaluations. Defaults to None.
        exclude (list, optional): List of evaluations to exclude from plot. Defaults to [].
        new_data (DataFrame, optional): DataFrame containing new data or extra data to plot. Defaults to None.
        new_data_label (str, optional): If new_data is being provided, a label for the legend needs to be provided.
            Defaults to "".
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        save_dir (str, optional): Directory where the figure will be saved. Defaults to "".
        z_order_dict (dict, optional): Dictionary containing the order on which to plot the evaluations. For example,
            z_order_dict = {"endf":1, "new_data":2, "exfor":3, "tendl":4, "jendl":5, "jeff":6} will plot the endf
            first followed by the new data if avaliable and so on. Defaults to {"endf":1, "new_data":2, "exfor":3,
            "tendl":4, "jendl":5, "jeff":6}.
        mode (str, optional): Which type of projectile is to be extracted. The only options are
            "neutrons" and "protons". Defaults to "neutrons".
        mev_to_ev (bool, optional): Converts energy from MeV to eV. Defaults to True.
        mb_to_b (bool, optional): Converts cross section from millibarns to barns. Defaults to True.

    Returns:
        None
    """
    endf, tendl, jendl, jeff = _load_all_evaluations(isotope, MT, mode, mev_to_ev, mb_to_b)

    plt.figure(figsize=(14, 9))
    all_not_in_exclude = "all" not in exclude
    all_dataframes = {'exfor': exfor, 'new_data': new_data, 'endf': endf, 'tendl': tendl, 'jendl': jendl, 'jeff': jeff}
    plot_labels = {
        'exfor': 'EXFOR', 'new_data': new_data_label, 'endf': 'ENDF/B-VIII', 'tendl': 'TENDL 2019',
        'jendl': 'JENDL 4.0', 'jeff': 'JEFF 3.3'}

    for key, dataframe in all_dataframes.items():
        if (dataframe is None) or (all_not_in_exclude and key not in ['exfor', 'new_data']):
            continue
        plot_kwargs = {"label": plot_labels[key], "zorder": z_order_dict[key]}
        if key == "new_data":
            plt.scatter(dataframe.Energy, dataframe.Data, **plot_kwargs)
        plt.loglog(dataframe.Energy, dataframe.Data, **plot_kwargs)

    plt.xlabel('Energy (eV)')
    plt.ylabel('Cross Section (b)')
    plt.legend()
    if save:
        save_name = "{}_{}_Evaluated_XS{}.png".format(isotope, MT, "_w_EXFOR" if exfor else "")
        plt.savefig(os.path.join(save, save_name), bbox_inches='tight', dpi=600)
