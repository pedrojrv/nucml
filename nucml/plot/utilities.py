"""Plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import imageio
import io
from PIL import Image


def create_gif(directory, extension, name, duration=2):
    """Gather all images in a directory and creates a GIF file.

    Args:
        directory (str): Path-like string where all images are saved.
        extension (str): Extension of images to be gathered (i.e. jpg).
        name (str): Name to save the GIF file as.
        duration (int, optional): Duration of the GIF file. Defaults to 2.

    Returns:
        None
    """
    images = []
    for file_name in os.listdir(directory):
        if file_name.endswith(extension):
            file_path = os.path.join(directory, file_name)
            images.append(imageio.imread(file_path))

    imageio.mimsave(os.path.join(directory, name), images, duration=duration)
    return None


def kdeplot(x, labels=[''], xlabel='', ylabel='', title='', figsize=(15, 10), save=False, path=''):
    """Create a KDE plot for a given array.

    Args:
        x (np.array or list): Numpy array or list of numpy arrays. If a list is given, all provided arrays will be
            ploted in the same figure.
        labels (list, optional): If x is a list, this argument represents the labels that will be plotted.
             Defaults to [''].
        xlabel (str, optional): Label for the X-axis. Defaults to ''.
        ylabel (str, optional): Label for the Y-axis. Defaults to ''.
        title (str, optional): Title of the plotted figure. Defaults to ''.
        figsize (tuple, optional): Figure size. Defaults to (15,10).
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        path (str, optional): Path-like string where the figure will be saved in case save=True. Defaults to ''.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    if isinstance(x, list):
        z = 0
        for i in x:
            g = sns.kdeplot(i, shade=True, label=labels[z])
            z = z + 1
    else:
        g = sns.kdeplot(x, shade=True)
    if len(xlabel) > 0:
        g.set(xlabel=xlabel)
    if len(ylabel) > 0:
        g.set(ylabel=ylabel)
    if len(title) > 0:
        plt.title(title)
    if save:
        plt.savefig(path, bbox_inches='tight')


def cat_plot(features, df, groupby, top=10, reverse=False, save=False, path=''):
    """Plot a categorical bar plot.

    Args:
        features (list): List of feature names to plot. Every single feature will be plotted individually.
        df (pd.DataFrame): Dataframe containing the features and data to be plotted.
        groupby (str): Feature by which the dataframe will be grouped for each bar plot.
        top (int, optional): In cases where there are a lot of categories, it is a good idea to limit the number of
            bars in the plot. This argument specifies the maximum number of categories to render. Defaults to 10.
        reverse (bool, optional): If True, the lowest frequent items are plotted rather than the most popular.
            Defaults to False.
        save (bool, optional): If True, the figure will be saved. Defaults to False.
        path (str, optional): Path-like string where the figure will be saved in case save=True. Defaults to ''.

    Returns:
        None
    """
    cat_cols_plot = features
    for i in cat_cols_plot:
        for_plotting = df[[i, groupby]].drop_duplicates()
        if reverse:
            sns.catplot(
                x=i, kind="count", data=for_plotting, order=for_plotting[i].value_counts().iloc[-top:].index,
                palette="GnBu_r", height=15, aspect=2)
        else:
            sns.catplot(
                x=i, kind="count", data=for_plotting, order=for_plotting[i].value_counts().iloc[:top].index,
                palette="GnBu_r", height=15, aspect=2)
        plt.title("{} Distribution".format(i))
        if save:
            if reverse:
                plt.savefig(path + '_{}_reverse.svg'.format(i), bbox_inches='tight')
            else:
                plt.savefig(path + '_{}.svg'.format(i), bbox_inches='tight')
    return None


def plotly_converter(plotly_object, convert_to="pil"):
    """Convert a Plotly figure into a PIL image object or a numpy array.

    Args:
        plotly_object (object): Plotly object to convert.
        convert_to (str, optional): Type of file to return. Type can be "pil" or "array". Defaults to "pil".

    Returns:
        object: Pil image or numpy array
    """
    fig_bytes = plotly_object.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    if convert_to == "pil":
        return img
    elif convert_to == "array":
        return np.asarray(img)
