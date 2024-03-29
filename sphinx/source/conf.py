"""Configuration file for Sphinx."""
import recommonmark  # noqa
from recommonmark.transform import AutoStructify
import os
import sys

source_suffix = ['.rst', '.md']
htmlhelp_basename = 'Recommonmarkdoc'

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath('../../nucml'))


# -- Project information -----------------------------------------------------

project = 'NucML'
copyright = '2021, Pedro Jr. Vicente-Valdez'
author = 'Pedro Jr. Vicente-Valdez'

# The full version, including alpha/beta/rc tags
release = '1.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'recommonmark',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 5,
}


# At the bottom of conf.py
def setup(app):
    """Sphinx setup."""
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,  # noqa
        'auto_toc_tree_section': 'Contents',
    }, True)
    app.add_transform(AutoStructify)
