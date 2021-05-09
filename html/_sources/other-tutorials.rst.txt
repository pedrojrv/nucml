.. _navigating-the-nde-label:

NucML contains many utilities to deal with everyday tasks. Here we list some that might be of interest to you.


Incorrect EXFOR Experimental Campaigns
======================================

Before fitting your models, it is important to generally inspect your data. In this particular case, visualizing and assessing
6 million data points are more or less impossible. This notebook contains a couple of reactions that have been found to have a negative 
impact on model performance. It is recommended that these experimental campaigns are eliminated from your dataset. 

.. toctree::
    :maxdepth: 1

    notebooks/4_Erroneous_Cross_Sections

ML-based EXFOR Outlier Detection
================================

To perform outlier detection in the EXFOR dataset, it is important that you do not rely on typical statistical techniques computed on the entire dataset (traditional ML). 
These will, in many cases, tag resonance data points as outliers. ML-based outlier detection can be used.

.. toctree::
    :maxdepth: 1

    notebooks/0_Evaluating_Dataset_Quality



Hybrid ML-ENDF ACE File Creation
================================

In the processing and validation stage of the nuclear data evaluation pipeline, a hybrid set of cross sections are generated previous to compilation into ACE files.
If you want more information on how these corrections are performed and unitarity is enforced, feel free to browse the following information.


.. toctree::
    :maxdepth: 1

    notebooks/2_Hybrid_ML_ENDF_XS_Generation


ENSDF NLD Linear Interpolation
==============================

In an attempt to include nuclear level information for cross section inference, simple linear interpolation techniques and utilities are incorporated into NucML.
In the following notebook, a set of nuclear level densities for all available isotopes up to 20 MeV is generated. 


.. toctree::
    :maxdepth: 1

    notebooks/0_ENSDF_Linear_Interpolation_NLD

