.. _getting-started-label:

Installation Guide
==================

NucML is a python toolbox and can be installed using pip along with all the needed dependencies.


1. Install NucML and Dependencies
---------------------------------

NucML uses `Numpy Arrays <https://numpy.org/doc/stable/contents.html>`_ and `Pandas DataFrame's <https://pandas.pydata.org>`_
as the main data objects. Each step of the evaluation pipeline requires different utilities including parsing utilities, data processing tools,
and Machine Learning packages including TensorFlow, XGBoost, and Scikit-learn. 

..	note::

	We recommend you install into a `conda environment <https://docs.conda.io/en/latest/>`_.

..	note::

	Since TensorFlow only supports Python versions up to 3.8, NucML must be install in an enviornment with Python Version =< 3.8.

..  code-block:: bash

	# create and activate conda environment
	conda create -n ml_nuclear_env python=3.8
	conda activate ml_nuclear_env

	# Make sure the python version is not higher than 3.8
	python -V

	# install nucml
	pip install nucml

Install Dependencies
^^^^^^^^^^^^^^^^^^^^

NucML needs the following high dependencies and their dependencies intalled in order to work:

* natsort
* wandb
* pandas
* plotly
* matplotlib
* scikit-learn
* seaborn
* imageio
* xgboost
* tensorflow

However, you can choose to install the package without these packages.

..  code-block:: bash

	# install nucml
	pip install nucml --no-deps

This is desirable when you first want to build, for example, TensorFlow with GPU support. XGBoost is automatically built with GPU support with pip. 


2. Configure NucML and Generate the EXFOR Datasets
--------------------------------------------------

Before exploring any functionalities all EXFOR datasets need to be generated. NucML is built to be versatile in terms of working directories, 
however, a suggested structure is provided here. The working directory is where all metadata files from all various Datasources including 
ACE, ENDF, EXFOR, and ENSDF files are located. The ML_Nuclear_Data repository contains some pre-generated datasets and everything needed to 
produce the heavier EXFOR-based datasets.

First, clone the repository by either downloading it directly from GitHub or using the command line.

..  code-block:: bash

    # navigate to the directory of your choice - change it to your own
    cd /Users/pedrovicentevaldez/Desktop/
    
	# clone the ml nuclear data repository
	git clone https://github.com/pedrojrv/ML_Nuclear_Data.git

In the rest of the setup we assume you cloned the repository in your Desktop. 

..	note::

	If downloaded directly from GitHub, make sure to unzip the file. 

..	note::

	Feel free to rename the ML_Nuclear_Data directory. If you do so, just change the rest of the instructions to fit your directory name.

NucML makes use of a configuration file to get the paths to the generated data based on the working directory structure. Since we are using
the ML_Nuclear_Data repository the configuration is easy. 


..  code-block:: bash

    # navigate to the cloned repo
    cd /Users/pedrovicentevaldez/Desktop/ML_Nuclear_Data
    
	# configure nucml paths
	python -c "import nucml.configure as config; config.configure('.', 'ACE/')"


Now we are ready to generate the EXFOR datasets. A python script is provided in the ML_Nuclear_Data repo to help you get started quickly.

..	note::

	This will generate EXFOR datasets for all avaliable projectiles and will therefore take a couple of minutes.

..	note::

    NOTE: This will create a database based on the latest version of the EXFOR files avaliable in the ML_Nuclear_Data repository. 

..  code-block:: bash

    # navigate to the cloned repo - if already there ignore
    cd /Users/pedrovicentevaldez/Desktop/ML_Nuclear_Data
    
	# generate the exfor datasets
	python generate_exfor.py


Running this script will create the CSV_Files directory within the EXFOR folder. Additionally, a tmp directory will also created containing 
helper files used in the creation of the final datasets. Feel free to delete this directory after the process has finish. 

3. Other Dependencies
---------------------

SERPENT2 and MATLAB must be installed if you want to validate your models using criticality benchmarks. These are not necessary for 
other tasks such as loading the data and training ML models.  