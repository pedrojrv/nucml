.. _installation-guide-label:


Installation Guide
==================


NucML uses `Numpy Arrays <https://numpy.org/doc/stable/contents.html>`_ and `Pandas DataFrame's <https://pandas.pydata.org>`_
as the main data objects. Each step of the evaluation pipeline requires different utilities including parsing utilities, data processing tools,
and Machine Learning packages including TensorFlow, XGBoost, and Scikit-learn.


1. Install NucML and Dependencies
---------------------------------

NucML is a python toolbox and can be installed using :code:`pip` along with all the needed dependencies. It is recommended that you create 
a :code:`conda` environment and then install :code:`nucml`. You can download :code:`conda` `here <https://docs.conda.io/en/latest/>`_.
Once downloaded you can write the following commands on your anaconda shell:

..  warning::

    Since TensorFlow only supports Python versions up to 3.8, NucML must be install in an enviornment with Python Version =< 3.8.

..  code-block:: bash

    # create and activate conda environment
    conda create -n ml_nuclear_env python=3.8
    conda activate ml_nuclear_env

    # Make sure the python version is not higher than 3.8
    python -V

    # install nucml and tensorflow docs
    pip install nucml
    pip install git+https://github.com/tensorflow/docs

Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^

Before moving foward, you need to install both :code:`XGBoost` and :code:`TensorFlow`. These are not provided by default
as a dependency in NucML. We leave these two packages out in case the user already has or plans to install both `XGBoost` and
`TensorFlow` with GPU support. Please follow the instructions in the packages documentation for installation instructions.
If you do not care about GPU support and just want to get started with :code:`NucML`, feel free to install both packages using:

..  code-block:: bash

    # install nucml
    pip install tensorflow xgboost



2. Configure NucML and Generate the EXFOR Datasets
--------------------------------------------------

Before exploring any functionalities, all EXFOR datasets need to be generated. NucML is built around a defined working directory. This
has been uploaded to GitHub as a repository for you to download. It contains all metadata files from all various datasources including 
ACE, ENDF, EXFOR, and ENSDF. The :code:`ML_Nuclear_Data` repository also contains some pre-generated datasets and everything needed to 
produce the heavier EXFOR-based datasets.

First, clone the repository by either downloading and unzipping it directly from `GitHub <https://github.com/pedrojrv/ML_Nuclear_Data>`_  or by 
using the command line.

..  code-block:: bash

    # navigate to the directory of your choice - change it to your own
    cd /Users/pedrovicentevaldez/Desktop/
    
    # clone the ml nuclear data repository
    git clone https://github.com/pedrojrv/ML_Nuclear_Data.git

In the rest of the setup it is assumeed you cloned the repository to your Desktop. 

..  note::

    Feel free to rename the :code:`ML_Nuclear_Data` directory now rather than later as it can cause some issues. 

NucML makes use of a configuration file to get the paths to the generated data based on the working directory structure. Having
downloaded the repository, it is time to tell :code:`NucML` where is it located. Write the following commands in your terminal:


..  code-block:: bash

    # activate your conda environment where nucml is installed
    conda activate ml_nuclear_env

    # navigate to the cloned repo
    cd /Users/pedrovicentevaldez/Desktop/ML_Nuclear_Data
    
    # configure nucml paths
    python -c "import nucml.configure as config; config.configure('.', 'ACE/')"


Now, we are ready to generate the EXFOR datasets. A utility python script is provided to help you get started quickly.

..  note::

    This will generate EXFOR datasets for all avaliable projectiles and will therefore take a couple of minutes.


..  code-block:: bash

    # navigate to the cloned repo - if already there ignore
    cd /Users/pedrovicentevaldez/Desktop/ML_Nuclear_Data
    
    # generate the exfor datasets
    python generate_exfor.py


Running this script will create a :code:`CSV_Files` directory within the :code:`EXFOR` folder. Additionally, a tmp directory will also created containing 
temporary files used in the creation of the final datasets. Feel free to delete the :code:`tmp` directory after the process has finish to save space.

3. Other Dependencies
---------------------

SERPENT2 and MATLAB must be installed if you want to validate your models using criticality benchmarks. These are not necessary for 
other tasks such as loading the data and training ML models.