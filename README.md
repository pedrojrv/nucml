# nucml
 
# NucML

NucML is the first and only end-to-end python-based supervised machine learning pipeline for enhanced bias-free nuclear data generation and evaluation to support the advancement of next-generation nuclear systems. It offers capabilities that allows researchers to navigate through each step of the ML-based nuclear data cross section evaluation pipeline. Some of the supported activities include include dataset parsing and compilation of reaction data, exploratory data analysis, data manipulation and feature engineering, model training and evaluation, and validation via criticality benchmarks. Some of the inherit benefits of this approach are the reduced human-bias in the generation and solution and the fast iteration times. Resulting data from these models can aid the current NDE and help decisions in uncertain scenarios.

<!-- # TODO: UNCOMMENT PERIODIC TABLE ONCE GITHUB PUBLIC in AME PARSING UTILITIES
# TODO: ADD LINK TO DOWNLOAD EVALUATED DATA
# TODO: FIX GET FOR EXFOR ENDF DATA UTILITIES IF NEEDED -->

# Installation


You can simply install NucML using `pip`:

```python
pip install nucml
```

# Quick-Start

Before exploring any functionalities all datasets need to be generated. There are many data srouces that enable NucML's core functionalities. All necessary are found in the follwoing github repository

 `generating_dataset.py` script as follow:

```python
cd path/to/the/directory/ML_Nuclear_Data
python -c "import nucml.configure as config; config.configure('."', 'ACE/')"
python generate_exfor.py
```

This will take a couple of minutes. Running this script will create the CSV_Files directory and all the content including all CSV files. Additionally, a `tmp` directory will be created containing files to create the CSV files. Feel free to delete it after the process has finish. 

Python tools to process nuclear data and test ML models in benchmark calculations.

- EXFOR_Parsing_Utilities.py: provides tools to process EXFOR C4 files into a format useful for futher data processing and ML applications.
- AME_Parsing_Utilities.py: provides tools to process the Atomic Mass Evaluation files. The resulting data files can be use in conjunction with the EXFOR files or independently.


## Quick-Start: Creating Datasets
Lets say you have your own C4 files and your own mass16, rct1, rct2 files and you want recreate the EXFOR database in csv format. The way you would do this is by calling out

This command creates two files: a pure AME csv or an iputed AME csv where NaN values can been filled using the mean from the avaliable isotopes. This command should take a few minutes.

`python -c "from Utilities import AME_Parsing_Utilities as ame_parsing; ame_parsing.get_all()"`

You'll need the proccessed AME files to create the EXFOR csv since the information will be appended to each measurment. To create an EXFOR csv either for protons or neutros or both you need to run the following command. This will process each c4 file and collect the necessary information, process it, convert it into a user-friendly format, and save a it into a single CSV file. Because of this, expect this command to last a while.

`python -c "from Utilities import EXFOR_Parsing_Utilities as exfor_parsing; exfor_parsing.get_all()"`

After having your dataset created you are welcome to use any other packages to traing your models on this dataset.

## Quick-Start: Testing your ML algorithms

- VISUALIZE ML ENDF EXFOR AND OTHER EVALUATED LIBRARIES
- GET STATISTICS FROM PREDICTINS ERRORS AMONGST EACH OTHER
- COMPILE ACE FILES IN CURRENT GRID
- RUN BENCHMARKS AND GET RESULTS NO TONLY KEFF


ADD VERBOSITY FUNCTIONS


Packages needed `natsort`

`python ./nucml/exfor/parsing_utilities.py "./AME"`


# Challenge Understanding

It is the intent of this projet to develop an algorithm for cross section predictions. Through this journey we are testign several ML models with different parameteres/hyperparameters. Cross sections data is used in every aspect of nuclear sciences. It describes the probability that a particular reaction channel will occurr. It is normaly used in monte carlo and deterministic codes to simulate a particular assembly. The accuracy of such models depend to a high degree on the cross section data used. Througout the years there have been a variety of experimental campaings that aim to calculate cross section data for many elements/isotopes. This campagins require sometimes extensive resources and calculating data for each existing isotope is unfeasiable/imposible. While there are physical reaction models that are used in tools like EMPIRE and TALLY which use theory to try and predict cross section data in unevaluated energy zones, it is known that some of these are innaccurate in the order of 5 or more. Motivated by this, and due to the avaliable computational power nowdays, we seek to use Machine Learning to try and come up with a model that will help us guide cross section data evaluations. 


**What factors influence cross section values?**

There are a variety of independent variables that are known to have a role in cross section data. Some obvious ones like the number of protons and neutrons, and some not like energy levels, and parities/spins. Therefore the first phase of this project consists on gathering only the experimental data and build a model around it to set a baseline accuracy.





# Analytic Approach

The task involves a regression problem. We need to collect features that might help the ML algorithm learn unhidden patterns and behavious. These features must make sense physically. Knowing that this is a regression problem we need to select a suitable techniques for the desired outcome. 

**Desicion Trees and Random Forests**

If we suspect cross sections follow a specific set of unknown tree rules then this might be an appropiate approach although it becomes apparent intially that the desiciont tree will need to be deep enough to capture specific patterns in all nuclei, it's isotopes, and energy regions. 





# Data Requirements

For this challenge we need real known expeirmental data that will form the basis for our entire problem. We will initially focuse on neutron induce reactions. This means we need energy and angle dependent cross section measurments. Additionally, we will require information about the individual isotopes. This involes masses, # of neutrons and protons, spins, parities, energy levels, etc. 




# Data Collection and Sources

We will use the EXFOR database which contains all experimental data avaliable. The latest database that was downloaded from the IAEA servers was `CE-2019-07-18`. As for the masses, binding energy, and beta decay information, they were gathered from the Atomic Mass Evaluation files. All other information will be collected from other nuclear databases by web scrapping. This will involve a great deal of cleaning and formatting data sources. 





# Data Understanding

Cleaned EXFOR Database Description:

- Prj: 1 for neutron.
- Isomer: G for Ground and M for Metastable
- MF: ENDF file labels. "Files" are usually used to store different types of data, thus:
MF=1 contains descriptive and miscellaneous data,
MF=2 contains resonance parameter data,
MF=3 contains reaction cross sections vs energy,
MF=4 contains angular distributions,
MF=5 contains energy distributions,
MF=6 contains energy-angle distributions,
MF=7 contains thermal scattering data,
MF=8 contains radioactivity data
MF=9-10 contain nuclide production data,
MF=12-15 contain photon production data, and
MF=30-36 contain covariance data.
- MT: Specifies the reaction to be measured. 
- Energy and Uncertainty: Energy at which the XS was measured and its given uncertainty.
- Cross Section Measurments and Uncertainty: XS measured along with its uncertainty.
- Angle and Uncertainty (if applicable): Some experiments do not specify angle (not even 0) so we are assuming those with empty values are at 0 degrees. The measured angle uncertainty is also given. 
- ELV/HL and dELV/HL: Something related to half-life?
- Protons, Neutrons and Mass Number: Derived from the original Target feature in EXFOR. 
- Product Meta State: The energy level state of the final product. 
- Center of Mass of Experiment: Lab or Center-of-Mass measurment. 


# Data Preparation

11/10/2019: The entire EXFOR database consisting of 6,007,126 data points and 18 features (including the two target variables) has been cleaned. There are still some questions regarding some unknown and default values that should be asked. The data is stored in `working_xs.csv`.

PUT ASSUMPTIONS HERE. 




# Modeling

### PHASE 1

In phase one we use only the EXFOR files. This files include features like:


We will build several ML models to set a baseline accuracy. 

### PHASE 2 

We collect masses from the Atomic Masses Evaluations. We add them to our original dataset and see improvment in accuracy. 

# EVALUATION


# Modify

Ace full path 

```python
ace_dir = "C:\\Users\\Pedro\\Documents\\Serpent\\xsdata\\endfb7\\acedata"

ame_dir_path = "C:\\Users\\Pedro\\Desktop\\ML_Nuclear_Data\\AME\\"
ame_originals_path = "C:\\Users\\Pedro\\Desktop\\ML_Nuclear_Data\\AME\\Originals\\"
```



# NucML

Pedro Vicente-Valdez\
Nuclear Engineering PhD Candidate - UC Berkeley \
AI/ML Moderator - Apple\
pedro.vicentevz@berkeley.edu

Python tools to process nuclear data and test ML models in benchmark calculations.

- EXFOR_Parsing_Utilities.py: provides tools to process EXFOR C4 files into a format useful for futher data processing and ML applications.
- AME_Parsing_Utilities.py: provides tools to process the Atomic Mass Evaluation files. The resulting data files can be use in conjunction with the EXFOR files or independently.


## Quick-Start: Creating Datasets
Lets say you have your own C4 files and your own mass16, rct1, rct2 files and you want recreate the EXFOR database in csv format. The way you would do this is by calling out

This command creates two files: a pure AME csv or an iputed AME csv where NaN values can been filled using the mean from the avaliable isotopes. This command should take a few minutes.

`python -c "from Utilities import AME_Parsing_Utilities as ame_parsing; ame_parsing.get_all()"`

You'll need the proccessed AME files to create the EXFOR csv since the information will be appended to each measurment. To create an EXFOR csv either for protons or neutros or both you need to run the following command. This will process each c4 file and collect the necessary information, process it, convert it into a user-friendly format, and save a it into a single CSV file. Because of this, expect this command to last a while.

`python -c "from Utilities import EXFOR_Parsing_Utilities as exfor_parsing; exfor_parsing.get_all()"`

After having your dataset created you are welcome to use any other packages to traing your models on this dataset.

## Quick-Start: Testing your ML algorithms

- VISUALIZE ML ENDF EXFOR AND OTHER EVALUATED LIBRARIES
- GET STATISTICS FROM PREDICTINS ERRORS AMONGST EACH OTHER
- COMPILE ACE FILES IN CURRENT GRID
- RUN BENCHMARKS AND GET RESULTS NO TONLY KEFF


ADD VERBOSITY FUNCTIONS


Packages needed `natsort`

`python ./nucml/exfor/parsing_utilities.py "./AME"`