.. _loading-datasets-label:

Datasets
========

.. Note::

    Documentation in progress. Please look at the ML_Nuclear_Data repository for hands-on tutorials on loading various datasets.

Nuclear data can be found in a variety of formats and in different data sources. While the modernization of various of these databases is underway, 
there is a need now for ready-to-use datasets for ML applications. Reaction data from EXFOR is available to download in either various $.x4$ files 
or a single $.xc4$ file, both of which are in formats not compatible with today's ML technologies. The structure and formats of these files were 
designed for compatibility with other types of software including codes like EMPIRE which converts EXFOR native files into individual isotopic $.C4$ 
files for later use. Another database of interest is the Experimental Unevaluated Nuclear Data List (XUNDL). The format in XUNDL files is also 
incompatible with current ML workflows. The Reference Input Parameter Library (RIPL) offers more easy-to-parse $.dat$ files but still offers 
challenges. These and many other nuclear data sources require modernization.

Identifying, parsing, and formatting all nuclear data sources can be a tedious time-consuming job. NucML contains a variety of utilities 
that make it easy to download the latest versions of the EXFOR, RIPL, ENDF, and AME libraries easily. To convert these into ML-friendly datasets, 
parsing utilities are available to read library-native formats, restructure the information, and store the resulting data structure into single 
easy-to-use files in various formats including CSV, JSON, hdf5, and even parquets and Google BigQuery tables.

Transforming these nuclear data sources is just one step towards the integration into ML pipelines. NucML uses the resulting restructured data 
to make core functionalities available. Datasets like EXFOR, even if converted to ML-friendly formats, cannot be considered $ML-ready$. 
Feature engineering is an important stage of any ML workflow. In this phase, features are filtered to only include relevant information, 
new features are engineered, data is transformed according to model-dependent requirements (i.e. one-hot encoding for categorical features), 
standardization and normalization take place, and much more. Engineering new features is just as important as any other step in the ML 
workflow and is known to be a quick and easy way to boost model performance. 

The AME database contains plenty of information including the precise atomic mass, binding energy, mass excess, beta decay energies, 
and Q-values for many types of reactions. These can complement EXFOR and the XUNDL/RIPL library. NucML Datasets makes incorporating 
AME features easy by automatically appending when loading every dataset. In cases where experimental campaigns used a natural target, AME 
data is interpolated using different methods to create usable information for these data points. Other core capabilities of all NucML Dataset 
loader functions include easy train-validation-test splits, recommended filters, feature filtering, normalization and standardization 
(i.e. power transformers, standard scalers, robust scalers), categorical data encoding, and more. It is through these optional capabilities 
that truly ML-ready datasets can be prepared and loaded. 

Some of the most popular ML libraries including Scikit-learn, XGBoost, TensorFlow, and PyTorch have different data requirements for optimized 
loading and training. By default, \textit{NucML Datasets} returns pandas DataFrame objects which are compatible with all scikit-learn models.
Although any user can subsequently transform these data structures, several functionalities are provided to return library dependent objects. 
For example, for TensorFlow, a tf.data.Dataset object can be loaded with user-specified options including batch sizes, shuffle buffers, and 
cache/prefetch instructions. Other objects supported are DataLoader and DMatrix for PyTorch and XGBoost respectively. 
