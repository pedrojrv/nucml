.. _loading-datasets-exfor:

Experimental Nuclear Reaction Data
==================================

.. Note::

    Documentation in progress. 

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



