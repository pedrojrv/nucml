.. _modeling-datasets-label:

Modeling Datasets
=================

.. Note::

    Documentation in progress. Please look at the ML_Nuclear_Data repository for hands-on tutorials on modeling ML-ready datasets using NucML.

In the evaluation phase of the traditional NDE pipeline, relevant data is used to guide physics-based model calculations which result in best estimates, 
dependent on data availability, of mean values including uncertainties and covariances. These values can then form part of one or more regional libraries (i.e., ENDF). 
As previously mentioned, the ML-NDE pipeline instead makes use of trained ML models to create reaction cross-section data and therefore to generate ML-based libraries. 

The NucML Model utilities provide various python script examples to train various ML algorithms including scikit-learn models (i.e. K-nearest-neighbors, 
Decision Trees), Gradient Boosting Machines, and Neural Networks. It is built around a strict ML management philosophy by keeping track of model 
hyperparameters and resulting performance metrics for the supported models. Other ML management tools like Comet ML and Weights and Biases can be 
configured and used using user-provided credentials. It is the goal of NucML to first and foremost provide researchers the framework and tools to create, 
train, and analyze their models rather than providing a set of optimized algorithms. 

Please refer to the following links for example python scripts for different models.


* `Decision-Trees (DT) <https://github.com/pedrojrv/ML_Nuclear_Data/blob/master/ML_EXFOR_neutrons/2_DT/dt.py>`_
* `K-Nearest-Neighbor (KNN) <https://github.com/pedrojrv/ML_Nuclear_Data/blob/master/ML_EXFOR_neutrons/1_KNN/knn.py>`_
* `XGBoost (Gradient Boosting Machines) <https://github.com/pedrojrv/ML_Nuclear_Data/blob/master/ML_EXFOR_neutrons/3_XGB/xgb.py>`_
* `Neural Networks <https://github.com/pedrojrv/ML_Nuclear_Data/tree/master/ML_EXFOR_neutrons/4_NN>`_ 
