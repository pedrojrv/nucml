.. _basic-walkthrough-label:


Basic Walkthrough
=================

The cloned repository does not only contain the information necessary to start generating the datasets but also a variety of 
tutorial notebooks that demonstrate the capabilities of NucML. Be sure to check them out. For more information in the contents please 
refer to the repository README file in GitHub. If you haven't installed NucmL yet, please follow the instructions in 
the :ref:`Installation Guide<getting-started-label>`.

In this section we summarize the main pipeline steps for a complete nuclear data neutron-induce cross section evaluation including 
loading the data, modeling, and validating using criticality benchmarks. 



1. Load the Neutrons EXFOR Data
-------------------------------

The `nucml.datasets` module offers the capability of loading various datasets from AME, ENSDF, ENDF, and EXFOR easily. In this example
we load the EXFOR/AME dataset for neutron-induce reactions. 

.. code-block:: python
    
    # first we import the datasets module
    import nucml.datasets as nuc_data

    # We can then load the EXFOR data.
    data, x_train, x_test, y_train, y_test, to_scale, scaler = nuc_data.load_exfor(mode="neutrons", log=True, low_en=True, max_en=2.0E7, num=True, basic=0, normalize=True)

There are a couple of things going on when loading the data. It starts by setting `mode="neutrons"` meaning we only want 
datapoints for neutron-induce reactions.  The `log` arguments specifies that we want the Cross 
Section and Incident Energy data in log form. These are highly skewed features that benefit from this transformation.
The `low_en` arguments tells the loading function that we only want low energy points (up to 2.0E7 eV). 

The EXFOR dataset contains a lot of information including Author, Institutions, Dates, and more. This is generally 
not useful for a ML model. We tell the loader function by setting `num=True` that we only want useful numerical 
and categorical data. This will automatically one-hot encode categorical variables. The `normalize=True` makes sure
that the data is normalized using a standard scaler. There are other transformers avaliable. 

The `basic` argument specifies the complexity of the dataset in terms of features included. See the documentation 
for more information. In this example `basic=0` means that we only want the most basic features 
(Energy, Data, Z, N, A, MT, Center of Mass Flag, Target Flag).

The loader returns eight objects:

- data: The original numerical dataset.
- x_train and y_train: The trainining data and labels.
- x_test and y_test: The testing data and labels.
- to_scale: A list of the features subject to normalization by the scaler.
- scaler: A scikit-learn scaler object. This is the transformer used to normalized the data. 



2. Building the ML Models 
-------------------------

Building the model is user dependent. There are a variety of models and an infinite number of hyperparameter combinations. Tuning
these models are outside the scope of this utility but here a simple Decision Tree model will be trained for demonstration purposes.


.. code-block:: python

    # importing the tree model from sklearn
    from sklearn import tree

    # initalizing the model instance
    dt_model = tree.DecisionTreeRegressor(max_depth=20, min_samples_split=3, min_samples_leaf=2)

    # training the model
    dt_model.fit(x_train, y_train)

You can evaluate the model with various performance metrics. NucML provides a convinience function that allows you to calculate
all performance metrics easily.

.. code-block:: python

    # import nucml modelling utilities
    import nucml.model.model_utilities as model_utils

    # making model predictions to calculate metrics
    y_hat_train = dt_model.predict(x_train)
    y_hat_test = dt_model.predict(x_test)

    # getting performance metrics using NucML
    train_error_metrics = model_utils.regression_error_metrics(y_hat_train, y_train)
    test_error_metrics = model_utils.regression_error_metrics(y_hat_test, y_test)

After training the model make sure to save it for later use in the validation phase. Remember to also save the scaler. 
This latter object is often forgotted but is needed to tranform the same or new data in later work. Make sure the model
name is unique since it is probable you will train various models of the same type.

..	note::

	The model name will be used to create a directory with the same name.


.. code-block:: python

    # import joblib.dump to save the model and scaler
    from joblib import dump

    # specify the path and names for the model and scaler
    model_saving_directory = "path/to/saving/directory/"
    model_name = "dt_model_mss3_msl2_maxdepth20.joblib"
    model_saving_path = os.path.join(model_saving_directory, model_name)
    scaler_saving_path = os.path.join(model_saving_directory, 'scaler.pkl')

    # save the models and scaler
    dump(dt_model, model_saving_path) 
    dump(scaler, open(scaler_saving_path, 'wb'))

In the next section we need to present a `DataFrame` to the ACE utilities to create model dependent benchmark files. The ACE module
is expecting a certain format containing at least the performance error metrics calculated in the previouse code snippet and the path
to the saved model and scaler. In this case we are saving a results `DataFrame` with only one row since we only trained one model
but in reality will include many more models.


.. code-block:: python

    # transform the obtained error metrics using the model utilities
    dt_results = model_utils.create_train_test_error_df(0, train_error_metrics, test_error_metrics)

    # adding the paths to dt_results 
    dt_results["model_path"] = os.path.abspath(model_saving_path)
    dt_results["scaler_path"] = os.path.abspath(scaler_saving_path)

    # You can also append extra information
    dt_results["normalizer"] = "standard_scaler"
    dt_results["max_depth"] = dt_model.get_depth()
    dt_results["mss"] = 3
    dt_results["msl"] = 2

    # save the results
    results_filepath = "path/to/saving/dir/dt_results.csv"
    dt_results.to_csv(results_filepath, index=False)

There are some python scripts included in the ML_Nuclear_Data to help you get started training scikit-learn and XGBoost models
easily. It is a great way to get get started and to experience some of the modeling capabilities of NucML and includes all necessary
code automate more of these tasks.


4. Generating Benchmark Files
-----------------------------

While the benchmark library is small, NucML allows the user to add more benchmark files by following a set of instructions. It 
is best practice to create benchmark files for all avaliable benchmarks. For demonstration we use only the U-233 Jezzebel 
criticality benchmark.

.. code-block:: python

    # import the ace utilities
    import nucml.ace.data_utilities as ace_utils

    # 1) specify directory where all benchmark files will be created
    dt_ml_ace_dir = "DT_B0/"

    # 2) Use the dt_results dataframe to generate benchmark files
    ace_utils.generate_bench_ml_xs(data, dt_results, "U233_MET_FAST_001", to_scale, dt_ml_ace_dir)


Under the hood, the ace utilities performs several things:

0. Creates a new directory with the `model_name` as the name within `dt_ml_ace_dir` 
1. Search for the queried benchmark template
2. Reads the composition and extracts isotopes for which ML cross sections are needed
3. Loads the model and scaler using the paths in the `dt_results`
4. Generates and processes cross sections
5. Creates the `.ace` files for the ML-generated cross sections and copies other needed isotopes from the main ACE directory
6. Creates the `.xsdir` file needed by SERPENT2

..	note::

	The benchmark name is based on the Benchmark Catalog. If a benchmark of choice is not incldued you can include
    your own by following the instructions in the Benchmarks documentation. 


Next, we can generate a `bash` script that will allow us to run all benchmark input files (in this case is just one) and transform 
the resulting `.m` file into a `.mat` file. 

.. code-block:: python

    ace_utils.generate_serpent_bash(dt_ml_ace_dir)

The generated bash script will be saved in the same directory (dt_ml_ace_dir).


5. Validating the ML Models
---------------------------

..	note::

	Both SERPENT2 and MATLAB need to be installed for the this section to work. MCNP is not supported.


You can run your SERPENT2 calculations by running the generated `bash` script.

.. code-block:: bash

    # navigate to the path where the bash script is
    cd path/to/dt_ml_ace_dir/

    # run the script
    ./serpent_script.sh

Once finished, you can gather all results using a simple `ace_utils` method.


.. code-block:: python

    # gather results
    dt_jezebel_results = ace_utils.gather_benchmark_results(dt_ml_ace_dir)


The resulting `DataFrame` contains the model name, benchmark name, the multiplication factor, and the error.


6. Next Steps
-------------

Congratulations, you have performed an end-to-end ML-enhanced nuclear data evaluation using the U-233 Jezebel Benchmark. These are just some of
the general submodules that NucML offers to help you navigate through the evaluation pipeline. Try going through the 
:ref:`Working with NucML<working-with-nucml-label>` section for more information and tutorials. 
