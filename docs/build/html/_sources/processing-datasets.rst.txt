.. _processing-datasets-label:

Processing Datasets
===================

.. Note::

    Documentation in progress. Please look at the ML_Nuclear_Data repository for hands-on tutorials on processing ML genereated cross sections using NucML.



In the processing stage, evaluated data is transformed into formats (i.e. ACE files) readable by user codes like SERPENT and MCNP. Having a great model 
is only half the process. Tools are needed to efficiently create entire evaluated libraries consisting of cross-sections for all needed isotopes and 
reaction channels. Currently, NucML offers the capabilities to create isotopic $.ace$ files by simply supplying a user-trained model. The python package 
will take care of querying the model for predictions using the same energy grid as the original .ACE files for compatibility. Additionally, some stability 
options are included. Since these algorithms are not expected to be perfect, \textit{NucML Ace} is built under the assumption that the best solution is a 
hybrid solution. In other words, traditional tools must work together with ML models to create a good evaluation. By default, \textit{NucML Ace} will 
stabilize the 1/v region using evaluated library values.



https://github.com/pedrojrv/ML_Nuclear_Data/blob/master/Benchmarks/0_Generating_MLXS_Benchmark.ipynb