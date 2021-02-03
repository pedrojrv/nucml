.. _faq-label:

FAQ
===

What versions of Python3 are supported?
---------------------------------------
NucML is being developed in Python 3.8. It is encourage that this version is installed. Lower versions might work but higher 
versions will not work at the moment due to TensorFlow support up to 3.8.

What if I want to work in two different directories? How do I deal with the configuration?
------------------------------------------------------------------------------------------
You can run the configuration file as many times as you need at the beggining of each script or notebook. 

.. code::

  # configure nucml paths
	python -c "import nucml.configure as config; config.configure('your/working/directory/here', 'path/to/ACE/')"

