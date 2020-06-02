lstm_ee
=======
Package to train NOvA LSTM Energy Estimators.

Overview
--------
This package is designed to simplify (re-)training of the energy estimators for
the NOvA Experiment that are based on recurrent neural networks. It contains
an extensive library of helper functions and a number of scripts to train
various kinds of energy estimators and evaluate them. Please refer to
`Documentation`_ for the details.

This package is inspired by the `rnnNeutrinoEnergyEstimator <original_>`_
which laid the groundwork for the LSTM energy estimator development.

Installation
------------
`lstm_ee` package is indented for developers. Therefore, the recommend way
to install the package is to:

1. Git clone this repository:

.. code-block:: bash

   git clone https://github.com/usert5432/lstm_ee

2. Add cloned repo to the ``PYTHONPATH`` environment variable.

.. code-block:: bash

   export PYTHONPATH="FULL_PATH_TO_CLONED_REPO:${PYTHONPATH}"

You may want to add the line above to your ``~/.bashrc``.

For the proper operation `lstm_ee` requires several other python packages to
be available on your system, see `Requirements`_.

If you are running `lstm_ee` for the first time it might be useful to run
its test suite to make sure that the package is not broken:

.. code-block:: bash

    python -m unittest tests.run_tests.suite


Requirements
------------

`lstm_ee` package is written in python v3 and won't work with python v2.
`lstm_ee` depends on the following packages:

* ``keras``   -- for training neural networks.
* ``pandas``, ``numpy`` -- for handling data.
* ``cython``  -- for compiling optimized data handling functions
* ``scipy``   -- for fitting curves.
* ``cafplot`` -- for plotting evaluation results.

Make sure that these packages are available on your system. You can install
them with ``pip`` by running

.. code-block:: bash

   pip install --user -r requirements.txt

Also, `lstm_ee` has a number of optional dependencies:

* ``tensorflow`` -- for exporting ``keras`` models into protobuf format that
  NOvASoft expects. Note that only ``tensorflow`` v1 is supported currently.

* ``pytables`` -- for working with HDF5 files.
* ``speval`` -- for parallelizing training across multiple machines.


Documentation
-------------

`lstm_ee` package comes with several layers of documentation. The basic
overview of the `lstm_ee` workings and examples of usage are documented in
sphinx format. You can find this documentation by the following
`link <prebuilt_doc_>`_ (requires nova credentials).

Alternatively, you can manually compile the sphinx documentation by running
the following command in the ``doc`` subdirectory:

.. code-block:: bash

   make html

It will build all available documentation, which can be viewed with a web
browser by pointing it to the ``build/html/index.html`` file.

In addition to the sphinx documentation the `lstm_ee` code is covered by a
numpy like docstrings. Please refer to the docstrings for the details about
inner `lstm_ee` workings.

.. _prebuilt_doc: https://nova-docdb.fnal.gov/cgi-bin/private/ShowDocument?docid=45821
.. _original: https://github.com/AlexanderRadovic/rnnNeutrinoEnergyEstimator

