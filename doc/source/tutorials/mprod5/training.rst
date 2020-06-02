Training LSTM EE Networks
=========================

Prerequisites
-------------

Before you can do the training you should install the `lstm_ee` package.
Please refer to :ref:`intro:Installation` for the instructions on how to do
that.

In addition to that you should setup the directory structure that `lstm_ee`
expects, c.f. :ref:`manuals/directory_structure:Directory Structure Setup`.

If you would like to perform training on the Wilson Cluster then refer to
the :ref:`manuals/wilson_cluster:Wilson Cluster Setup` about how to setup
training environment on the Wilson Cluster.

Training
--------

The scripts to train mprod5 networks can be found in the `lstm_ee` package
under the directory ``scripts/train/numu/mprod5/``. In that directory there are
four scripts named ``train_{fd,nd}_{fhc,rhc}.py`` -- one for each flavor of
training.

If you want to train say Far Detector FHC network then simply run

.. code-block:: bash

   python train_fd_fhc.py

Note, however that this script expects to find a dataset by a path specified
in a training configuration (in the file ``train_fd_fhc.py``):

.. code-block:: python

    config =
    ...
        'dataset'      : (
            'numu/mprod5/fd_fhc'
            '/dataset_lstm_ee_fd_fhc_nonswap_loose_cut.csv.xz'
        ),
    ...

which is itself located under the ``${LSTM_EE_DATADIR}`` directory. So, make
sure that you have moved the dataset obtained in the previous step to the
following location

::

    "${LSTM_EE_DATADIR}/numu/mprod5/fd_fhc/dataset_lstm_ee_fd_fhc_nonswap_loose_cut.csv.xz"

It may take anywhere from 30 minutes to several hours for the training to
complete, depending on your machine. You can speed up training by using
multiprocessing data generation and/or data caches. Please refer to the
:ref:`manuals/data:Caches and Multiprocessing` for the details about speed
optimization.

Training Results
----------------

Once training is complete the trained model and related files will be stored
under a directory also specified in the training configuration (in the file
``train_fd_fhc.py``):

.. code-block:: python

    config =
    ...
        'outdir'          : 'numu/mprod5/final/fd_fhc',
    ...

which is itself located under the ``${LSTM_EE_OUTDIR}`` directory. In other
words, you can find the trained model (``model.h5``), along with its
configuration (``config.json``) and training history (``log.csv``) under:

::

    "${LSTM_EE_OUTDIR}/numu/mprod5/final/fd_fhc/model_hash(HASHSTRING)"

where HASHSTRING is an **MD5** hash of the training configuration. In the
remainder of this tutorial I will be referring to this directory as
**NETWORK_PATH**.


