Training RNN Energy Estimator
=============================

Prerequisites
-------------

Before you can do the training you should install the `lstm_ee` package.
Please refer to :ref:`intro:Installation` for the instructions on how to do
that.

In addition to that you should setup the directory structure that `lstm_ee`
expects, c.f. :ref:`manuals/directory_structure:Directory Structure Setup`.

You also need a GPU machine with appropriate packages for training of the
energy estimator.


Training
--------

You can train the RNN energy estimator with the help of the
``scripts/train/dune/numu/01_lstm_v1.py`` script that comes with the `lstm_ee`
package:

.. code-block:: bash

   python scripts/train/dune/numu/01_lstm_v1.py

Note, however that this script expects to find the training dataset by a path
specified in a training configuration (in the file ``01_lstm_v1.py``):

.. code-block:: python

    config =
    ...
        'dataset'      : 'dune/numu/dataset_rnne_dune_numu.csv.xz'
    ...

which is located under the ``${LSTM_EE_DATADIR}`` directory. So, make sure that
you have moved the dataset obtained in the previous step to the following
location

::

    "${LSTM_EE_DATADIR}/dune/numu/dataset_rnne_dune_numu.csv.xz"

It may take anywhere from 20 minutes to several hours for the training to
complete, depending on your machine. You can speed up training by using
multiprocessing data generation and/or data caches. Please refer to the
:ref:`manuals/data:Caches and Multiprocessing` for the details about speed
optimization.

Training Results
----------------

Once training is complete the trained model and related files will be stored
under a directory specified in the training configuration (in the file
``01_lstm_v1.py``):

.. code-block:: python

    config =
    ...
        'outdir'          : 'dune/numu/01_rnne_v1',
    ...

which is located under the ``${LSTM_EE_OUTDIR}`` directory. In other words, you
can find the trained model (``model.h5``), along with its configuration
(``config.json``) and training history (``log.csv``) under:

::

    "${LSTM_EE_OUTDIR}/dune/numu/01_rnne_v1/model_hash(HASHSTRING)"

where HASHSTRING is an **MD5** hash of the training configuration. In the
remainder of this tutorial I will be referring to this directory as
**NETWORK_PATH**.


