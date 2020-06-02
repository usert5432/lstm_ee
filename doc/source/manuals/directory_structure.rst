Directory Structure Setup
=========================

The `lstm_ee` package scripts assume that the user datasets are located under
the directory specified by the ``${LSTM_EE_DATADIR}`` environment variable.
Therefore, before you begin training you should setup this environment variable
like (maybe worth putting this to your ``~/.bashrc``):

.. code-block:: bash

   export LSTM_EE_DATADIR=PATH_TO_DIRECTORY_WITH_DATA

Similarly, `lstm_ee` scripts will save the trained models and associated data
under the directory specified by the ``${LSTM_EE_OUTDIR}`` environment
variable. Therefore, you should setup this directory as well:

.. code-block:: bash

   export LSTM_EE_OUTDIR=PATH_TO_DIRECTORY_WHERE_RESULTS_WILL_BE_SAVED


