Using Trained Networks with NOvASoft
====================================

So, you have trained trained your `lstm_ee` network and hopefully satisfied
with its performance. Next, we can try using it in the ``NOvASoft``.
``NOvASoft`` support using `lstm_ee` networks at both NOvA-*art* and ``CAFAna``
levels. However, before the network can be used with ``NOvASoft`` it needs to
be converted from the ``keras`` format into the ``protobuf`` format that
``NOvASoft`` expects.

Converting keras Network into Protobuf Format
---------------------------------------------

In order to convert the trained network into the protobuf format `lstm_ee`
comes with a script ``scripts/tf/export_model.py``. For the proper operation
requires ``tensorflow`` version 1 package to be available on your system.
The usage of this script is simple:

.. code-block:: bash

   python scripts/tf/export_model.py NETWORK_PATH

This script should produce a directory ``NETWORK_PATH/tf`` with two files:

1. ``model.pb`` -- `lstm_ee` network saved in protobuf format.
   This network is optimized for evaluation.
2. ``config.json`` -- network configuration that includes names of input
   variables that it uses, and names of input/output graph nodes.

You should copy this directory ``NETWORK_PATH/tf`` to the machine where it
will be used and maybe rename it to something nice like ``my_awesome_network``.


Using Network at NOvA-*art* Level
---------------------------------

This section assumes that you know how NOvA-*art* works. If this is not the
case then please refer to its tutorial.

To evaluate energies at the NOvA-*art* level you would need to use
``FillLSTME_module.cc`` *art* producer that is located under
``TensorFlowEvaluator/LSTME/art/producer/``. To make it use your network simply
modify its *fcl* configuration file ``FillLSTMEConfig.fcl`` and replace there
values of ``modelFDFHC``, ``modelFDRHC``, ``modelNDFHC``, ``modelNDRHC`` to
point to your network(-s), for example

::

    modelFDFHC : "my_awesome_network"


Using Network at CAFAna Level
-----------------------------

``NOvASoft`` comes with a number of helpers to allow simple evaluation of
`lstm_ee` networks at ``CAFAna`` level. In order to use `lstm_ee` networks
at ``CAFAna`` level, first you would have to include the header with relevant
definitions in your ``CAFAna`` script:

.. code-block:: cpp

    #include "TensorFlowEvaluator/LSTME/cafana/LSTMEVar.h"

Then you will need to create a ``CAFAnaModel`` object that will load network's
graph and evaluate it:

.. code-block:: cpp

   auto model = LSTME::initCAFAnaModel("my_awesome_network");

Finally, this ``CAFAnaModel`` can be used to construct ``CAFAna`` variables
that perform the actual energy evaluation:

.. code-block:: cpp

   Var muE   = LSTME::muonEnergy(model);
   Var hadE  = LSTME::hadEnergy(model);
   Var numuE = LSTME::numuEnergy(model);

.. versionchanged:: r44349
    ``r44349`` renames LSTM energy constructor functions:
    `muonEnergy` -> `primaryEnergy`; `hadEnergy` -> `secondaryEnergy`;
    `numuEnergy` -> `totalEnergy`.

These variables ``muE``, ``hadE``, ``numuE`` will behave as the standard
``CAFAna`` variables for all intents and purposes. You can use them to create
Spectra, Predictions, etc.


.. warning::
    The ``CAFAnaModel`` was written in order to cache results of energy
    evaluation and distribute them between ``muE``, ``hadE``, ``numuE``.
    However, it relies on an undocumented and not fully tested method to
    detect presence of systematic shifts. That is, it examines
    ``SRProxySystController::fgSeqNo`` to check if systematic shift is present.

    Since this method is not fully tested it may result in numerous subtle
    bugs, where for example energy evaluated on a sample without systematics
    will be reused on a sample with systematics. Bump #CAFAna channel on slack
    if this happens.

