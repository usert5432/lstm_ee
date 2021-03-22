Using Trained Networks with dunetpc
===================================

The `VLNets` package of the ``dunetpc`` has an *art* producer that is capable
of evaluating RNN energy estimator and storing the results in the *art* files.
However, before the network can be used with ``dunetpc`` it needs to
be converted from the ``keras`` format into the ``protobuf`` format that
``dunetpc`` expects.

Converting keras Network into Protobuf Format
---------------------------------------------

Please refer to the NOvA's tutorial about how to convert ``keras`` network
into the ``protobuf`` format:
:ref:`tutorials/mprod5/exporting:Converting keras Network into Protobuf Format`
.

Using Network with dunetpc
--------------------------

This section assumes that you know how *art* producers work.

To evaluate RNN energy estimator you can use the **VLNEnergyProducer** producer
that is a part of the `VLNets` ``dunetpc`` module. You can use an *art* job
``vlnenergyevalnumujob.fcl`` to perform the network evaluation. Simply change
"ModelPath" of the ``vlnenergyreco`` producer in ``vlnenergyevalnumujob.fcl``
job config to point to your model:

::

    physics.producers.vlnenergyreco.ModelPath : "my_awesome_network"


