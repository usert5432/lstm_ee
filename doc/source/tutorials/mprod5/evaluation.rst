Evaluation of Trained Networks
==============================

Now that you have trained your LSTM energy estimator you probably want to know
how it performs. To assist you with this task `lstm_ee` package comes with a
number of scripts. Here a few of them that might be interesting:

1. ``scripts/eval/eval_model.py`` -- script to plot 1D energy resolution
   histogram for the trained energy estimator.

2. ``scripts/eval/make_binstat_plots.py`` -- script to plot Mean/RMS of the
   energy resolution vs true energy.

3. ``scripts/eval/make_auxiliary_plots.py`` -- script to plot 2D histograms
   of energy resolution vs true energy. And to plot the energy histograms
   themselves.

To run one of these scripts you can use the following simplified command:

.. code-block:: bash

   python SCRIPT.py [-e EXT] --preset PRESET_NAME NETWORK_PATH

where ``SCRIPT.py`` is a name of the script that you would like to run,
``EXT`` is a plot extension (e.g. *pdf*, *png*), ``PRESET`` is a name of the
evaluation preset (e.g. if you are evaluating on a sample with energies from
0 to 7 GeV then you should use *numu_7GeV* preset), and ``NETWORK_PATH`` is a
directory where network is saved. The complete list of ``SCRIPT.py`` options
can be obtained by running it with ``--help`` flag.


Example of Plotting Energy Resolution Histograms
------------------------------------------------

Let us try to plot energy resolution histograms for the network trained in
the previous part :doc:`training`. To create *pdf* plots of the energy
resolution histograms you would need to run

.. code-block:: bash

   python scripts/eval/eval_model.py -e pdf --preset numu_7GeV NETWORK_PATH

where **NETWORK_PATH** is the path with saved network

::

    "${LSTM_EE_OUTDIR}/numu/mprod5/final/fd_fhc/model_hash(HASHSTRING)"

and we have used preset *numu_7GeV* since the dataset that we obtained
in :doc:`data` contains events with energies from 0 to 7 GeV.  When the script
completes its execution it will produce a number of files in the directory

::

    "NETWORK_PATH/evals/noise(none)_preset(numu_7GeV)_weights(weight)"

These files are:

1. ``stats.csv`` are the statistical properties of energy resolution histograms
   for different energy types (Lepton, Hadronic, Neutrino) for a given model.
2. ``stats_base.csv`` are the statistical properties of energy resolution
   histograms for the baseline energies (usually for the spline based EE).
3. ``plots/fom_primary.pdf`` -- lepton energy resolution histogram plot.
4. ``plots/fom_secondary.pdf`` -- hadronic energy resolution histogram plot.
5. ``plots/fom_total.pdf`` -- neutrino energy resolution histogram plot.


Other Types of Evaluations
--------------------------

Other evaluation scripts summarized at the beginning of
`Evaluation of Trained Networks`_ work similar to ``eval_model.py`` and
produce additional plots.

You may also be interested in advanced evaluation
scripts found in  ``eval``, ``studies``, ``model``, ``plot`` subdirectories
of the `lstm_ee` ``script`` directory. Please refer to their usage summary for
the details.


