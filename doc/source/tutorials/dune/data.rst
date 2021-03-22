Generating Training/Validation Samples
======================================

In order to begin training of the RNN energy estimator you need to get a
training sample. Training sample can be extracted from any *art* dataset
with the help of **VLNEnergyDataGen** ``dunetpc`` module.

In this tutorial we will extract training sample from the MCC11 DUNE FD
dataset: ``prodgenie_nu_dune10kt_1x2x6_mcc11_lbl_reco``.

Grid Job Submission
-------------------

The ``VLNets`` package of ``dunetpc`` comes with a sample job
``vlnenergydatagenjob.fcl`` that can be readily used to create training samples
for NuMu CC energy estimation.

The preferred job submission method on DUNE is with the help of `project.py
<projectpy_>`_. Below, you can find a ``project.py`` stage configuration that
can be used to extract training dataset from the DUNE files:

.. code-block:: xml

  <stage name="training_sample_generation">
    <fcl>vlnenergydatagenjob.fcl</fcl>
    <inputdef>prodgenie_nu_dune10kt_1x2x6_mcc11_lbl_reco</inputdef>
    <datafiletypes>csv</datafiletypes>
    <numjobs>250</numjobs>
    <schema>root</schema>
    <outdir>&OUTDIR;</outdir>
    <workdir>&WORKDIR;</workdir>
  </stage>

Where it is expected that ``OUTDIR`` and ``WORKDIR`` variables are set by
the user. After running the ``vlnenergydatagenjob.fcl`` on the grid, the
training dataset will be extracted and stored in the *csv* files in
``${OUTDIR}/*/*.csv``.  To complete training sample generation you need to
merge multiple extracted *csv* files into one.

.. _projectpy: https://cdcvs.fnal.gov/redmine/projects/dunetpc/wiki/Using_project_python

Merging Job Output Files
------------------------

The `lstm_ee` package provides a bash script called ``merge_csv.sh`` that can
be used to merge multiple csv files into one. You can find this script in the
``scripts/data`` directory of the `lstm_ee` package. In addition to merging
the output files together it will compress the result with the *xz* compressor.

In order to use ``merge_csv.sh`` to merge job output files you may run the
following command:

.. code-block:: bash

   bash merge_csv.sh MERGED_FILE_NAME.csv.xz "${OUTDIR}"/*/*.csv

After ``merge_csv.sh`` has finished running you can use the resulting file
``MERGED_FILE_NAME.csv.xz`` for training `lstm_ee` networks.

