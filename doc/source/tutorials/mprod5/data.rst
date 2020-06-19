Obtaining Training/Validation Data for MiniProd5 Training
=========================================================

In order to obtain training and validation datasets for the `lstm_ee`
retraining for the MiniProd5 campaign you have two options:

1. You can regenerate datasets manually from the mprod5 **caf** files.
2. Or you can reuse already generated datasets.

I will provide instructions for each option below.

1. Manual Data Generation
-------------------------

The scripts to manually generate dataset from the mprod5 **caf** files are
committed to the NOvA devsrepo. You can fetch them using the following command

.. code-block:: bash

   export DEVSREPO=svn+ssh://p-novaart@cdcvs.fnal.gov/cvs/projects/novaart-devs
   svn checkout "${DEVSREPO}/trunk/users/torbunov/lstm_ee/scripts/mprod5"

Inside the fetched directory *mprod5* you will find four files named
``exporter_lstm_ee_{fd,nd}_{fhc,rhc}_nonswap.C``. They can be used to generate
training and validation datasets for the Far and Near Detectors, FHC and RHC
horn currents respectively. These scripts are known to be working at the
``R19-10-30-final-prod4.b`` release of NOvaSoft, and may work in the later
releases.


Job Submission
^^^^^^^^^^^^^^

Let say you want to generate data for the Far Detector FHC training. To do that
you would need to submit a grid job that runs the script
``exporter_lstm_ee_fd_fhc_nonswap.C`` in parallel. The job can be submitted to
the grid via the following command:

.. code-block:: bash

   submit_cafana.py \
        --njobs 250 --print_jobsub --rel R19-10-30-final-prod4.b \
        --outdir OUTDIR \
        exporter_lstm_ee_fd_fhc_nonswap.C

where ``OUTDIR`` is a directory under **/pnfs** where job output files will be
stored. Once the grid job has completed you will find multiple csv files under
``OUTDIR`` with names ``dataset_lstm_ee_fd_fhc_nonswap_*_of_*.csv``. These
output files need to be merged together before they can be used for training.


Merging Job Output Files
^^^^^^^^^^^^^^^^^^^^^^^^

The `lstm_ee` package provides a bash script called ``merge_csv.sh`` that can
be used to merge multiple csv files into one. You can find this script in the
``scripts/data`` directory of the `lstm_ee` package. In addition to merging
the output files together it will compress the result with the *xz* compressor.

In order to use ``merge_csv.sh`` to merge job output files you may run the
following command:

.. code-block:: bash

   bash merge_csv.sh MERGED_FILE_NAME.csv.xz OUTDIR/dataset_*.csv

After ``merge_csv.sh`` has finished running you can use the resulting file
``MERGED_FILE_NAME.csv.xz`` for training `lstm_ee` networks.


2. Retrieving Old Datasets
--------------------------

The old mprod5 datasets are stored under the SAM system. The SAM definition
that contains these datasets is ``dataset_lstm_ee_mprod5``. There are four
different datasets available:

1. ``dataset_lstm_ee_mprod5_fd_fhc_nonswap.csv.xz`` -- FD FHC
2. ``dataset_lstm_ee_mprod5_fd_rhc_nonswap.csv.xz`` -- FD RHC
3. ``dataset_lstm_ee_mprod5_nd_fhc_nonswap.csv.xz`` -- ND FHC
4. ``dataset_lstm_ee_mprod5_nd_rhc_nonswap.csv.xz`` -- ND RHC

To retrieve any of those datasets you can use ``ifdh_fetch`` command, e.g.

::

    ifdh_fetch dataset_lstm_ee_mprod5_fd_fhc_nonswap.csv.xz

