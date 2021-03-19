Wilson Cluster Setup
====================

.. warning::
    Wilson Cluster undergone a major upgrade in December of 2020.
    Unfortunately, things still have not settled.
    This guide no longer applies, and should be updated once Wilson Cluster
    environment stabilizes.
    If you need nelp with Wilson Cluster, please check out #nova_reco channel
    on NOvA Slack.

This is a quick guide to get you started with training networks on the Wilson
Cluster.

First, you need to have access to the Wilson Cluster. Ask your Computing
Coordinator if you do not have access yet. You can verify whether you can
connect to the Wilson Cluster by running:

.. code-block:: bash

   ssh LOGIN@tev.fnal.gov

Wilson Cluster uses ``slurm`` for managing cluster jobs. You can find its
quick overview `here <slurm_overview_>`_. Example below is going to show
how to train network interactively with ``slurm``. It assumes that you have
already :ref:`installed <intro:Installation>` and
:ref:`setup <manuals/directory_structure:Directory Structure Setup>` the
`lstm_ee` package.

Interactive training of MiniProd5 Network on Wilson Cluster
-----------------------------------------------------------

First, we need to start an interactive job on a GPU enabled machine:

.. code-block:: bash

   srun --pty --time=24:00:00 --gres=gpu:1 -p gpu -w gpu3 bash

this command will try to start an interactive session on a *gpu3* node
lasting *24* hours.

.. note::
    The *gpu3* node may be occupied, so you would have to look for other nodes.
    There are 4 different gpu node groups in Wilson Cluster.
    The fastest ones are *gpu3*, but there are only 2 *gpu3* nodes available.
    Slightly slower are *gpu4* nodes, but there are 8 of them.
    There are a few *gpu1* and *gpu2* nodes but they are slow and you may
    run out of RAM on them.
    You can check which gpu nodes are currently occupied by running

    ::

        squeue -p gpu

Next, the software that `lstm_ee` requires for training is packaged in a
``Singularity`` container. You would need to load that container before the
training can begin:

.. code-block:: bash

   singularity exec --nv -B /usr/bin:/opt  /lfstev/nnet/singularity/singularity-ML-tf1.12.simg  bash

.. note::
    The ``Singularity`` container

    ::

        /lfstev/nnet/singularity/singularity-ML-tf1.12.simg

    may get moved or outdated. Please bump ``#nova_reco`` slack channel if this
    happens and update this document.

Now, once everything is set up we can start the training. To do that you would
have to launch the training script with ``python3.5`` interpreter, like

.. code-block:: bash

   python3.5 scripts/train/numu/mprod5/final/train_fd_fhc.py

.. note::
    You would have to use ``python3.5`` instead of plain ``python`` for all
    python interpreter invocations. This may change however if the
    ``Singularity`` container above gets updated. Please update this document
    in such event.



.. _slurm_overview: https://slurm.schedmd.com/quickstart.html

