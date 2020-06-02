Next Steps
==========

Okay, so we have trained and evaluated MiniProd5 networks and managed to use
them in ``NOvaSoft``. But you may wonder where to get more information about
`lstm_ee`. There are a couple of way you can get more info:

1. Looking over :doc:`../../manuals/index` of this documentation.
   However, :doc:`../../manuals/index` document only few selected topics.

2. The bulk of the documentation is written in form of docstrings in the
   `lstm_ee` source code. To view them you can either directly read the source
   code, or run ``pydoc`` or view docstrings in interactive python shell (e.g.
   ``ipython``). For example, to view documentation of training configuration
   options with ``pydoc`` run

.. code-block:: bash

   pydoc lstm_ee.args.Config

3. Additionally, `lstm_ee` contains a large collection of scripts that were
   used during the NuMu energy estimator development. The scripts to train
   various flavors of energy estimators can be found in
   ``scripts/train/numu/prod4/``.  If can use them as examples.

4. Also, fell free to email the author of this package if you cannot easily
   find answer to your question. You can reach the author by
   ``<torbu001@umn.edu>``.
