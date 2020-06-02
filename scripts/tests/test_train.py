"""Script to verify that model training is working at all."""

import os
import logging

from lstm_ee.consts  import ROOT_OUTDIR
from lstm_ee.args    import join_dicts
from lstm_ee.presets import PRESETS_TRAIN
from lstm_ee.train   import create_and_train_model
from lstm_ee.utils   import setup_logging, parse_concurrency_cmdargs

config = join_dicts(
    PRESETS_TRAIN['numu_v3'],
    {
    # Config:
        'batch_size'   : 1024,
        #'vars_input_slice',
        #'vars_input_png2d',
        #'vars_input_png3d',
        #'vars_target_total',
        #'vars_target_primary',
        'dataset'      : 'test_10k.csv',
        'early_stop'   : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'   : 'val_loss',
                'min_delta' : 0,
                'patience'  : 40,
            },
        },
        'epochs'       : 200,
        'loss'         : 'mean_absolute_percentage_error',
        'max_prongs'   : None,
        'model'        : {
            'name'   : 'lstm_v3',
            'kwargs' : {
                'batchnorm'    : True,
                'layers_pre'   : [ 128, 128, 128],
                'lstm_units2d' : 32,
                'lstm_units3d' : 32,
                'layers_post'  : [ 128, 128, 128],
                'n_resblocks'  : 0,
            },
        },
        'noise'          : None,
        'optimizer'      : {
            'name'   : 'RMSprop',
            'kwargs' : { 'lr' : 0.001 },
        },
        'prong_sorters'  : None,
        'regularizer'    : {
            'name'   : 'l1',
            'kwargs' : { 'l' : 0.001 },
        },
        'schedule'       : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'  : 'val_loss',
                'factor'   : 0.5,
                'patience' : 5,
                'cooldown' : 0
            },
        },
        'seed'            : 1337,
        'steps_per_epoch' : 250,
        'test_size'       : 0.1,
        'weights'         : None,
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : None,
        'outdir'          : 'test_train/',
    }
)

parse_concurrency_cmdargs(config)

setup_logging(
    logging.DEBUG, os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

create_and_train_model(**config)

