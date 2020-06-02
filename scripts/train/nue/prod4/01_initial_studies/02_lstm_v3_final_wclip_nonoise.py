"""Train family of numu like LSTM EE network with different weights"""

import os
from speval import speval

from lstm_ee.args    import join_dicts
from lstm_ee.consts  import ROOT_OUTDIR
from lstm_ee.presets import PRESETS_TRAIN
from lstm_ee.train   import create_and_train_model
from lstm_ee.utils   import parse_concurrency_cmdargs, setup_logging

config = join_dicts(
    PRESETS_TRAIN['nue_v3'],
    {
    # Config:
        'batch_size'   : 1024,
        #'vars_input_slice',
        #'vars_input_png2d',
        #'vars_input_png3d',
        #'vars_target_total',
        #'vars_target_primary',
        'dataset'      :
            'nue/prod4/fd_fhc/dataset_lstm_ee_fd_fhc_fluxswap_std_cut.csv.xz',
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
                'layers_pre'   : [ 128, 128, 128 ],
                'layers_post'  : [ 128, 128, 128 ],
                'lstm_units2d' : 32,
                'lstm_units3d' : 32,
                'n_resblocks'  : 0,
            },
        },
        'noise'        : None,
        'optimizer'    : {
            'name'   : 'RMSprop',
            'kwargs' : { 'lr' : 0.001 },
        },
        'prong_sorters' : None,
        'regularizer'   : {
            'name'   : 'l1',
            'kwargs' : { 'l' : 0.001 },
        },
        'schedule'     : {
            'name'   : 'standard',
            'kwargs' : {
                'monitor'  : 'val_loss',
                'factor'   : 0.5,
                'patience' : 5,
                'cooldown' : 0
            },
        },
        'seed'            : 0,
        'steps_per_epoch' : 250,
        'test_size'       : 200000,
        'weights'         : None,
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : None,
        'outdir'          : 'nue/prod4/01_initial_studies/02_lstm_v3_final_wclip_nonoise/',
    }
)

parse_concurrency_cmdargs(config)

search_space = [
    {
        'weights' : 'weight'
    },
    {
        'weights' : {
            'name'   : 'flat',
            'kwargs' : { 'bins' : 25, 'range' : (0, 5)},
        }
    },
    {
        'weights' : {
            'name'   : 'flat',
            'kwargs' : { 'bins' : 25, 'range' : (0, 5), 'clip' : 50 },
        }
    },
]

logger = setup_logging(
    log_file = os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

speval(
    lambda x : create_and_train_model(**config, extra_kwargs = x),
    search_space,
    os.path.join(ROOT_OUTDIR, config['outdir'], "trials.db"),
    timeout = 3 * 60 * 60
)

