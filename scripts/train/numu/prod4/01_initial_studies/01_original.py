"""Train simplest LSTM EE network without optimizations."""

import os

from lstm_ee.args    import join_dicts
from lstm_ee.consts  import ROOT_OUTDIR
from lstm_ee.presets import PRESETS_TRAIN
from lstm_ee.train   import create_and_train_model
from lstm_ee.utils   import parse_concurrency_cmdargs, setup_logging

config = join_dicts(
    PRESETS_TRAIN['numu_v2'],
    {
    # Config:
        'batch_size'   : 1024,
        #'vars_input_slice',
        #'vars_input_png2d',
        #'vars_input_png3d',
        #'vars_target_total',
        #'vars_target_primary',
        'dataset'      :
            'numu/prod4/fd_fhc/dataset_lstm_ee_fd_fhc_nonswap_std_cut.csv.xz',
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
        'max_prongs'   : 5,
        'model'        : {
            'name'   : 'lstm_v1',
            'kwargs' : {
                'batchnorm'  : False,
                'lstm_units' : 32,
            },
        },
        'noise'        : None,
        'optimizer'    : {
            'name'   : 'RMSprop',
            'kwargs' : { 'lr' : 0.001 },
        },
        'prong_sorters' : None,
        'regularizer'   : None,
        'schedule'      : {
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
        'test_size'       : 200000,
        #'test_size'       : 0.1,
        'weights'         : None,
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : [
            '-png.cvnpart.neutronid',
            '-png.cvnpart.pizeroid',
            '-png.bpf[2].pid',
        ],
        'vars_mod_slice'  : None,
        'outdir'          : 'numu/prod4/initial_studies/original/',
    }
)


parse_concurrency_cmdargs(config)

logger = setup_logging(
    log_file = os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

create_and_train_model(**config)

