"""Train family of v3 LSTM networks with stacked LSTM units"""

import os

from speval import speval

from lstm_ee.args    import join_dicts
from lstm_ee.consts  import ROOT_OUTDIR
from lstm_ee.presets import PRESETS_TRAIN
from lstm_ee.train   import create_and_train_model
from lstm_ee.utils   import parse_concurrency_cmdargs, setup_logging

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
        'max_prongs'   : None,
        'model'        : {
            'name'   : 'lstm_v3_stack',
            'kwargs' : {
                'batchnorm'   : True,
                'layers_pre'  : [],
                'layers_post' : [],
                'lstm3d_spec' : [ (32, 'forward') ],
                'lstm2d_spec' : [ (32, 'forward') ],
                'n_resblocks' : 0,
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
        'seed'            : 1337,
        'steps_per_epoch' : 250,
        'test_size'       : 200000,
        'weights'         : {
            'name'   : 'flat',
            'kwargs' : { 'bins' : 50, 'range' : (0, 5) },
        },
    # Args:
        'vars_mod_png2d'  : None,
        'vars_mod_png3d'  : None,
        'vars_mod_slice'  : None,
        'outdir'          : 'numu/prod4/initial_studies/stack_of_lstms/',
    }
)

parse_concurrency_cmdargs(config)

logger = setup_logging(
    log_file = os.path.join(ROOT_OUTDIR, config['outdir'], "train.log")
)

search_space = [ {} ]
search_space += [
    {
        'model' : {
            'kwargs' : { 'lstm3d_spec' : [ (32, 'forward'), ] * N, }
        }
    } for N in range(2, 4)
]
search_space += [
    {
        'model' : {
            'kwargs' : { 'lstm3d_spec' : [ (32, 'backward'), ] * N, }
        }
    } for N in range(2, 4)
]
search_space += [
    {
        'model' : {
            'kwargs' : { 'lstm3d_spec' : [ (32, 'bidirectional'), ] * N, }
        }
    } for N in range(2, 4)
]


speval(
    lambda x : create_and_train_model(**config, extra_kwargs = x),
    search_space,
    os.path.join(ROOT_OUTDIR, config['outdir'], "trials.db"),
    timeout = 3 * 60 * 60
)

