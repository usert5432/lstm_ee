"""
Constants that are widely used in lstm_ee
"""

import os

DEF_SEED  = 1337
DEF_MASK  = 0.

LABEL_TOTAL     = 'total'
LABEL_PRIMARY   = 'primary'
LABEL_SECONDARY = 'secondary'

if 'LSTM_EE_DATADIR' in os.environ:
    ROOT_DATADIR = os.environ['LSTM_EE_DATADIR']
else:
    ROOT_DATADIR = '/'

if 'LSTM_EE_OUTDIR' in os.environ:
    ROOT_OUTDIR = os.environ['LSTM_EE_OUTDIR']
else:
    ROOT_OUTDIR = '/'

