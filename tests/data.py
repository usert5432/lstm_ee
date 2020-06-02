"""
Common dataset definition for various tests.
"""

import numpy as np

def nan_equal(a, b):
    """Compare `a` and `b` taking into account possible NaN values"""
    a = np.array(a)
    b = np.array(b)
    return (np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).all()

TEST_INPUT_VARS_SLICE   = [ 'x_slice1', 'x_slice2' ]
TEST_INPUT_VARS_PNG3D   = [ 'x_png3d1', 'x_png3d2' ]
TEST_INPUT_VARS_PNG2D   = [ 'x_png2d1', 'x_png2d2' ]
TEST_TARGET_VAR_TOTAL   = 'target_total'
TEST_TARGET_VAR_PRIMARY = 'target_primary'

X_SLICE_1 = [ 1, 2, 3, 2, 1 ]
X_SLICE_2 = [ 2, 3, 4, 3, 2 ]

TARGET_TOTAL   = [ 0, 0, 5, 0, 3 ]
TARGET_PRIMARY = [ 1, 1, 6, 1, 3 ]

X_PNG3D_1 = [ [ 1,2,3 ], [ 4 ],   [  ],      [ 4 ], [ 1,2 ] ]
X_PNG3D_2 = [ [ 4,5,6 ], [ 5 ],   [  ],      [ 5 ], [ 4,5 ] ]
X_PNG2D_1 = [ [ 9,8 ],   [ 9,1 ], [ 1,6,3 ], [  ],  [ 4 ]   ]
X_PNG2D_2 = [ [ 6,5 ],   [ 8,9 ], [ 4,2,5 ], [  ],  [ 3 ]   ]

TEST_DATA = {
    'x_slice1'       : X_SLICE_1,
    'x_slice2'       : X_SLICE_2,
    'target_total'   : TARGET_TOTAL,
    'target_primary' : TARGET_PRIMARY,
    'x_png3d1'       : X_PNG3D_1,
    'x_png3d2'       : X_PNG3D_2,
    'x_png2d1'       : X_PNG2D_1,
    'x_png2d2'       : X_PNG2D_2,
}

TEST_DATA_VARS = list(TEST_DATA.keys())
TEST_DATA_LEN  = len(TEST_DATA['x_slice1'])

