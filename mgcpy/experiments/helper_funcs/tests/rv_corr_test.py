import numpy as np

from mgcpy.experiments.helper_funcs.rv_corr import rv_corr

def test_local_corr():
    a = np.array([1, 4, 6, 5, 1, 9, 12, 3])
    b = np.arange(8)
    A = np.vstack([a,b])
    assert rv_corr(A, A)[0] == 1
    assert np.array_equal(rv_corr(A, A, 1)[0], np.array([[1., 1.], [1., 1.]]))

test_local_corr()
