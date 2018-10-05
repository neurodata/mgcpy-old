import pytest
import numpy as np
from mgcpy.independence_tests.dcorr import dcorr


def test_dcorr():
    # small simulated example of exponential dependency
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504, 1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917, 7.1748, 6.5792, 2.4012])[:, np.newaxis]
    assert np.round(dcorr(X, Y, 'dcorr'), 4) == 0.9707
    assert np.round(dcorr(X, Y, 'mcorr'), 4) == 0.9550
    assert np.round(dcorr(X, Y, 'mantel'), 4) == 0.8351
    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    assert dcorr(X, Y, 'dcorr') == 0
