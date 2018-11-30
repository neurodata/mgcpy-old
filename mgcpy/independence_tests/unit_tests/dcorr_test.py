import pytest
import numpy as np
from mgcpy.independence_tests.dcorr import DCorr


def test_dcorr():
    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    mcorr = DCorr(which_test='mcorr')
    assert np.allclose(mcorr.test_statistic(X, Y)[0], 0)

    # small simulated example of quadratic dependency: quad_sim(10, 2, noise=0, indep=False)
    X = np.array(
        [
            [0.95737163, -0.66346496],
            [0.71554598, -0.26426413],
            [0.68515833, -0.75692817],
            [-0.21809123, -0.44901085],
            [0.8511713, -0.76584218],
            [-0.55365671, -0.92794556],
            [-0.20974912, -0.47266052],
            [-0.42393941, -0.30563822],
            [0.08633873, 0.99153115],
            [-0.44444384, -0.99182435]])
    Y = np.array(
        [
            [0.39142434],
            [0.3403718],
            [0.09406136],
            [0.1958918],
            [0.21925826],
            [1.03556978],
            [0.19898681],
            [0.33265039],
            [0.33884542],
            [0.88426943]])
    mcorr = DCorr(which_test='mcorr')
    dcorr = DCorr(which_test='dcorr')
    mantel = DCorr(which_test='mantel')

    # test get_name
    assert mcorr.get_name() == 'mcorr'
    assert dcorr.get_name() == 'dcorr'
    assert mantel.get_name() == 'mantel'

    # test statistic
    assert np.allclose(mcorr.test_statistic(X, Y)[0], 0.3117760199455171)
    assert np.allclose(dcorr.test_statistic(X, Y)[0], 0.4454977629359435)
    assert np.allclose(mantel.test_statistic(X, Y)[0], 0.2725479362090295)

    # test p value
    assert np.allclose(mcorr.p_value(X, Y)[0], 0.03207910931266045)
    assert np.allclose(dcorr.p_value(X, Y)[0], 0.07384, atol=0.1)
    assert np.allclose(mantel.p_value(X, Y)[0], 0.25674, atol=0.1)
