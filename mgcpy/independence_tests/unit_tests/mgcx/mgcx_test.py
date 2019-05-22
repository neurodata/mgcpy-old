import numpy as np
import pytest
import random
from mgcpy.independence_tests.mgcx import MGCX
from mgcpy.independence_tests.mgc import MGC


def test_mgcx():
    # test zero variance dataset with multiple lags.
    X = np.array([1, 1, 1, 1, 1, 1, 1])
    Y = np.array([1, 2, 3, 4, 5, 6, 7])
    mgcx = MGCX(max_lag = 3)
    assert np.allclose(mgcx.test_statistic(X, Y)[0], 0)

    dir_name = './mgcpy/independence_tests/unit_tests/dcorr/data/'
    X = np.genfromtxt(dir_name + 'test_stat_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'test_stat_Y_mtx.csv', delimiter=',')
    Y = Y[:, np.newaxis]
    mgcx = MGCX(max_lag = 0)
    mgc = MGC()

    # test get_name
    assert mgcx.get_name() == 'mgcx'

    assert np.allclose(mgcx.test_statistic(X, Y)[0], mgc.test_statistic(X, Y)[0], atol=1e-4)

    '''
    Generate independent random variables and ensure that the test is valid with hypothesis test.
    '''
    random.seed(123)
    mgcx = MGCX(max_lag = 2)
    n = 20
    num_sims = 20
    alpha = 0.05
    num_rejects = 0

    for sim in range(num_sims):
        X = np.random.normal(0.0, 1.0, n)
        Y = np.random.normal(0.0, 1.0, n)
        if mgcx.p_value(X, Y, replication_factor = 100)[0] < alpha:
            num_rejects += 1

    assert np.allclose(num_rejects, num_sims*alpha, atol=1.96*num_sims*alpha*(1-alpha))
