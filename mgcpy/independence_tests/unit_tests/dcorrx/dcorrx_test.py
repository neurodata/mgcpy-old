import numpy as np
import pytest
import random
from mgcpy.independence_tests.dcorrx import DCorrX


def test_dcorrx():
    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1, 1])
    Y = np.array([1, 2, 3, 4])
    unbiased = DCorrX(which_test='unbiased', max_lag = 0)
    assert np.allclose(unbiased.test_statistic(X, Y)[0], 0)

    # same test with multiple lags.
    X = np.array([1, 1, 1, 1, 1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3, 4, 5, 6, 7])[:, np.newaxis]
    unbiased = DCorrX(which_test='biased', max_lag = 3)
    assert np.allclose(unbiased.test_statistic(X, Y)[0], 0)

    # test if n <= 3 for unbiased estimator.
    X = np.array([1, 2, 3])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    unbiased = DCorrX(which_test='unbiased', max_lag = 0)
    with pytest.raises(ValueError) as excinfo:
        unbiased.test_statistic(X, Y)
    assert "n <= 3" in str(excinfo.value)

    dir_name = './mgcpy/independence_tests/unit_tests/dcorr/data/'
    X = np.genfromtxt(dir_name + 'test_stat_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'test_stat_Y_mtx.csv', delimiter=',')
    Y = Y[:, np.newaxis]
    unbiased = DCorrX(which_test='unbiased', max_lag = 0)
    biased = DCorrX(which_test='biased', max_lag = 0)

    # test that it must be biased on unbiased.
    with pytest.raises(ValueError) as excinfo:
        notbiased = DCorrX(which_test='notbiased', max_lag = 0)
    assert "which_test" in str(excinfo.value)

    # test get_name
    assert unbiased.get_name() == 'unbiased'
    assert biased.get_name() == 'biased'

    # test statistic
    assert np.allclose(unbiased.test_statistic(X, Y)[0], 0.1174, atol=1e-4)
    assert np.allclose(biased.test_statistic(X, Y)[0], 0.1548, atol=1e-4)

    '''
    Generate independent random variables and ensure that the test is valid with hypothesis test.
    '''
    random.seed(123)
    unbiased = DCorrX(which_test='unbiased', max_lag = 2)
    n = 25
    num_sims = 30
    alpha = 0.05
    num_rejects = 0

    for sim in range(num_sims):
        X = np.random.normal(0.0, 1.0, n)
        Y = np.random.normal(0.0, 1.0, n)
        if unbiased.p_value(X, Y, replication_factor = 100)[0] < alpha:
            num_rejects += 1

    # p value
    assert np.allclose(num_rejects, num_sims*alpha, atol=1.96*num_sims*alpha*(1-alpha))
