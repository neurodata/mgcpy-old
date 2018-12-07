import pytest
import numpy as np
from mgcpy.independence_tests.dcorr import DCorr


def test_dcorr():
    # test the special case when one of the dataset has zero variance
    X = np.array([1, 1, 1])[:, np.newaxis]
    Y = np.array([1, 2, 3])[:, np.newaxis]
    unbiased = DCorr(which_test='unbiased')
    assert np.allclose(unbiased.test_statistic(X, Y)[0], 0)

    dir_name = './mgcpy/independence_tests/unit_tests/dcorr/data/'
    X = np.genfromtxt(dir_name + 'test_stat_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'test_stat_Y_mtx.csv', delimiter=',')
    Y = Y[:, np.newaxis]
    unbiased = DCorr(which_test='unbiased')
    biased = DCorr(which_test='biased')
    mantel = DCorr(which_test='mantel')

    # test get_name
    assert unbiased.get_name() == 'unbiased'
    assert biased.get_name() == 'biased'
    assert mantel.get_name() == 'mantel'

    # test statistic
    assert np.allclose(unbiased.test_statistic(X, Y)[0], 0.1174, atol=1e-4)
    assert np.allclose(biased.test_statistic(X, Y)[0], 0.1179, atol=1e-4)
    assert np.allclose(mantel.test_statistic(X, Y)[0], 0.2255, atol=1e-4)

    '''
    test p value
    analytical p value for unbiased dcorr is compared with R package energy
    other p values are compared with the permutation tests in mgc-paper
    the value is the mean and atol is set to 4 times standard deviation
    '''
    X = np.genfromtxt(dir_name + 'pvalue_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'pvalue_Y_mtx.csv', delimiter=',')
    Y = Y[:, np.newaxis]

    assert np.allclose(unbiased.p_value(X, Y)[0], 0.04827, atol=0.01)
    assert np.allclose(biased.p_value(X, Y)[0], 0.0621, atol=0.03)
    assert np.allclose(mantel.p_value(X, Y)[0], 0.1823, atol=0.04)
