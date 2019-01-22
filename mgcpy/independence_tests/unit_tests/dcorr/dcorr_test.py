import numpy as np
import pytest
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
    assert np.allclose(biased.test_statistic(X, Y)[0], 0.1548, atol=1e-4)
    assert np.allclose(mantel.test_statistic(X, Y)[0], 0.2421, atol=1e-4)

    # test statistic (fast versions)
    assert np.allclose(unbiased.test_statistic(X, Y, is_fast=True)[0], 0.1562, atol=1e-4)
    assert np.allclose(biased.test_statistic(X, Y, is_fast=True)[0], 0.3974, atol=1e-4)
    assert np.allclose(mantel.test_statistic(X, Y, is_fast=True)[0], 0.3392, atol=1e-4)

    # additional test for mantel
    X = np.genfromtxt(dir_name + 'mantel_test_stat_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'mantel_test_stat_Y_mtx.csv', delimiter=',')
    X = X[:, np.newaxis]
    Y = Y[:, np.newaxis]
    assert np.allclose(mantel.test_statistic(X, Y)[0], 0.7115, atol=1e-4)
    assert np.allclose(mantel.test_statistic(X, Y, is_fast=True)[0], 0.7552, atol=1e-4) # faster version

    '''
    test p value
    analytical p value for unbiased dcorr is compared with R package energy
    other p values are compared with the permutation tests in mgc-paper
    the value is the mean and atol is set to 4 times standard deviation
    '''
    X = np.genfromtxt(dir_name + 'pvalue_X_mtx.csv', delimiter=',')
    Y = np.genfromtxt(dir_name + 'pvalue_Y_mtx.csv', delimiter=',')
    Y = Y[:, np.newaxis]

    # p value
    assert np.allclose(unbiased.p_value(X, Y)[0], 0.0640, atol=0.03)
    assert np.allclose(biased.p_value(X, Y)[0], 0.0510, atol=0.03)
    assert np.allclose(mantel.p_value(X, Y)[0], 0.1020, atol=0.03)

    # p value (faster versions)
    assert np.allclose(unbiased.p_value(X, Y, is_fast=True)[0], 0.7429, atol=0.03)
    assert np.allclose(biased.p_value(X, Y, is_fast=True)[0], 0, atol=0.03)
    assert np.allclose(mantel.p_value(X, Y, is_fast=True)[0], 0, atol=0.03)
