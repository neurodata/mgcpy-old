import mgcpy.benchmarks.simulations as sims
import numpy as np
import pytest
from mgcpy.independence_tests.rv_corr import RVCorr


def test_local_corr():
    # Against a randomly defined data set
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504,
                  1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917,
                  7.1748, 6.5792, 2.4012])[:, np.newaxis]
    rvcorr = RVCorr(None)
    rvcorr2 = RVCorr(None, 'pearson')
    rvcorr3 = RVCorr(None, 'cca')

    test_stat1 = rvcorr.test_statistic(X, Y)[0]
    test_stat2 = rvcorr2.test_statistic(X, Y)[0]
    test_stat3 = rvcorr3.test_statistic(X, Y)[0]
    assert np.round(test_stat1, decimals=2) == 0.90
    assert np.round(test_stat2, decimals=2) == 0.95
    assert np.round(test_stat3, decimals=2) == 0.90

    # Against linear simulations
    np.random.seed(0)
    X, Y = sims.linear_sim(100, 1)
    rvcorr = RVCorr(None)
    rvcorr2 = RVCorr(None, 'pearson')
    rvcorr3 = RVCorr(None, 'cca')

    assert rvcorr.get_name() == 'rv'
    assert rvcorr2.get_name() == 'pearson'
    assert rvcorr3.get_name() == 'cca'

    test_stat1 = rvcorr.test_statistic(X, Y)[0]
    test_stat2 = rvcorr2.test_statistic(X, Y)[0]
    test_stat3 = rvcorr3.test_statistic(X, Y)[0]
    assert np.round(test_stat1, decimals=2) == 0.24
    assert np.round(test_stat2, decimals=2) == 0.49
    assert np.round(test_stat3, decimals=2) == 0.24
