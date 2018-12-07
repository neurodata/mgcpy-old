import mgcpy.benchmarks.simulations as sims
import numpy as np
import pytest
from mgcpy.independence_tests.hhg import HHG


def test_hhg():
    # Against a randomly defined data set
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504,
                  1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917,
                  7.1748, 6.5792, 2.4012])[:, np.newaxis]
    hhg = HHG()
    test_stat = hhg.test_statistic(X, Y)[0]

    assert np.round(test_stat, decimals=2) == 411.88

    # Against linear simulations
    np.random.seed(0)
    X, Y = sims.linear_sim(100, 1)
    test_stat = hhg.test_statistic(X, Y)[0]

    assert np.round(test_stat, decimals=2) == 28986.52

    X, Y = sims.linear_sim(100, 1, noise=0)
    test_stat = hhg.test_statistic(X, Y)[0]

    assert np.round(test_stat, decimals=2) == 950600.00
