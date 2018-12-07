import mgcpy.benchmarks.simulations as sims
import numpy as np
import pytest
from mgcpy.independence_tests.kendall_spearman import KendallSpearman


def test_kendall_spearman():
    # Against a randomly defined data set
    X = np.array([1.1728, 2.4941, 2.4101, 0.1814, 1.1978, 1.5806, 1.2504,
                  1.9706, 1.8839, 0.8760])[:, np.newaxis]
    Y = np.array([3.2311, 12.1113, 11.1350, 1.1989, 3.3127, 4.8580, 3.4917,
                  7.1748, 6.5792, 2.4012])[:, np.newaxis]
    kspear = KendallSpearman(None)
    kspear2 = KendallSpearman(None, 'spearman')

    test_stat1 = kspear.test_statistic(X, Y)[0]
    test_stat2 = kspear2.test_statistic(X, Y)[0]

    assert np.round(test_stat1, decimals=2) == 1.00
    assert np.round(test_stat2, decimals=2) == 1.00

    # Against linear simulations
    np.random.seed(0)
    X, Y = sims.linear_sim(100, 1)
    kspear = KendallSpearman(None)
    kspear2 = KendallSpearman(None, 'spearman')

    assert kspear.get_name() == 'kendall'
    assert kspear2.get_name() == 'spearman'

    test_stat1 = kspear.test_statistic(X, Y)[0]
    test_stat2 = kspear2.test_statistic(X, Y)[0]

    assert np.round(test_stat1, decimals=2) == 0.33
    assert np.round(test_stat2, decimals=2) == 0.48
