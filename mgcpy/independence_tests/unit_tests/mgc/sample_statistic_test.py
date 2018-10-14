import pytest
import numpy as np

from mgcpy.independence_tests.mgc.sample_statistic import mgc_sample
from mgcpy.benchmarks.simulations import linear_sim


def test_mgc_sample_linear():
    X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                  0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
    Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                  1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)

    mgc_statistic = 0.4389398
    optimal_scale = [10, 10]

    result = mgc_sample(X, Y)

    assert np.allclose(mgc_statistic, result["mgc_statistic"])
    assert np.allclose(optimal_scale, result["optimal_scale"])
