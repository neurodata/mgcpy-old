import numpy as np

from mgcpy.independence_tests.mdmr import MDMR


def test_mdmr():
    # load data from csv files
    X = np.genfromtxt('./mgcpy/independence_tests/unit_tests/mdmr/data/X_mdmr.csv', delimiter=",")

    Y = np.genfromtxt('./mgcpy/independence_tests/unit_tests/mdmr/data/Y_mdmr.csv', delimiter=",")

    mdmr = MDMR()
    a, results1 = mdmr.test_statistic(X, Y, individual=1)
    b, c = mdmr.p_value(X, Y)
    results2 = mdmr.ind_p_value(X, Y)

    # test get_name
    assert mdmr.get_name() == 'mdmr'

    # test statistic
    assert np.allclose(a, 79.43449382)

    # p-value
    assert np.allclose(b, 0.0)  # 0.000999

    # individual statistics
    assert np.allclose(results1[0, 1], 10.772903841496253)
    assert np.allclose(results2[0, 1], 10.772903841496248)
    assert np.allclose(results2[0, 2], 0.0)
    assert np.allclose(results1[1, 1], 3.993675101550822)
    assert np.allclose(results2[1, 1], 3.9936751015508376)
    assert np.allclose(results2[1, 2], 0.0)
    assert np.allclose(results1[2, 1], 12.421294413838664)
    assert np.allclose(results2[2, 1], 12.421294413838748)
    assert np.allclose(results2[2, 2], 0.0)
