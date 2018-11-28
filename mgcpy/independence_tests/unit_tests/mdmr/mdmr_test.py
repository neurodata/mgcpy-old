from mgcpy.independence_tests.mdmr.mdmr import MDMR
from mgcpy.independence_tests.mdmr.mdmrfunctions import compute_distance_matrix
import numpy as np


def test_mdmr():
    #load data from csv files
    X = np.genfromtxt('./mgcpy/independence_tests/unit_tests/mdmr/data/X_mdmr.csv', delimiter=",")
    
    Y = np.genfromtxt('./mgcpy/independence_tests/unit_tests/mdmr/data/Y_mdmr.csv', delimiter=",")

    mdmr = MDMR(compute_distance_matrix)
    
    #test get_name
    assert mdmr.get_name() == 'MDMR'
    
    #test statistic
    assert np.allclose(mdmr.test_statistic(X, Y)[0], 25.03876688)
    
    #p-value
    assert np.allclose(mdmr.test_statistic(X, Y)[1], 0.000999)
    