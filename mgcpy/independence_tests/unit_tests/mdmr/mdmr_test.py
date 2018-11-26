from mgcpy.independence_tests.mdmr.mdmr import MDMR
from mgcpy.independence_tests.mdmr.mdmrfunctions import compute_distance_matrix
import numpy as np


def test_mdmr():
    #load data from csv files
    csv1 = np.genfromtxt('data/X_mdmr.csv', delimiter=",")
    X = csv1
    
    csv1 = np.genfromtxt('data/Y_mdmr.csv', delimiter=",")
    Y = csv1
    
    mdmr = MDMR(csv1, csv2, compute_distance_matrix)
    
    #test get_name
    assert mdmr.get_name() == 'MDMR'
    
    #test statistic
    assert np.allclose(mdmr.test_statistic()[0], 25.03876688)
    
    #p-value
    assert np.allclose(mdmr.test_statistic()[1], 0.000999)
    