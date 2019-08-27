import numpy as np
import pytest
import random
from mgcpy.benchmarks.ts_benchmarks import IndependentAR1, CorrelatedAR1, NonlinearLag1
from scipy.stats import pearsonr
from mgcpy.independence_tests.dcorr import DCorr

def test_ts_benchmarks():
    n = 1000
    random.seed(456)
    alpha = 0.05

    # Independent simulation.
    indep = IndependentAR1()
    X, Y = indep.simulate(n)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))

    # Correlated simulation.
    corr = CorrelatedAR1()
    X, Y = corr.simulate(n, phi = 0.9)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))

    # reject the null that the series is not correlated.
    p_value = pearsonr(X[1:n],Y[0:(n-1)])[1]
    assert np.greater(alpha, p_value)

    # Nonlinear simulation.
    nonlin = NonlinearLag1()
    X, Y = nonlin.simulate(n)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))
