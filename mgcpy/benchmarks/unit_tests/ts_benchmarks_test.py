import numpy as np
import pytest
import random
from mgcpy.benchmarks.ts_benchmarks import IndependentAR1, CorrelatedAR1, NonlinearLag1
from scipy.stats import pearsonr
from mgcpy.independence_tests.dcorr import DCorr

def test_ts_benchmarks():
    n = 1000
    random.seed(123)
    alpha = 0.05

    # Independent simulation.
    indep = IndependentAR1()
    X, Y = indep.simulate(n)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))

    # Independent simulation.
    corr = CorrelatedAR1()
    X, Y = corr.simulate(n)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))

    # reject the null that the series is not correlated.
    p_value = pearsonr(X[1:n],Y[0:(n-1)])[1]
    assert np.greater(alpha, p_value)

    # Independent simulation.
    nonlin = NonlinearLag1()
    X, Y = nonlin.simulate(n)
    assert np.all(X.shape == (n,))
    assert np.all(Y.shape == (n,))

    dcorr = DCorr()
    p_value = dcorr.p_value(X[1:n].reshape(n-1, 1),Y[0:(n-1)].reshape(n-1, 1), is_fast = True)[0]
    assert np.greater(alpha, p_value)
