import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from joblib import Parallel, delayed

# Time series simulation classes.

class TimeSeriesProcess(ABC):
    """
    TimeSeriesProcess abstract class

    Specifies the generic interface that must be implemented by
    all the processes.
    """

    def __init__(self):
        self.name = None
        self.filename = None

    @abstractmethod
    def simulate(self, n):
        """
        Abstract method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """

        pass

class IndependentAR1(TimeSeriesProcess):
    def __init__(self):
        self.name = 'Independent AR(1)'
        self.filename = 'indep_ar1'

    def simulate(self, n, phi = 0.5, sigma2 = 1.0):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :param phi: AR coefficient.
        :type phi: float

        :param sigma2: Variance of noise.
        :type sigma2: float

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """
        # X_t and Y_t are univarite AR(1) with phi = 0.5 for both.
        # Innovations follow N(0, sigma2).

        # Innovations.
        epsilons = np.random.normal(0.0, sigma2, n)
        etas = np.random.normal(0.0, sigma2, n)

        X = np.zeros(n)
        Y = np.zeros(n)
        X[0] = epsilons[0]
        Y[0] = etas[0]

        # AR(1) process.
        for t in range(1,n):
            X[t] = phi*X[t-1] + epsilons[t]
            Y[t] = phi*Y[t-1] + etas[t]

        return X, Y

class CorrelatedAR1(TimeSeriesProcess):
    def __init__(self):
        self.name = 'Correlated AR(1)'
        self.filename = 'corr_ar1'

    def simulate(self, n, phi = 0.5, sigma2 = 1.0):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :param phi: AR coefficient.
        :type phi: float

        :param sigma2: Variance of noise.
        :type sigma2: float

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """
        # X_t and Y_t are together a bivarite AR(1) with Phi = [0 0.5; 0.5 0].
        # Innovations follow N(0, sigma2).

        # Innovations.
        epsilons = np.random.uniform(0.0, 1.0, n)
        etas = np.random.normal(0.0, 1.0, n)

        X = np.zeros(n)
        Y = np.zeros(n)
        X[0] = epsilons[0]
        Y[0] = etas[0]

        for t in range(1,n):
            X[t] = phi*Y[t-1] + epsilons[t]
            Y[t] = phi*X[t-1] + etas[t]

        return X, Y

class NonlinearLag1(TimeSeriesProcess):
    def __init__(self):
        self.name = 'Nonlinearly Related Lag 1'
        self.filename = 'nonlin_lag1'

    def simulate(self, n, sigma2 = 1.0):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :param sigma2: Variance of noise.
        :type sigma2: float

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """
        # X_t and Y_t are together a bivarite nonlinear process.
        # Innovations follow N(0, sigma2).

        # Innovations.
        epsilons = np.random.normal(0.0, sigma2, n)
        etas = np.random.normal(0.0, sigma2, n)

        X = np.zeros(n)
        Y = np.zeros(n)
        X[0] = epsilons[0]
        Y[0] = etas[0]

        for t in range(1, n):
            X[t] = epsilons[t]*Y[t-1]
            Y[t] = etas[t]

        return X, Y

# Power computation functions.

def power_curve(tests, process, num_sims, alpha, sample_size, verbose = False):
    """
    Method to generate power curves for time series.

    :param tests: An array-like object containing TimeSeriesIndependenceTest objects.
    :type tests: 1-D array-like

    :param process: A TimeSeriesProcess object for which to profile the test.
    :type process: TimeSeriesProcess

    :param num_sims: number of simulation at each sample size.
    :type num_sims: integer

    :param alpha: significance level.
    :type alpha: float

    :param verbose: whether to display output.
    :type verbose: boolean

    :param sample_sizes: range of sample sizes for which to estimate power.
    :type sample_sizes: 1-D array-like
    """
    # Store simulate processes.
    n_full = sample_sizes[len(sample_sizes) - 1]
    X_full = np.zeros((n_full, num_sims))
    Y_full = np.zeros((n_full, num_sims))
    for s in range(num_sims):
        X_full[:, s], Y_full[:, s] = process.simulate(n_full)

    for test in tests:
        powers = np.zeros(len(sample_sizes))
        for i in range(len(sample_sizes)):
            n = sample_sizes[i]
            if verbose: print("Estimating power at sample size: %d" % n)
            powers[i] = compute_power(test, X_full, Y_full, num_sims, alpha, n)
        test['powers'] = powers

    # Display.
    plot_power(tests, sample_sizes, alpha, process)

def compute_power(test, X_full, Y_full, num_sims, alpha, n):
    """
    Helper method estimate power of a test on a given simulation.

    :param test: Test to profile, either DCorrX or MGCX.
    :type test: TimeSeriesIndependenceTest

    :param X_full: An ``[n*num_sims]`` data matrix where ``n`` is the highest sample size.
    :type X_full: 2D ``numpy.array``

    :param Y_full: An ``[n*num_sims]`` data matrix where ``n`` is the highest sample size.
    :type Y_full: 2D ``numpy.array``

    :param num_sims: number of simulation at each sample size.
    :type num_sims: integer

    :param alpha: significance level.
    :type alpha: float

    :param n: sample size.
    :type n: integer

    :return: returns the estimated power.
    :rtype: float
    """
    num_rejects = 0.0

    def worker(s):
        X = X_full[range(n), s]
        Y = Y_full[range(n), s]

        p_value, _ = test['object'].p_value(X, Y, is_fast = test['is_fast'])
        if p_value <= alpha:
            return 1

        return 0

    num_rejects = np.sum(Parallel(n_jobs=-2, verbose=10)(delayed(worker)(s) for s in range(num_sims)))

    return num_rejects / num_sims

def plot_power(tests, sample_sizes, alpha, process):
    """
    Helper method to generate power curves for time series.

    :param tests: An array-like object containing TimeSeriesIndependenceTest objects.
    :type tests: 1-D array-like

    :param sample_sizes: range of sample sizes for which to estimate power.
    :type sample_sizes: 1-D array-like

    :param alpha: significance level.
    :type alpha: float

    :param process: A TimeSeriesProcess object for which to profile the test.
    :type process: TimeSeriesProcess
    """
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots()
    plt.title(process.name)
    plt.xlabel("n")
    plt.ylabel("Rejection Probability")
    plt.ylim((-0.05, 1.05))

    for test in tests:
        plt.plot(sample_sizes, test['powers'], linestyle = '-', color = test['color'])
    ax.legend([test['name'] for test in tests], loc = 'upper left', prop={'size': 12})

    ax.axhline(y = alpha, color = 'black', linestyle = '--')
    plt.show()
