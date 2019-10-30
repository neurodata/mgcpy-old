import numpy as np
from abc import ABC, abstractmethod

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

    def simulate(self, n, phi_1 = 0.5, phi_3 = 0.0, sigma2 = 1.0):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :param phi_1: AR(1) coefficient.
        :type phi_1: float

        :param phi_3: AR(3) coefficient.
        :type phi_3: float

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
        epsilons = np.random.normal(0.0, sigma2, n)
        etas = np.random.normal(0.0, sigma2, n)

        X = np.zeros(n)
        Y = np.zeros(n)
        for s in range(3):
            X[s] = epsilons[s]
            Y[s] = etas[s]
    
        # AR(1) process, unless phi_3 is specified.
        for t in range(3,n):
            X[t] = phi_1*Y[t-1] + phi_3*Y[t-3] + epsilons[t]
            Y[t] = phi_1*X[t-1] + phi_3*X[t-3] + etas[t]

        return X, Y

class Nonlinear(TimeSeriesProcess):
    def __init__(self, lag = 1):
        self.lag = int(lag)
        self.name = 'Nonlinearly Related Lag %d' % self.lag
        self.filename = 'nonlin_lag%d' % self.lag

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
        for s in range(self.lag):
            X[s] = epsilons[s]
            Y[s] = etas[s]

        for t in range(self.lag, n):
            X[t] = epsilons[t]*Y[t-self.lag]
            Y[t] = etas[t]

        return X, Y
