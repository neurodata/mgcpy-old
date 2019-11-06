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
        self.name = "Independent AR(1)"
        self.filename = "indep_ar1"

    def simulate(self, n, phi=0.5, sigma2=1.0):
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
        for t in range(1, n):
            X[t] = phi * X[t - 1] + epsilons[t]
            Y[t] = phi * Y[t - 1] + etas[t]

        return X, Y


class CorrelatedAR1(TimeSeriesProcess):
    def __init__(self):
        self.name = "Correlated AR(1)"
        self.filename = "corr_ar1"

    def simulate(self, n, phi_1=0.5, phi_3=0.0, sigma2=1.0):
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
        for t in range(3, n):
            X[t] = phi_1 * Y[t - 1] + phi_3 * Y[t - 3] + epsilons[t]
            Y[t] = phi_1 * X[t - 1] + phi_3 * X[t - 3] + etas[t]

        return X, Y


class Nonlinear(TimeSeriesProcess):
    def __init__(self, lag=1):
        self.lag = int(lag)
        self.name = "Nonlinearly Related Lag %d" % self.lag
        self.filename = "nonlin_lag%d" % self.lag

    def simulate(self, n, sigma2=1.0):
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
            X[t] = epsilons[t] * Y[t - self.lag]
            Y[t] = etas[t]

        return X, Y


class NonlinearDependence(TimeSeriesProcess):
    """
    Nonlinear dependence setting proposed by Gretton et al 2014.
    Parameter defaults are set to replicate results from paper.
    """

    def __init__(self):
        self.name = "Nonlinear Dependence"
        self.filename = "nonlin_dependence"

    def simulate(self, n, extinction_rate, radius=1, alpha=0.2):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :param extinction_rate: Rate of extinction \in [0, 1]
        :type extinction_rate: float

        :param radius: radius (default: 1)
        :type radius: float

        :param alpha: autoregressive component (default: 0.2)
        :type alpha: float

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """
        if (extinction_rate < 0) or (extinction_rate > 1):
            msg = "extinction_rate must be between 0 and 1, inclusive."
            raise ValueError(msg)

        # Innovations.
        eta, epsilon = self._sample_innovations(extinction_rate, radius)

        X = np.zeros(n)
        Y = np.zeros(n)
        X[0] = epsilon
        Y[0] = eta

        for t in range(1, n):
            eta, epsilon = self._sample_innovations(extinction_rate, radius)
            X[t] = alpha * X[t - 1] + epsilon
            Y[t] = alpha * Y[t - 1] + eta

        return X, Y

    def _sample_innovations(self, extinction_rate, radius):
        while True:
            eta = np.random.normal(0, 1)
            epsilon = np.random.normal(0, 1)
            d = np.random.uniform()

            if (eta ** 2 + epsilon ** 2 > radius ** 2) or (d > extinction_rate):
                return eta, epsilon


class EconometricProcess(TimeSeriesProcess):
    """
    Econometric process that models market volatility.
    Proposed by Gretton et al. 2016
    """

    def __init__(self, shift = 1, scale = 0.45):
        self.name = "Econometric Process"
        self.filename = "econometric_proc"
        self.shift = shift
        self.scale = scale

    def simulate(self, n):
        """
        Method to simulate observations of the process.

        :param n: sample_size
        :type n: integer

        :return: returns a list of two items, that contains:

            - :X: a ``[n*1]`` data matrix, a matrix with n samples
            - :Y: a ``[n*1]`` data matrix, a matrix with n samples
        :rtype: list
        """
        # Innovations.
        epsilons = np.random.normal(0, 1, 2)

        X = np.zeros(n)
        Y = np.zeros(n)
        X[0] = epsilons[0]
        Y[0] = epsilons[1]

        for t in range(1, n):
            epsilons = np.random.normal(0, 1, 2)
            sigma = self.shift + self.scale * (X[t - 1] ** 2 + Y[t - 1] ** 2)
            X[t] = epsilons[0] * sigma
            Y[t] = epsilons[1] * sigma

        return X, Y


class DynamicProcess(TimeSeriesProcess):
    """
    Dynamic process.
    Proposed by Gretton et al. 2016
    """

    def __init__(self):
        self.name = "Dynamic Process"
        self.filename = "dynamic_proc"

    def simulate(self, n):
        # Setting parameters
        f_1 = 4
        f_2 = 20
        T_s = 1 / 100
        C = 0.4

        epsilons = np.random.normal(0, 1, 2)

        X = np.zeros(n)
        Y = np.zeros(n)

        phi_1 = np.zeros(n)
        phi_2 = np.zeros(n)
        phi_1[0] = 0.1 * epsilons[0] + 2 * np.pi * f_1 * T_s
        phi_2[0] = 0.1 * epsilons[1] + 2 * np.pi * f_2 * T_s

        X[0] = np.cos(phi_1[0])
        Y[0] = (2 + C * np.sin(phi_1[0])) * np.cos(phi_2[0])

        for t in range(1, n):
            epsilons = np.random.normal(0, 1, 2)
            phi_1[t] = phi_1[t - 1] + 0.1 * epsilons[0] + 2 * np.pi * f_1 * T_s
            phi_2[t] = phi_2[t - 1] + 0.1 * epsilons[0] + 2 * np.pi * f_2 * T_s

            X[t] = np.cos(phi_1[t])
            Y[t] = (2 + C * np.sin(phi_1[t])) * np.cos(phi_2[t])

        return X, Y
