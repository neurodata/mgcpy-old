import numpy as np
import math
from mgcpy.independence_tests.dcorr import DCorr
from transform_matrices import Transform_Matrices
from scipy.ndimage import rotate

def power(independence_test, sample_generator, num_samples=100, theta=0, num_dimensions=2, noise=0.0, repeats=1000, alpha=.05):
    '''
    Estimate power
    :param independence_test: an object whose class inherits from the Independence_Test abstract class
    :type: Object(Independence_Test)
    :param sample_generator: a function used to generate simulation from simulations.py with parameters given by the following arguments
        - num_samples: default to 100
        - num_dimensions: default to 1
        - noise: default to 0
    :type: function
    :param num_samples: the number of samples generated by the simulation
    :type: int
    :param num_dimensions: the number of dimensions of the samples generated by the simulation
    :type: int
    :param noise: the noise used in simulation
    :type: float
    :param repeats: the number of times we generate new samples to estimate the null/alternative distribution
    :type: int
    :param alpha: the type I error level
    :type: float
    :return empirical_power: the estimated power
    :type: float
    '''
    def compute_distance_matrix(data_matrix_X, data_matrix_Y):
        # obtain the pairwise distance matrix for X and Y
        dist_mtx_X = squareform(pdist(data_matrix_X, metric='euclidean'))
        dist_mtx_Y = squareform(pdist(data_matrix_Y, metric='euclidean'))
        return (dist_mtx_X, dist_mtx_Y)

    # test statistics under the null, used to estimate the cutoff value under the null distribution
    test_stats_null = np.zeros(repeats)
    # test statistic under the alternative
    test_stats_alternative = np.zeros(repeats)
    for rep in range(repeats):
        # generate new samples for each iteration
        X, Y = sample_generator(num_samples, num_dimensions, noise)
        data_matrix_X=Transform_Matrices(X,Y)[0]
        data_matrix_Y=Transform_Matrices(X,Y)[1]
        data_matrix_Y=data_matrix_Y[:,np.newaxis]
        data_matrix_X=rotate(data_matrix_X,theta)
       # data_matrix_X,data_matrix_Y=compute_distance_matrix(X,Y)
        # permutation test
        permuted_y = np.random.permutation(data_matrix_Y)
        test_stats_null[rep], _ = independence_test.test_statistic(data_matrix_X=data_matrix_X, data_matrix_Y=permuted_y)
        test_stats_alternative[rep], _ = independence_test.test_statistic(data_matrix_X=data_matrix_X, data_matrix_Y=data_matrix_Y)

    # the cutoff is determined so that 1-alpha of the test statistics under the null distribution
    # is less than the cutoff
    cutoff = np.sort(test_stats_null)[math.ceil(repeats*(1-alpha))]
    # the proportion of test statistics under the alternative which is no less than the cutoff (in which case
    # the null is rejected) is the empirical power
    empirical_power = np.where(test_stats_alternative >= cutoff)[0].shape[0] / repeats
    return empirical_power