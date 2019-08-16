"""
    **Common Functions used in Fast Dcorr and Fast MGC**
"""

import math
from statistics import mean, stdev

import numpy as np
from scipy.stats import norm


def _sample_atrr(matrix_X, sub_samples):
    """
    Computes the number of samples, sub samples, and the observed test statistics

    :param matrix_X: is interpreted as a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
    :type matrix_X: 2D numpy.array

    :param sub_samples: the number of sub samples that will be used in the calculation
    :type sub_samples: integer

    :return: returns a list of two items, that contains:

        - :num_samples: the number of samples that will be used when calculating the fast test statistic
        - :sub_samples: the number of sub_samples that will be used in the calculation
    :rtype: list
    """
    total_samples = matrix_X.shape[0]
    num_samples = total_samples // sub_samples

    # if full data size (total_samples) is not more than 4 times of sub_samples, split to 4 samples
    # too few samples will fail the normal approximation and cause the test to be invalid

    if total_samples < 4 * sub_samples:
        sub_samples = total_samples // 4
        num_samples = 4

    return num_samples, sub_samples


def _fast_pvalue(test_statistic, test_statistic_metadata):
    """
    Computes the number of samples, sub samples, and the observed test statistics

    :param test_statistic: the test statistic that will be used when calculating the p value
    :type test_statistic: 2D numpy.array

    :param test_statistic_metadata: a ``dict`` containing the sigma and mu to be used in calculation
    :type test_statistic_metadata: dict

    :return: calculated p value of the test statistic
    :rtype: float
    """
    sigma = test_statistic_metadata["sigma"]
    mu = test_statistic_metadata["mu"]

    # compute p value
    p_value = 1 - norm.cdf(test_statistic, mu, sigma)

    return p_value


def _sub_sample(matrix_X, matrix_Y, test_statistic, num_samples, sub_samples, which_test):
    """
    Sub samples the data and calculates the sub sampled test statistic

    :param matrix_X: is interpreted as a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
    :type matrix_X: 2D numpy.array

    :param matrix_Y: is interpreted as a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
    :type matrix_X: 2D numpy.array

    :param test_statistic: the test statistic that will be used when calculating the p value
    :type test_statistic: 2D numpy.array

    :param num_samples: total number of samples
    :type num_samples: integer

    :param sub_samples: total number of sub samples
    :type sub_samples: integer

    :param which_test: the type of global correlation to use, can be 'unbiased', 'biased' 'mantel', 'mgc'
    :type which_test: string

    :return: calculated test statistic by sub sampling
    :rtype: float
    """
    # the observed statistics by subsampling
    test_statistic_sub_sampling = np.zeros(num_samples)

    if which_test == 'mgc':
        permuted_Y = np.random.permutation(matrix_Y)
    else:
        permuted_Y = matrix_Y
    for i in range(num_samples):
        sub_matrix_X = matrix_X[(sub_samples*i):sub_samples*(i+1), :]
        sub_matrix_Y = permuted_Y[(sub_samples*i):sub_samples*(i+1), :]

        test_statistic_sub_sampling[i], _ = test_statistic(sub_matrix_X, sub_matrix_Y)

    return test_statistic_sub_sampling


def _approx_null_dist(num_samples, test_statistic_sub_sampling, which_test):
    """
    Approximates the null distribution of the p value calculation

    :param test_statistic: the test statistic that will be used when calculating the p value
    :type test_statistic: 2D numpy.array

    :param test_statistic_metadata: a ``dict`` containing the sigma and mu to be used in calculation
    :type test_statistic_metadata: integer

    :return: calculated p value of the test statistic
    :rtype: float
    """
    if which_test == 'mgc':
        sigma = stdev(test_statistic_sub_sampling) / num_samples
        mu = max(0, mean(test_statistic_sub_sampling))
    else:
        sigma = stdev(test_statistic_sub_sampling) / math.sqrt(num_samples)
        mu = 0

    return sigma, mu
