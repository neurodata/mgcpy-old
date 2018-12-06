"""
    **Faster version of MGC**
"""

import numpy as np
from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.mgc.threshold_smooth import (smooth_significant_local_correlations,
                                                           threshold_local_correlations)


def faster_mgc(matrix_X, matrix_Y, sub_samples=100, null_only=True, alpha=0.01):
    '''
    MGC test statistic computation and permutation test by fast subsampling.
    Note that trivial amount of noise is added to matrix_X and matrix_Y,
    to break possible ties in data for MGC.

    :param matrix_X: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*d]`` data matrix, a square matrix with n samples in d dimensions
    :type matrix_X: 2D numpy.array

    :param matrix_Y: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*d]`` data matrix, a square matrix with n samples in d dimensions
    :type matrix_Y: 2D numpy.array

    :param sub_samples: specifies the number of subsamples.
                        generally total_samples/sub_samples should be more than 4,
                        and ``sub_samples`` should be large than 30.
    :type sub_samples: integer

    :param null_only: specifies if subsampling is to be used for estimating the null only OR to compute the observed statistic as well

        - *True:* uses subsampled statistics for estimating the null only and computes the observed statistic by full data,
                this runs in ``O(total_samples^2 + sub_samples * total_samples)``
        - *False:* uses subsampled statistics for estimating the null and also computes the observed statistic by subsampling,
                 this runs in ``O(sub_samples*total_samples)``
    :type null_only: boolean

    :param alpha: specifies the type 1 error level.
                  this is is used to derive the confidence interval and estimate required sample size to achieve power 1.
    :type alpha: float

    :return: a ``dict`` of results with the following keys:

        - :p_value: P-value of MGC
        - :test_statistic: the sample MGC statistic within ``[-1, 1]``
        - :local_correlation_matrix: a 2D matrix of all local correlations within ``[-1,1]``
        - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
        - :confidence_interval: a ``[1*2]`` matrix representing the confidence_interval for the local correlation with 1-alpha confidence.
        - :required_size: the required estimated sample size to have power 1 at level alpha
    :rtype: dictionary
    '''
    total_samples = matrix_Y.shape[0]
    num_samples = total_samples // sub_samples

    # if full data size (total_samples) is not more than 4 times of sub_samples, split to 4 samples
    # too few samples will fail the normal approximation and cause the test to be invalid

    if total_samples < 4 * sub_samples:
        sub_samples = total_samples // 4
        num_samples = 4

    # the observed statistics by subsampling
    test_statistic_sub_sampling = np.zeros((1, num_samples))

    # add trivial noise to break any ties
    matrix_X += 1e-10 * np.random.uniform(size=matrix_X.shape)
    matrix_Y += 1e-10 * np.random.uniform(size=matrix_Y.shape)

    # the local correlations by subsampling
    local_correlation_matrix_sub_sampling = np.zeros((sub_samples, sub_samples, num_samples))

    # create MGC object
    mgc = MGC()

    # subsampling computation
    for i in range(num_samples):
        sub_matrix_X = matrix_X[(sub_samples*i):(sub_samples*(i+1)-1)]
        sub_matrix_Y = matrix_Y[(sub_samples*i):(sub_samples*(i+1)-1)]

        mgc_statistic, test_statistic_metadata = mgc.test_statistic(sub_matrix_X, sub_matrix_Y)
        test_statistic_sub_sampling[i], local_correlation_matrix_sub_sampling[:, :, i] = \
            mgc_statistic, test_statistic_metadata["local_correlation_matrix"]

    local_correlation_matrix = np.mean(local_correlation_matrix_sub_sampling, axis=2)
    sigma = np.std(test_statistic_sub_sampling) / np.sqrt(num_samples)

    sample_size = len(matrix_X) - 1  # sample size minus 1

    # find a connected region of significant local correlations, by thresholding
    significant_connected_region = threshold_local_correlations(
        local_correlation_matrix, sample_size)

    # find the maximum within the significant region
    result = smooth_significant_local_correlations(
        significant_connected_region, local_correlation_matrix)
    mgc_statistic, optimal_scale = result["mgc_statistic"], result["optimal_scale"]
