"""
    MGCPY Sample Statistic
"""

import math
import numpy as np
import scipy.ndimage
import scipy.stats

from mgcpy.independence_tests.mgc.local_correlation import local_correlations


def threshold_local_correlations(local_correlation_matrix, sample_size):
    """
    Finds a connected region of significance in the local correlation map by thresholding

    :param local_correlation_matrix: all local correlations within [-1,1]
    :type local_covariance_matrix: 2D numpy.array

    :param sample_size: the sample size of original data
                        (which may not equal m or n in case of repeating data).
    :type sample_size: int

    :return: a binary matrix of size m and n, with 1's indicating the significant region.
    :rtype: 2D numpy.array
    """
    m, n = local_correlation_matrix.shape

    # parametric threshold
    # a threshold is estimated based on the normal distribution approximation (from Szekely2013)
    significant_percentile = 1 - (0.02 / sample_size)  # percentile to consider as significant
    threshold = sample_size * (sample_size - 3) / 4 - 1 / 2  # beta approximation
    threshold = scipy.stats.beta.ppf(significant_percentile, threshold, threshold) * 2 - 1

    # non-paratemetric threshold
    # set option = 1 to compute a non-parametric and data-adaptive threshold
    # (using the negative local correlation)
    option = 0
    if option == 1:
        np_threshold = local_correlation_matrix

        # all negative correlations
        np_threshold = np_threshold[np_threshold < 0]

        # the standard deviation of negative correlations
        np_threshold = 5 * np.sqrt(np.sum(np_threshold ** 2) / len(np_threshold))

        # use the max of paratemetric and non-parametric thresholds
        if not math.isnan(np_threshold) and np_threshold > threshold:
            threshold = np_threshold

    # take the max of threshold and local correlation at the maximal scale
    threshold = max(threshold, local_correlation_matrix[m - 1][n - 1])

    # find the largest connected component of significant correlations
    significant_connected_region = local_correlation_matrix > threshold
    if np.sum(significant_connected_region) > 0:
        significant_connected_region, _ = scipy.ndimage.measurements.label(
            significant_connected_region)
        _, label_counts = np.unique(significant_connected_region, return_counts=True)
        max_label = np.argmax(label_counts) + 1
        significant_connected_region = significant_connected_region == max_label
    else:
        significant_connected_region = np.array([[False]])

    return significant_connected_region


def smooth_significant_local_correlations(significant_connected_region, local_correlation_matrix):
    """
    Finds the smoothed maximal within the significant region R:
    - If area of R is too small it returns the last local correlation
    - Otherwise, returns the maximum within significant_connected_region.

    :param significant_connected_region: a binary matrix of size m and n, with 1's indicating the significant region.
    :type significant_connected_region: 2D numpy.array

    :param local_correlation_matrix: all local correlations within [-1,1]
    :type local_covariance_matrix: 2D numpy.array

    :return: A ``dict`` with the following keys:
    :rtype: dict
        - :mgc_statistic: the sample MGC statistic within [-1, 1]
        - :optimal_scale: the estimated optimal scale as an [x, y] pair.
    """
    m, n = local_correlation_matrix.shape

    # default sample mgc to local corr at max scale
    mgc_statistic = local_correlation_matrix[m - 1][n - 1]
    optimal_scale = [m, n]  # default the optimal scale to max scale

    if np.linalg.norm(significant_connected_region) != 0:

        # proceed only when the connected region's area is sufficiently large
        if np.sum(significant_connected_region) >= 2 * min(m, n):
            max_local_correlation = np.max(local_correlation_matrix[significant_connected_region])

            # find all scales within significant_connected_region that maximize the local correlation
            max_local_correlation_index = np.where((local_correlation_matrix >= max_local_correlation) & significant_connected_region)

            # adding 1s to match R indexing
            k = max_local_correlation_index[0][0] + 1
            l = max_local_correlation_index[1][0] + 1

            if max_local_correlation >= mgc_statistic:
                mgc_statistic = max_local_correlation
                optimal_scale = [k, l]

    return {"mgc_statistic": mgc_statistic,
            "optimal_scale": optimal_scale}


def mgc_sample(matrix_A, matrix_B, base_global_correlation="mgc"):
    """
    Computes the MGC measure between two datasets.
    - It first computes all the local correlations
    - Then, it returns the maximal statistic among all local correlations based on thresholding.

    :param matrix_A: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type matrix_A: 2D numpy.array

    :param matrix_B: is interpreted as either:
        - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a [n*d] data matrix, a square matrix with n samples in d dimensions
    :type matrix_B: 2D numpy.array

    :param base_global_correlation: specifies which global correlation to build up-on,
                                    including 'mgc','dcor','mantel', and 'rank'.
                                    Defaults to mgc.
    :type base_global_correlation: str

    :return: A ``dict`` with the following keys:
    :rtype: dict
        - :mgc_statistic: the sample MGC statistic within [-1, 1]
        - :correlation_matrix: a 2D matrix of all local correlations within [-1,1]
        - :optimal_scale: the estimated optimal scale as an [x, y] pair.

    **Example:**
    >>> import numpy as np
    >>> from mgcpy.mgc.sample_statistic import mgc_sample

    >>> X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    >>> Y = np.array([[30, 20, 10], [5, 10, 20], [8, 16, 32]])
    >>> result = mgc_sample(X, Y)
    """
    # compute all local correlations
    local_correlation_matrix = local_correlations(matrix_A, matrix_B, base_global_correlation)[
                                                  "local_correlation_matrix"]
    m, n = local_correlation_matrix.shape
    if m == 1 or n == 1:
        mgc_statistic = local_correlation_matrix[m - 1][n - 1]
        optimal_scale = m * n
    else:
        sample_size = len(matrix_A) - 1  # sample size minus 1

        # find a connected region of significant local correlations, by thresholding
        significant_connected_region = threshold_local_correlations(
            local_correlation_matrix, sample_size)

        # find the maximum within the significant region
        result = smooth_significant_local_correlations(
            significant_connected_region, local_correlation_matrix)
        mgc_statistic, optimal_scale = result["mgc_statistic"], result["optimal_scale"]

    return {"mgc_statistic": mgc_statistic,
            "local_correlation_matrix": local_correlation_matrix,
            "optimal_scale": optimal_scale}
