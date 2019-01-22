"""
    **MGC's Sample Statistic Module**
"""

import numpy as np
import scipy.ndimage
import scipy.stats


def threshold_local_correlations(local_correlation_matrix, sample_size):
    """
    Finds a connected region of significance in the local correlation map by thresholding

    :param local_correlation_matrix: all local correlations within ``[-1,1]``
    :type local_covariance_matrix: 2D numpy.array

    :param sample_size: the sample size of original data
                        (which may not equal ``m`` or ``n`` in case of repeating data).
    :type sample_size: integer

    :return: a binary matrix of size ``m`` and ``n``, with 1's indicating the significant region.
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
    # option = 0
    # if option == 1:
    #     np_threshold = local_correlation_matrix
    #
    #     # all negative correlations
    #     np_threshold = np_threshold[np_threshold < 0]
    #
    #     # the standard deviation of negative correlations
    #     np_threshold = 5 * np.sqrt(np.sum(np_threshold ** 2) / len(np_threshold))
    #
    #     # use the max of paratemetric and non-parametric thresholds
    #     if not np.isnan(np_threshold) and np_threshold > threshold:
    #         threshold = np_threshold

    # take the max of threshold and local correlation at the maximal scale
    threshold = max(threshold, local_correlation_matrix[m - 1][n - 1])

    # find the largest connected component of significant correlations
    significant_connected_region = local_correlation_matrix > threshold
    if np.sum(significant_connected_region) > 0:
        significant_connected_region, _ = scipy.ndimage.measurements.label(
            significant_connected_region)
        _, label_counts = np.unique(significant_connected_region, return_counts=True)
        # skip the first element in label_counts, as it is count(zeros)
        max_label = np.argmax(label_counts[1:]) + 1
        significant_connected_region = significant_connected_region == max_label
    else:
        significant_connected_region = np.array([[False]])

    return significant_connected_region


def smooth_significant_local_correlations(significant_connected_region, local_correlation_matrix):
    """
    Finds the smoothed maximal within the significant region R:

        - If area of R is too small it returns the last local correlation
        - Otherwise, returns the maximum within significant_connected_region.

    :param significant_connected_region: a binary matrix of size ``m`` and ``n``, with 1's indicating the significant region.
    :type significant_connected_region: 2D numpy.array

    :param local_correlation_matrix: all local correlations within ``[-1,1]``
    :type local_covariance_matrix: 2D numpy.array

    :return: A ``dict`` with the following keys:

            - :mgc_statistic: the sample MGC statistic within ``[-1, 1]``
            - :optimal_scale: the estimated optimal scale as an ``[x, y]`` pair.
    :rtype: dictionary
    """

    m, n = local_correlation_matrix.shape

    # default sample mgc to local corr at max scale
    mgc_statistic = local_correlation_matrix[m - 1][n - 1]
    optimal_scale = [m, n]  # default the optimal scale to max scale

    if np.linalg.norm(significant_connected_region) != 0:

        # proceed only when the connected region's area is sufficiently large
        # if np.sum(significant_connected_region) >= min(m, n):
        # if np.sum(significant_connected_region) >= 2 * min(m, n):
        if np.sum(significant_connected_region) >= np.ceil(0.02*max(m,n))*min(m,n):
            max_local_correlation = np.max(local_correlation_matrix[significant_connected_region])

            # find all scales within significant_connected_region that maximize the local correlation
            max_local_correlation_indices = np.where(
                (local_correlation_matrix >= max_local_correlation) & significant_connected_region)

            if max_local_correlation >= mgc_statistic:
                mgc_statistic = max_local_correlation

                k, l = max_local_correlation_indices
                one_d_indices = k * n + l  # 2D to 1D indexing
                k = np.max(one_d_indices) // n
                l = np.max(one_d_indices) % n
                optimal_scale = [k+1, l+1]  # adding 1s to match R indexing

    return {"mgc_statistic": mgc_statistic,
            "optimal_scale": optimal_scale}
