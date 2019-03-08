import numpy as np
import math
from mgcpy.hypothesis_tests.transforms import k_sample_transform
import scipy.io
import os


def power_given_data(base_path, independence_test, simulation_type, num_samples, repeats=1000, alpha=.05, additional_params={}):
    # test statistics under the null, used to estimate the cutoff value under the null distribution
    test_stats_null = np.zeros(repeats)
    # test statistic under the alternative
    test_stats_alternative = np.zeros(repeats)

    # direct p values on permutation (now, only for fast_mgc)
    p_values = np.zeros(repeats)

    # absolute path to the benchmark directory
    file_name_prefix = os.path.join(base_path, 'sample_data_power_sample_sizes/type_{}_size_{}'.format(simulation_type, num_samples))

    all_matrix_X = scipy.io.loadmat(file_name_prefix + '_X.mat')['x_mtx'][..., np.newaxis]
    all_matrix_Y = scipy.io.loadmat(file_name_prefix + '_Y.mat')['y_mtx'][..., np.newaxis]

    # rotation transform matrix
    c, s = np.cos(math.radians(60)), np.sin(math.radians(60))
    rotation_matrix = np.array([[c, s], [-s, c]])

    for rep in range(repeats):
        matrix_X = all_matrix_X[rep, :, :]
        matrix_Y = all_matrix_Y[rep, :, :]

        # apply two sample transform
        data_matrix = np.concatenate([matrix_X, matrix_Y], axis=1)
        rotated_data_matrix = np.dot(rotation_matrix, data_matrix.T).T
        matrix_U, matrix_V = k_sample_transform(data_matrix, rotated_data_matrix)

        # permutation test
        if additional_params and additional_params["is_fast"]:
            p_values[rep], _ = independence_test.p_value(matrix_U, matrix_V, **additional_params)
        else:
            permuted_V = np.random.permutation(matrix_V)
            test_stats_null[rep], _ = independence_test.test_statistic(
                matrix_U, permuted_V, **additional_params)
            test_stats_alternative[rep], _ = independence_test.test_statistic(
                matrix_U, matrix_V, **additional_params)

        # if the test is pearson, use absolute value of the test statistic
        # so the more extreme test statistic is still in a one-sided interval
        if independence_test.get_name() == 'pearson':
            test_stats_null[rep] = abs(test_stats_null[rep])
            test_stats_alternative[rep] = abs(test_stats_alternative[rep])

    if additional_params and additional_params["is_fast"]:
        empirical_power = np.where(p_values <= alpha)[0].shape[0] / repeats
    else:
        # the cutoff is determined so that 1-alpha of the test statistics under the null distribution
        # is less than the cutoff
        cutoff = np.sort(test_stats_null)[math.ceil(repeats*(1-alpha))]

        # the proportion of test statistics under the alternative which is no less than the cutoff (in which case
        # the null is rejected) is the empirical power
        empirical_power = np.where(test_stats_alternative >= cutoff)[0].shape[0] / repeats

    return empirical_power
