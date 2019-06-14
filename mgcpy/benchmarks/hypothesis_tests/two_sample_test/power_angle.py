import numpy as np
import math
from mgcpy.hypothesis_tests.transforms import k_sample_transform
import scipy.io
import os


def power_given_data(independence_test, simulation_type, num_samples=100, num_dimensions=1, theta=0, repeats=1000, alpha=.05, additional_params={}):
    # test statistics under the null, used to estimate the cutoff value under the null distribution
    test_stats_null = np.zeros(repeats)
    # test statistic under the alternative
    test_stats_alternative = np.zeros(repeats)
    p_values = np.zeros(repeats)
    # absolute path to the benchmark directory
    file_name_prefix = 'matlabsim_{}'.format(
                simulation_type)
    all_matrix_X = scipy.io.loadmat(file_name_prefix + '_X_final.mat')['x_mat']
    all_matrix_Y = scipy.io.loadmat(file_name_prefix + '_Y_final.mat')['y_mat']
    theta = math.radians(theta)
    a = [[0 for x in range(2)] for y in range(2)]
    a[0][0] = math.cos(theta)
    a[0][1] = math.sin(theta)*(-1)
    a[1][0] = math.sin(theta)
    a[1][1] = math.cos(theta)
    a = np.asarray(a)
    for rep in range(repeats):
        matrix_X = all_matrix_X[:, :, rep]
        matrix_Y = all_matrix_Y[:, :, rep]
        data_matrix = k_sample_transform(matrix_X.T, matrix_Y.T)[0]
        r_matrix = np.dot(a, data_matrix)
        long_matrix = k_sample_transform(data_matrix.T, r_matrix.T)[0]
        label_matrix = k_sample_transform(data_matrix.T, r_matrix.T)[1]
        mat_X = long_matrix
        mat_Y = label_matrix
        # permutation test
        if additional_params and additional_params["is_fast"]:
            p_values[rep], _ = independence_test.p_value(mat_X, mat_Y, **additional_params)
        else:
            permuted_Y = np.random.permutation(mat_Y)
            test_stats_null[rep], _ = independence_test.test_statistic(
                mat_X, permuted_Y, **additional_params)
            test_stats_alternative[rep], _ = independence_test.test_statistic(
                mat_X, mat_Y, **additional_params)
        '''
        # if the test is pearson, use absolute value of the test statistic
        # so the more extreme test statistic is still in a one-sided interval
        if independence_test.get_name() == 'pearson':
            test_stats_null[rep] = abs(test_stats_null[rep])
            test_stats_alternative[rep] = abs(test_stats_alternative[rep])
        '''

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