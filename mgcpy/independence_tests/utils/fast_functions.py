import numpy as np
from scipy.stats import norm


def _sample_atrr(matrix_Y, sub_samples):
    total_samples = matrix_Y.shape[0]
    num_samples = total_samples // sub_samples

    # if full data size (total_samples) is not more than 4 times of sub_samples, split to 4 samples
    # too few samples will fail the normal approximation and cause the test to be invalid

    if total_samples < 4 * sub_samples:
        sub_samples = total_samples // 4
        num_samples = 4

    # the observed statistics by subsampling
    test_statistic_sub_sampling = np.zeros(num_samples)

    return num_samples, sub_samples, test_statistic_sub_sampling


def _fast_pvalue(test_statistic, test_statistic_metadata):
    sigma = test_statistic_metadata["sigma"]
    mu = test_statistic_metadata["mu"]

    # compute p value
    p_value = 1 - norm.cdf(test_statistic, mu, sigma)

    return p_value


def _sub_sample(matrix_X, matrix_Y, test_statistic, num_samples, sub_samples, test_statistic_sub_sampling, which_test):
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
    if which_test == 'mgc':
        sigma = stdev(test_statistic_sub_sampling) / num_samples
        mu = max(0, mean(test_statistic_sub_sampling))
    else:
        sigma = stdev(test_statistic_sub_sampling) / math.sqrt(num_samples)
        mu = 0

    return sigma, mu
