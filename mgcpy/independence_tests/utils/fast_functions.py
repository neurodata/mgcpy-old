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
