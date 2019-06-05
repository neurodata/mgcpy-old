import numpy as np
from mgcpy.independence_tests.dcorr import DCorr
from sklearn import preprocessing


def k_sample_transform(x, y, is_y_categorical=False):
    '''
    Transform to represent a k-sample test as an independence test

    :param X: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type X: 2D numpy.array

    :param Y: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
        - a ``[n*1]`` label matrix, categorical data for X, if ``is_y_categorical`` is set to True
    :type Y: 2D numpy.array

    :param is_y_categorical: if set to True, ``Y`` has categorical data ans is a labels array for X,
                             else, it is a plain data matrix
    :type is_y_categorical: boolean

    :return:

        - :u: a concatenated data matrix of dimensions ``[2*n, p]``
        - :v: a label matrix for ``u``, which indicates to which category each data entry in ``u`` belongs to
    :rtype: list
    '''
    if not is_y_categorical:
        assert x.shape == y.shape, "Matrices X and Y need to be of same dimensions [n, p]"
    else:
        assert x.shape[0] == y.shape[0] and y.shape[1] == 1, "Matrices X and Y need to be of dimensions [n, p], [n, 1]"

    if not is_y_categorical:
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    else:
        u = x
        v = preprocessing.LabelEncoder().fit_transform(y.flatten()) + 1

    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]

    return u, v


def paired_two_sample_transform(x, y):
    '''
    Transform to represent a paired two-sample test as an independence test
    Steps:
        - combine x and y to get the joint_distribution
        - sample n pairs from the joint_distribution
        - compute the eucledian distance between the sampled n pairs, which is ``randomly_sampled_pairs_distance``
        - compute the eucledian distance between the actual x and y, which is ``actual_pairs_distance``
        - compute the two sample transformed matrices of ``randomly_sampled_pairs_distance`` and ``actual_pairs_distance``
    :param X: is interpreted as either:
        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type X: 2D numpy.array
    :param Y: is interpreted as either:
        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type Y: 2D numpy.array
    :return:
        - :u: a data matrix of dimensions ``[2*n, p]``
        - :v: a label matrix for ``u``, which indicates to which category each data entry in ``u`` belongs to
    :rtype: list
    '''
    assert x.shape == y.shape, "Matrices X and Y need to be of same dimensions [n, p]"

    joint_distribution = np.concatenate([x, y], axis=0)  # (2n, p) shape

    pairwise_sampled_xy = np.array([joint_distribution[np.random.randint(joint_distribution.shape[0], size=2), :]
                                    for _ in range(x.shape[0])])  # (n, 2, p) shape
    pairwise_sampled_x = pairwise_sampled_xy[:, 0]  # (n, p) shape
    pairwise_sampled_y = pairwise_sampled_xy[:, 1]  # (n, p) shape

    # compute the eucledian distances
    randomly_sampled_pairs_distance = np.linalg.norm(pairwise_sampled_x - pairwise_sampled_y, axis=1)
    actual_pairs_distance = np.linalg.norm(x - y, axis=1)

    u, v = k_sample_transform(randomly_sampled_pairs_distance, actual_pairs_distance)

    return u, v


def paired_two_sample_test_dcorr(x, y, which_test="biased", compute_distance_matrix=None, is_fast=False):
    '''
    Compute paired two sample test's DCorr test_statistic

    :param X: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type X: 2D numpy.array

    :param Y: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type Y: 2D numpy.array

    :return: paired two sample DCorr test_statistic
    :rtype: float
    '''
    assert x.shape == y.shape, "Matrices X and Y need to be of same dimensions [n, p]"

    dcorr = DCorr(is_paired=True, which_test=which_test, compute_distance_matrix=compute_distance_matrix)

    return dcorr.p_value(x, y, is_fast=is_fast)
