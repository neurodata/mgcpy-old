import numpy as np
from scipy.spatial import distance


def dist_transform(dist_mtx_X, dist_mtx_Y, corr_type):
    '''Transform the distance matrices in a specified way

    :param dist_mtx_X, B: n*n distance matrix
    :param corr_type: a string indicating which global correlation to build upon, e.g. 'dcorr'
    :return: C, D: the transformed matrices
    '''

    transformed_dist_mtx_X = dist_center(dist_mtx_X, corr_type)
    transformed_dist_mtx_Y = dist_center(dist_mtx_Y, corr_type)

    return (transformed_dist_mtx_X, transformed_dist_mtx_Y)


def dist_center(dist_mtx_X, corr_type):
    '''Center the distance matrix as specified by the correlation test
    that uses it

    :param dist_mtx_X: an n*n distance matrix to be centered
    :param corr_type: a string specifying which centering scheme to use, e.g. 'dcorr'
    :return: C: the centered matrix
    '''
    # the dimension of dist_mtx_X
    n = dist_mtx_X.shape[0]

    # the centering scheme makes the difference among dcorr, mcorr
    # and mantel

    # unbiased centering, used to compute mcorr statistic
    if corr_type == 'mcorr':
        '''
        all the means are not divided by n exactly so that the transform is unbiased
        the mean taken over the rows
        convert the vector of row means into matrix so that entries in
        the same column has the same row mean
        can be directly subtract off later
        '''
        # the mean taken over the columns
        # convert into matrix so that entries in the same row has the same column mean
        row_mean = np.tile(np.sum(dist_mtx_X, axis=1)[:, np.newaxis]/(n-2), (1, n))
        col_mean = np.tile(np.sum(dist_mtx_X, axis=0)/(n-2), (n, 1))
        # mean of all the entries
        grand_mean = np.sum(dist_mtx_X)/(n-1)/(n-2)
        # the quantity which we adjust dist_mtx_X with
        adjustment = row_mean + col_mean - grand_mean

    # biased centering, used to compute dcorr statistic
    elif corr_type == 'dcorr':
        row_mean = np.tile(np.sum(dist_mtx_X, axis=1)[:, np.newaxis]/n, (1, n))
        col_mean = np.tile(np.sum(dist_mtx_X, axis=0)/n, (n, 1))
        # the mean taken over the columns
        # convert into matrix so that entries in the same row has the same column mean
        # mean of all the entries
        grand_mean = np.sum(dist_mtx_X)/np.square(n)
        adjustment = row_mean + col_mean - grand_mean

    # mantel transform
    elif corr_type == 'mantel':
        # mean of all the entries (scaled differently than in dcorr)
        adjustment = np.sum(dist_mtx_X)/n/(n-1)

    # the centered matrix
    centered_mtx_X = dist_mtx_X - adjustment

    if corr_type == 'mcorr':
        # the diagonal entries should always be zero
        for j in range(n):
            centered_mtx_X[j, j] = 0

    return centered_mtx_X
