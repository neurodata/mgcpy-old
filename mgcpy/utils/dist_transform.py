import numpy as np
from scipy.spatial import distance


def dist_transform(A, B, corr_type):
    '''Transform the distance matrices in a specified way

    :param A, B: n*n distance matrix
    :param corr_type: a string indicating which global correlation to build upon, e.g. 'dcorr'
    :return: C, D: the transformed matrices
    '''

    C = dist_center(A, corr_type)
    D = dist_center(B, corr_type)

    return (C, D)


def dist_center(A, corr_type):
    '''Center the distance matrix as specified by the correlation test
    that uses it

    :param A: an n*n distance matrix to be centered
    :param corr_type: a string specifying which centering scheme to use, e.g. 'dcorr'
    :return: C: the centered matrix
    '''
    # the dimension of A
    n = A.shape[0]

    # the centering scheme makes the difference among dcorr, mcorr
    # and mantel

    # unbiased dcorr transform
    if corr_type == 'dcorr':
        '''
        all the means are not divided by n exactly so that the transform is unbiased
        the mean taken over the rows
        convert the vector of row means into matrix so that entries in
        the same column has the same row mean
        can be directly subtract off later
        '''
        row_mean = np.tile(np.sum(A, axis=0)/(n-2), (n, 1))
        # the mean taken over the columns
        # convert into matrix so that entries in the same row has the same column mean
        col_mean = np.tile(np.sum(A, axis=1)[:, np.newaxis]/(n-2), (1, n))
        # mean of all the entries
        grand_mean = np.sum(A)/(n-1)/(n-2)
        # the quantity which we adjust A with
        adjustment = row_mean + col_mean - grand_mean

    # mantel transform
    elif corr_type == 'mantel':
        # mean of all the entries (scaled differently than in dcorr)
        adjustment = np.sum(A)/n/(n-1)

    # "default mgc transform" used in fastmgc
    elif corr_type == 'mcorr':
        adjustment = np.tile(np.sum(A, axis=0)/(n-1), (n, 1))

    # the centered matrix
    C = A - adjustment
    # the diagonal entries should always be zero
    for j in range(n):
        C[j, j] = 0

    return C
