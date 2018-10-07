import numpy as np
from scipy.spatial.distance import pdist, squareform
from mgcpy.utils.dist_transform import dist_transform


def dcorr(X, Y, corr_type='dcorr', metric='euclidean'):
    '''
    Compute the correlation between X and Y using dcorr/mcorr/mantel

    Procedure: compute two distance matrices A and B, each n*n using pdist and squareform
    then perform distance transformation using dist_transform(A, B)
    calculate correlation by computing all global covariance and variance using global_cov(A, B)

    :param X: data matrix, size n*p, n: number of examples, p: dimension of each example
    :param Y: data matrix, size n*q, p and q can be different
    :param corr_type: a string specifying which global correlation to use, can be 'dcorr', 'mcorr', 'mantel'
    :param metric: the type of metric used to compute distance
    :return: a single value (test statistic)
    '''
    # obtain the pairwise distance matrix for X and Y
    A = squareform(pdist(X, metric=metric))
    B = squareform(pdist(Y, metric=metric))

    # perform distance transformation
    A, B = dist_transform(A, B, corr_type)
    # after distance transformation, A and B need not be symmetric
    # e.g. not after mcorr transform, so transpose is necessary
    cov = global_cov(A, np.transpose(B))
    varA = global_cov(A, np.transpose(A))
    varB = global_cov(B, np.transpose(B))

    # check the case when one of the dataset has zero variance
    if varA <= 0 or varB <= 0:
        corr = 0
    else:
        corr = cov/np.real(np.sqrt(varA*varB))

    # use absolute value for mantel coefficients
    if corr_type == 'mantel':
        return np.abs(corr)

    return corr


def global_cov(A, B):
    '''
    Compute the global covariance using distance matrix A and B

    :param A, B: n*n distance matrix
    :return: float representing the covariance/variance
    '''
    return np.sum(np.multiply(A, B))
