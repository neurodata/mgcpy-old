# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:09:14 2018

@author: sunda
"""
import numpy as np
import scipy.spatial as scp

DTYPE = np.float64
ITYPE = np.int32

def check_rank(X):
    """
    This function checks if X is rank deficient.
    """
    k    = X.shape[1]
    rank = np.linalg.matrix_rank(X)
    if rank < k:
        raise Exception("matrix is rank deficient (rank %i vs cols %i)" % (rank, k))

def compute_distance_matrix(X, disttype):
    D = scp.distance.pdist(X, disttype)
    return D
        
def hatify(X):
    """
    Returns the "hat" matrix.
    """
    return X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)

def gower_center(Y):
    """
    Computes Gower's centered similarity matrix.
    """
    n = Y.shape[0]
    I = np.eye(n,n)
    uno = np.ones((n, 1))
    
    A = -0.5 * (Y ** 2)
    C = I - (1.0 / n) * uno.dot(uno.T)
    G = C.dot(A).dot(C)
    
    return G

def gower_center_many(Ys):
    """
    Gower centers each matrix in the input.
    """
    observations = int(np.sqrt(Ys.shape[0]))
    tests        = Ys.shape[1]
    Gs           = np.zeros_like(Ys)
    
    for i in range(tests):
#        print(type(observations))
        D        = Ys[:, i].reshape(observations, observations)
        Gs[:, i] = gower_center(D).flatten()
    
    return Gs


def gen_H2_perms(X, predictors, permutation_indexes):
    """
    Return H2 for each permutation of X indices, where H2 is the hat matrix
    minus the hat matrix of the untested columns.
    """
    permutations, observations = permutation_indexes.shape
    variables = X.shape[1]
    
    covariates = [i for i in range(variables) if i not in predictors]
    H2_permutations = np.zeros((observations ** 2, permutations))
    for i in range(permutations):
        perm_X = X[permutation_indexes[i]]
        H2 = hatify(perm_X) - hatify(perm_X[:, covariates])
        H2_permutations[:, i] = H2.flatten()
    
    return H2_permutations


def gen_IH_perms(X, predictors, permutation_indexes):
    """
    Return I-H where H is the hat matrix and I is the identity matrix.
    
    The function calculates this correctly for multiple column tests.
    """
    permutations, observations = permutation_indexes.shape
    I            = np.eye(observations, observations)
    
    IH_permutations = np.zeros((observations ** 2, permutations))
    for i in range(permutations):
        IH = I - hatify(X[permutation_indexes[i, :]])
        IH_permutations[:,i] = IH.flatten()
    
    return IH_permutations


def calc_ftest(Hs, IHs, Gs, m2, nm):
    """
    This function calculates the pseudo-F statistic.
    """
    N = Hs.T.dot(Gs)
    D = IHs.T.dot(Gs)
    F = (N / m2) / (D / nm)
    return F


def fperms_to_pvals(F_perms):
    """
    This function calculates the permutation p-value from the test statistics of all permutations.
    """
    permutations, tests = F_perms.shape
    permutations -= 1
    pvals = np.zeros(tests)
    for i in range(tests):
        j        = (F_perms[1:, i] >= F_perms[0, i]).sum().astype('float')
        pvals[i] = (j+1) / (permutations+1)
    return pvals