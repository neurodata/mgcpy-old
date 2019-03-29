import numpy as np

DTYPE = np.float64
ITYPE = np.int32


def check_rank(X):
    """
    This function checks if X is rank deficient.

    :param matrix_X: is interpreted as:

        - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
    :type matrix_X: 2D `numpy.array`

    :rtype: None

    :raise: Raises Exception if X matrix is rank deficient.
    """
    rank = np.linalg.matrix_rank(X)
    if rank < min(X.shape[0], X.shape[1]):
        raise Exception("matrix is rank deficient (rank %i vs cols %i)" % (rank, X.shape[1]))


def hatify(X):
    """
    Calculates the "hat" matrix.

    :param X: is interpreted as:

        - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
    :type X: 2D `numpy.array`

    :return: returns the hat matrix of the data matrix input.
    :rtype: 2D `numpy.array`
    """
    return X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)


def gower_center(Y):
    """
    Computes Gower's centered similarity matrix.

    :param Y: is interpreted as:

        - a ``[n*n]`` distance matrix
    :type Y: 2D `numpy.array`

    :return: returns the gower centered similarity matrix of the input matrix.
    :rtype: 2D `numpy.array`
    """
    n = Y.shape[0]
    I = np.eye(n, n)
    uno = np.ones((n, 1))

    A = -0.5 * (Y ** 2)
    C = I - (1.0 / n) * uno.dot(uno.T)
    G = C.dot(A).dot(C)

    return G


def gower_center_many(Ys):
    """
    Gower centers each matrix in the input.

    :param Ys: is interpreted as:

        - an array of ``[n^2*1]`` distance matrices
    :type Ys: 2D `numpy.array`
        Note: in practice this function is only run on one matrix currently.
        Due to this, Ys will just be a 1D `numpy.array`

    :return: returns the gower centered similarity matrix of the all input matrices.
    :rtype: 2D `numpy.array`
    """
    observations = int(np.sqrt(Ys.shape[0]))
    tests = Ys.shape[1]
    Gs = np.zeros_like(Ys)

    for i in range(tests):
        #        print(type(observations))
        D = Ys[:, i].reshape(observations, observations)
        Gs[:, i] = gower_center(D).flatten()

    return Gs


def gen_H2_perms(X, predictors, permutation_indexes):
    """
    Return H2 for each permutation of X indices, where H2 is the hat matrix
    minus the hat matrix of the untested columns.

    :param X: is interpreted as:

        - a ``[n*d+1]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        and a column of ones placed before the matrix
    :type X: 2D `numpy.array`

    :param predictors: is interpreted as:

        - a ``[1*d]`` array with the number of each variable in X used as a predictor
    :type predictors: 1D `numpy.array`

    :param permutation_indexes: is interpreted as:

        - a ``[p+1*n]`` matrix where p is the number of permutations given in the main code.
        This matrix has p permutations of indexes of the X data.
    :type permutation_indexes: 2D `numpy.array`

    :return: a ``[p+1*n^2]`` array of the flattened H2 matrices for each permutation
    :rtype: 2D `numpy.array`
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

    The function calculates this correctly for multiple predictor tests.

    :param X: is interpreted as:

        - a ``[n*d+1]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        and a column of ones placed before the matrix
    :type X: 2D `numpy.array`

    :param predictors: is interpreted as:

        - a ``[1*d]`` array with the number of each variable in X used as a predictor
    :type predictors: 1D `numpy.array`

    :param permutation_indexes: is interpreted as:

        - a ``[p+1*n]`` matrix where p is the number of permutations given in the main code.
        This matrix has p permutations of indexes of the X data.
    :type permutation_indexes: 2D `numpy.array`

    :return: a ``[p+1*n^2]`` array of the flattened arrays of the IH matrix for each permutation
    :rtype: 2D `numpy.array`
    """
    permutations, observations = permutation_indexes.shape
    I = np.eye(observations, observations)

    IH_permutations = np.zeros((observations ** 2, permutations))
    for i in range(permutations):
        IH = I - hatify(X[permutation_indexes[i, :]])
        IH_permutations[:, i] = IH.flatten()

    return IH_permutations


def calc_ftest(Hs, IHs, Gs, m2, nm):
    """
    This function calculates the pseudo-F statistic.

    :param Hs: is interpreted as:

        - a ``[p+1*n^2]`` array with the flattened H2 matrix for each permutation
    :type Hs: 2D `numpy.array`

    :param IHs: is interpreted as:

        - a ``[p+1*n^2]`` array with the flattened IH matrix for each permutation
    :type IHs: 2D `numpy.array`

    :param Gs: is interpreted as:

        - a [n^2*a] array with the gower centered distance matrix where a is in practice 1
    :type Gs: 2D `numpy.array`

    :param m2: is interpreted as:

        - a float equal to the number of predictors minus the number of tests (which will be 1)
    :type m2: `float`

    :param nm: is interpreted as:

        - a float equal to the number of subjects minus the number of predictors
    :type nm: `float`

    :return: a ``[p+1*1]`` array of F statistics for each permutation
    :rtype: 1D `numpy.array`
    """
    N = Hs.T.dot(Gs)
    D = IHs.T.dot(Gs)
    F = (N / m2) / (D / nm)
    return F


def fperms_to_pvals(F_perms):
    """
    This function calculates the permutation p-value from the test statistics of all permutations.

    :param F_perms: is interpreted as:

        - a ``[p+1*1]`` array of F statistics for each permutation
    :type F_perms: 1D `numpy.array`

    :return: a float which is the permutation p-value of the F-statistic
    :rtype: `float`
    """
    permutations, tests = F_perms.shape
    permutations -= 1
    pvals = np.zeros(tests)
    for i in range(tests):
        j = (F_perms[1:, i] >= F_perms[0, i]).sum().astype('float')
        pvals[i] = (j) / (permutations)
    return pvals
