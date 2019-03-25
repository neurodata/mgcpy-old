"""
    **Main MDMR Independence Test Module**
"""

import numpy as np
import scipy.spatial as scp

from mgcpy.independence_tests.abstract_class import IndependenceTest
from mgcpy.independence_tests.utils.compute_distance_matrix import \
    compute_distance
from mgcpy.independence_tests.utils.mdmr_functions import (calc_ftest,
                                                           check_rank,
                                                           fperms_to_pvals,
                                                           gen_H2_perms,
                                                           gen_IH_perms,
                                                           gower_center_many)


class MDMR(IndependenceTest):
    def __init__(self, compute_distance_matrix=None):
        '''
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: ``FunctionType`` or ``callable()``
        '''
        IndependenceTest.__init__(self, compute_distance_matrix)
        self.which_test = "mdmr"

    def get_name(self):
        return self.which_test

    def test_statistic(self, matrix_X, matrix_Y, permutations=0, individual=0, disttype='cityblock'):
        """
        Computes MDMR Pseudo-F statistic between two datasets.

        - It first takes the distance matrix of Y (by )
        - Next it regresses X into a portion due to Y and a portion due to residual
        - The p-value is for the null hypothesis that the variable of X is not correlated with Y's distance matrix

        :param data_matrix_X: (optional, default picked from class attr) is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with n samples in d dimensions
        :type data_matrix_X: 2D `numpy.array`

        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with n samples in d dimensions
        :type data_matrix_Y: 2D `numpy.array`

        :parameter 'individual':

            -integer, `0` or `1`
            with value `0` tests the entire X matrix (default)
            with value `1` tests the entire X matrix and then each predictor variable individually

        :return: with individual = `0`, returns 1 values, with individual = `1` returns 2 values, containing:

            -the test statistic of the entire X matrix
            -for individual = 1, an array with the variable of X in the first column,
                the test statistic in the second, and the permutation p-value in the third (which here will always be 1)
        :rtype: list
        """
        X = matrix_X
        Y = matrix_Y

        # calculate distance matrix of Y
        D, _ = compute_distance(Y, np.identity(1), self.compute_distance_matrix)
        a = D.shape[0]**2
        D = D.reshape((a, 1))

        predictors = np.arange(X.shape[1])
        predsingle = X.shape[1]
        check_rank(X)

        # check number of subjects compatible
        subjects = X.shape[0]
        if subjects != np.sqrt(D.shape[0]):
            raise Exception("# of subjects incompatible between X and D")

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        predictors = np.array(predictors)
        predictors += 1

        # Gower Center the distance matrix of Y
        Gs = gower_center_many(D)

        m2 = float(X.shape[1] - predictors.shape[0])
        nm = float(subjects - X.shape[1])

        # form permutation indexes
        permutation_indexes = np.zeros((permutations + 1, subjects), dtype=np.int)
        permutation_indexes[0, :] = range(subjects)
        for i in range(1, permutations + 1):
            permutation_indexes[i, :] = np.random.permutation(subjects)

        H2perms = gen_H2_perms(X, predictors, permutation_indexes)
        IHperms = gen_IH_perms(X, predictors, permutation_indexes)

        # Calculate test statistic
        F_perms = calc_ftest(H2perms, IHperms, Gs, m2, nm)

        # Calculate p-value
        p_vals = None
        if permutations > 0:
            p_vals = fperms_to_pvals(F_perms)
        F_permtotal = F_perms[0, :]
        self.test_statistic_ = F_permtotal
        if individual == 0:
            return self.test_statistic_, self.test_statistic_metadata_

        # code for individual test
        if individual == 1:
            results = np.zeros((predsingle, 3))
            for predictors in range(1, predsingle+1):
                predictors = np.array([predictors])

                Gs = gower_center_many(D)

                m2 = float(X.shape[1] - predictors.shape[0])
                nm = float(subjects - X.shape[1])

                permutation_indexes = np.zeros((permutations + 1, subjects), dtype=np.int)
                permutation_indexes[0, :] = range(subjects)
                for i in range(1, permutations + 1):
                    permutation_indexes[i, :] = np.random.permutation(subjects)

                H2perms = gen_H2_perms(X, predictors, permutation_indexes)
                IHperms = gen_IH_perms(X, predictors, permutation_indexes)

                F_perms = calc_ftest(H2perms, IHperms, Gs, m2, nm)

                p_vals = None
                if permutations > 0:
                    p_vals = fperms_to_pvals(F_perms)
                results[predictors-1, 0] = predictors
                results[predictors-1, 1] = F_perms[0, :]
                results[predictors-1, 2] = p_vals

            return F_permtotal, results

    def p_value(self, matrix_X, matrix_Y, replication_factor=1000):
        """
        Tests independence between two datasets using MGC and permutation test.

        :param matrix_X: is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D `numpy.array`

        :param replication_factor: specifies the number of replications to use for
                                   the permutation test. Defaults to ``1000``.
        :type replication_factor: integer

        :return: returns a list of two items,that contains:

            - :p_value: P-value of MGC
            - :p_value_metadata:
        :rtype: list
        """
        return super(MDMR, self).p_value(matrix_X, matrix_Y)

    def ind_p_value(self, matrix_X, matrix_Y, permutations=1000, individual=1, disttype='cityblock'):
        """
        Individual predictor variable p-values calculation

        :param matrix_X: is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_X: 2D `numpy.array`

        :param matrix_Y: is interpreted as:

            - a ``[n*d]`` data matrix, a matrix with ``n`` samples in ``d`` dimensions
        :type matrix_Y: 2D `numpy.array`
        """
        results = self.test_statistic(matrix_X, matrix_Y, permutations, individual)[1]
        return results
