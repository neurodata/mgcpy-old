from mgcpy.independence_tests.abstract_class import IndependenceTest

import copy
from mgcpy.independence_tests.mdmr.mdmrfunctions import *
#from mdmrfunctions import *

class MDMR(IndependenceTest):
    def __init__(self, compute_distance_matrix):
        '''
        :param data_matrix_X: is interpreted as:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array
        :param data_matrix_Y: is interpreted as either:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()
        '''
        IndependenceTest.__init__(self, compute_distance_matrix)
    
    def get_name(self):
        return "MDMR"
    
    def test_statistic(self, data_matrix_X, data_matrix_Y, permutations = 0, individual = 0, disttype = 'cityblock'):
        """
        Computes MDMR between two datasets.
        - It first takes the distance matrix of Y (by )
        - Next it regresses X into a portion due to Y and a portion due to residual for each variable of X
        - The p-value is for the null hypothesis that the variable of X is not correlated with Y's distance matrix
        :param data_matrix_X: (optional, default picked from class attr) is interpreted as:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array
        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array
        :parameter 'individual':
            with value 0 tests the entire X matrix by (default), returns 2 values
            with value 1 it tests the entire X matrix and then each column (variable) individually, returns 3 values
        :return: with individual = 0, returns the test statistic of the entire X matrix and the associated p-value
        with individual = 1, returns the above as well as an array with the 
        variable of X in the first column, the test statistic in the 2nd, and the permutation p-value in the 3rd
        """
        X = data_matrix_X
        Y = data_matrix_Y
        
        D = self.compute_distance_matrix(Y, disttype)
        D = scp.distance.squareform(D)
        a = D.shape[0]**2
        D = D.reshape((a,1))
        
        columns = np.arange(X.shape[1])
        columnsingle = X.shape[1]
        check_rank(X)
    
        subjects = X.shape[0]
        if subjects != np.sqrt(D.shape[0]):
            raise Exception("# of subjects incompatible between X and D")
        
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        col = copy.copy(columns)
        col += 1
    
        Gs = gower_center_many(D)
    
        df_among = float(col.shape[0])
        df_resid = float(subjects - X.shape[1])
    
        permutation_indexes = np.zeros((permutations + 1, subjects), dtype=np.int)
        permutation_indexes[0, :] = range(subjects)
        for i in range(1, permutations + 1):
            permutation_indexes[i,:] = np.random.permutation(subjects)
    
        H2perms = gen_H2_perms(X, col, permutation_indexes)
        IHperms = gen_IH_perms(X, col, permutation_indexes)
        
        F_perms = calc_ftest(H2perms, IHperms, Gs,
                            df_among, df_resid)
    
        p_vals = fperms_to_pvals(F_perms)
        F_permtotal = F_perms[0, :]
        pvaltotal = p_vals
        if individual == 0:
            return F_permtotal, pvaltotal
        if individual == 1:
            results = np.zeros((columnsingle,3))
            for col in range(1, columnsingle+1):
                col = copy.copy(col)
                #    columns += 1
                
                Gs = gower_center_many(D)
                
                df_among = float(col)
                df_resid = float(subjects - X.shape[1])
                
                permutation_indexes = np.zeros((permutations + 1, subjects), dtype=np.int)
                permutation_indexes[0, :] = range(subjects)
                for i in range(1, permutations + 1):
                    permutation_indexes[i,:] = np.random.permutation(subjects)
                    
                H2perms = gen_H2_perms_single(X, col, permutation_indexes)
                IHperms = gen_IH_perms_single(X, col, permutation_indexes)
                    
                F_perms = calc_ftest(H2perms, IHperms, Gs,
                                    df_among, df_resid)
                
                p_vals = fperms_to_pvals(F_perms)
                results[col-1,0] = col
                results[col-1,1] = F_perms[0, :]
                results[col-1,2] = p_vals
    
    
            return F_permtotal, pvaltotal, results

    def p_value(self, data_matrix_X, data_matrix_Y, permutations = 1000, individual = 0, disttype = 'cityblock'):
        """
        Computes the p-value of the pseudo-F test statistic.
        """
        p_value = self.test_statistic(data_matrix_X, data_matrix_Y, permutations)[1]
        
        return p_value
