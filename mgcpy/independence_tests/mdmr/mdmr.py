from mgcpy.independence_tests.abstract_class import IndependenceTest
from mdmrfunctions import *

class MDMR(IndependenceTest):
    def __init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix):
                '''
        :param data_matrix_X: is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array
        :param data_matrix_Y: is interpreted as either:
            - a [n*n] distance matrix, a square matrix with zeros on diagonal for n samples OR
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array
        :param compute_distance_matrix: a function to compute the pairwise distance matrix, given a data matrix
        :type compute_distance_matrix: FunctionType or callable()
        '''
        IndependenceTest.__init__(self, data_matrix_X, data_matrix_Y, compute_distance_matrix)
    
    
    def test_statistic(self, data_matrix_X=None, data_matrix_Y=None, permutations = 1000):
        """
        Computes MDMR between two datasets.
        - It first takes the distance matrix of Y
        - Next it regresses X into a portion due to Y and a portion due to residual for each variable of X
        - The p-value is for the null hypothesis that the variable of X is not correlated with Y's distance matrix
        :param data_matrix_X: (optional, default picked from class attr) is interpreted as:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_X: 2D numpy.array
        :param data_matrix_Y: (optional, default picked from class attr) is interpreted as:
            - a [n*d] data matrix, a square matrix with n samples in d dimensions
        :type data_matrix_Y: 2D numpy.array
        :return: returns an array with the variable of X in the first column, the statistic in the 2nd, and the permutation p-value in the 3rd:
        """
        X = data_matrix_X
        Y = data_matrix_Y
        columns = X.shape[1]
        check_rank(X)
        
        D = scp.spatial.distance.pdist(Y, 'cityblock')
        D = scp.spatial.distance.squareform(D)
        a = D.shape[0]**2
        D = D.reshape((a,1))
        
        subjects = X.shape[0]
        if subjects != np.sqrt(D.shape[0]):
            raise Exception("# of subjects incompatible between X and D")

        X = np.hstack((np.ones((X.shape[0], 1)), X))
        results = np.zeros((columns,3))
        for col in range(1, columns+1):
            col = copy.copy(col)
            #    columns += 1
    
            Gs = gower_center_many(D)
            
            df_among = float(col)
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
        results[col-1,0] = col
        results[col-1,1] = F_perms[0, :]
        results[col-1,2] = p_vals

    return results