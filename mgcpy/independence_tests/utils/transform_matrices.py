import numpy as np


def transform_matrices(matrix1=None, matrix2=None):
    """
    :param matrix_1: matrix corresponding to the first distribution
    :type matrix_1: 2D `numpy.array`

    :param matrix_2: matrix corresponding to the second distribution
    :type matrix_2: 2D `numpy.array`
    """
    flattened_matrix1 = matrix1.flatten()
    flattened_matrix2 = matrix2.flatten()
    data_matrix = np.concatenate(flattened_matrix1, flattened_matrix2)
    label_matrix = np.concatenate(np.zeros(flattened_matrix1.shape), np.ones(flattened_matrix2.shape))

    return(data_matrix, label_matrix)

# def transform_matrices(A, B):
#    U = A.tolist()
#    V = B.tolist()
#    if isinstance(U[0], list):
#        col = len(U[0])+len(V[0])
#        row = len(U)
#    else:
#        col = len(U)+len(V)
#        row = 1
#
#    data = [[0 for t1 in range(col)] for t2 in range(row)]
#    num = [0 for t1 in range(col)]
#
#    if isinstance(U[0], list):
#        for n1 in range(row):
#            for n2 in range(len(U[0])):
#                data[n1][n2] = U[n1][n2]
#                num[n2] = 0
#        for n3 in range(row):
#            for n4 in range(len(V[0])):
#                data[n3][n4+len(U[0])] = V[n3][n4]
#                num[n4+len(U[0])] = 1
#    else:
#        for n1 in range(row):
#            for n2 in range(len(U)):
#                data[n1][n2] = U[n2]
#                num[n2] = 0
#        for n3 in range(row):
#            for n4 in range(len(V)):
#                data[n3][n4+len(U)] = V[n4]
#                num[n4+len(U)] = 1
#
#    x = np.asarray(data)
#    y = np.asarray(num)
#    return x, y
