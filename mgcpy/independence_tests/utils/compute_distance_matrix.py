def compute_distance(matrix_X, matrix_Y, compute_distance):
    # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
    if matrix_X.shape[0] != matrix_X.shape[1] or sum(matrix_X.diagonal()**2) > 0:
        matrix_X = compute_distance(matrix_X)
    if matrix_Y.shape[0] != matrix_Y.shape[1] or sum(matrix_Y.diagonal()**2) > 0:
        matrix_Y = compute_distance(matrix_Y)

    return matrix_X, matrix_Y
