"""
    **Common Distance Calculation Matrix**
"""


def compute_distance(matrix_X, matrix_Y, _compute_distance):
    """
    Computes the distance matrix among both independence tests

    :param matrix_X: is interpreted as a ``[n*p]`` data matrix, a matrix with ``n`` samples in ``p`` dimensions
    :type matrix_X: 2D numpy.array

    :param matrix_Y: is interpreted as a ``[n*q]`` data matrix, a matrix with ``n`` samples in ``q`` dimensions
    :type matrix_Y: 2D numpy.array

    :param _compute_distance: is interpreted as the distance matrix calculation with the specified metric
    :type _compute_distance: ``FunctionType`` or ``callable()``

    :return: returns a list of two items, that contains:

        - :matrix_X: the calculated distance matrix for ``matrix_X``
        - :matrix_Y: the calculated distance matrix for ``matrix_Y``
    :rtype: list
    """
    # use the matrix shape and diagonal elements to determine if the given data is a distance matrix or not
    if matrix_X.shape[0] != matrix_X.shape[1] or sum(matrix_X.diagonal()**2) > 0:
        matrix_X = _compute_distance(matrix_X)
    if matrix_Y.shape[0] != matrix_Y.shape[1] or sum(matrix_Y.diagonal()**2) > 0:
        matrix_Y = _compute_distance(matrix_Y)

    return matrix_X, matrix_Y
