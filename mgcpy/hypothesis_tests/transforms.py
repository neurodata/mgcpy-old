import numpy as np
from sklearn import preprocessing


def k_sample_transform(x, y, is_y_categorical=False):
    '''
    Transform to represent a k-sample test as an independence test

    :param X: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
    :type X: 2D numpy.array

    :param Y: is interpreted as either:

        - a ``[n*n]`` distance matrix, a square matrix with zeros on diagonal for n samples OR
        - a ``[n*p]`` data matrix, a matrix with n samples in p dimensions
        - a ``[n*1]`` label matrix, categorical data for X, if ``is_y_categorical`` is set to True
    :type Y: 2D numpy.array

    :param is_y_categorical: if set to True, ``Y`` has categorical data ans is a labels array for X,
                             else, it is a plain data matrix
    :type is_y_categorical: boolean

    :return:

        - :u: a concatenated data matrix of dimensions ``[2*n, p]``
        - :v: a label matrix for ``u``, which indicates to which category each label in ``u`` belongs to
    :rtype: list
    '''
    assert x.shape == y.shape, "Matrices X and Y need to be of same dimensions [n, p]"

    if not is_y_categorical:
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    else:
        u = x
        v = preprocessing.LabelEncoder().fit_transform(y) + 1

    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]

    return u, v
