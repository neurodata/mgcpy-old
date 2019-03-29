import numpy as np
import pytest
from mgcpy.independence_tests.utils.distance_transform import (center_distance_matrix,
                                                             rank_distance_matrix,
                                                             transform_distance_matrix)
from scipy.spatial import distance_matrix


def test_rank_distance_matrix():
    a = np.array([[1, 4, 6],
                  [2, 5, 7],
                  [1, 4, 6]], dtype=np.float)
    ranked_a = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [1, 1, 1]], dtype=np.float)
    assert np.array_equal(ranked_a, rank_distance_matrix(a))


def test_center_distance_matrix_mgc():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]], dtype=np.float)
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_mgc = np.array([[0.00000000, 42.95233, 43.04942],
                                               [-0.09708753, 0.00000, -43.04942],
                                               [0.09708753, -42.95233, 0.00000]], dtype=np.float)

    assert np.allclose(X_centered_distance_matrix_mgc,
                       center_distance_matrix(X_distance_matrix)["centered_distance_matrix"])


def test_center_distance_matrix_rank():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_rank = np.array([[0, 0, 0],
                                                [-1, 0, -1],
                                                [0, -1, 0]])

    assert np.allclose(X_centered_distance_matrix_rank,
                       center_distance_matrix(X_distance_matrix, "rank")["centered_distance_matrix"])


def test_center_distance_matrix_unbiased():
    X = np.array([[2, 1, 100, 90, 1000], [4, 2, 10, 80, 900], [8, 3, 10, 20, 500], [7, 2, 9, 19, 20], [10, 5, 12, 22, 502]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_dcor = np.array([[0.0000, -387.9321,  130.6687,  128.2148,  129.0485],
                                                [-387.9321,    0.0000,  129.3331,  130.5881,  128.0110],
                                                [130.6687,  129.3331,    0.0000, -130.8726, -129.1292],
                                                [128.2148,  130.5881, -130.8726,    0.0000, -127.9303],
                                                [129.0485,  128.0110, -129.1292, -127.9303,    0.0000]])

    assert np.allclose(X_centered_distance_matrix_dcor,
                       center_distance_matrix(X_distance_matrix, "unbiased")["centered_distance_matrix"])


def test_center_distance_matrix_biased():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_biased = np.array([[-79.19474188, 39.5326, 39.6621],
                                                  [39.5326, -21.79551326, -17.7371],
                                                  [39.6621, -17.7371, -21.9249633]])

    assert np.allclose(X_centered_distance_matrix_biased, center_distance_matrix(X_distance_matrix, "biased")["centered_distance_matrix"])


def test_center_distance_matrix_mantel():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_mantel = np.array([[-61.45760922, 28.57016, 28.76434],
                                                  [28.57016, -61.45760922, -57.33450],
                                                  [28.76434, -57.33450, -61.45760922]])

    assert np.allclose(X_centered_distance_matrix_mantel,
                       center_distance_matrix(X_distance_matrix, "mantel")["centered_distance_matrix"])


def test_transform_distance_matrix():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    Y = np.array([[30, 20, 10], [5, 10, 20], [8, 16, 32]])
    X_distance_matrix = distance_matrix(X, X)
    Y_distance_matrix = distance_matrix(Y, Y)
    X_centered_distance_matrix = np.array([[0.00000000, 42.95233, 43.04942],
                                           [-0.09708753, 0.00000, -43.04942],
                                           [0.09708753, -42.95233, 0.00000]])
    X_ranked_distance_matrix = np.array([[1, 3, 3],
                                         [2, 1, 2],
                                         [3, 2, 1]])
    Y_centered_distance_matrix = np.array([[0.000000, 7.487543, 8.810524],
                                           [-1.322981, 0.000000, -8.810524],
                                           [1.322981, -7.487543, 0.000000]])
    Y_ranked_distance_matrix = np.array([[1, 3, 3],
                                         [2, 1, 2],
                                         [3, 2, 1]])
    transformed_distance_matrix_X_Y = transform_distance_matrix(
        X_distance_matrix, Y_distance_matrix)

    assert np.allclose(X_centered_distance_matrix,
                       transformed_distance_matrix_X_Y["centered_distance_matrix_A"])
    assert np.allclose(Y_centered_distance_matrix,
                       transformed_distance_matrix_X_Y["centered_distance_matrix_B"])
    assert np.allclose(X_ranked_distance_matrix,
                       transformed_distance_matrix_X_Y["ranked_distance_matrix_A"])
    assert np.allclose(Y_ranked_distance_matrix,
                       transformed_distance_matrix_X_Y["ranked_distance_matrix_B"])
