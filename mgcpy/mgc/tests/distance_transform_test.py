import pytest
import numpy as np
from scipy.spatial import distance_matrix

from mgcpy.mgc.distance_transform import rank_distance_matrix, center_distance_matrix


def test_rank_distance_matrix():
    a = np.array([[1, 4, 6],
                  [2, 5, 7],
                  [1, 4, 6]])
    ranked_a = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [1, 1, 1]])
    assert np.array_equal(ranked_a,
                          rank_distance_matrix(a))


def test_center_distance_matrix_mgc():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_mgc = np.array([[0.00000000, 42.95233, 43.04942],
                                               [-0.09708753, 0.00000, -43.04942],
                                               [0.09708753, -42.95233, 0.00000]])

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


def test_center_distance_matrix_dcor():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_dcor = np.array([[0.00000, - 30.009258, - 30.073983],
                                                [-30.00926, 0.000000, -1.374369],
                                                [-30.07398, -1.374369, 0.000000]])

    assert np.allclose(X_centered_distance_matrix_dcor,
                       center_distance_matrix(X_distance_matrix, "dcor")["centered_distance_matrix"])


def test_center_distance_matrix_mantel():
    X = np.array([[2, 1, 100], [4, 2, 10], [8, 3, 10]])
    X_distance_matrix = distance_matrix(X, X)
    X_centered_distance_matrix_mantel = np.array([[0.00000, 28.57016, 28.76434],
                                                  [28.57016, 0.00000, -57.33450],
                                                  [28.76434, -57.33450, 0.00000]])

    assert np.allclose(X_centered_distance_matrix_mantel,
                       center_distance_matrix(X_distance_matrix, "mantel")["centered_distance_matrix"])
