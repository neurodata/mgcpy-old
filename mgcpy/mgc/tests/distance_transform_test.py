import pytest
import numpy as np

from mgcpy.mgc.distance_transform import rank_distance_matrix


def test_rank_distance_matrix():
    a = np.array([[1, 4, 6],
                  [2, 5, 7],
                  [1, 4, 6]])
    ranked_a = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [1, 1, 1]])
    assert np.array_equal(rank_distance_matrix(a), ranked_a)
