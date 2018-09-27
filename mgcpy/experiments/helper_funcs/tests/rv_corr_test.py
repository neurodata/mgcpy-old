import pytest
import numpy as np
from scipy.stats import pearsonr

from mgcpy.experiments.helper_funcs.rv_corr import rv_corr


def test_rank_distance_matrix():
    a = np.array([1, 4, 6])
    b = np.array([1, 1, 1])
    assert np.array_equal(rv_corr(a,b), pearsonr(a, b))