import numpy as np

from mgcpy.independence_tests.mgc.mgc import MGC
from mgcpy.independence_tests.rv_corr import RVCorr


def create_independence_test(type, data_matrix_X, data_matrix_Y, base_global_correlation=None):
    if type == "mgc":
        if base_global_correlation:
            return MGC(data_matrix_X, data_matrix_Y, None, base_global_correlation)
        else:
            return MGC(data_matrix_X, data_matrix_Y, None)
    elif type == "rvcorr":
        return RVCorr(data_matrix_X, data_matrix_Y, None)


if __name__ == '__main__':
    # demo test
    X = np.array([0.07487683, -0.18073412, 0.37266440, 0.06074847, 0.76899045,
                  0.51862516, -0.13480764, -0.54368083, -0.73812644, 0.54910974]).reshape(-1, 1)
    Y = np.array([-1.31741173, -0.41634224, 2.24021815, 0.88317196, 2.00149312,
                  1.35857623, -0.06729464, 0.16168344, -0.61048226, 0.41711113]).reshape(-1, 1)

    # MGC
    mgc_independence_test = create_independence_test("mgc", X, Y)
    mgc_statistic, mgc_metadata = mgc_independence_test.test_statistic()
    mgc_p_value, _ = mgc_independence_test.p_value()

    print("Test Statistic:", mgc_statistic)
    print("P-Value:", mgc_p_value)
    print("Optimal Scale:", mgc_metadata["optimal_scale"])

    # RV CORR
    rvcorr_independence_test = create_independence_test("rvcorr", X, Y)
    rvcorr_statistic, _ = rvcorr_independence_test.test_statistic()

    print("Test Statistic:", rvcorr_statistic)
