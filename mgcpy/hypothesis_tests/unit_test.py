import numpy as np
import pandas as pd
import pytest
from mgcpy.hypothesis_tests.transforms import (k_sample_transform,
                                               paired_two_sample_test_dcorr,
                                               paired_two_sample_transform)
from mgcpy.independence_tests.mgc import MGC


def test_k_sample():
    np.random.seed(1234)

    # prepare data
    salary_data = pd.read_csv("./mgcpy/hypothesis_tests/salary_data.csv")

    # 2 sample case
    men_salaries = salary_data.loc[salary_data['Gender'] == "M"]["Current Annual Salary"].values
    women_salaries = salary_data.loc[salary_data['Gender'] == "F"]["Current Annual Salary"].values
    u, v = k_sample_transform(np.random.choice(men_salaries, 1000), np.random.choice(women_salaries, 1000))
    mgc = MGC()
    p_value, p_value_metadata = mgc.p_value(u, v, is_fast=True)
    assert np.allclose(p_value, 0.0, atol=0.01)

    # k sample case
    salaries = salary_data["Current Annual Salary"].values
    department_labels = salary_data["Department"].values
    u, v = k_sample_transform(salaries[:100], department_labels[:100], is_y_categorical=True)
    mgc = MGC()
    p_value, p_value_metadata = mgc.p_value(u, v)
    assert np.allclose(p_value, 0.0, atol=0.01)

    # 2 sample case (H_0 is valid)

    # generate 100 samples from the same distribution (x = np.random.randn(100))
    x = np.array([0.34270011,  1.30064541, -0.41888945,  1.40367111,  0.31901975, -1.83695735, -0.70370144,  0.89338428,  0.86047303, -0.98841287,
                  0.78325279,  0.55864254,  0.33317265,  2.22286831, -0.22349382, 0.40376754, -1.05356267,  0.54994568, -1.39765046,  0.41427267,
                  -0.24457334,  0.2464725, -0.32179342, -1.77106008, -0.52824522, 1.57839019, -1.66455582, -0.97663735, -0.55176702, -1.95347702,
                  1.01934119,  1.05765468, -0.69941067, -1.12479123,  0.85236935, -0.77356459,  0.30217738,  0.95246919, -0.61210025,  1.09253269,
                  0.13576324,  0.62642456,  0.1859519,  0.32209166,  1.98633424, -0.57271182,  1.18247811,  2.05352048, -0.28297455,  0.25754106,
                  0.80790087, -0.26995007,  1.8223331, -1.80151834,  0.71496981, -0.5119113, -1.45558062,  1.24115387,  1.44295579, -0.24726018,
                  -2.07078337,  1.90810404, -1.36892494, -0.39004086,  1.35998082, 1.50891149, -1.29257757,  0.05513461, -1.58889596,  0.48703248,
                  0.83443891,  0.46234541,  2.20457643,  1.47884097, -0.05316384, 0.72591566,  0.14339927, -1.29137912,  0.07908333,  0.80684167,
                  0.22417797,  0.45241074, -1.03024521,  0.6615743,  0.27216365, 2.4188678,  0.20561134,  0.71095061, -1.02478312,  0.54512964,
                  0.16582386, -0.39648338, -0.77905918, -0.33196771,  0.69407125, -0.81484451,  3.01568098, -0.49053868, -0.60987204,  1.72967348])
    # assign half of them as samples from 1 and the other half as samples from 2
    y = np.concatenate([np.repeat(1, 50), np.repeat(2, 50)], axis=0)

    u, v = k_sample_transform(x, y, is_y_categorical=True)
    mgc = MGC()
    p_value, p_value_metadata = mgc.p_value(u, v)
    assert np.allclose(p_value, 0.819, atol=0.1)


def test_paired_two_sample_transform():
    np.random.seed(1234)
    constant = 0.3

    # case 1: paired data
    paired_X = np.random.normal(0, 1, 1000).reshape(-1, 1)
    paired_Y = paired_X + constant

    # use MGC to perform independence test on "unpaired" data
    u, v = paired_two_sample_transform(paired_X, paired_Y)
    mgc = MGC()
    p_value, p_value_metadata = mgc.p_value(u, v, is_fast=True)

    print(p_value, p_value_metadata)
    # assert np.allclose(p_value, 1.0, atol=0.1)

    # case 2: unpaired data
    unpaired_X = np.random.normal(0, 1, 1000).reshape(-1, 1)
    unpaired_Y = np.random.normal(constant, 1, 1000).reshape(-1, 1)

    # use MGC to perform independence test on "unpaired" data
    u, v = paired_two_sample_transform(unpaired_X, unpaired_Y)
    mgc = MGC()
    p_value, p_value_metadata = mgc.p_value(u, v, is_fast=True)

    print(p_value, p_value_metadata)
    # assert np.allclose(p_value, 0.0, atol=0.1)


def test_paired_two_sample_dcorr():
    np.random.seed(1234)
    constant = 0.3

    # case 1: paired data
    paired_X = np.random.normal(0, 1, 1000).reshape(-1, 1)
    paired_Y = paired_X + constant

    # use DCorr to perform independence test on "paired" data
    p_value, p_value_metadata = paired_two_sample_test_dcorr(paired_X, paired_Y)

    print(p_value, p_value_metadata)
    # assert np.allclose(p_value, 1.0, atol=0.1)

    # case 2: unpaired data
    unpaired_X = np.random.normal(0, 1, 1000).reshape(-1, 1)
    unpaired_Y = np.random.normal(constant, 1, 1000).reshape(-1, 1)

    # use DCorr to perform independence test on "unpaired" data
    p_value, p_value_metadata = paired_two_sample_test_dcorr(unpaired_X, unpaired_Y)

    print(p_value, p_value_metadata)
    # assert np.allclose(p_value, 0.0, atol=0.1)
