import numpy as np
from mgcpy.independence_tests.utils.two_sample import TwoSample


def test_two_sample_tests():
    # read in the data
    mycsv = np.genfromtxt('./mgcpy/independence_tests/unit_tests/two_sample/data/twosample_data.csv', delimiter=',')
    men = []
    women = []

    # create data matrices for men's and women's salaries
    for i in range(1, 5593):
        men.append(mycsv[i][1])
    for j in range(5593, 9399):
        women.append(mycsv[j][1])

    # convert lists to numpy arrays
    men = np.asarray(men)
    women = np.asarray(women)

    # create instances to run tests
    dcorr_unbiased = TwoSample('dcorr_unbiased')
    dcorr_biased = TwoSample('dcorr_biased')
    mantel = TwoSample('mantel')
    rv_corr = TwoSample('rv_corr')
    cca = TwoSample('cca')
    pearson = TwoSample('pearson')
    kendall = TwoSample('kendall')
    spearman = TwoSample('spearman')
    hhg = TwoSample('hhg')
    mgc = TwoSample('mgc')

    # get test statistics and p values for each test
    dcorr_ut, dcorr_up = dcorr_unbiased.test(men[:100], women[:100])
    dcorr_bt, dcorr_bp = dcorr_biased.test(men[:100], women[:100])
    mantel_t, mantel_p = mantel.test(men[:100], women[:100])
    rv_corr_t, rv_corr_p = rv_corr.test(men[:100], women[:100])
    cca_t, cca_p = cca.test(men[:100], women[:100])
    pearson_t, pearson_p = pearson.test(men[:100], women[:100])
    kendall_t, kendall_p = kendall.test(men[:100], women[:100])
    spearman_t, spearman_p = spearman.test(men[:100], women[:100])
    hhg_t, hhg_p = hhg.test(men[:20], women[:20])
    mgc_t, mgc_p = mgc.test(men[:100], women[:100])

    # test statistics
    assert np.allclose(dcorr_ut, -0.0008913292636794623)
    assert np.allclose(dcorr_bt, -0.0008446435181938056)
    assert np.allclose(mantel_t, 0.0005541922995149036)
    assert np.allclose(rv_corr_t, 0.005450544032308922)
    assert np.allclose(cca_t, 0.005450544032308922)
    assert np.allclose(pearson_t, -0.0738278)
    assert np.allclose(kendall_t, -0.048544835126699805)
    assert np.allclose(spearman_t, -0.05923832207225863)
    assert np.allclose(hhg_t, 491.9622752)
    assert np.allclose(mgc_t, -0.00088954874649)

    # p-values
    assert np.allclose(dcorr_up, 0.4502224743821557)
    assert np.allclose(dcorr_bp, 0.415)
    assert np.allclose(mantel_p, 0.904)
    assert np.allclose(rv_corr_p, 0.296)
    assert np.allclose(cca_p, 0.273)
    assert np.allclose(pearson_p, 0.857)
    assert np.allclose(kendall_p, 0.775)
    assert np.allclose(spearman_p, 0.827)
    assert np.allclose(hhg_p, 0.679)
    assert np.allclose(mgc_p, 0.410)
