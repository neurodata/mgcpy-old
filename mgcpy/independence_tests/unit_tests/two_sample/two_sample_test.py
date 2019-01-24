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
    print("DCorr (Un-Biased)")
    (dcorr_ut, _), (dcorr_up, _) = dcorr_unbiased.test(men[:100], women[:100])
    assert np.allclose(dcorr_ut, -0.0008913292636794623)
    assert np.allclose(dcorr_up, 0.4502224743821557, rtol=1.e-1)

    print("DCorr (Biased)")
    (dcorr_bt, _), (dcorr_bp, _) = dcorr_biased.test(men[:100], women[:100])
    assert np.allclose(dcorr_bt, 0.008266367785130459)
    assert np.allclose(dcorr_bp, 0.415, rtol=1.e-1)

    print("Mantel")
    (mantel_t, _), (mantel_p, _) = mantel.test(men[:100], women[:100])
    assert np.allclose(mantel_t, 0.005235443585975389)
    assert np.allclose(mantel_p, 0.417, rtol=1.e-1)

    print("RVCorr")
    (rv_corr_t, _), (rv_corr_p, _) = rv_corr.test(men[:100], women[:100])
    assert np.allclose(rv_corr_t, 0.005450544032308922)
    assert np.allclose(rv_corr_p, 0.296, atol=1.e-1)

    print("CCA")
    (cca_t, _), (cca_p, _) = cca.test(men[:100], women[:100])
    assert np.allclose(cca_t, 0.005450544032308922)
    assert np.allclose(cca_p, 0.273, atol=1.e-1)

    print("Pearson")
    (pearson_t, _), (pearson_p, _) = pearson.test(men[:100], women[:100])
    assert np.allclose(pearson_t, -0.0738278)
    assert np.allclose(pearson_p, 0.29882475, rtol=1.e-1)

    print("Kendall")
    (kendall_t, _), (kendall_p, _) = kendall.test(men[:100], women[:100])
    assert np.allclose(kendall_t, -0.048544835126699805)
    assert np.allclose(kendall_p, 0.4033465589917702, rtol=1.e-1)

    print("Spearman")
    (spearman_t, _), (spearman_p, _) = spearman.test(men[:100], women[:100])
    assert np.allclose(spearman_t, -0.05923832207225863)
    assert np.allclose(spearman_p, 0.40471089369759095, rtol=1.e-1)

    print("HHG")
    (hhg_t, _), (hhg_p, _) = hhg.test(men[:10], women[:10])
    assert np.allclose(hhg_t, 114.26451890021742)
    assert np.allclose(hhg_p, 0.679, rtol=1.e-1)

    print("MGC")
    (mgc_t, _), (mgc_p, _) = mgc.test(men[:25], women[:25])
    assert np.allclose(mgc_t, -0.012762686636339487)
    assert np.allclose(mgc_p, 0.6110000000000004, rtol=1.e-1)
