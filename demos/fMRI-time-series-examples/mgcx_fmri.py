# Import dependencies.
import numpy as np
import nibabel.cifti2 as ci
import math

from mgcpy.independence_tests.dcorrx import DCorrX
from mgcpy.independence_tests.mgcx import MGCX

# Load image - individual 100307.
img = ci.load("rfMRI_REST1_LR_Atlas_hp2000_clean_filt_sm6.HCPMMP.ptseries.nii")
fmri_data = np.array(img.get_fdata())

# Parameters and constants.
verbose = True # Print output to track progress.
M = 0 # Number of lags in the past to inspect.
timesteps = range(300)

# Initialize
n = len(timesteps) # Number of samples.
p = fmri_data.shape[1] # Number of parcels.
dcorrx = DCorrX(max_lag = M)
mgcx = MGCX(max_lag = M)

# i and j represent the parcels of which to test independence.
def compute_dcorrx(i, j):
    X = fmri_data[sample_indices,i].reshape(n, 1)
    Y = fmri_data[sample_indices,j].reshape(n, 1)

    dcorrx_statistic, metadata = dcorrx.test_statistic(X, Y)
    dcorrx_optimal_lag = metadata['optimal_lag']

    return dcorrx_statistic, dcorrx_optimal_lag

def compute_mgcx(i, j):
    X = fmri_data[sample_indices,i].reshape(n, 1)
    Y = fmri_data[sample_indices,j].reshape(n, 1)

    mgcx_statistic, metadata = mgcx.test_statistic(X, Y)
    mgcx_optimal_lag = metadata['optimal_lag']

    return mgcx_statistic, mgcx_optimal_lag

i = 1
j = 1
mgcx_statistic, mgcx_optimal_lag = compute_mgcx(i, j)
print(mgcx_statistic, mgcx_optimal_lag)
