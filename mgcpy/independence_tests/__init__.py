# add your independence test imports here to simplify deeper imports to shallower ones
# Example: "from mgcpy import MGC", instead of "from mgcpy.independence_tests.mgc.mgc import MGC"

from .mgc import MGC
from .dcorr import DCorr
from .hhg import HHG
from .kendall_spearman import KendallSpearman
from .rv_corr import RVCorr
