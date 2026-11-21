from .funcs import rms
from .interpolate import interpolate
from .outlier import grubbs_outlier_test, tau_outlier_test
from .results import FitResults

__all__ = [
    "FitResults",
    "grubbs_outlier_test",
    "interpolate",
    "rms",
    "tau_outlier_test",
]
