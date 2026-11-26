import numpy as np
from scipy.stats import t, ttest_ind_from_stats


def rms(x):
    """Return root mean square"""
    return np.sqrt(np.mean(x**2))


def t_statistics(x, stderr, n, alpha=0.05):
    """Compare two means and return the t-test diagnostics."""
    stderr = np.sqrt(stderr**2)
    std = stderr * np.sqrt(n)
    t_exp, pval = ttest_ind_from_stats(
        x[0], std[0], n[0], x[1], std[1], n[1], equal_var=False
    )
    df = np.sum(stderr**2) / np.sum(stderr**2 / (n - 2))
    t_crit = t.ppf(1 - alpha / 2, df)
    passed = False
    if pval > alpha:
        passed = True
    return (t_exp, pval, t_crit, df, passed)
