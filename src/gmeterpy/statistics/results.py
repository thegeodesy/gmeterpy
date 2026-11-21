"""Shared access to least-squares regression results."""

import numpy as np


class FitResults:
    """Diagnostics from a fitted statsmodels least-squares regression.

    Parameters
    ----------
    regression : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted OLS, WLS or GLS regression.
    model : object, optional
        Source model or measurements associated with the regression.

    Notes
    -----
    Residuals use fitted minus observed values. Parameter counts include all
    coefficients, including constants; residual degrees of freedom come from
    the backend and account for rank. Public arrays and tables are copies.
    The underlying regression is shared for inspection and should be treated
    as read-only. Its native residuals retain the opposite statsmodels sign.
    Units are those of the supplied regression.
    """

    def __init__(self, regression, *, model=None):
        """Retain the backend regression."""
        self._regression = regression
        self.model = model

    @property
    def regression(self):
        """Underlying statsmodels result for inspection.

        Returns
        -------
        statsmodels.regression.linear_model.RegressionResultsWrapper
            Full fit and diagnostics. This object is shared, not copied;
            treat it as read-only.
        """
        return self._regression

    @property
    def residuals(self):
        """Unweighted residuals: fitted minus observed, in observation units.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Unweighted residuals: fitted minus observed, in observation units."""
        return -self.regression.resid.copy()

    @property
    def whitened_residuals(self):
        """Whitened residuals using the geodetic sign convention.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Fitted minus observed residuals after the backend's whitening
            transformation. These are not necessarily studentized residuals.
        """
        return -self.regression.wresid.copy()

    @property
    def params(self):
        """Fitted regression parameters.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Fitted regression parameters.
            Units and parameter ordering are defined by the fitted model."""
        return self.regression.params.copy()

    @property
    def sigma(self):
        """Parameter standard errors, with the same units as ``params``.

        Returns
        -------
        pandas.Series or numpy.ndarray
            Parameter standard errors, with the same units as ``params``."""
        return self.regression.bse.copy()

    @property
    def number_of_observations(self):
        """Number of observations used in the regression.

        Returns
        -------
        float
            Number of observations used in the regression."""
        return self.regression.nobs

    @property
    def number_of_unknowns(self):
        """Number of fitted parameters, including any constant terms.

        Returns
        -------
        int
            Number of fitted parameters, including any constant terms."""
        return len(self.params)

    @property
    def degrees_of_freedom(self):
        """Residual degrees of freedom, accounting for model rank.

        Returns
        -------
        float
            Residual degrees of freedom, accounting for model rank."""
        return self.regression.df_resid

    @property
    def s0(self):
        """Standard deviation of unit weight, the square root of regression scale.

        Returns
        -------
        float
            Standard deviation of unit weight, the square root of regression scale."""
        return np.sqrt(self.regression.scale)

    @property
    def parameter_covariance(self):
        """Covariance of all fitted parameters.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            Independent covariance in backend parameter order. Each entry
            has the product of the corresponding parameter units.
        """
        return self.regression.cov_params().copy()

    def has_params(self, name):
        """Check whether a named parameter matches a regular expression.

        Parameters
        ----------
        name : str
            Regular expression matched against the parameter index.

        Returns
        -------
        bool
            Whether at least one named parameter matches.
        """
        return bool(self.params.index.str.contains(name).any())

    def chi2(self, alpha=0.95, scale=1.0):
        """Compare residual variance with a reference scale.

        Parameters
        ----------
        alpha : float, optional
            Confidence level. Default is 0.95.
        scale : float, optional
            Reference standard deviation of unit weight. Default is 1.

        Returns
        -------
        statistic, lower, upper : float
            Chi-square statistic and reference interval. Assumes the usual
            Gaussian least-squares error model and positive residual degrees
            of freedom.
        """
        from scipy.stats import chi2

        dof = self.degrees_of_freedom
        lower, upper = chi2.interval(alpha, df=dof)
        return self.s0**2 / scale**2 * dof, lower, upper

    def qqplot(self, ax=None, **kwargs):
        """Plot residual quantiles against a fitted normal distribution.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Destination axes. A new figure is created when omitted.
        **kwargs
            Arguments passed to statsmodels' Q-Q plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            Axes containing the plot. Callers own newly created figures.
        """
        import matplotlib.pyplot as plt
        from statsmodels.graphics.gofplots import qqplot

        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3))
        qqplot(self.residuals, fit=True, line="s", ax=ax, **kwargs)
        ax.set_title("Normal Q-Q plot")
        return ax
