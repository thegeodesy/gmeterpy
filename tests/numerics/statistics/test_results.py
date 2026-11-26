"""Shared regression diagnostics and adjustment compatibility."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from statsmodels.api import OLS, RLM, WLS

from gmeterpy.core.network import NetworkAdjustmentResults
from gmeterpy.core.relative import RelativeReadingsResults
from gmeterpy.numerics.statistics import FitResults


def test_shared_parameter_search_and_diagnostic_plots():
    design = pd.DataFrame({"constant": [1] * 5, "slope": [0, 1, 2, 3, 4]})
    regression = OLS([1, 2, 4, 5, 8], design).fit()
    result = FitResults(regression)
    assert result.has_params("^slope$")
    assert not result.has_params("offset")
    statistic, lower, upper = result.chi2(scale=2)
    assert statistic == pytest.approx(np.sum(result.residuals**2) / 4)
    assert 0 < lower < upper

    figure, ax = plt.subplots()
    try:
        assert result.qqplot(ax=ax) is ax
        assert ax.get_title() == "Normal Q-Q plot"
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("estimator", [OLS, WLS])
def test_least_squares_diagnostics(estimator):
    design = pd.DataFrame({"constant": [1] * 5, "slope": [0, 1, 2, 3, 4]})
    observed = np.array([1, 2, 4, 5, 8])
    kwargs = {"weights": [1, 2, 1, 3, 1]} if estimator is WLS else {}
    regression = estimator(observed, design, **kwargs).fit()
    result = FitResults(regression)

    assert result.regression is regression
    assert result.number_of_observations == 5
    assert result.number_of_unknowns == 2
    assert result.degrees_of_freedom == 3
    np.testing.assert_allclose(result.residuals, design @ result.params - observed)
    np.testing.assert_allclose(result.sigma**2, np.diag(result.parameter_covariance))
    np.testing.assert_allclose(result.residuals, -regression.resid)
    np.testing.assert_allclose(result.whitened_residuals, -regression.wresid)
    assert result.s0 == pytest.approx(np.sqrt(regression.scale))
    for attribute in [
        "params",
        "sigma",
        "residuals",
        "whitened_residuals",
        "parameter_covariance",
    ]:
        original = getattr(result, attribute)
        edited = getattr(result, attribute)
        edited.iloc[:] = 999
        assert getattr(result, attribute).equals(original)


@pytest.mark.parametrize("estimator", [OLS, WLS, RLM])
@pytest.mark.parametrize(
    "result_class", [RelativeReadingsResults, NetworkAdjustmentResults]
)
def test_adjustment_uses_shared_results(estimator, result_class):
    design = np.column_stack([np.ones(5), np.arange(5)])
    regression = estimator([1, 2, 4, 5, 8], design).fit()
    model = SimpleNamespace(_proc={"drift_order": 1}, data=pd.DataFrame({"jd": [0]}))
    result = result_class(model, regression)

    assert isinstance(result, FitResults)
    assert result.model is model
    assert result.regression is regression
    assert result.number_of_unknowns == 2
    assert result.number_of_observations == 5
    assert result.degrees_of_freedom == 3
    assert result.s0 == pytest.approx(np.sqrt(regression.scale))
    np.testing.assert_allclose(result.parameter_covariance, regression.cov_params())
    edited = result.params
    edited[:] = 999
    np.testing.assert_array_equal(result.params, regression.params)


def test_network_tie_residual_sign(monkeypatch):
    design = pd.DataFrame({"constant": [1] * 5, "slope": [0, 1, 2, 3, 4]})
    observed = np.array([1, 2, 4, 5, 8])
    regression = OLS(observed, design).fit()
    ties = pd.DataFrame({"from": ["A"] * 5, "to": ["B"] * 5, "stdev": [1.0] * 5})
    model = SimpleNamespace(ties=SimpleNamespace(data=ties), _method="OLS")
    result = NetworkAdjustmentResults(model, regression)
    monkeypatch.setattr(
        result,
        "adjusted_ties",
        lambda: pd.DataFrame({"from": ["A"], "to": ["B"], "stdev": [0.1]}),
    )

    table = result.ties()

    np.testing.assert_allclose(table.resid, design @ regression.params - observed)
    np.testing.assert_allclose(table.stresid, result.whitened_residuals)
