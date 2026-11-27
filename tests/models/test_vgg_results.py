"""Vertical gravity results and Jaakko Mäkinen's reference fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gmeterpy.units as u
from gmeterpy.models.vgg import (
    FloatingGravityFitResults,
    PolynomialVerticalGravityFitResults,
)
from gmeterpy.models.vgg.floating import FloatingGravityFit


def _fit_ties(data, degree=2):
    return FloatingGravityFit(degree=degree).fit(
        data.level_1.to_numpy(dtype=float) / 1000 * u.m,
        data.level_2.to_numpy(dtype=float) / 1000 * u.m,
        data.delta_g.to_numpy(dtype=float) * u.uGal,
        set_ids=data.runn,
    )


def _quadratic_data():
    # g(h) = -300 h + 12 h², with input heights in millimetres.
    data = pd.DataFrame(
        {
            "runn": [1, 1, 1, 2, 2, 2],
            "level_1": [0, 0, 0, 0, 0, 0],
            "level_2": [500, 1000, 1500, 500, 1000, 1500],
            "delta_g": [-147, -288, -423, -147, -288, -423],
        }
    )

    return data


@pytest.fixture
def quadratic_fit():
    return _fit_ties(_quadratic_data())


def test_fit_recovers_quadratic(quadratic_fit, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _fit_ties(_quadratic_data(), degree=2)

    assert isinstance(result, FloatingGravityFitResults)
    assert type(result.regression.model).__name__ == "OLS"
    np.testing.assert_allclose(
        result.fitted_model.coefficients, [0, -300, 12], atol=1e-10
    )
    np.testing.assert_allclose(result.observations.resid, 0, atol=1e-10)
    assert list(result.parameter_covariance.columns[-2:]) == ["a1", "a2"]
    assert result.fitted_model.covariance.shape == (3, 3)

    assert list(tmp_path.iterdir()) == []


def test_correction_and_gradient(quadratic_fit):
    assert quadratic_fit.fitted_model.gravity_difference(
        np.asarray(0.5) * u.m, np.asarray(1.5) * u.m
    )[0].to_value(u.uGal) == pytest.approx(-276)
    assert quadratic_fit.fitted_model.gravity_difference(
        np.asarray(1.5) * u.m, np.asarray(0.5) * u.m
    )[0].to_value(u.uGal) == pytest.approx(276)
    assert (
        quadratic_fit.fitted_model.gravity_difference(
            np.asarray(0.5) * u.m, np.asarray(0.5) * u.m
        )[0].to_value(u.uGal)
        == 0
    )
    np.testing.assert_allclose(
        quadratic_fit.fitted_model.gravity_difference(
            np.asarray(0) * u.m, np.asarray([0.5, 1, 1.5]) * u.m
        )[0].to_value(u.uGal),
        [-147, -288, -423],
    )
    assert quadratic_fit.fitted_model.gradient(np.asarray(1) * u.m)[0].to_value(
        u.uGal / u.m
    ) == pytest.approx(-276)
    np.testing.assert_allclose(
        quadratic_fit.fitted_model.gradient(np.asarray([0.5, 1.5]) * u.m)[0].to_value(
            u.uGal / u.m
        ),
        [-288, -264],
    )


def test_fit_owns_one_fitted_model(quadratic_fit):
    model = quadratic_fit.fitted_model
    np.testing.assert_allclose(model.coefficients, [0, -300, 12], atol=1e-10)
    assert quadratic_fit.fitted_model is model
    np.testing.assert_array_equal(
        model.covariance,
        np.pad(quadratic_fit.parameter_covariance.iloc[-2:, -2:], ((1, 0), (1, 0))),
    )
    assert model.gravity_difference(np.asarray(0.5) * u.m, np.asarray(1.5) * u.m)[
        0
    ].to_value(u.uGal) == pytest.approx(
        quadratic_fit.fitted_model.gravity_difference(
            np.asarray(0.5) * u.m, np.asarray(1.5) * u.m
        )[0].to_value(u.uGal)
    )
    covariance = model.covariance
    covariance[:] = 999
    assert quadratic_fit.fitted_model.gravity_difference(
        np.asarray(0.5) * u.m, np.asarray(1.5) * u.m
    )[0].to_value(u.uGal) == pytest.approx(-276)


def test_fit_diagnostics(quadratic_fit):
    assert quadratic_fit.number_of_observations == 8
    assert quadratic_fit.number_of_unknowns == 4
    assert quadratic_fit.degrees_of_freedom == 4
    np.testing.assert_allclose(quadratic_fit.residuals, 0, atol=1e-10)
    np.testing.assert_allclose(quadratic_fit.params, [0, 0, -300, 12], atol=1e-10)
    np.testing.assert_allclose(quadratic_fit.sigma, 0, atol=1e-10)
    assert quadratic_fit.s0 == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("rename_runs", [False, True])
def test_fit_is_invariant_to_input_order_and_run_names(rename_runs):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    data = pd.read_csv(source)
    expected = _fit_ties(data)
    shuffled = data.sample(frac=1, random_state=12)
    names = {
        run: f"set-{10 - run}" if rename_runs else run for run in data.runn.unique()
    }
    shuffled = shuffled.assign(runn=shuffled.runn.map(names))

    result = _fit_ties(shuffled)

    np.testing.assert_allclose(
        result.fitted_model.coefficients,
        expected.fitted_model.coefficients,
    )
    np.testing.assert_allclose(
        result.fitted_model.covariance, expected.fitted_model.covariance
    )
    assert result.fitted_model.gravity_difference(
        np.asarray(0.258) * u.m, np.asarray(1) * u.m
    )[0].to_value(u.uGal) == pytest.approx(
        expected.fitted_model.gravity_difference(
            np.asarray(0.258) * u.m, np.asarray(1) * u.m
        )[0].to_value(u.uGal)
    )
    assert result.fitted_model.gravity_difference(
        np.asarray(0.258) * u.m, np.asarray(1) * u.m
    )[1].to_value(u.uGal) == pytest.approx(
        expected.fitted_model.gravity_difference(
            np.asarray(0.258) * u.m, np.asarray(1) * u.m
        )[1].to_value(u.uGal)
    )
    assert result.params.index.is_unique
    for run_id, name in names.items():
        assert result.params[f"offset[run={name}]"] == pytest.approx(
            expected.params[f"offset[run={run_id}]"]
        )


def test_jaakko_makinen_2010_polynomial_fit_ties():
    # Table 1, using one reference per set. The other two differences in
    # sets 4 and 5 are implied by these level values, not extra observations.
    data = pd.DataFrame(
        [
            (1, 47, 982, -398.0, 0.9),
            (2, 47, 982, -396.4, 0.7),
            (3, 47, 982, -402.1, 2.3),
            (4, 49, 788, -312.9, 1.6),
            (4, 49, 1311, -527.0, 1.6),
            (5, 49, 788, -311.5, 1.3),
            (5, 49, 1311, -524.0, 1.4),
        ],
        columns=["runn", "level_1", "level_2", "delta_g", "std"],
    )
    result = _fit_ties(data)

    # Section 5 / Fig. 2a: published coefficients and one-sigma errors.
    np.testing.assert_allclose(
        result.params.iloc[-2:], [-441.2, 16.8], atol=0.05, rtol=0
    )
    np.testing.assert_allclose(result.sigma.iloc[-2:], [6.2, 5.0], atol=0.05, rtol=0)
    assert result.fitted_model.gravity_difference(
        np.asarray(1.2) * u.m, np.asarray(0.05) * u.m
    )[0].to_value(u.uGal) == pytest.approx(483.2, abs=0.05)
    assert result.fitted_model.gravity_difference(
        np.asarray(1.2) * u.m, np.asarray(0.05) * u.m
    )[1].to_value(u.uGal) == pytest.approx(1.8, abs=0.05)
    assert result.number_of_observations == 12
    assert result.number_of_unknowns == 7
    assert result.degrees_of_freedom == 5


def test_fit_preserves_input_and_converts_heights(quadratic_fit):
    data = _quadratic_data().astype(
        {"level_1": "Int64", "level_2": "Int64", "delta_g": "Float64"}
    )
    original = data.copy()
    result = _fit_ties(data)
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_allclose(sorted(result.observations.h.unique()), [0, 0.5, 1, 1.5])
    assert result.fitted_model.gravity_difference(
        np.asarray(0) * u.m, np.asarray(1) * u.m
    )[0].to_value(u.uGal) == pytest.approx(-288)


@pytest.mark.parametrize(
    "attribute",
    ["observations", "params", "sigma", "residuals", "parameter_covariance"],
)
def test_results_return_independent_copies(quadratic_fit, attribute):
    original = getattr(quadratic_fit, attribute)
    edited = getattr(quadratic_fit, attribute)
    edited.iloc[:] = 999

    assert getattr(quadratic_fit, attribute).equals(original)
    assert quadratic_fit.fitted_model.gravity_difference(
        np.asarray(0) * u.m, np.asarray(1) * u.m
    )[0].to_value(u.uGal) == pytest.approx(-288)


def test_results_snapshot_observations(quadratic_fit):
    observations = quadratic_fit.observations
    result = FloatingGravityFitResults(
        observations, quadratic_fit.regression, quadratic_fit.fitted_model
    )
    observations.loc[:, "h"] = 999
    observations.loc[:, "resid"] = 999
    pd.testing.assert_frame_equal(result.observations, quadratic_fit.observations)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_results_prepare_model_outputs(quadratic_fit, degree):
    fitted = _fit_ties(_quadratic_data(), degree=degree)
    observations = fitted.observations.drop(columns="resid")
    original = observations.copy()

    result = FloatingGravityFitResults(
        observations, fitted.regression, fitted.fitted_model
    )

    assert result.fitted_model.degree == degree
    assert result.fitted_model.covariance.shape == (degree + 1, degree + 1)
    np.testing.assert_allclose(
        result.fitted_model.gravity(np.asarray(fitted.observations.h) * u.m)[
            0
        ].to_value(u.uGal),
        fitted.fitted_model.gravity(np.asarray(fitted.observations.h) * u.m)[
            0
        ].to_value(u.uGal),
    )
    np.testing.assert_allclose(result.observations.resid, -fitted.regression.resid)
    pd.testing.assert_frame_equal(observations, original)


def test_report_preserves_pandas_formatting(quadratic_fit, tmp_path):
    formatter = "{:.8e}".format
    with pd.option_context("display.float_format", formatter):
        quadratic_fit.report(tmp_path / "report.txt")
        assert pd.get_option("display.float_format") is formatter


def test_plot_leaves_figure_with_caller(quadratic_fit, tmp_path):
    figure = quadratic_fit.plot()
    try:
        figure.savefig(tmp_path / "plot.png")
        assert plt.fignum_exists(figure.number)
        ax = figure.axes[0]
        assert ax.get_autoscalex_on()
        assert ax.get_autoscaley_on()
        curve = ax.lines[2]
        heights = curve.get_ydata()
        np.testing.assert_allclose(
            curve.get_xdata(),
            quadratic_fit.fitted_model.gravity(np.asarray(heights) * u.m)[0].to_value(
                u.uGal
            ),
        )
        np.testing.assert_allclose(heights[[0, -1]], [0, 1.5])
        assert "substructed" not in ax.get_title()
    finally:
        plt.close(figure)


def test_plot_presentation_options(quadratic_fit):
    observations = quadratic_fit.observations.copy()
    coefficients = quadratic_fit.fitted_model.coefficients.copy()
    figure = quadratic_fit.plot(
        h_ref=0.8,
        subtract_gradient=-300,
        xlim=(-10, 50),
        ylim=(-0.1, 2),
        height_range=(0, 1.8),
    )
    try:
        ax = figure.axes[0]
        assert ax.get_xlim() == (-10, 50)
        assert ax.get_ylim() == (-0.1, 2)
        curve = ax.lines[2]
        heights = curve.get_ydata()
        np.testing.assert_allclose(heights[[0, -1]], [0, 1.8])
        np.testing.assert_allclose(curve.get_xdata(), 12 * heights**2, atol=1e-10)
        for line, (_, group) in zip(
            ax.lines[:2], observations.groupby("set_ids"), strict=True
        ):
            np.testing.assert_allclose(
                line.get_xdata(), 12 * group.h**2 - group.resid, atol=1e-10
            )
        uncertainty = quadratic_fit.fitted_model.gravity_difference(
            np.asarray(heights) * u.m, np.asarray(0.8) * u.m
        )[1].to_value(u.uGal)
        np.testing.assert_allclose(
            ax.lines[3].get_xdata(), curve.get_xdata() - uncertainty
        )
        np.testing.assert_allclose(
            ax.lines[4].get_xdata(), curve.get_xdata() + uncertainty
        )
        assert "-300.0" in ax.get_title()
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(quadratic_fit.observations, observations)
    np.testing.assert_array_equal(quadratic_fit.fitted_model.coefficients, coefficients)


def test_plot_reconstructs_noisy_observations():
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    result = _fit_ties(pd.read_csv(source))
    figure = result.plot()
    try:
        groups = list(result.observations.groupby("set_ids"))
        for line, (run_id, group) in zip(
            figure.axes[0].lines[: len(groups)], groups, strict=True
        ):
            offset = result.params[f"offset[run={run_id}]"]
            np.testing.assert_allclose(line.get_xdata(), group.g - offset, atol=1e-10)
            predicted = offset + result.fitted_model.gravity(np.asarray(group.h) * u.m)[
                0
            ].to_value(u.uGal)
            np.testing.assert_allclose(group.resid, predicted - group.g, atol=1e-10)
    finally:
        plt.close(figure)


def test_fit_requires_positive_degree():
    with pytest.raises(ValueError, match="degree must be at least one"):
        _fit_ties(_quadratic_data(), degree=0)


def test_prediction_table_from_shared_results(quadratic_fit):
    results = PolynomialVerticalGravityFitResults(
        quadratic_fit.regression, quadratic_fit.fitted_model
    )
    h1, h2 = 100 * u.cm, [50, 150] * u.cm

    table = results.prediction_table(h1, h2)

    np.testing.assert_allclose(table.h1, [1, 1])
    np.testing.assert_allclose(table.h2, [0.5, 1.5])
    np.testing.assert_allclose(table.gravity_difference, [141, -135])
    np.testing.assert_allclose(table.mean_gradient, [-282, -270])
