"""Vertical gravity results and Jaakko Mäkinen's reference fit."""

from ast import literal_eval
from datetime import UTC, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gmeterpy.units as u
from gmeterpy.cli.vgfit.report import write_report
from gmeterpy.models.vgg import (
    FloatingGravityFitResults,
    PolynomialVerticalGravity,
    PolynomialVerticalGravityFitResults,
)
from gmeterpy.models.vgg.floating import FloatingGravityFit

SOCHI_TIES = Path(__file__).resolve().parents[1] / "data/sochi.csv"


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


def test_influence_table_uses_geodetic_residuals():
    data = _quadratic_data()
    data.loc[0, "delta_g"] += 3
    result = _fit_ties(data)
    influence = result.regression.get_influence()

    table = result.influence_table()

    pd.testing.assert_frame_equal(
        table[result.observations.columns], result.observations
    )
    np.testing.assert_allclose(table.leverage, influence.hat_matrix_diag)
    np.testing.assert_allclose(
        table.studentized_residual, -influence.resid_studentized_internal
    )
    np.testing.assert_allclose(table.cooks_distance, influence.cooks_distance[0])


def test_influence_table_handles_unit_leverage(tmp_path):
    result = FloatingGravityFit().fit(
        [0, 0, 0, 0] * u.m,
        [0.5, 0.5, 0.5, 1.0] * u.m,
        [-147, -146, -148, -288] * u.uGal,
        set_ids=[1, 1, 1, 1],
    )

    table = result.influence_table()
    target = table.loc[table.h == 1.0].iloc[0]

    assert target.leverage == pytest.approx(1)
    assert np.isnan(target.studentized_residual)
    assert np.isnan(target.cooks_distance)

    path = tmp_path / "report.txt"
    assert target.unit_leverage
    write_report(result, path)
    assert any(
        row.split()[-3:] == ["n/a", "n/a", "*"]
        for row in path.read_text(encoding="utf-8").splitlines()
    )


@pytest.mark.parametrize("rename_runs", [False, True])
def test_fit_is_invariant_to_input_order_and_run_names(rename_runs):
    data = pd.read_csv(SOCHI_TIES)
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
    source = (
        Path(__file__).resolve().parents[2] / "examples/vgfit/makinen2010_table1.csv"
    )
    table = pd.read_csv(source)
    lowest = table.groupby("set")["lower_height_m"].transform("min")
    ties = table.loc[table.lower_height_m == lowest]
    result = FloatingGravityFit(degree=2).fit(
        ties.lower_height_m.to_numpy() * u.m,
        ties.upper_height_m.to_numpy() * u.m,
        ties.gravity_difference_ugal.to_numpy() * u.uGal,
        set_ids=ties["set"],
    )

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
        write_report(quadratic_fit, tmp_path / "report.txt")
        assert pd.get_option("display.float_format") is formatter


def test_report_starts_with_fit_summary(quadratic_fit, tmp_path):
    path = tmp_path / "report.txt"
    started = datetime.now(UTC).replace(microsecond=0)
    write_report(quadratic_fit, path)

    report = path.read_text(encoding="utf-8")
    summary = report.split("Fit summary:\n", 1)[1].split("\n\n", 1)[0]
    timestamp = summary.splitlines()[1].removeprefix("Generated at").strip()
    generated = datetime.fromisoformat(timestamp)
    assert generated.utcoffset() == timedelta(0)
    assert started <= generated <= datetime.now(UTC)
    summary = " ".join(summary.split())
    assert "Method Floating gravity OLS" in summary
    assert "Observed height range 0.000 to 1.500 m" in summary
    assert "Input ties 6 Measurement sets 2 Fitted level observations 8" in summary
    assert (
        f"Residual standard deviation {quadratic_fit.s0:.3g} uGal "
        f"Sum of squared residuals {quadratic_fit.regression.ssr:.6g} uGal^2" in summary
    )
    assert "Attention" not in summary
    assert report.index("Fit summary:") < report.index("Transfer model:")


def test_report_lists_full_fit_and_recreates_transfer_model(quadratic_fit, tmp_path):
    path = tmp_path / "report.txt"
    write_report(quadratic_fit, path)

    report = path.read_text(encoding="utf-8")
    transfer_section, full_fit = report.split("Transfer model:\n", 1)[1].split(
        "\n\nFull fit:\n", 1
    )
    assert "Units (gravity, height)  uGal, m" in transfer_section
    assert "a = [a0, a1, a2]" in transfer_section
    assert "g(h) = [1, h, h^2] @ a" in transfer_section
    assert "d = [0, h2 - h1, h2^2 - h1^2]" in transfer_section
    assert "delta_g(h1, h2) = g(h2) - g(h1) = d @ a" in transfer_section
    assert "u(delta_g) = sqrt(d @ C @ d.T)" in transfer_section
    vector, matrix = transfer_section.rsplit("\na = ", 1)[1].split("\nC = ", 1)
    coefficients = literal_eval(vector)
    covariance = literal_eval(matrix)
    recreated = PolynomialVerticalGravity(coefficients, covariance, units=(u.uGal, u.m))
    np.testing.assert_array_equal(
        recreated.coefficients, quadratic_fit.fitted_model.coefficients
    )
    np.testing.assert_array_equal(
        recreated.covariance, quadratic_fit.fitted_model.covariance
    )

    model_section, covariance_section, _ = full_fit.split("\n\n", 2)
    assert "Floating gravity = gravity polynomial + set offset" not in model_section
    rows = [line.split() for line in model_section.splitlines()[2:]]
    assert rows[:3] == [
        ["a0", "0", "fixed", "uGal"],
        ["a1", "-300", f"{quadratic_fit.sigma['a1']:.6g}", "uGal/m"],
        ["a2", "12", f"{quadratic_fit.sigma['a2']:.6g}", "uGal/m^2"],
    ]
    for run, row in enumerate(rows[3:], start=1):
        name = f"offset[run={run}]"
        assert row == [
            f"offset[{run}]",
            "0.000",
            f"{quadratic_fit.params[name]:.6g}",
            f"{quadratic_fit.sigma[name]:.6g}",
            "uGal",
        ]
    assert len(rows) == 5
    assert covariance_section.startswith("Full covariance matrix\n")
    covariance_rows = [line.split() for line in covariance_section.splitlines()[2:]]
    labels = ["offset[1]", "offset[2]", "a1", "a2"]
    assert covariance_rows[0] == labels
    assert [row[0] for row in covariance_rows[1:]] == labels
    np.testing.assert_allclose(
        np.array([row[1:] for row in covariance_rows[1:]], dtype=float),
        quadratic_fit.parameter_covariance.to_numpy(),
        rtol=5e-6,
    )
    assert "h (m)" in report
    assert "g (uGal)" in report
    assert "resid (uGal)" in report


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
        # The secant from 0 to 1 m for g(h) = -300 h + 12 h² is -288 µGal/m.
        np.testing.assert_allclose(
            curve.get_xdata(),
            -12 * heights + 12 * heights**2,
            atol=1e-10,
        )
        np.testing.assert_allclose(heights[[0, -1]], [0, 1.5])
        assert ax.get_xlabel().startswith("Detrended gravity")
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
        np.testing.assert_allclose(heights[[0, -1]], [0, 1.5])
        np.testing.assert_allclose(curve.get_xdata(), 12 * heights**2, atol=1e-10)
        extrapolated = next(
            line for line in ax.lines if line.get_label() == "Extrapolated fit"
        )
        assert extrapolated.get_linestyle() == "--"
        extrapolated_heights = extrapolated.get_ydata()[
            np.isfinite(extrapolated.get_xdata())
        ]
        np.testing.assert_allclose(extrapolated_heights[[0, -1]], [1.5, 1.8])
        for line, (_, group) in zip(
            ax.lines[:2], observations.groupby("set_ids"), strict=True
        ):
            group = group.sort_values("h", kind="stable")
            np.testing.assert_allclose(
                line.get_xdata(), 12 * group.h**2 - group.resid, atol=1e-10
            )
        band = ax.collections[0]
        assert band.get_label() == "Standard uncertainty relative to 0.800 m"
        vertices = band.get_paths()[0].vertices
        reference = vertices[vertices[:, 1] == 0.8, 0]
        assert reference.size > 0
        np.testing.assert_allclose(reference, 12 * 0.8**2)
        np.testing.assert_allclose(ax.lines[-1].get_ydata(), [0.8, 0.8])
        assert ax.get_xlabel().startswith("Detrended gravity")
        assert "subtracted" in ax.get_title()
        assert "Ref. height 0.800 m" in ax.get_title()
        assert "\n" not in ax.get_title()
        assert not figure.legends
    finally:
        plt.close(figure)
    pd.testing.assert_frame_equal(quadratic_fit.observations, observations)
    np.testing.assert_array_equal(quadratic_fit.fitted_model.coefficients, coefficients)


def test_plot_without_covariance_leaves_no_figure(quadratic_fit):
    result = FloatingGravityFitResults(
        quadratic_fit.observations,
        quadratic_fit.regression,
        PolynomialVerticalGravity(quadratic_fit.fitted_model.coefficients),
    )
    figures = plt.get_fignums()

    with pytest.raises(ValueError, match="Coefficient covariance is required"):
        result.plot()

    assert plt.get_fignums() == figures


def test_plot_connects_observations_in_height_order():
    result = FloatingGravityFit().fit(
        [0, 0, 0] * u.m,
        [0.500, 0.501, 1.0] * u.m,
        [-149, -148, -290] * u.uGal,
        set_ids=[1, 1, 1],
    )

    figure = result.plot()
    try:
        np.testing.assert_array_equal(
            figure.axes[0].lines[0].get_ydata(), [0, 0.500, 0.501, 1.0]
        )
    finally:
        plt.close(figure)


def test_plot_reconstructs_noisy_observations():
    result = _fit_ties(pd.read_csv(SOCHI_TIES))
    figure = result.plot(subtract_gradient=-300)
    try:
        ax = figure.axes[0]
        groups = list(result.observations.groupby("set_ids"))
        for line, (run_id, group) in zip(ax.lines[: len(groups)], groups, strict=True):
            group = group.sort_values("h", kind="stable")
            offset = result.params[f"offset[run={run_id}]"]
            np.testing.assert_allclose(
                line.get_xdata(), group.g - offset + 300 * group.h, atol=1e-10
            )
            predicted = offset + result.fitted_model.gravity(np.asarray(group.h) * u.m)[
                0
            ].to_value(u.uGal)
            np.testing.assert_allclose(group.resid, predicted - group.g, atol=1e-10)
        height = result.observations.h.min()
        fitted, _ = result.fitted_model.gravity(height * u.m)
        _, uncertainty = result.fitted_model.gravity_difference(
            0.710 * u.m, height * u.m
        )
        vertices = ax.collections[0].get_paths()[0].vertices
        limits = np.unique(vertices[vertices[:, 1] == height, 0])
        np.testing.assert_allclose(
            limits,
            (fitted + np.array([-1, 1]) * uncertainty).to_value(u.uGal) + 300 * height,
        )
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
