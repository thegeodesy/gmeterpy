"""Regression checks for the vertical gradient fitting workflow."""

from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gmeterpy.cli.vgfit.command import main
from gmeterpy.models.vgg import FloatingGravityFitResults

SOCHI_TIES = Path(__file__).resolve().parents[1] / "data/sochi.csv"


@pytest.mark.parametrize(
    ("station", "include_sensitivity"), [("", False), ("Sochi", True)]
)
def test_main_writes_station_outputs(
    tmp_path, monkeypatch, station, include_sensitivity
):
    monkeypatch.chdir(tmp_path)
    figures = plt.get_fignums()
    args = [str(SOCHI_TIES), "-n", station]
    review_note = "Checked heights and retained both sets."
    if include_sensitivity:
        args.extend(["--leave-one-set-out", "--review-note", review_note])
    main(args)

    assert plt.get_fignums() == figures
    assert plt.imread(tmp_path / "sochi_vgg_fit.png").size > 0
    report = (tmp_path / "sochi_vgg_fit.txt").read_text()
    assert f"vertical gravity gradients at {station}\n" in report
    assert "0.000 --> 1.400" in report
    assert "h1 --> h2 (m)" in report
    assert "Gravity diff. (uGal)" in report
    assert "Mean grad. (uGal/m)" in report
    assert "Extrap." in report
    prediction_rows = report.split("Prediction table:", 1)[1].splitlines()
    outside = next(row for row in prediction_rows if row.startswith("0.000 --> 1.000"))
    short = next(row for row in prediction_rows if row.startswith("0.700 --> 0.720"))
    assert outside.endswith("yes")
    assert short.endswith("no")
    assert short.split()[3:6] == ["-6.711", "+/-", "0.024"]
    data_section = report.split("Data:\n", 1)[1].split("\n\n", 1)[0]
    source = pd.read_csv(SOCHI_TIES)
    assert source.to_string(index=False, na_rep="", float_format=str) in data_section
    assert "Influence diagnostics:" in report
    summary, inspection = report.split("Inspection:\n", 1)
    assert summary.split("\n\n")[-2].startswith("Fit summary:")
    inspection = inspection.split("\n\n", 1)[0]
    assert (
        "Set 1: check all ties. The shared reference height (258 mm) is flagged."
        in inspection
    )
    assert "Set 5: check the 711 to 1292 mm tie." in inspection
    assert (
        f"Review note: {review_note if include_sensitivity else 'Pending'}"
        in inspection
    )
    assert "*  Observation flagged for inspection." in report
    assert "Lev. > 2p/n (1.176 for this fit)" in report
    assert "Cook's D > 4/n (0.235 for this fit)" in report
    assert (
        "For a flagged shared reference height, check all ties in that set." in report
    )
    assert any(
        row.split()[:3] == ["0.258", "0.000", "1"] and row.split()[-1] == "*"
        for row in report.splitlines()
    )
    assert ("Leave-one-set-out summary:" in report) == include_sensitivity
    if include_sensitivity:
        tie_summary = report.split(
            "Held-out tie errors (predicted minus observed)\n", 1
        )[1].split("\n\n", 1)[0]
        first_set = tie_summary.splitlines()[1].split()
        assert first_set == ["1", "2", "-4.60", "4.72"]
        assert "Largest transfer shift" not in report


def test_fit_with_mapped_columns_and_source_units(tmp_path, capsys):
    data = pd.read_csv(SOCHI_TIES).rename(
        columns={
            "runn": "set",
            "level_1": "lower",
            "level_2": "upper",
            "delta_g": "diff",
        }
    )
    data[["lower", "upper", "diff"]] /= 1000
    source = tmp_path / "ties.csv"
    data.to_csv(source, index=False)

    main(
        [
            "-i",
            str(source),
            "-o",
            str(tmp_path),
            "--columns",
            "set_id=set",
            "h1=lower",
            "h2=upper",
            "delta_g=diff",
            "--height-unit",
            "m",
            "--gravity-unit",
            "mGal",
        ]
    )

    plot_path = tmp_path / "ties_vgg_fit.png"
    report_path = tmp_path / "ties_vgg_fit.txt"
    assert plot_path.is_file()
    assert capsys.readouterr().out.splitlines() == [
        f"Wrote plot to {plot_path}",
        f"Wrote report to {report_path}",
    ]
    report = report_path.read_text()
    assert (
        "Input heights (lower, upper) in m. Gravity differences (diff) in mGal."
        in report
    )
    assert "Match Set to set in the original ties." in report
    assert f"Input source {source}" in " ".join(report.split())
    assert "Input columns  set_id=set, h1=lower, h2=upper, delta_g=diff" in report
    assert "Set 5: check the 0.711 to 1.292 m tie." in report
    assert "-6.711 +/- 0.024" in report
    assert (
        pd.read_csv(source).to_string(index=False, na_rep="", float_format=str)
        in report
    )


@pytest.mark.parametrize("automatic", [False, True])
def test_exclude_sets_from_cli_preserves_input(tmp_path, automatic):
    source = tmp_path / "ties.csv"
    original = SOCHI_TIES.read_bytes()
    source.write_bytes(original)
    options = (
        ["--auto-exclude"]
        if automatic
        else ["--exclude-set", "1", "--exclude-set", "5"]
    )
    main(["-i", str(source), "-o", str(tmp_path), "--leave-one-set-out", *options])

    assert source.read_bytes() == original
    report = (tmp_path / "ties_vgg_fit.txt").read_text()
    summary, held_out = report.split("Leave-one-set-out summary:", 1)
    assert "Excluded sets 1, 5" in " ".join(summary.split())
    if automatic:
        assert "Auto-excluded sets 1, 5" in " ".join(summary.split())
        assert "Set 7: check" in summary
    assert "Ties used 6" in " ".join(summary.split())
    held_out, data = held_out.split("Data:", 1)
    assert [row.split()[0] for row in held_out.splitlines()[4:] if row] == [
        "2",
        "3",
        "4",
        "6",
        "7",
        "8",
    ]
    assert "Original input ties, including excluded sets." in data
    assert (
        pd.read_csv(source).to_string(index=False, na_rep="", float_format=str) in data
    )


def test_plot_options(tmp_path, monkeypatch):
    plot = FloatingGravityFitResults.plot
    figures = []

    def capture_plot(result, *args, **kwargs):
        assert kwargs["h_ref"] == 0.8
        figure = plot(result, *args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(FloatingGravityFitResults, "plot", capture_plot)
    main(
        [
            "-i",
            str(SOCHI_TIES),
            "-n",
            "Sochi",
            "-o",
            str(tmp_path),
            "--h-ref",
            "0.8",
            "--xlim",
            "-20",
            "50",
            "--ylim",
            "0",
            "2",
            "--subtract-gradient",
            "-300",
            "--highlight-set",
            "2",
        ]
    )

    ax = figures[0].axes[0]
    assert ax.get_xlim() == (-20, 50)
    assert ax.get_ylim() == (0, 2)
    extrapolated = next(
        line for line in ax.lines if line.get_label() == "Extrapolated fit"
    )
    assert extrapolated.get_ydata()[0] == pytest.approx(0)
    assert extrapolated.get_ydata()[-1] == pytest.approx(2)
    assert "-300.0" in ax.get_title()
    set_one, set_two = ax.lines[:2]
    assert set_two.get_alpha() > set_one.get_alpha()
    assert set_two.get_zorder() > set_one.get_zorder()
    assert figures[0].texts[0].get_text() == "Sochi · Set 2"
    assert (tmp_path / "sochi_vgg_fit.png").is_file()
    assert not plt.fignum_exists(figures[0].number)


def test_stdin_input(tmp_path, monkeypatch):
    stream = StringIO(SOCHI_TIES.read_text())
    monkeypatch.setattr("sys.stdin", stream)
    plot = FloatingGravityFitResults.plot

    def check_automatic_limits(result, *args, **kwargs):
        assert kwargs["ylim"] is None
        assert kwargs["height_range"] is None
        return plot(result, *args, **kwargs)

    monkeypatch.setattr(FloatingGravityFitResults, "plot", check_automatic_limits)
    main(["-", "-o", str(tmp_path), "--auto-limits"])
    assert not stream.closed
    assert (tmp_path / "vgg_fit.png").is_file()
    report = (tmp_path / "vgg_fit.txt").read_text()
    assert "Input source stdin" in " ".join(report.split())


def test_overwrite_requires_an_explicit_option(tmp_path, capsys):
    report = tmp_path / "sochi_vgg_fit.txt"
    report.write_text("First fit")
    plot = tmp_path / "sochi_vgg_fit.png"
    args = [str(SOCHI_TIES), "-o", str(tmp_path)]

    with pytest.raises(SystemExit) as error:
        main(args)

    assert error.value.code == 2
    assert "--overwrite" in capsys.readouterr().err
    assert report.read_text() == "First fit"
    assert not plot.exists()

    main([*args, "--overwrite"])
    assert "Fit summary:" in report.read_text()
    assert plot.is_file()


def test_no_plot_writes_only_the_report(tmp_path, monkeypatch, capsys):
    plot = tmp_path / "sochi_vgg_fit.png"
    plot.write_bytes(b"First fit")

    def unexpected_plot(*args, **kwargs):
        pytest.fail("--no-plot tried to create a figure")

    monkeypatch.setattr(FloatingGravityFitResults, "plot", unexpected_plot)
    main([str(SOCHI_TIES), "-o", str(tmp_path), "--no-plot"])

    report = tmp_path / "sochi_vgg_fit.txt"
    assert report.is_file()
    assert plot.read_bytes() == b"First fit"
    assert capsys.readouterr().out.strip() == f"Wrote report to {report}"


def test_missing_input_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["-i", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "missing.csv" in capsys.readouterr().err


def test_invalid_number_names_the_input_column_and_row(tmp_path, capsys):
    source = tmp_path / "ties.csv"
    source.write_text("set_id,h1,h2,gravity_diff\n1,0,500,-150\n1,0,1000,abc\n")
    with pytest.raises(SystemExit) as error:
        main([str(source), "--columns", "delta_g=gravity_diff"])

    assert error.value.code == 2
    assert "'gravity_diff' requires a finite number in data row 2. Found 'abc'." in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ([], "Supply an input FILE"),
        (["one.csv", "-i", "two.csv"], "Supply the input as FILE or with --input"),
    ],
)
def test_requires_one_input(args, message, capsys):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_command_preserves_existing_figures(tmp_path):
    existing = plt.figure()
    figures = plt.get_fignums()
    try:
        main(["-i", str(SOCHI_TIES), "-o", str(tmp_path)])
        assert plt.get_fignums() == figures
    finally:
        plt.close(existing)


def test_command_closes_figure_when_saving_fails(tmp_path):
    figures = plt.get_fignums()
    with pytest.raises(FileNotFoundError):
        main(["-i", str(SOCHI_TIES), "-o", str(tmp_path / "missing")])
    assert plt.get_fignums() == figures


@pytest.mark.parametrize("base_mm, expected_gradient", [(0, -288), (1000, -276)])
def test_automatic_detrending(tmp_path, monkeypatch, base_mm, expected_gradient):
    h1 = base_mm / 1000
    h2 = np.array([0.5, 1, 1.5]) + h1
    data = pd.DataFrame(
        {
            "runn": [1, 1, 1],
            "level_1": [base_mm] * 3,
            "level_2": h2 * 1000,
            "delta_g": -300 * (h2 - h1) + 12 * (h2**2 - h1**2),
        }
    )
    data = pd.concat([data, data.assign(runn=2)], ignore_index=True)
    source = tmp_path / "ties.csv"
    data.to_csv(source, index=False)
    plot = FloatingGravityFitResults.plot
    titles = []

    def capture_plot(result, *args, **kwargs):
        figure = plot(result, *args, **kwargs)
        titles.append(figure.axes[0].get_title())
        return figure

    monkeypatch.setattr(FloatingGravityFitResults, "plot", capture_plot)
    with np.errstate(divide="raise", invalid="raise"):
        main(["-i", str(source), "-o", str(tmp_path)])
    assert f"{expected_gradient:.1f}" in titles[0]
