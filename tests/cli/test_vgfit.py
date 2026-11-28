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
        args.extend(["--loso-transfer", "0.270", "1.200", "--review-note", review_note])
    main(args)

    assert plt.get_fignums() == figures
    assert plt.imread(tmp_path / "sochi_vgg_fit.png").size > 0
    report = (tmp_path / "sochi_vgg_fit.txt").read_text()
    assert report.splitlines()[0] == "Vertical gravity fit report"
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
    assert "Influence diagnostics:" not in report
    summary, review = report.split("Review:\n", 1)
    assert summary.split("\n\n")[-2].startswith("Fit summary:")
    assert f"Station {station or 'Unspecified'}" in " ".join(summary.split())
    assert "Excluded sets 0" in " ".join(summary.split())
    assert "Observed height range 0.258 to 1.293 m" in " ".join(summary.split())
    assert "Leave-one-set-out" not in summary
    review = review.split("\n\nTransfer model:\n", 1)[0]
    assert "Review status" not in review
    if include_sensitivity:
        assert f"Review note    {review_note}" in review
        diagnostics = review.split("Leave-one-set-out check\n", 1)[1]
        overview, details = diagnostics.split("\n\n", 1)
        set_one, set_five, other_shifts = details.split("\n\n")
        assert set_one.splitlines() == [
            "Set 1 retained (2 of 2 ties flagged)",
            "  Row 1  0.258 to 0.678 m  -3.54 uGal",
            "  Row 2  0.258 to 1.165 m  -5.66 uGal",
            "  Lowest fitted height on omission  0.258 to 0.261 m",
            "  Transfer shift on omission (uGal)  -1.20",
            "  Standard uncertainty (uGal)  1.06 to 0.799",
        ]
        assert "Set 5 retained (1 of 1 tie flagged)" in set_five
        assert "Row 6  0.711 to 1.292 m  +3.66 uGal" in set_five
        assert "Lowest fitted height" not in set_five
        assert "set 5" not in other_shifts
        assert "8 sets checked, 2 flagged, 0 with unavailable predictions." in overview
        assert "Inspection threshold 3 uGal" in " ".join(summary.split())
        assert "exceeds the Inspection threshold in Fit summary" in overview
        assert "Final-fit inspection" not in review
        assert "predicted minus observed" in overview
    else:
        assert "Review note" not in review
        assert "Inspection threshold" not in summary
        assert "Leave-one-set-out check" not in report
        assert (
            "Set 1: check all ties. The shared reference height (258 mm) is flagged."
            in review
        )
        assert "Set 5: check the 711 to 1292 mm tie." in review


def test_leave_one_set_out_report_uses_original_rows(tmp_path):
    main(
        [
            str(SOCHI_TIES),
            "-o",
            str(tmp_path),
            "--no-plot",
            "--exclude-set",
            "1",
            "--leave-one-set-out",
        ]
    )

    report = (tmp_path / "sochi_vgg_fit.txt").read_text()
    diagnostics = report.split("Leave-one-set-out check\n", 1)[1]
    _, details = diagnostics.split("\n\n", 1)
    assert details.split("\n\n", 1)[0].splitlines() == [
        "Set 5 retained (1 of 1 tie flagged)",
        "  Row 6  0.711 to 1.292 m  +3.53 uGal",
    ]


@pytest.mark.parametrize(
    ("options", "flagged_sets"),
    [(["--leave-one-set-out"], {1, 5}), (["--tie-tolerance", "6"], set())],
)
def test_leave_one_set_out_inspects_errors_above_tolerance(
    tmp_path, options, flagged_sets
):
    main([str(SOCHI_TIES), "-o", str(tmp_path), "--no-plot", *options])

    report = (tmp_path / "sochi_vgg_fit.txt").read_text()
    diagnostics = report.split("Leave-one-set-out check\n", 1)[1]
    check = diagnostics.split("\n\nTransfer model:", 1)[0]
    assert {
        int(line.split()[1]) for line in check.splitlines() if line.startswith("Set ")
    } == flagged_sets
    assert f"8 sets checked, {len(flagged_sets)} flagged" in diagnostics


def test_leave_one_set_out_reports_unavailable_predictions(tmp_path):
    source = tmp_path / "ties.csv"
    source.write_text(
        "set_id,h1,h2,delta_g\nA,0,250,-75\nA,0,500,-148\nA,0,750,-224\nA,0,1000,-295\n"
    )
    main([str(source), "-o", str(tmp_path), "--leave-one-set-out", "--no-plot"])

    report = (tmp_path / "ties_vgg_fit.txt").read_text()
    check = report.split("Leave-one-set-out check\n", 1)[1]
    assert "Set A has 4 ties with unavailable predictions." in check
    assert "No measurement sets remain." in check
    assert "1 set checked, 0 flagged, 1 with unavailable predictions." in check


def test_leave_one_set_out_flags_large_errors_in_review(tmp_path):
    source = tmp_path / "ties.csv"
    source.write_text(
        "set_id,h1,h2,delta_g\n"
        "A,0,1000,-250\nA,0,2000,-500\n"
        "B,0,1000,-299\nB,0,2000,-601\n"
        "C,0,1000,-301\nC,0,2000,-599\n"
        "D,0,1000,-300\nD,0,2000,-601\n"
    )
    main(
        [
            str(source),
            "-d",
            "1",
            "-o",
            str(tmp_path),
            "--leave-one-set-out",
            "--no-plot",
        ]
    )

    report = (tmp_path / "ties_vgg_fit.txt").read_text()
    check = report.split("Leave-one-set-out check\n", 1)[1]
    assert "Set A retained (2 of 2 ties flagged)" in check
    assert "Row 1  0 to 1 m" in check
    assert "Row 2  0 to 2 m" in check


def test_set_diagnostics_include_requested_transfer_shift(tmp_path):
    main(
        [
            str(SOCHI_TIES),
            "-o",
            str(tmp_path),
            "--no-plot",
            "--exclude-set",
            "1",
            "--loso-transfer",
            "0.270",
            "1.200",
            "--tie-tolerance",
            "4",
        ]
    )

    report = (tmp_path / "sochi_vgg_fit.txt").read_text()
    summary, diagnostics = report.split("Leave-one-set-out check\n", 1)
    overview, explanation = diagnostics.split("\n\n", 1)
    assert "7 sets checked, 0 flagged, 0 with unavailable predictions." in overview
    assert "Inspection threshold 4 uGal" in " ".join(summary.split())
    assert "from 0.27 to 1.2 m" in explanation
    assert "set 5 +0.38" in explanation
    assert "set 7 -0.30" in explanation


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
    assert "Input source" not in report
    assert str(source) not in report
    assert "Input columns  set_id=set, h1=lower, h2=upper, delta_g=diff" in report
    assert "Set 5: check the 0.711 to 1.292 m tie." in report
    assert "-6.711 +/- 0.024" in report
    assert (
        pd.read_csv(source).to_string(index=False, na_rep="", float_format=str)
        in report
    )


def test_exclude_sets_from_cli_preserves_input(tmp_path):
    source = tmp_path / "ties.csv"
    original = SOCHI_TIES.read_bytes()
    source.write_bytes(original)
    main(
        [
            "-i",
            str(source),
            "-o",
            str(tmp_path),
            "--leave-one-set-out",
            "--exclude-set",
            "1",
            "--exclude-set",
            "5",
            "--exclude-reason",
            "1",
            "Reference height could not be verified.",
        ]
    )

    assert source.read_bytes() == original
    report = (tmp_path / "ties_vgg_fit.txt").read_text()
    summary, diagnostics = report.split("Leave-one-set-out check\n", 1)
    assert "Excluded sets 2 (IDs 1, 5)" in " ".join(summary.split())
    assert (
        "Manual exclusions\n  Set 1: Reference height could not be verified.\n"
        "  Set 5: Unspecified reason\n"
    ) in summary
    assert "Ties used 6" in " ".join(summary.split())
    overview = diagnostics.split("\n\n", 1)[0]
    data = diagnostics.split("\nData:\n", 1)[1]
    assert overview == "6 sets checked, 0 flagged, 0 with unavailable predictions."
    assert "Original input ties, including excluded sets." in data
    assert (
        pd.read_csv(source).to_string(index=False, na_rep="", float_format=str) in data
    )


def test_cli_rejects_removed_auto_exclusion(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(SOCHI_TIES), "-o", str(tmp_path), "--auto-exclude"])

    assert error.value.code == 2
    assert "unrecognized arguments: --auto-exclude" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_exclusion_reason_requires_a_manual_exclusion(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(SOCHI_TIES), "-o", str(tmp_path), "--exclude-reason", "1", "Reason"])

    assert error.value.code == 2
    assert (
        "--exclude-reason for '1' requires --exclude-set '1'" in capsys.readouterr().err
    )
    assert not list(tmp_path.iterdir())


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
    assert "Input source" not in report


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
