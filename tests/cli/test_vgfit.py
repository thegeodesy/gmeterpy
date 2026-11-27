"""Regression checks for the vertical gradient fitting workflow."""

from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from gmeterpy.cli.vgfit import main
from gmeterpy.models.vgg import FloatingGravityFitResults


@pytest.mark.parametrize("station", ["", "Sochi"])
def test_main_writes_station_outputs(tmp_path, monkeypatch, station):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    monkeypatch.chdir(tmp_path)
    figures = plt.get_fignums()
    main(["-i", str(source), "-n", station])

    assert plt.get_fignums() == figures
    assert plt.imread(tmp_path / f"{station}_all.png").size > 0
    report = (tmp_path / f"vg_{station}_fit_all.txt").read_text()
    assert f"vertical gravity gradients at {station}\n" in report
    assert "0.000 --> 1.400 m" in report


def test_plot_options(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
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
            str(source),
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
            "--height-range",
            "0.1",
            "1.8",
            "--subtract-gradient",
            "-300",
        ]
    )

    ax = figures[0].axes[0]
    assert ax.get_xlim() == (-20, 50)
    assert ax.get_ylim() == (0, 2)
    extrapolated = next(
        line for line in ax.lines if line.get_label() == "Extrapolated fit"
    )
    assert extrapolated.get_ydata()[0] == pytest.approx(0.1)
    assert extrapolated.get_ydata()[-1] == pytest.approx(1.8)
    assert "-300.0" in ax.get_title()
    assert (tmp_path / "Sochi_all.png").is_file()
    assert not plt.fignum_exists(figures[0].number)


def test_stdin_input(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    stream = StringIO(source.read_text())
    monkeypatch.setattr("sys.stdin", stream)
    main(["-i", "-", "-o", str(tmp_path), "--auto-limits"])
    assert not stream.closed
    assert (tmp_path / "_all.png").is_file()


def test_missing_input_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["-i", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "missing.csv" in capsys.readouterr().err


def test_command_preserves_existing_figures(tmp_path):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    existing = plt.figure()
    figures = plt.get_fignums()
    try:
        main(["-i", str(source), "-o", str(tmp_path)])
        assert plt.get_fignums() == figures
    finally:
        plt.close(existing)


def test_command_closes_figure_when_saving_fails(tmp_path):
    source = Path(__file__).resolve().parents[2] / "examples/vgfit/sochi.csv"
    figures = plt.get_fignums()
    with pytest.raises(FileNotFoundError):
        main(["-i", str(source), "-o", str(tmp_path / "missing")])
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
