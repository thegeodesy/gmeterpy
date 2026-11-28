"""Preparation of common-reference vertical gravity ties."""

import pandas as pd
import pytest

import gmeterpy.units as u
from gmeterpy.cli.vgfit.report import write_report
from gmeterpy.models.vgg.floating import FloatingGravityFit


def test_prepare_observations():
    data = pd.DataFrame(
        {
            "runn": [2, 1, 1],
            "level_1": [200, 100, 100],
            "level_2": [700, 600, 1100],
            "delta_g": [-145, -146, -290],
            "std": [99, 1, 2],
        }
    )
    original = data.copy()
    expected = pd.DataFrame(
        {
            "g": [0.0, -146, -290, 0, -145],
            "h": [0.1, 0.6, 1.1, 0.2, 0.7],
            "set_id": [1, 1, 1, 2, 2],
        }
    )

    result = FloatingGravityFit().fit(
        data.level_1.to_numpy() * u.mm,
        data.level_2.to_numpy() * u.mm,
        data.delta_g.to_numpy() * u.uGal,
        set_id=data.runn,
    )
    observations = result.observations.drop(columns="resid")[["g", "h", "set_id"]]

    pd.testing.assert_frame_equal(observations, expected)
    pd.testing.assert_frame_equal(data, original)


def test_fit_requires_common_reference_within_each_set():
    with pytest.raises(ValueError) as error:
        FloatingGravityFit().fit(
            [0, 0.5] * u.m,
            [0.5, 1] * u.m,
            [-147, -141] * u.uGal,
            set_id=[1, 1],
        )
    assert str(error.value) == "Set 1 has multiple reference heights: 0.0 m, 0.5 m"


def test_equal_target_ties_remain_separate():
    result = FloatingGravityFit(degree=1).fit(
        [0, 0, 0] * u.m,
        [0.5, 0.5, 1] * u.m,
        [-150, -150, -300] * u.uGal,
        set_id=[1, 1, 1],
    )

    assert result.observations.g.tolist() == [0, -150, -150, -300]
    assert result.degrees_of_freedom == 2


def test_fit_requires_identifiable_coefficients():
    with pytest.raises(ValueError, match="cannot identify a degree-2 polynomial"):
        FloatingGravityFit().fit(
            [0, 0, 0] * u.m,
            [0.5, 0.5, 0.5] * u.m,
            [-149, -150, -151] * u.uGal,
            set_id=[1, 1, 1],
        )


def test_fit_requires_residual_degrees_of_freedom():
    with pytest.raises(ValueError, match="needs residual degrees of freedom"):
        FloatingGravityFit().fit(
            [0, 0] * u.m,
            [0.5, 1] * u.m,
            [-147, -288] * u.uGal,
            set_id=[1, 1],
        )


def test_leave_one_set_out_compares_transfers_and_predicts_original_ties():
    transfers, held_out = FloatingGravityFit(degree=1).leave_one_set_out(
        [0, 0, 0, 0] * u.mm,
        [500, 500, 1000, 1000] * u.mm,
        [-150, -160, -300, -320] * u.uGal,
        set_id=["A", "B", "A", "B"],
        transfer_h1=0.5 * u.m,
        transfer_h2=[1, 1.5] * u.m,
    )

    assert transfers.omitted_set_id.tolist() == ["A", "A", "B", "B"]
    assert transfers.full_fit_delta_g.tolist() == pytest.approx([-155, -310] * 2)
    assert transfers.refit_delta_g.tolist() == pytest.approx([-160, -320, -150, -300])
    assert transfers.transfer_shift.tolist() == pytest.approx([-5, -10, 5, 10])
    assert transfers.full_fit_extrapolated.tolist() == [False, True, False, True]
    assert transfers.refit_extrapolated.tolist() == [False, True, False, True]
    assert transfers.fit_error.isna().all()
    assert held_out.set_id.tolist() == ["A", "B", "A", "B"]
    assert held_out.prediction_error.tolist() == pytest.approx([-10, 10, -20, 20])


def test_fit_and_refits_exclude_the_same_sets():
    fitter = FloatingGravityFit(degree=1)
    ties = (
        [0, 0, 0, 0, 0, 0] * u.m,
        [0.5, 1, 0.5, 1, 1, 2] * u.m,
        [-150, -300, -160, -320, -420, -840] * u.uGal,
    )
    set_id = ["A", "A", "B", "B", "C", "C"]

    result = fitter.fit(*ties, set_id=set_id, exclude_sets="C")
    _, held_out = fitter.leave_one_set_out(*ties, set_id=set_id, exclude_sets=["C"])

    assert result.excluded_set_ids == ("C",)
    assert result.observations.set_id.unique().tolist() == ["A", "B"]
    assert result.fitted_height_range == (0, 1)
    assert result.params["a1"] == pytest.approx(-310)
    assert held_out.set_id.tolist() == ["A", "A", "B", "B"]
    assert held_out.prediction_error.tolist() == pytest.approx([-10, -20, 10, 20])


def test_report_shows_held_out_ties_without_transfer_heights(tmp_path):
    fitter = FloatingGravityFit(degree=1)
    ties = (
        [0, 0, 2, 2] * u.m,
        [0.5, 1, 2.5, 3] * u.m,
        [-150, -300, -160, -320] * u.uGal,
    )
    set_id = [1, 1, 2, 2]

    transfers, held_out = fitter.leave_one_set_out(*ties, set_id=set_id)

    assert transfers.empty
    assert held_out.prediction_error.tolist() == pytest.approx([-10, -20, 10, 20])
    path = tmp_path / "report.txt"
    result = fitter.fit(*ties, set_id=set_id)
    write_report(result, path, held_out_ties=held_out)
    report = path.read_text(encoding="utf-8")
    assert "Held-out tie errors (predicted minus observed)" in report


def test_leave_one_set_out_flags_refit_without_residual_degrees_of_freedom():
    transfers, held_out = FloatingGravityFit().leave_one_set_out(
        [0, 0, 0, 0, 0] * u.m,
        [0.5, 1, 1.5, 0.5, 1] * u.m,
        [-147, -288, -423, -148, -289] * u.uGal,
        set_id=[1, 1, 1, 2, 2],
        transfer_h1=0.5 * u.m,
        transfer_h2=1 * u.m,
    )

    assert "residual degrees of freedom" in transfers.loc[0, "fit_error"]
    assert pd.isna(transfers.loc[0, "transfer_shift"])
    assert pd.isna(transfers.loc[0, "refit_extrapolated"])
    assert pd.isna(transfers.loc[1, "fit_error"])
    assert held_out.loc[held_out.set_id == 1, "prediction_error"].isna().all()
    assert held_out.loc[held_out.set_id == 2, "prediction_error"].tolist() == (
        pytest.approx([1, 1])
    )
