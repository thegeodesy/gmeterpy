"""Preparation of common-reference vertical gravity ties."""

import pandas as pd
import pytest

import gmeterpy.units as u
from gmeterpy.models.vgrad.floating import FloatingGravityFit


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
            "set_ids": [1, 1, 1, 2, 2],
        }
    )

    result = FloatingGravityFit().fit(
        data.level_1.to_numpy() * u.mm,
        data.level_2.to_numpy() * u.mm,
        data.delta_g.to_numpy() * u.uGal,
        set_ids=data.runn,
    )
    observations = result.observations.drop(columns="resid")[["g", "h", "set_ids"]]

    pd.testing.assert_frame_equal(observations, expected)
    pd.testing.assert_frame_equal(data, original)


def test_fit_requires_common_reference_within_each_set():
    with pytest.raises(ValueError, match="common reference height"):
        FloatingGravityFit().fit(
            [0, 0.5] * u.m,
            [0.5, 1] * u.m,
            [-147, -141] * u.uGal,
            set_ids=[1, 1],
        )


def test_equal_target_ties_remain_separate():
    result = FloatingGravityFit(degree=1).fit(
        [0, 0, 0] * u.m,
        [0.5, 0.5, 1] * u.m,
        [-150, -150, -300] * u.uGal,
        set_ids=[1, 1, 1],
    )

    assert result.observations.g.tolist() == [0, -150, -150, -300]
    assert result.degrees_of_freedom == 2


def test_fit_requires_identifiable_coefficients():
    with pytest.raises(ValueError, match="cannot identify a degree-2 polynomial"):
        FloatingGravityFit().fit(
            [0, 0, 0] * u.m,
            [0.5, 0.5, 0.5] * u.m,
            [-149, -150, -151] * u.uGal,
            set_ids=[1, 1, 1],
        )


def test_fit_requires_residual_degrees_of_freedom():
    with pytest.raises(ValueError, match="needs residual degrees of freedom"):
        FloatingGravityFit().fit(
            [0, 0] * u.m,
            [0.5, 1] * u.m,
            [-147, -288] * u.uGal,
            set_ids=[1, 1],
        )
