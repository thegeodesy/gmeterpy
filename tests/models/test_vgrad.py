"""Polynomial gravity models without a regression backend."""

import numpy as np
import pandas as pd
import pytest

import gmeterpy.units as u
from gmeterpy.models import PolynomialVerticalGravity


def test_polynomial_gravity_model():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])

    assert model.degree == 2
    np.testing.assert_array_equal(model.coefficients, [0, -300, 12])
    assert model.gravity_difference(np.asarray(0.5) * u.m, np.asarray(1.5) * u.m)[
        0
    ].to_value(u.uGal) == pytest.approx(-276)
    np.testing.assert_allclose(
        model.gradient(np.asarray([0.5, 1.5]) * u.m)[0].to_value(u.uGal / u.m),
        [-288, -264],
    )
    # Sensitivities for the correction are 1 and 2, including covariance.
    expected = np.sqrt(4 + 2 * 1 * 2 + 9 * 2**2)
    np.testing.assert_allclose(
        model.gravity_difference(np.asarray([0.5, 1.5]) * u.m, np.asarray(0.5) * u.m)[
            1
        ].to_value(u.uGal),
        [0, expected],
    )


@pytest.mark.parametrize(
    "heights, expected",
    [
        (0, 0),
        (1, -288),
        ([0, 0.5, 1.5], [0, -147, -423]),
        ([[0, 0.5], [1, 1.5]], [[0, -147], [-288, -423]]),
    ],
)
def test_evaluate_gravity_changes(heights, expected):
    model = PolynomialVerticalGravity([0, -300, 12], units=(u.uGal, u.m))
    np.testing.assert_allclose(
        model.gravity(np.asarray(heights) * u.m)[0].to_value(u.uGal), expected
    )
    np.testing.assert_allclose(
        model.gravity_difference(np.asarray(0) * u.m, np.asarray(heights) * u.m)[
            0
        ].to_value(u.uGal),
        expected,
    )


def test_coefficients_without_covariance():
    model = PolynomialVerticalGravity([0, -300, 12], units=(u.uGal, u.m))
    correction, uncertainty = model.gravity_difference(0 * u.m, 1 * u.m)
    assert correction.to_value(u.uGal) == -288
    assert uncertainty is None
    assert model.covariance is None


def test_model_owns_its_data():
    coefficients = np.array([0, -300.0, 12])
    covariance = pd.DataFrame([[0, 0, 0], [0, 4.0, 1], [0, 1, 9]])
    model = PolynomialVerticalGravity(coefficients, covariance, units=(u.uGal, u.m))
    coefficients[:] = 0
    covariance.iloc[:] = 0
    model.coefficients[:] = 0
    returned_covariance = model.covariance
    returned_covariance[:] = 0
    model.coefficients[:] = 0

    np.testing.assert_array_equal(model.coefficients, [0, -300, 12])
    np.testing.assert_array_equal(model.covariance, [[0, 0, 0], [0, 4, 1], [0, 1, 9]])


def test_model_fit(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "height": [0, 0.5, 1, 1.5],
            "gravity": [0, -147, -288, -423],
            "runn": [1, 1, 1, 1],
        }
    )
    original = data.copy()
    monkeypatch.chdir(tmp_path)

    result = PolynomialVerticalGravity.fit_gravity_differences(
        [0, 0, 0] * u.m,
        np.asarray(data.height.iloc[1:]) * u.m,
        np.asarray(data.gravity.iloc[1:]) * u.uGal,
        set_ids=data.runn.iloc[1:],
    )

    model = result.fitted_model
    np.testing.assert_allclose(model.coefficients, [0, -300, 12], atol=1e-10)
    assert model.gravity_difference(np.asarray(0.5) * u.m, np.asarray(1.5) * u.m)[
        0
    ].to_value(u.uGal) == pytest.approx(-276)
    assert model.covariance.shape == (3, 3)
    pd.testing.assert_frame_equal(data, original)
    assert list(tmp_path.iterdir()) == []


def test_observation_arrays_are_paired_by_position():
    result = PolynomialVerticalGravity.fit_gravity_differences(
        h1=[0, 0, 0] * u.m,
        h2=[0.5, 1, 1.5] * u.m,
        delta_g=[-147, -288, -423] * u.uGal,
        set_ids=pd.Series([1, 1, 1], index=[40, 10, 20]),
    )
    model = result.fitted_model
    np.testing.assert_allclose(model.coefficients, [0, -300, 12], atol=1e-10)


def test_changing_set_reference_does_not_affect_model():
    heights = np.array([0, 0.5, 1, 1.5]) * u.m
    gravity = np.array([0, -147.1, -287.8, -423.2]) * u.uGal
    first = PolynomialVerticalGravity.fit_gravity_differences(
        [0, 0, 0] * u.m, heights[1:], gravity[1:], set_ids=[1, 1, 1]
    ).fitted_model
    last = PolynomialVerticalGravity.fit_gravity_differences(
        [1.5, 1.5, 1.5] * u.m,
        heights[:-1],
        gravity[:-1] - gravity[-1],
        set_ids=[1, 1, 1],
    ).fitted_model
    np.testing.assert_allclose(first.coefficients, last.coefficients)
    np.testing.assert_allclose(first.covariance, last.covariance)


def test_fit_and_evaluate_convert_units():
    result = PolynomialVerticalGravity.fit_gravity_differences(
        [0, 0, 0] * u.cm,
        [50, 100, 150] * u.cm,
        [-0.147, -0.288, -0.423] * u.mGal,
        set_ids=[1, 1, 1],
    )
    model = result.fitted_model
    change = model.gravity_difference(50 * u.cm, 1.5 * u.m)[0]
    assert change.unit == u.uGal
    assert change.value == pytest.approx(-276)
    assert model.gradient(100 * u.cm)[0].unit == u.uGal / u.m
    assert model.gravity_difference(50 * u.cm, 1.5 * u.m)[1].unit == u.uGal
    with pytest.raises(u.UnitsError):
        model.gravity(1 * u.s)[0]


def test_mean_gradient():
    model = PolynomialVerticalGravity(
        [0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]], units=(u.uGal, u.m)
    )
    lower = 50 * u.cm
    upper = [1.0, 1.5] * u.m
    unit = u.uGal / u.m

    gradient, uncertainty = model.mean_gradient(lower, upper)

    assert gradient.unit == uncertainty.unit == unit
    np.testing.assert_allclose(gradient.value, [-282, -276])
    np.testing.assert_allclose(uncertainty.value, np.sqrt([27.25, 44]))
    reverse_gradient, reverse_uncertainty = model.mean_gradient(upper, lower)
    np.testing.assert_allclose(reverse_gradient.value, gradient.value)
    np.testing.assert_allclose(reverse_uncertainty.value, uncertainty.value)

    with pytest.raises(ValueError, match="distinct heights"):
        model.mean_gradient(lower, [0.5, 1.0] * u.m)


def test_gradient_uncertainty():
    model = PolynomialVerticalGravity(
        [0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]], units=(u.uGal, u.m)
    )
    uncertainty = model.gradient([0, 50, 150] * u.cm)[1]
    assert uncertainty.unit == u.uGal / u.m
    np.testing.assert_allclose(uncertainty.value, np.sqrt([4, 15, 91]))


def test_constructor_converts_coefficient_and_covariance_units():
    covariance = pd.DataFrame(
        [[0, 0, 0], [0, 4e-10, 1e-12], [0, 1e-12, 9e-14]],
        index=["a0", "a1", "a2"],
        columns=["a0", "a1", "a2"],
    )
    model = PolynomialVerticalGravity(
        [0, -0.003, 0.0000012], covariance, units=(u.mGal, u.cm)
    )

    np.testing.assert_allclose(model.coefficients, [0, -300, 12])
    np.testing.assert_allclose(model.covariance, [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    assert model.gravity(1 * u.m)[0].to_value(u.uGal) == pytest.approx(-288)


def test_potential_difference():
    model = PolynomialVerticalGravity([0, -300, 12])
    g0 = 9.81 * u.m / u.s**2
    heights = [100, 150] * u.cm

    difference = model.potential_difference(
        50 * u.cm, heights, g0, reference_height=0 * u.m
    )[0]

    assert difference.unit == u.m**2 / u.s**2
    # The antiderivative of the relative gravity is -150 h² + 4 h³.
    np.testing.assert_allclose(
        difference.value, [-4.905 + 109e-8, -9.81 + 287e-8], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        model.potential_difference(heights, 50 * u.cm, g0, reference_height=0 * u.m)[
            0
        ].value,
        -difference.value,
    )

    # At 1 m the absolute gravity is g0 minus 288 microgals.
    from_reference = model.potential_difference(
        50 * u.cm, heights, g0 - 288 * u.uGal, reference_height=100 * u.cm
    )[0]
    np.testing.assert_allclose(
        from_reference.value, difference.value, rtol=0, atol=1e-12
    )


def test_transfer_gravity():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    reference_height = 50 * u.cm
    reference_gravity = 9.81 * u.m / u.s**2

    gravity, uncertainty = model.transfer_gravity(
        1 * u.m,
        reference_gravity + [0, 10] * u.uGal,
        reference_height=reference_height,
        reference_gravity_uncertainty=2 * u.uGal,
    )

    np.testing.assert_allclose(
        (gravity - reference_gravity).to_value(u.uGal), [-141, -131]
    )
    assert uncertainty.shape == gravity.shape
    np.testing.assert_allclose(uncertainty.to_value(u.uGal), np.sqrt(6.8125 + 4))


def test_interval_gravity_and_potential_uncertainty():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    low, high = 0 * u.m, 1 * u.m
    reference_gravity = 9.81 * u.m / u.s**2
    reference_height = 50 * u.cm

    anchored = model.with_reference_gravity(
        reference_gravity,
        reference_height=reference_height,
        reference_gravity_uncertainty=2 * u.uGal,
    )

    # At the interval midpoint the linear term cancels. The quadratic
    # sensitivity is 1/3 - 1/4 = 1/12 m².
    expected = np.hypot(3 / 12, 2)
    for h1, h2 in [(low, high), (high, low)]:
        _, potential_uncertainty = model.potential_difference(
            h1,
            h2,
            reference_gravity,
            reference_height=reference_height,
            reference_gravity_uncertainty=2 * u.uGal,
        )
        mean, mean_uncertainty = anchored.mean_gravity(h1, h2)
        assert potential_uncertainty.to_value(u.m**2 / u.s**2) == pytest.approx(
            expected * 1e-8
        )
        assert mean_uncertainty.to_value(u.uGal) == pytest.approx(expected)

        assert (mean - reference_gravity).to_value(u.uGal) == pytest.approx(1)
    with pytest.raises(ValueError, match="distinct heights"):
        model.mean_gravity(low, low)


def test_curvature_and_uncertainty():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    heights = [0, 50, 100] * u.cm

    curvature, uncertainty = model.curvature(heights)
    np.testing.assert_allclose(curvature.to_value(u.uGal / u.m**2), 24)
    np.testing.assert_allclose(uncertainty.to_value(u.uGal / u.m**2), 6)


def test_anchor_and_replace_reference_gravity():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    reference_gravity = 9.81 * u.m / u.s**2
    anchored = model.with_reference_gravity(
        reference_gravity,
        reference_height=50 * u.cm,
        reference_gravity_uncertainty=2 * u.uGal,
    )
    reanchored = anchored.with_reference_gravity(
        reference_gravity, reference_height=1 * u.m
    )

    gravity, uncertainty = anchored.gravity([0.5, 1.0] * u.m)
    np.testing.assert_allclose(
        (gravity - reference_gravity).to_value(u.uGal), [0, -141], atol=1e-6
    )
    np.testing.assert_allclose(uncertainty.to_value(u.uGal), [2, np.sqrt(10.8125)])
    np.testing.assert_allclose(
        (reanchored.gravity(1 * u.m)[0] - reference_gravity).to_value(u.uGal),
        0,
        atol=1e-6,
    )
    assert model.gravity(0 * u.m)[0] == 0 * u.uGal
    assert anchored.gravity_difference(0.5 * u.m, 1 * u.m) == model.gravity_difference(
        0.5 * u.m, 1 * u.m
    )


def test_reference_gravity_at_construction():
    model = PolynomialVerticalGravity(
        [0, -300, 12],
        [[0, 0, 0], [0, 4, 1], [0, 1, 9]],
        reference_gravity=9.81 * u.m / u.s**2,
        reference_height=50 * u.cm,
        reference_gravity_uncertainty=2 * u.uGal,
    )

    gravity, uncertainty = model.gravity(50 * u.cm)
    assert gravity.to_value(u.m / u.s**2) == pytest.approx(9.81)
    assert uncertainty.to_value(u.uGal) == pytest.approx(2)

    with pytest.raises(ValueError, match="must be supplied together"):
        PolynomialVerticalGravity([0, -300, 12], reference_height=50 * u.cm)


def test_potential_difference_uses_stored_reference():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])
    reference_gravity = 9.81 * u.m / u.s**2
    reference_height = 50 * u.cm
    anchored = model.with_reference_gravity(
        reference_gravity,
        reference_height=reference_height,
        reference_gravity_uncertainty=2 * u.uGal,
    )
    interval = (0 * u.m, 1 * u.m)
    calculate = model.potential_difference
    calculate_anchored = anchored.potential_difference

    assert calculate_anchored(*interval) == calculate(
        *interval,
        reference_gravity,
        reference_height=reference_height,
        reference_gravity_uncertainty=2 * u.uGal,
    )
    assert calculate_anchored(
        *interval, reference_gravity, reference_height=1 * u.m
    ) == calculate(*interval, reference_gravity, reference_height=1 * u.m)
    relative_potential, _ = calculate(*interval)
    assert relative_potential.to_value(u.uGal * u.m) == pytest.approx(146)
    with pytest.raises(ValueError, match="must be supplied together"):
        calculate_anchored(*interval, reference_height=1 * u.m)


def test_mean_gravity_without_a_reference():
    model = PolynomialVerticalGravity([0, -300, 12], [[0, 0, 0], [0, 4, 1], [0, 1, 9]])

    mean, uncertainty = model.mean_gravity(0 * u.m, 1 * u.m)

    assert mean.to_value(u.uGal) == pytest.approx(-146)
    assert uncertainty.to_value(u.uGal) == pytest.approx(np.sqrt(7 / 3))


def test_full_polynomial_uses_constant_and_cross_covariance():
    model = PolynomialVerticalGravity(
        [980_000, -0.3],
        [[4e-6, 1e-6], [1e-6, 9e-6]],
        units=(u.mGal, u.m),
    )
    h1, h2 = 0 * u.m, 2 * u.m

    assert model.degree == 1
    np.testing.assert_allclose(model.coefficients, [980_000_000, -300])
    np.testing.assert_allclose(model.covariance, [[4, 1], [1, 9]])

    gravity, uncertainty = model.gravity(h2)
    assert gravity.to_value(u.uGal) == pytest.approx(979_999_400, rel=0, abs=1e-6)
    assert uncertainty.to_value(u.uGal) == pytest.approx(np.sqrt(44))
    difference, uncertainty = model.gravity_difference(h1, h2)
    assert difference.to_value(u.uGal) == pytest.approx(-600)
    assert uncertainty.to_value(u.uGal) == pytest.approx(6)

    mean, uncertainty = model.mean_gravity(h1, h2)
    assert mean.to_value(u.uGal) == pytest.approx(979_999_700, rel=0, abs=1e-6)
    assert uncertainty.to_value(u.uGal) == pytest.approx(np.sqrt(15))

    potential, uncertainty = model.potential_difference(h1, h2)
    assert potential.to_value(u.m**2 / u.s**2) == pytest.approx(
        -19.599994, rel=0, abs=1e-12
    )
    assert uncertainty.to_value(u.uGal * u.m) == pytest.approx(np.sqrt(60))

    anchored = model.with_reference_gravity(9.81 * u.m / u.s**2, reference_height=h1)
    _, uncertainty = anchored.mean_gravity(h1, h2)
    assert uncertainty.to_value(u.uGal) == pytest.approx(3)


def test_gravity_difference_with_height_uncertainty():
    model = PolynomialVerticalGravity([0, -300], [[0, 0], [0, 4]])
    h1, h2 = 0 * u.m, 1 * u.m

    difference, uncertainty = model.gravity_difference(
        h1, h2, height_uncertainty=1 * u.mm
    )
    assert difference.to_value(u.uGal) == pytest.approx(-300)
    assert uncertainty.to_value(u.uGal) == pytest.approx(np.sqrt(4 + 2 * 0.3**2))

    _, uncertainty = model.gravity_difference(
        h1, h2, height_uncertainty=(0.5 * u.mm, [1, 2] * u.mm)
    )
    np.testing.assert_allclose(
        uncertainty.to_value(u.uGal), np.sqrt(4 + 0.15**2 + np.array([0.3, 0.6]) ** 2)
    )


def test_transfer_gravity_with_height_uncertainty():
    model = PolynomialVerticalGravity([0, -300, 12], np.zeros((3, 3)))

    _, uncertainty = model.transfer_gravity(
        target_height=1 * u.m,
        reference_gravity=9.81 * u.m / u.s**2,
        reference_height=0 * u.m,
        reference_gravity_uncertainty=2 * u.uGal,
        reference_height_uncertainty=1 * u.mm,
        target_height_uncertainty=2 * u.mm,
    )

    assert uncertainty.to_value(u.uGal) == pytest.approx(
        np.sqrt(2**2 + 0.3**2 + 0.552**2)
    )


def test_gravity_with_height_uncertainty():
    model = PolynomialVerticalGravity([0, -300], [[0, 0], [0, 4]])

    gravity, uncertainty = model.gravity(1 * u.m, height_uncertainty=[1, 2] * u.mm)
    np.testing.assert_allclose(gravity.to_value(u.uGal), [-300, -300])
    np.testing.assert_allclose(uncertainty.to_value(u.uGal), np.hypot(2, [0.3, 0.6]))

    anchored = model.with_reference_gravity(
        980_000_000 * u.uGal, reference_height=1 * u.m
    )
    _, uncertainty = anchored.gravity(1 * u.m, height_uncertainty=1 * u.mm)
    assert uncertainty.to_value(u.uGal) == pytest.approx(0.3)


@pytest.mark.parametrize("quadratic", [0, 12])
def test_mean_gradient_with_height_uncertainty(quadratic):
    model = PolynomialVerticalGravity([0, -300, quadratic], np.diag([0, 4, 0]))

    gradient, uncertainty = model.mean_gradient(
        0 * u.m, 1 * u.m, height_uncertainty=(1 * u.mm, 2 * u.mm)
    )
    unit = u.uGal / u.m
    assert gradient.to_value(unit) == pytest.approx(-300 + quadratic)
    assert uncertainty.to_value(unit) == pytest.approx(
        np.hypot(2, quadratic * np.hypot(0.001, 0.002))
    )


def test_derivative_height_uncertainties():
    model = PolynomialVerticalGravity([0, -300, 12, 2], np.zeros((4, 4)))

    _, gradient_sigma = model.gradient(1 * u.m, height_uncertainty=1 * u.mm)
    _, curvature_sigma = model.curvature(1 * u.m, height_uncertainty=1 * u.mm)

    assert gradient_sigma.to_value(u.uGal / u.m) == pytest.approx(0.036)
    assert curvature_sigma.to_value(u.uGal / u.m**2) == pytest.approx(0.012)


def test_mean_and_potential_height_uncertainties():
    model = PolynomialVerticalGravity([0, -300], np.zeros((2, 2)))
    start, end = 0 * u.m, 1 * u.m
    height_uncertainty = (1 * u.mm, 2 * u.mm)
    anchored = model.with_reference_gravity(9.81 * u.m / u.s**2, reference_height=start)

    _, mean_sigma = anchored.mean_gravity(
        start, end, height_uncertainty=height_uncertainty
    )
    _, potential_sigma = anchored.potential_difference(
        start, end, height_uncertainty=height_uncertainty
    )
    assert mean_sigma.to_value(u.uGal) == pytest.approx(150 * np.hypot(0.001, 0.002))
    assert potential_sigma.to_value(u.m**2 / u.s**2) == pytest.approx(
        np.hypot(9.81 * 0.001, 9.809997 * 0.002)
    )

    _, explicit_sigma = model.potential_difference(
        start,
        end,
        9.81 * u.m / u.s**2,
        reference_height=start,
        height_uncertainty=height_uncertainty,
    )
    assert explicit_sigma == potential_sigma


def test_reference_height_uncertainty_in_anchored_evaluations():
    model = PolynomialVerticalGravity([0, -300], np.zeros((2, 2)))
    reference = {
        "reference_gravity": 9.81 * u.m / u.s**2,
        "reference_height": 0.5 * u.m,
        "reference_height_uncertainty": 1 * u.mm,
    }
    anchored = model.with_reference_gravity(**reference)
    h1, h2 = 0 * u.m, 2 * u.m

    _, gravity_sigma = anchored.gravity(h2)
    _, mean_sigma = anchored.mean_gravity(h1, h2)
    _, potential_sigma = anchored.potential_difference(h1, h2)
    _, explicit_sigma = model.potential_difference(h1, h2, **reference)

    assert gravity_sigma.to_value(u.uGal) == pytest.approx(0.3)
    assert mean_sigma.to_value(u.uGal) == pytest.approx(0.3)
    assert potential_sigma.to_value(u.uGal * u.m) == pytest.approx(0.6)
    assert explicit_sigma == potential_sigma
