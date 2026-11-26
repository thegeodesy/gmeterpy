import numpy as np

from gmeterpy.numerics.polynomial import Polynomial, polyder, polyint, polyval


def test_polyval_propagates_cross_covariance():
    values, sigma = polyval([0, 2], [3, 2], covariance=[[4, 1], [1, 9]])

    np.testing.assert_allclose(values, [3, 7])
    np.testing.assert_allclose(sigma, [2, np.sqrt(44)])


def test_polyder_transforms_covariance():
    coefficients, covariance = polyder(
        [3, 2, 5], covariance=[[4, 1, 0], [1, 9, 2], [0, 2, 16]]
    )

    np.testing.assert_allclose(coefficients, [2, 10])
    np.testing.assert_allclose(covariance, [[9, 4], [4, 64]])


def test_polyint_with_fixed_value_at_lower_bound():
    coefficients, covariance = polyint([3, 2], k=7, lbnd=1, covariance=[[4, 1], [1, 9]])

    np.testing.assert_allclose(coefficients, [3, 3, 1])
    values, sigma = polyval([1, 3], coefficients, covariance=covariance)
    np.testing.assert_allclose(values, [7, 21])
    np.testing.assert_allclose(sigma, [0, np.sqrt(176)])


def test_polynomial_preserves_covariance_through_integration_and_differentiation():
    p = Polynomial([3, 2], covariance=[[4, 1], [1, 9]])
    integral = p.integ(k=7, lbnd=1)

    assert p.degree() == 1
    assert integral.degree() == 2
    np.testing.assert_allclose(integral([1, 3]), ([7, 21], [0, np.sqrt(176)]))

    derivative = integral.deriv()
    np.testing.assert_allclose(derivative.coef, p.coef)
    np.testing.assert_allclose(derivative.covariance, p.covariance)
    assert Polynomial([3, 2]).deriv()(1) == (2, None)


def test_polynomial_difference_cancels_constant_uncertainty():
    p = Polynomial([3, 2], covariance=[[4, 1], [1, 9]])

    values, sigma = p.difference(1, [0, 1, 3])

    np.testing.assert_allclose(values, [-2, 0, 4])
    np.testing.assert_allclose(sigma, [3, 0, 6])


def test_interval_mean_propagates_coefficients_and_endpoint_errors():
    p = Polynomial([3, 2], covariance=[[4, 1], [1, 9]])

    mean, sigma = p.interval_mean(1, 3, x_uncertainty=(0.1, 0.2))
    reverse = p.interval_mean(3, 1, x_uncertainty=(0.2, 0.1))

    np.testing.assert_allclose((mean, sigma), (7, np.sqrt(44 + 0.1**2 + 0.2**2)))
    np.testing.assert_allclose(reverse, (mean, sigma))

    constant = Polynomial([3], covariance=[[0]])
    assert constant.interval_mean(1, 3, x_uncertainty=0.1) == (3, 0)
