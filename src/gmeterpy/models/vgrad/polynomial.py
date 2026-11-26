"""Vertical gravity predictions and uncertainty propagation."""

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

import gmeterpy.units as u
from gmeterpy.numerics import polynomial
from gmeterpy.numerics.statistics.results import FitResults

if TYPE_CHECKING:
    from statsmodels.regression.linear_model import RegressionResultsWrapper


def _gravity_quantities(
    value: ArrayLike, uncertainty: ArrayLike | None, *, unit: u.UnitBase
) -> tuple[u.Quantity, u.Quantity | None]:
    """Attach units to a value and its standard uncertainty."""
    return (
        u.Quantity(value, unit),
        None if uncertainty is None else u.Quantity(uncertainty, unit),
    )


def _height_sigma_metres(uncertainty: u.Quantity) -> NDArray[np.float64]:
    """Convert a nonnegative standard height uncertainty to metres."""
    sigma = np.asarray(uncertainty.to_value(u.m), dtype=np.float64)
    if np.any(~np.isfinite(sigma) | (sigma < 0)):
        raise ValueError("Height uncertainties must be finite and nonnegative")
    return sigma


def _endpoint_height_sigmas(
    uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert a shared uncertainty or an endpoint pair to metres."""
    if isinstance(uncertainty, u.Quantity):
        sigma = _height_sigma_metres(uncertainty)
        return sigma, sigma
    h1_sigma, h2_sigma = uncertainty
    return _height_sigma_metres(h1_sigma), _height_sigma_metres(h2_sigma)


class _GravityReference(NamedTuple):
    gravity: u.Quantity
    height: u.Quantity
    gravity_uncertainty: u.Quantity
    height_uncertainty: u.Quantity


class PolynomialVerticalGravity:
    r"""Polynomial model of vertical gravity changes above the station.

    Parameters
    ----------
    coefficients : array_like
        Coefficients in ascending power order, including the constant.
        Coefficient of order k is expressed in gravity_unit / height_unit**k.
    covariance : pandas.DataFrame or array_like, optional
        Coefficient covariance in the same order. Each entry has the product
        of the corresponding coefficient units. If omitted, evaluation methods
        return None for the standard uncertainty.
    units : tuple of astropy.units.UnitBase, optional
        Unit pair ``(gravity_unit, height_unit)``. Defaults to microgals and
        metres. Coefficients and covariance are converted to the metre and
        microgal basis at construction.

    Notes
    -----
    The model describes gravity changes over a local height interval relative
    to a measurement mark. Evaluations outside the observed height range are
    extrapolations. The propagated uncertainty accounts for coefficient
    uncertainty and any supplied reference gravity and height uncertainties.
    It excludes departures of the actual gravity field from the polynomial.

    For degree :math:`n` and coefficients :math:`a_k`, the gravity change
    from height zero and its local gradient are

    .. math::

       g(h) = \sum_{k=0}^{n} a_k h^k = \mathbf a^{\mathsf T}\mathbf h,
       \qquad g'(h) = \sum_{k=1}^{n} k a_k h^{k-1}.

    Here :math:`\mathbf h=(1,h,\ldots,h^n)^{\mathsf T}`. The convention
    :math:`a_0=0` fixes the gravity datum for difference fits. Supply a
    measured gravity value to ``transfer_gravity()`` or
    ``potential_difference()`` for a calculation in its datum. The gravity
    difference from :math:`h_i` to :math:`h_j`
    is :math:`\Delta g_{ij}=g(h_j)-g(h_i)=\mathbf a^{\mathsf T}\mathbf h_{ij}`,
    where :math:`\mathbf h_{ij}=\mathbf h_j-\mathbf h_i`. Given :math:`C_a`,
    the standard uncertainty of the difference is

    .. math::

       u(\Delta g_{ij}) = \sqrt{\mathbf h_{ij}^{\mathsf T} C_a \mathbf h_{ij}}.

    Evaluation methods return a (value, uncertainty) tuple and treat heights
    as exact unless height uncertainties are supplied. Heights are expressed
    in metres and gravity changes in microgals.
    """

    def __init__(
        self,
        coefficients: ArrayLike,
        covariance: pd.DataFrame | ArrayLike | None = None,
        *,
        units: tuple[u.UnitBase, u.UnitBase] = (u.uGal, u.m),
    ) -> None:
        """Convert coefficients and covariance to the metre and microgal basis."""
        gravity_unit, height_unit = units
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.size == 0:
            raise ValueError("At least the constant coefficient is required")
        orders = np.arange(coefficients.size)
        scale = gravity_unit.to(u.uGal) / height_unit.to(u.m) ** orders
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape != (coefficients.size, coefficients.size):
                raise ValueError(
                    "Covariance must have one row and column per coefficient"
                )
            covariance = covariance * np.outer(scale, scale)
        self._polynomial = polynomial.Polynomial(
            coefficients * scale, covariance=covariance
        )

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Supplied coefficients in ascending power order.

        Returns
        -------
        numpy.ndarray
            Independent coefficient array; coefficient n is in microgals/m**n.
        """
        return self._polynomial.coef

    @property
    def covariance(self) -> NDArray[np.float64] | None:
        """Polynomial coefficient covariance.

        Returns
        -------
        numpy.ndarray or None
            Independent covariance matrix in ascending power order, or None
            when coefficient uncertainties are not supplied.
        """
        return self._polynomial.covariance

    @property
    def degree(self) -> int:
        """Highest power represented by the coefficients.

        Returns
        -------
        int
            Polynomial model degree, including any leading zero coefficients.
        """
        return self._polynomial.degree()

    @staticmethod
    def _prepare_reference(
        reference_gravity: u.Quantity | None,
        reference_height: u.Quantity | None,
        reference_gravity_uncertainty: u.Quantity | None,
        reference_height_uncertainty: u.Quantity | None,
    ) -> _GravityReference | None:
        """Validate and copy an optional reference in the model's units."""
        if (
            reference_gravity is None
            and reference_height is None
            and reference_gravity_uncertainty is None
            and reference_height_uncertainty is None
        ):
            return None
        if reference_gravity is None or reference_height is None:
            raise ValueError(
                "Reference gravity and reference height must be supplied together"
            )
        if reference_gravity_uncertainty is None:
            reference_gravity_uncertainty = u.Quantity(0, u.uGal)
        height_sigma = (
            0.0
            if reference_height_uncertainty is None
            else _height_sigma_metres(reference_height_uncertainty)
        )
        return _GravityReference(
            reference_gravity.to(u.uGal),
            reference_height.to(u.m),
            reference_gravity_uncertainty.to(u.uGal),
            u.Quantity(height_sigma, u.m, copy=True),
        )

    @u.quantity_input(heights=u.m)
    def gravity(
        self, heights: u.Quantity, *, height_uncertainty: u.Quantity | None = None
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the gravity profile and its standard uncertainty.

        Parameters
        ----------
        heights : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.
        height_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of the evaluation heights, independent of
            the fit. Must broadcast with heights.
            Defaults to exact evaluation heights.

        Returns
        -------
        value, uncertainty : tuple
            Gravity and its standard uncertainty in microgals, in the
            polynomial's gravity datum.
            Uncertainty is None without coefficient covariance.
        """
        return self._evaluate_at_height(heights, height_uncertainty, 0, u.uGal)

    @u.quantity_input(h1=u.m, h2=u.m)
    def gravity_difference(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        *,
        height_uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate a gravity difference and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Initial and final heights from the same datum.
        height_uncertainty : ~astropy.units.Quantity or tuple, optional
            Total standard uncertainty of the sensor heights. Supply one
            quantity for both endpoints or a pair ordered as ``(h1, h2)``.
            Omit to treat heights as exact.
            Both endpoint uncertainties must broadcast with the heights.

        Returns
        -------
        value, uncertainty : tuple
            :math:`\Delta g_{ij}=g(h_j)-g(h_i)` and its standard uncertainty,
            both in microgals. Uncertainty is None without coefficient covariance.

        Notes
        -----
        Height uncertainty is propagated to first order using the endpoint
        sensitivities :math:`-g'(h_1)` and :math:`g'(h_2)`. The height
        errors are assumed uncorrelated with each other and independent
        of the fitted coefficients.
        """
        h1_m, h2_m = np.broadcast_arrays(h1.to_value(u.m), h2.to_value(u.m))
        height_sigmas = None
        if height_uncertainty is not None:
            u1, u2 = _endpoint_height_sigmas(height_uncertainty)
            h1_m, h2_m, u1_m, u2_m = np.broadcast_arrays(h1_m, h2_m, u1, u2)
            height_sigmas = u1_m, u2_m
        value, uncertainty = self._polynomial.difference(h1_m, h2_m)
        if height_sigmas is not None and uncertainty is not None:
            u1_m, u2_m = height_sigmas
            s1 = self._polynomial.value(h1_m, derivative=1) * u1_m
            s2 = self._polynomial.value(h2_m, derivative=1) * u2_m
            height_sigma = np.hypot(s1, s2)
            uncertainty = np.hypot(uncertainty, height_sigma)
        return _gravity_quantities(value, uncertainty, unit=u.uGal)

    @u.quantity_input(
        target_height=u.m,
        reference_gravity=u.uGal,
        reference_height=u.m,
        reference_gravity_uncertainty=u.uGal,
    )
    def transfer_gravity(
        self,
        target_height: u.Quantity,
        reference_gravity: u.Quantity,
        *,
        reference_height: u.Quantity,
        reference_gravity_uncertainty: u.Quantity = 0 * u.uGal,
        reference_height_uncertainty: u.Quantity | None = None,
        target_height_uncertainty: u.Quantity | None = None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Transfer reference gravity to a target height and propagate its standard uncertainty.

        Parameters
        ----------
        target_height : ~astropy.units.Quantity
            Target height in a compatible length unit.
        reference_gravity : ~astropy.units.Quantity
            Supplied gravity value at the reference height.
        reference_height : ~astropy.units.Quantity
            Reference height measured from the same datum as the target.
        reference_gravity_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of reference gravity, independent of the
            fitted coefficients and height errors. Defaults to zero.
        reference_height_uncertainty, target_height_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainties of the reference and target heights,
            respectively. Each defaults to an exact height. Height errors
            are assumed uncorrelated and independent of the fit. Inputs
            must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g_r+g(h)-g(h_r)` and its standard uncertainty, both in
            microgals. Uncertainty is None without coefficient covariance.
        """
        difference, uncertainty = self.gravity_difference(
            reference_height,
            target_height,
            height_uncertainty=(
                None
                if reference_height_uncertainty is None
                and target_height_uncertainty is None
                else (
                    0 * u.m
                    if reference_height_uncertainty is None
                    else reference_height_uncertainty,
                    0 * u.m
                    if target_height_uncertainty is None
                    else target_height_uncertainty,
                )
            ),
        )
        reference, difference_values, reference_sigma = np.broadcast_arrays(
            reference_gravity.to_value(u.uGal),
            difference.to_value(u.uGal),
            reference_gravity_uncertainty.to_value(u.uGal),
        )
        value = u.Quantity(reference + difference_values, u.uGal)
        if uncertainty is not None:
            uncertainty = u.Quantity(
                np.hypot(uncertainty.to_value(u.uGal), reference_sigma), u.uGal
            )
        return value, uncertainty

    @u.quantity_input(h1=u.m, h2=u.m)
    def potential_difference(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        reference_gravity: u.Quantity | None = None,
        *,
        reference_height: u.Quantity | None = None,
        reference_gravity_uncertainty: u.Quantity | None = None,
        reference_height_uncertainty: u.Quantity | None = None,
        height_uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the gravity potential difference and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Initial and final heights relative to the model datum.
        reference_gravity : ~astropy.units.Quantity, optional
            Supplied gravity value at the reference height. Supply both reference
            gravity and height to use the measurement for this calculation.
        reference_height : ~astropy.units.Quantity, optional
            Height of the reference gravity, measured from the same datum.
        reference_gravity_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of reference gravity, independent of the
            fitted coefficients. Defaults to zero for an explicit reference.
            Omit the reference arguments to use the model's current
            gravity datum.
            All inputs must broadcast.

        reference_height_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of an explicit reference height. Supply it
            with reference gravity and height. Defaults to an exact height.
            Assumed independent of the fit, reference gravity, and endpoints.
        height_uncertainty : ~astropy.units.Quantity or tuple, optional
            One standard uncertainty for both endpoints or a pair ordered
            as ``(h1, h2)``. Endpoint errors are treated as uncorrelated
            and independent of the fit and gravity reference. Omit to
            treat both endpoints as exact.

        Returns
        -------
        value, uncertainty : tuple
            Final minus initial gravity potential and its standard uncertainty,
            in square metres per square second. Uncertainty is None without
            coefficient covariance.

        Notes
        -----
        Height increases upward, and gravity magnitude is positive downward.
        With a supplied reference and :math:`d=h_2-h_1`,

        .. math::

           \Delta W=-d g_r-\sum_{k=1}^n q_k a_k,
           \qquad q_k=\frac{h_2^{k+1}-h_1^{k+1}}{k+1}-d h_r^k,

           u^2(\Delta W)=\mathbf q^{\mathsf T} C_a\mathbf q+d^2u^2(g_r)
               +d^2g'(h_r)^2u^2(h_r)+g(h_1)^2u^2(h_1)+g(h_2)^2u^2(h_2).

        Without a reference, integrate the full polynomial in its current
        gravity datum and apply :math:`dW/dh=-g(h)`. For a relative model,
        this gives a potential difference in that gravity datum. An unknown
        constant gravity offset contributes that offset times the signed height
        separation to the integral. The reference height sensitivity is
        :math:`d g'(h_r)`.
        Endpoint height uncertainties, if supplied, enter through the
        sensitivities :math:`g(h_1)` and :math:`-g(h_2)`.
        """
        reference = self._prepare_reference(
            reference_gravity,
            reference_height,
            reference_gravity_uncertainty,
            reference_height_uncertainty,
        )
        integral, uncertainty = self._integrated_gravity(h1, h2, reference)
        if height_uncertainty is not None:
            u1, u2 = _endpoint_height_sigmas(height_uncertainty)
            unit = u.uGal * u.m
            integral_value, u1, u2 = np.broadcast_arrays(
                integral.to_value(unit), u1, u2
            )
            integral = u.Quantity(integral_value, unit)
            if uncertainty is not None:
                g1, g2 = self._endpoint_gravities(h1, h2, reference)
                uncertainty = u.Quantity(
                    np.hypot(uncertainty.to_value(unit), np.hypot(g1 * u1, g2 * u2)),
                    unit,
                )
        potential_unit = u.m**2 / u.s**2
        if uncertainty is not None:
            uncertainty = uncertainty.to(potential_unit)
        return (-integral).to(potential_unit), uncertainty

    def _integrated_gravity(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        gravity_reference: _GravityReference | None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        """Integrate the polynomial and propagate its uncertainty."""
        reference_inputs = ()
        if gravity_reference is not None:
            reference_inputs = (
                gravity_reference.height.to_value(u.m),
                gravity_reference.gravity.to_value(u.uGal),
                gravity_reference.gravity_uncertainty.to_value(u.uGal),
                gravity_reference.height_uncertainty.to_value(u.m),
            )
        h1_m, h2_m, *reference_values = np.broadcast_arrays(
            np.asarray(h1.to_value(u.m), dtype=float),
            np.asarray(h2.to_value(u.m), dtype=float),
            *reference_inputs,
        )
        basis = polynomial.integral_basis(h1_m, h2_m, self.degree)
        if gravity_reference is None:
            return _gravity_quantities(
                *self._polynomial.evaluate_basis(basis), unit=u.uGal * u.m
            )

        reference, gravity, gravity_sigma, reference_sigma = reference_values
        separation = h2_m - h1_m
        basis -= separation * polynomial.evaluation_basis(reference, self.degree)
        value, uncertainty = self._polynomial.evaluate_basis(basis)
        value = value + separation * gravity
        if uncertainty is not None:
            slope = self._polynomial.value(reference, derivative=1)
            reference_gravity_sigma = np.hypot(gravity_sigma, slope * reference_sigma)
            uncertainty = np.hypot(uncertainty, separation * reference_gravity_sigma)
        return _gravity_quantities(value, uncertainty, unit=u.uGal * u.m)

    @u.quantity_input(h1=u.m, h2=u.m)
    def mean_gravity(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        *,
        height_uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the interval mean of the gravity polynomial.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct endpoints relative to the model datum. Arrays must broadcast.

        height_uncertainty : ~astropy.units.Quantity or tuple, optional
            One standard uncertainty for both endpoints or a pair ordered
            as ``(h1, h2)``. Endpoint errors are treated as uncorrelated
            and independent of the fit. Omit to treat both endpoints as exact.

        Returns
        -------
        value, uncertainty : tuple
            The uniform height average of the same polynomial evaluated by
            ``gravity()`` and its standard uncertainty, both in microgals.
            Uncertainty is None without coefficient covariance.
        """
        return self._polynomial_interval_mean(
            self._polynomial, h1, h2, height_uncertainty, unit=u.uGal
        )

    @u.quantity_input(height=u.m)
    def gradient(
        self, height: u.Quantity, *, height_uncertainty: u.Quantity | None = None
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the local gradient and its standard uncertainty.

        Parameters
        ----------
        height : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.
        height_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of the evaluation height, independent of
            the fit. Must broadcast with height. Omit to treat height as exact.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g'(h)=\sum_{k=1}^n k a_k h^{k-1}` and its standard
            uncertainty in microgals per metre. Uncertainty is None without
            covariance.
        """
        return self._evaluate_at_height(height, height_uncertainty, 1, u.uGal / u.m)

    @u.quantity_input(h1=u.m, h2=u.m)
    def mean_gradient(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        *,
        height_uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity] | None = None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the interval-average gradient and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct endpoints from the same datum.
        height_uncertainty : ~astropy.units.Quantity or tuple, optional
            One standard uncertainty for both endpoints or a pair ordered
            as ``(h1, h2)``. The endpoint errors are assumed uncorrelated
            and independent of the fit. All inputs must broadcast.
            Omit to treat both heights as exact.

        Returns
        -------
        value, uncertainty : tuple
            :math:`\Delta g/(h_2-h_1)` and its standard uncertainty, both
            in microgals per metre. Uncertainty is None without covariance.
        """
        return self._polynomial_interval_mean(
            self._polynomial.deriv(), h1, h2, height_uncertainty, unit=u.uGal / u.m
        )

    @staticmethod
    def _polynomial_interval_mean(
        curve: polynomial.Polynomial,
        h1: u.Quantity,
        h2: u.Quantity,
        height_uncertainty: u.Quantity | tuple[u.Quantity, u.Quantity] | None,
        *,
        unit: u.UnitBase,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        """Evaluate a polynomial mean with height uncertainties in metres."""
        sigmas = (
            None
            if height_uncertainty is None
            else _endpoint_height_sigmas(height_uncertainty)
        )
        return _gravity_quantities(
            *curve.interval_mean(
                h1.to_value(u.m), h2.to_value(u.m), x_uncertainty=sigmas
            ),
            unit=unit,
        )

    @u.quantity_input(height=u.m)
    def curvature(
        self, height: u.Quantity, *, height_uncertainty: u.Quantity | None = None
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate curvature and its standard uncertainty.

        Parameters
        ----------
        height : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.
        height_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of the evaluation height, independent of
            the fit. Must broadcast with height. Omit to treat height as exact.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g''(h)=\sum_{k=2}^n k(k-1)a_k h^{k-2}` and its standard
            uncertainty in microgals per metre squared. Uncertainty is None
            without covariance.
        """
        return self._evaluate_at_height(height, height_uncertainty, 2, u.uGal / u.m**2)

    def _evaluate_at_height(
        self,
        height: u.Quantity,
        height_uncertainty: u.Quantity | None,
        order: int,
        unit: u.UnitBase,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        """Evaluate a polynomial derivative and propagate height uncertainty."""
        height_m = np.asarray(height.to_value(u.m))
        height_sigma = None
        if height_uncertainty is not None:
            height_m, height_sigma = np.broadcast_arrays(
                height_m, _height_sigma_metres(height_uncertainty)
            )
        curve = self._polynomial.deriv(order) if order else self._polynomial
        value, uncertainty = curve(height_m)
        if height_sigma is not None and uncertainty is not None:
            sensitivity = curve.value(height_m, derivative=1)
            uncertainty = np.hypot(uncertainty, sensitivity * height_sigma)
        return _gravity_quantities(value, uncertainty, unit=unit)

    def _endpoint_gravities(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        reference: _GravityReference | None,
    ):
        """Evaluate the endpoints in the gravity datum used for integration."""
        offset = 0.0
        if reference is not None:
            offset = np.asarray(
                reference.gravity.to_value(u.uGal), dtype=float
            ) - self._polynomial.value(reference.height.to_value(u.m))
        return (
            self._polynomial.value(h1.to_value(u.m)) + offset,
            self._polynomial.value(h2.to_value(u.m)) + offset,
        )


class PolynomialVerticalGravityFitResults(FitResults):
    """Regression diagnostics and predictions from a fitted gravity polynomial.

    Parameters
    ----------
    regression : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted regression.
    fitted_model : PolynomialVerticalGravity
        Gravity polynomial estimated by the regression.
    """

    def __init__(
        self,
        regression: "RegressionResultsWrapper",
        fitted_model: PolynomialVerticalGravity,
    ) -> None:
        super().__init__(regression)
        self._fitted_model = fitted_model

    @property
    def fitted_model(self) -> PolynomialVerticalGravity:
        """Gravity polynomial estimated by this fit."""
        return self._fitted_model

    @u.quantity_input(h1=u.m, h2=u.m)
    def prediction_table(self, h1: u.Quantity, h2: u.Quantity) -> pd.DataFrame:
        """Return gravity differences and mean gradients for height pairs.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct initial and final heights. Arrays must broadcast.

        Returns
        -------
        pandas.DataFrame
            One row per pair. Heights are in metres, gravity differences
            in microgals, and gradients in microgals per metre. The
            uncertainty columns contain standard uncertainties or NaN
            when coefficient covariance is unavailable.
        """
        start, end = np.broadcast_arrays(h1.to_value(u.m), h2.to_value(u.m))
        separation = end - start
        if np.any(separation == 0):
            raise ValueError("An interval average requires distinct heights")
        difference, uncertainty = self.fitted_model.gravity_difference(h1, h2)
        dg = np.asarray(difference.to_value(u.uGal))
        sigma = (
            np.nan if uncertainty is None else np.asarray(uncertainty.to_value(u.uGal))
        )
        return pd.DataFrame(
            {
                "h1": start.ravel(),
                "h2": end.ravel(),
                "gravity_difference": dg.ravel(),
                "gravity_difference_uncertainty": np.broadcast_to(
                    sigma, dg.shape
                ).ravel(),
                "mean_gradient": (dg / separation).ravel(),
                "mean_gradient_uncertainty": (sigma / np.abs(separation)).ravel(),
            }
        )
