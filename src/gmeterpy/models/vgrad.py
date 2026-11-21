"""Vertical gravity gradient correction.

This module fits polynomial models to relative gravity observations and
evaluates gravity changes and their uncertainties between heights.

"""

import itertools
from copy import copy
from os import PathLike
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyder, polyint, polyval, polyvander
from numpy.typing import ArrayLike, NDArray

import gmeterpy.units as u
from gmeterpy.statistics.results import FitResults

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from statsmodels.regression.linear_model import RegressionResultsWrapper


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
    reference_gravity, reference_height : ~astropy.units.Quantity, optional
        Absolute gravity and its height relative to the model datum. Supply
        both to anchor the profile. Without a reference, the
        constant term sets the gravity at height zero.
    reference_gravity_uncertainty : ~astropy.units.Quantity, optional
        Standard uncertainty of the reference gravity, independent of the
        fitted coefficients. Defaults to zero when a reference is supplied.

    Notes
    -----
    The model describes gravity changes over a local height interval relative
    to a measurement mark. Evaluations outside the observed height range are
    extrapolations. The propagated uncertainty accounts for coefficient
    uncertainty and any supplied reference gravity uncertainty. It excludes
    departures of the actual gravity field from the polynomial.

    For degree :math:`n` and coefficients :math:`a_k`, the gravity change
    from height zero and its local gradient are

    .. math::

       g(h) = \sum_{k=0}^{n} a_k h^k = \mathbf a^{\mathsf T}\mathbf h,
       \qquad g'(h) = \sum_{k=1}^{n} k a_k h^{k-1}.

    Here :math:`\mathbf h=(1,h,\ldots,h^n)^{\mathsf T}`. The convention
    :math:`a_0=0` fixes the gravity datum for difference fits. Use ``with_reference_gravity()`` to
    anchor the profile to a measured gravity value. The gravity difference
    from :math:`h_i` to :math:`h_j`
    is :math:`\Delta g_{ij}=g(h_j)-g(h_i)=\mathbf a^{\mathsf T}\mathbf h_{ij}`,
    where :math:`\mathbf h_{ij}=\mathbf h_j-\mathbf h_i`. Given :math:`C_a`,
    the standard uncertainty of the difference is

    .. math::

       u(\Delta g_{ij}) = \sqrt{\mathbf h_{ij}^{\mathsf T} C_a \mathbf h_{ij}}.

    Evaluation methods return a (value, uncertainty) tuple and treat heights
    as exact. Heights are expressed in metres and gravity changes in microgals.
    The floating-gravity fit follows Eq. (3-2) of Jaakko Mäkinen (2010)
    [Makinen2010]_. The model covariance for difference fits has a zero row
    and column for the fixed constant. Figure 2a illustrates the quadratic
    fit and propagation of its coefficient covariance.
    """

    def __init__(
        self,
        coefficients: ArrayLike,
        covariance: pd.DataFrame | ArrayLike | None = None,
        *,
        units: tuple[u.UnitBase, u.UnitBase] = (u.uGal, u.m),
        reference_gravity: u.Quantity | None = None,
        reference_height: u.Quantity | None = None,
        reference_gravity_uncertainty: u.Quantity | None = None,
    ) -> None:
        """Convert coefficients and covariance to the metre and microgal basis."""
        gravity_unit, height_unit = units
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.size == 0:
            raise ValueError("At least the constant coefficient is required")
        orders = np.arange(coefficients.size)
        scale = gravity_unit.to(u.uGal) / height_unit.to(u.m) ** orders
        self._reference = self._prepare_reference(
            reference_gravity, reference_height, reference_gravity_uncertainty
        )
        self._coefficients = np.asarray(coefficients * scale, dtype=np.float64)
        self._basis = np.eye(coefficients.size)
        self._covariance = None
        if covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            if covariance.shape != (coefficients.size, coefficients.size):
                raise ValueError(
                    "Covariance must have one row and column per coefficient"
                )
            self._covariance = np.asarray(
                covariance * np.outer(scale, scale), dtype=np.float64
            )

    @property
    def coefficients(self) -> NDArray[np.float64]:
        """Supplied coefficients in ascending power order.

        Returns
        -------
        numpy.ndarray
            Independent coefficient array; coefficient n is in microgals/m**n.
        """
        return self._coefficients.copy()

    @property
    def covariance(self) -> NDArray[np.float64] | None:
        """Polynomial coefficient covariance.

        Returns
        -------
        numpy.ndarray or None
            Independent covariance matrix in ascending power order, or None
            when coefficient uncertainties are not supplied.
        """
        return None if self._covariance is None else self._covariance.copy()

    @property
    def degree(self) -> int:
        """Highest power represented by the coefficients.

        Returns
        -------
        int
            Polynomial model degree, including any leading zero coefficients.
        """
        return self._basis.shape[0] - 1

    @u.quantity_input(
        reference_gravity=u.uGal,
        reference_height=u.m,
        reference_gravity_uncertainty=u.uGal,
    )
    def with_reference_gravity(
        self,
        reference_gravity: u.Quantity,
        *,
        reference_height: u.Quantity,
        reference_gravity_uncertainty: u.Quantity = 0 * u.uGal,
    ) -> Self:
        r"""Return a new model with gravity anchored to a reference measurement.

        Parameters
        ----------
        reference_gravity : ~astropy.units.Quantity
            Absolute gravity at the reference height.
        reference_height : ~astropy.units.Quantity
            Height of the reference measurement relative to the model datum.
        reference_gravity_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of the reference gravity, independent of
            the fitted coefficients. Defaults to zero.

        Returns
        -------
        PolynomialVerticalGravity
            Model whose ``gravity()`` evaluates
            :math:`g_r + [g(h)-g(h_r)]`. A new reference replaces any previous
            one. The original model is unchanged.
        """
        anchored = copy(self)
        anchored._reference = self._prepare_reference(
            reference_gravity, reference_height, reference_gravity_uncertainty
        )
        return anchored

    @staticmethod
    def _prepare_reference(
        reference_gravity: u.Quantity | None,
        reference_height: u.Quantity | None,
        reference_gravity_uncertainty: u.Quantity | None,
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity] | None:
        """Validate and copy an optional reference in the model's units."""
        if (
            reference_gravity is None
            and reference_height is None
            and reference_gravity_uncertainty is None
        ):
            return None
        if reference_gravity is None or reference_height is None:
            raise ValueError(
                "Reference gravity and reference height must be supplied together"
            )
        if reference_gravity_uncertainty is None:
            reference_gravity_uncertainty = u.Quantity(0, u.uGal)
        return (
            reference_gravity.to(u.uGal),
            reference_height.to(u.m),
            reference_gravity_uncertainty.to(u.uGal),
        )

    @u.quantity_input(heights=u.m)
    def gravity(self, heights: u.Quantity) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the gravity profile and its standard uncertainty.

        Parameters
        ----------
        heights : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.

        Returns
        -------
        value, uncertainty : tuple
            Gravity and its standard uncertainty in microgals, in the
            polynomial's gravity datum or anchored to a reference measurement.
            Uncertainty is None without coefficient covariance.
        """
        if self._reference is not None:
            reference_gravity, reference_height, reference_gravity_uncertainty = (
                self._reference
            )
            return self.transfer_gravity(
                heights,
                reference_gravity,
                reference_height=reference_height,
                reference_gravity_uncertainty=reference_gravity_uncertainty,
            )
        basis = polyval(np.asarray(heights.to_value(u.m)), self._basis)
        return self._gravity_quantities(*self._evaluate_basis(basis), unit=u.uGal)

    @u.quantity_input(h1=u.m, h2=u.m)
    def gravity_difference(
        self, h1: u.Quantity, h2: u.Quantity
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate a gravity difference and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Initial and final heights from the same datum. Arrays must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            :math:`\Delta g_{ij}=g(h_j)-g(h_i)` and its standard uncertainty,
            both in microgals. Uncertainty is None without covariance.
        """
        h1_m, h2_m = np.broadcast_arrays(h1.to_value(u.m), h2.to_value(u.m))
        basis = polyval(h2_m, self._basis) - polyval(h1_m, self._basis)
        return self._gravity_quantities(*self._evaluate_basis(basis), unit=u.uGal)

    @u.quantity_input(
        height=u.m,
        reference_gravity=u.uGal,
        reference_height=u.m,
        reference_gravity_uncertainty=u.uGal,
    )
    def transfer_gravity(
        self,
        height: u.Quantity,
        reference_gravity: u.Quantity,
        *,
        reference_height: u.Quantity,
        reference_gravity_uncertainty: u.Quantity = 0 * u.uGal,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Transfer reference gravity to a target height and propagate its standard uncertainty.

        Parameters
        ----------
        height : ~astropy.units.Quantity
            Target height in a compatible length unit.
        reference_gravity : ~astropy.units.Quantity
            Absolute gravity at the reference height.
        reference_height : ~astropy.units.Quantity
            Reference height measured from the same datum as the target.
        reference_gravity_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of reference gravity, independent of the
            fitted coefficients. Defaults to zero. All inputs must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g_r+g(h)-g(h_r)` and its standard uncertainty, both in
            microgals. Uncertainty is None without coefficient covariance.
        """
        difference, uncertainty = self.gravity_difference(reference_height, height)
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
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the gravity potential difference and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Initial and final heights relative to the model datum.
        reference_gravity : ~astropy.units.Quantity, optional
            Absolute gravity at the reference height. Supply both reference
            gravity and height to override the model's stored reference.
        reference_height : ~astropy.units.Quantity, optional
            Height of the reference gravity, measured from the same datum.
        reference_gravity_uncertainty : ~astropy.units.Quantity, optional
            Standard uncertainty of reference gravity, independent of the
            fitted coefficients. Defaults to zero for an explicit reference.
            Omit all three reference arguments to use the model's current
            gravity datum.
            All inputs must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            Final minus initial gravity potential and its standard uncertainty,
            in square metres per square second. Uncertainty is None without
            coefficient covariance.

        Notes
        -----
        Height increases upward, and gravity magnitude is positive downward.
        With an explicit or stored reference and :math:`d=h_2-h_1`,

        .. math::

           \Delta W=-d g_r-\sum_{k=1}^n q_k a_k,
           \qquad q_k=\frac{h_2^{k+1}-h_1^{k+1}}{k+1}-d h_r^k,

           u^2(\Delta W)=\mathbf q^{\mathsf T} C_a\mathbf q+d^2u^2(g_r).

        Without a reference, integrate the full polynomial in its current
        gravity datum and apply :math:`dW/dh=-g(h)`. For a relative model,
        this gives a potential difference in that gravity datum. An unknown
        constant gravity offset contributes that offset times the signed height
        separation to the integral. Heights are treated as exact.
        """
        reference = self._prepare_reference(
            reference_gravity, reference_height, reference_gravity_uncertainty
        )
        if reference is None:
            reference = self._reference
        integral, uncertainty = self._integrated_gravity(h1, h2, reference)
        potential_unit = u.m**2 / u.s**2
        if uncertainty is not None:
            uncertainty = uncertainty.to(potential_unit)
        return (-integral).to(potential_unit), uncertainty

    def _integrated_gravity(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        gravity_reference: tuple[u.Quantity, u.Quantity, u.Quantity] | None,
    ) -> tuple[u.Quantity, u.Quantity | None]:
        """Integrate the polynomial and propagate its uncertainty."""
        h1_m = np.asarray(h1.to_value(u.m), dtype=float)
        h2_m = np.asarray(h2.to_value(u.m), dtype=float)
        antiderivative = polyint(self._basis)
        if gravity_reference is None:
            h1_m, h2_m = np.broadcast_arrays(h1_m, h2_m)
            basis = polyval(h2_m, antiderivative) - polyval(h1_m, antiderivative)
            return self._gravity_quantities(
                *self._evaluate_basis(basis), unit=u.uGal * u.m
            )

        reference_gravity, reference_height, reference_gravity_uncertainty = (
            gravity_reference
        )
        h1_m, h2_m, reference, gravity, gravity_uncertainty = np.broadcast_arrays(
            h1_m,
            h2_m,
            reference_height.to_value(u.m),
            reference_gravity.to_value(u.uGal),
            reference_gravity_uncertainty.to_value(u.uGal),
        )
        separation = h2_m - h1_m
        basis = polyval(h2_m, antiderivative) - polyval(h1_m, antiderivative)
        basis -= separation * polyval(reference, self._basis)
        value, uncertainty = self._evaluate_basis(basis)
        value = value + separation * gravity
        if uncertainty is not None:
            uncertainty = np.hypot(uncertainty, separation * gravity_uncertainty)
        return self._gravity_quantities(value, uncertainty, unit=u.uGal * u.m)

    @u.quantity_input(h1=u.m, h2=u.m)
    def mean_gravity(
        self, h1: u.Quantity, h2: u.Quantity
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the interval mean of the gravity polynomial.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct endpoints relative to the model datum. Arrays must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            The uniform height average of the same polynomial evaluated by
            ``gravity()`` and its standard uncertainty, both in microgals.
            Uncertainty is None without coefficient covariance.
        """
        separation = u.Quantity(self._height_difference(h1, h2), u.m)
        integral, uncertainty = self._integrated_gravity(h1, h2, self._reference)
        if uncertainty is not None:
            uncertainty = (uncertainty / abs(separation)).to(u.uGal)
        return (integral / separation).to(u.uGal), uncertainty

    @u.quantity_input(height=u.m)
    def gradient(self, height: u.Quantity) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the local gradient and its standard uncertainty.

        Parameters
        ----------
        height : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g'(h)=\sum_{k=1}^n k a_k h^{k-1}` and its standard
            uncertainty in microgals per metre. Uncertainty is None without
            covariance.
        """
        basis = polyval(np.asarray(height.to_value(u.m)), polyder(self._basis, 1))
        return self._gravity_quantities(*self._evaluate_basis(basis), unit=u.uGal / u.m)

    @u.quantity_input(h1=u.m, h2=u.m)
    def mean_gradient(
        self, h1: u.Quantity, h2: u.Quantity
    ) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate the interval-average gradient and its standard uncertainty.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct endpoints from the same datum. Arrays must broadcast.

        Returns
        -------
        value, uncertainty : tuple
            :math:`\Delta g/(h_2-h_1)` and its standard uncertainty, both
            in microgals per metre. Uncertainty is None without covariance.
        """
        separation = u.Quantity(self._height_difference(h1, h2), u.m)
        difference, uncertainty = self.gravity_difference(h1, h2)
        if uncertainty is not None:
            uncertainty = uncertainty / abs(separation)
        return difference / separation, uncertainty

    @u.quantity_input(height=u.m)
    def curvature(self, height: u.Quantity) -> tuple[u.Quantity, u.Quantity | None]:
        r"""Evaluate curvature and its standard uncertainty.

        Parameters
        ----------
        height : ~astropy.units.Quantity
            Scalar or array of heights above the model datum.

        Returns
        -------
        value, uncertainty : tuple
            :math:`g''(h)=\sum_{k=2}^n k(k-1)a_k h^{k-2}` and its standard
            uncertainty in microgals per metre squared. Uncertainty is None
            without covariance.
        """
        basis = polyval(np.asarray(height.to_value(u.m)), polyder(self._basis, 2))
        return self._gravity_quantities(
            *self._evaluate_basis(basis), unit=u.uGal / u.m**2
        )

    def _evaluate_basis(self, basis: NDArray[np.floating]):
        """Apply a linear functional and propagate its coefficient covariance."""
        value = np.einsum("i,i...->...", self._coefficients, basis)
        uncertainty = None
        if self._covariance is not None:
            variance = np.einsum("i...,ij,j...->...", basis, self._covariance, basis)
            uncertainty = np.sqrt(variance)
        return value, uncertainty

    @staticmethod
    def _gravity_quantities(
        value: ArrayLike, uncertainty: ArrayLike | None, *, unit: u.UnitBase
    ) -> tuple[u.Quantity, u.Quantity | None]:
        """Attach units to a value and its standard uncertainty."""
        return (
            u.Quantity(value, unit),
            None if uncertainty is None else u.Quantity(uncertainty, unit),
        )

    @staticmethod
    def _height_difference(h1: u.Quantity, h2: u.Quantity):
        """Return the signed separation in metres for distinct heights."""
        separation = np.asarray(h2.to_value(u.m)) - np.asarray(h1.to_value(u.m))
        if np.any(separation == 0):
            raise ValueError("An interval average requires distinct heights")
        return separation

    @classmethod
    @u.quantity_input(h1=u.m, h2=u.m, delta_g=u.uGal)
    def fit_gravity_differences(
        cls,
        h1: u.Quantity,
        h2: u.Quantity,
        delta_g: u.Quantity,
        *,
        set_ids: ArrayLike,
        degree: int = 2,
        **kwargs: Any,
    ):
        r"""Fit a vertical gravity gradient from measured gravity differences.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Reference and target heights for each tie, in compatible length
            units. All ties in a set must share the same reference height.
        delta_g : ~astropy.units.Quantity
            Measured differences :math:`\Delta g_{ij}=g(h_j)-g(h_i)`, where
            :math:`h_i` is the reference and :math:`h_j` is the target.
        set_ids : array_like
            Set identifier for each tie. Arrays are paired by position.
        degree : int, optional
            Positive polynomial degree. Default is 2.
        **kwargs
            Arguments passed to statsmodels OLS.

        Returns
        -------
        gmeterpy.models.vgrad.PolynomialVerticalGravityFitResults
            Fitted model and diagnostics for the reconstructed level values.

        Raises
        ------
        ValueError
            If the degree is below one, a set has multiple reference heights,
            the coefficients cannot be estimated uniquely, or the fit has no
            residual degrees of freedom to estimate uncertainty.

        Notes
        -----
        Each set's reference is assigned zero relative gravity. Differences
        give the target level values. The fit uses these floating levels with
        one offset per set, following Jaakko Mäkinen (2010) [Makinen2010]_.
        Supply ties from the common reference to each target level. Each set
        contributes one reference row and one row per supplied target tie.
        Equal target values are retained as separate observations.
        """
        from statsmodels.api import OLS

        if degree < 1:
            raise ValueError("Polynomial degree must be at least one")

        set_ids = np.asarray(set_ids)
        references = pd.DataFrame(
            {"set_ids": set_ids, "h": np.asarray(h1.to_value(u.m), dtype=float)}
        )
        if np.any(references.groupby("set_ids")["h"].nunique().to_numpy() > 1):
            raise ValueError("Each set must have a common reference height")

        references = references.drop_duplicates(subset="set_ids").assign(g=0.0)
        targets = pd.DataFrame(
            {
                "set_ids": set_ids,
                "h": np.asarray(h2.to_value(u.m), dtype=float),
                "g": np.asarray(delta_g.to_value(u.uGal), dtype=float),
            }
        )
        observations = (
            pd.concat([references, targets])
            .sort_values(["set_ids", "g"], ascending=[True, False], ignore_index=True)
            .reindex(columns=["h", "g", "set_ids"])
        )

        # design matrix
        polynomial = polyvander(observations.h.to_numpy(), degree)[:, 1:]
        offsets = pd.get_dummies(observations.set_ids)
        names = [f"offset[run={run}]" for run in offsets.columns]
        names.extend(f"a{k}" for k in range(1, degree + 1))
        exog = pd.DataFrame(np.hstack((offsets, polynomial)), columns=pd.Index(names))

        adjustment = OLS(observations.g.to_numpy(copy=True), exog, **kwargs)
        design = adjustment.exog
        assert design is not None
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise ValueError(
                f"The observations cannot identify a degree-{degree} polynomial "
                "and all run offsets"
            )
        if design.shape[0] <= design.shape[1]:
            raise ValueError(
                "The fit needs residual degrees of freedom to estimate uncertainty"
            )
        regression = adjustment.fit()
        coefficient_slice = slice(len(offsets.columns), None)
        coefficients = regression.params.iloc[coefficient_slice].to_numpy()
        covariance = regression.cov_params().iloc[coefficient_slice, coefficient_slice]
        fitted_model = cls(
            np.concatenate(([0.0], coefficients)),
            np.pad(covariance.to_numpy(), ((1, 0), (1, 0))),
            units=(u.uGal, u.m),
        )
        return PolynomialVerticalGravityFitResults(
            observations, regression, fitted_model
        )


class PolynomialVerticalGravityFitResults(FitResults):
    r"""Floating gravity observations, diagnostics and their fitted model.

    Parameters
    ----------
    observations : pandas.DataFrame
        Prepared level values: ``h`` in metres, ``g`` in microgals, and
        ``set_ids`` identifying measurement sets. Copied at construction.
    regression : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted equal-weight regression, including set offsets.
    fitted_model : PolynomialVerticalGravity
        Gravity model with coefficient covariance, constructed once by fit.

    Notes
    -----
    For measured level :math:`i` in set :math:`s`, the fitted gravity value
    includes a set offset :math:`b_s`. The geodetic residual is

    .. math::

       \hat g_{is} = b_s + g(h_{is}),
       \qquad v_{is} = \hat g_{is} - g_{is}.

    The fit minimizes :math:`\sum_{i,s} v_{is}^2` with equal observation
    weights, following Jaakko Mäkinen (2010) [Makinen2010]_. The full
    parameter covariance includes the set offsets and polynomial
    coefficients. The fitted model uses the coefficient block to propagate
    gravity difference uncertainty.

    Public tables are independent copies.
    The underlying statsmodels object is shared for read-only inspection.
    Plot returns a caller-owned figure; report writes a text file.
    """

    def __init__(
        self,
        observations: pd.DataFrame,
        regression: "RegressionResultsWrapper",
        fitted_model: PolynomialVerticalGravity,
    ) -> None:
        """Retain the fitted model and snapshot the prepared observations."""
        super().__init__(regression)
        self._observations = observations.drop(columns="resid", errors="ignore").copy()
        self._fitted_model = fitted_model

    @property
    def observations(self) -> pd.DataFrame:
        """Prepared floating observations with geodetic residuals.

        Returns
        -------
        pandas.DataFrame
            Independent table with ``h`` in metres, ``g`` and ``resid`` in
            microgals, and set identifiers ``set_ids``.
        """
        return self._observations.assign(resid=self.residuals)

    @property
    def fitted_model(self) -> PolynomialVerticalGravity:
        """Gravity model estimated by this fit.

        Returns
        -------
        PolynomialVerticalGravity
            The single fitted model owned by these results. Public model
            coefficients and covariance are returned as copies.
        """
        return self._fitted_model

    def plot(
        self,
        station: str = "",
        *,
        h_ref: float = 0.710,
        subtract_gradient: float | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        height_range: tuple[float, float] | list[float] | None = None,
    ) -> "Figure":
        r"""Plot a fitted vertical gravity model and its uncertainty.

        Parameters
        ----------
        station : str, optional
            Station name used in the title.
        h_ref : float, optional
            Uncertainty reference height in metres. Default is 0.710.
        subtract_gradient : float, optional
            Linear gradient to subtract in microgals per metre. By default,
            display the original polynomial without detrending.
        xlim : tuple or list of float, optional
            Displayed gravity limits in microgals. Default uses automatic limits.
        ylim : tuple or list of float, optional
            Height axis limits in metres. Default uses automatic limits.
        height_range : tuple or list of float, optional
            Height range in metres over which to sample the model curve.
            Default is the observed height range.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with heights in metres and gravity in microgals.
            The caller is responsible for closing it.

        Notes
        -----
        With display gradient :math:`s` (zero when omitted), the center
        curve is :math:`g(h)-s h` and the observations are plotted at
        :math:`g(h)-s h-v`, where :math:`v` is fitted minus observed.
        The band is centered on the curve with half-width
        :math:`u\bigl(g(h)-g(h_{\mathrm{ref}})\bigr)`.
        """
        import matplotlib.pyplot as plt

        df = self.observations
        model = self.fitted_model

        gradient = 0 if subtract_gradient is None else subtract_gradient
        gpr = lambda x: (
            model.gravity(np.asarray(x) * u.m)[0].to_value(u.uGal) - gradient * x
        )

        fig, ax = plt.subplots(figsize=(6, 7))
        ax.set_ylabel("Height ($m$)", fontsize=12)
        ax.set_xlabel(r"Gravity ($\mu Gal$)", fontsize=12)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        markers = itertools.cycle(("*", "s", "^", "p", "o", "D"))
        colors = itertools.cycle(("g", "r", "m", "c", "y"))

        # plot source data
        for _eqn, group in df.groupby("set_ids"):
            ga = gpr(group.h) - group.resid
            ax.plot(
                ga,
                group.h,
                "--",
                marker=next(markers),
                color=next(colors),
                markersize=10,
                alpha=0.8,
                linewidth=1.0,
            )

        # plot curve
        if height_range is None:
            height_range = (df.h.min(), df.h.max())
        height_min, height_max = height_range
        h = np.linspace(height_min, height_max, 100)
        fitted_gravity = gpr(h)
        ax.plot(fitted_gravity, h, "b-", linewidth=2.0)

        # 1-sigma diff with h_ref height
        _, uncertainty = model.gravity_difference(h * u.m, h_ref * u.m)
        if uncertainty is None:
            raise ValueError("Coefficient covariance is required to plot uncertainty")
        sigma = uncertainty.to_value(u.uGal)
        ci_l = fitted_gravity - sigma
        ci_u = fitted_gravity + sigma
        ax.plot(ci_l, h, "b", ci_u, h, "b", linestyle="dashed")
        ax.fill_betweenx(h, ci_l, ci_u, alpha=0.05, color="0.05")

        title = f"{station}"
        subtitle = f"degree={model.degree}, "
        if subtract_gradient is not None:
            subtitle += rf"${gradient:.1f}\,\mu$Gal / m substructed, "
        subtitle += "$u_{k = 1}$"
        fig.suptitle(title, fontsize=14, y=0.95)
        ax.set_title(subtitle, fontsize=12)

        return fig

    def report(
        self,
        path: str | PathLike[str],
        station: str = "",
        *,
        data: pd.DataFrame | None = None,
    ) -> None:
        r"""Write a vertical gravity fit report.

        Parameters
        ----------
        path : str or os.PathLike
            Destination file.
        station : str, optional
            Station name used in the report.
        data : pandas.DataFrame, optional
            Original tie input for the data section. Defaults to prepared
            observations when no tie table is supplied.

        Notes
        -----
        Reports heights in metres, gravity differences in microgals, and average
        gradients in microgals per metre. Uncertainties have coverage factor k = 1.

        For each pair of control heights :math:`h_j>h_i`, the reported
        average gradient and its standard uncertainty are

        .. math::

           \bar\gamma = \frac{\Delta g_{ij}}{h_j-h_i},
           \qquad u(\bar\gamma)=\frac{u(\Delta g_{ij})}{|h_j-h_i|}.
        """

        data = self.observations if data is None else data
        model = self.fitted_model
        res = self.regression
        offset_labels = {
            f"offset[run={run}]": f"offset[run={run}, h_min={group.h.min():.3f} m]"
            for run, group in self._observations.groupby("set_ids")
        }

        h_control = np.array(
            [0.000, 0.05, 0.130, 0.270, 0.700, 0.720, 0.900, 1.000, 1.200, 1.300, 1.4]
        )
        initial, final = np.triu_indices(len(h_control), k=1)
        h1, h2 = h_control[initial], h_control[final]
        differences, uncertainties = model.gravity_difference(h1 * u.m, h2 * u.m)
        if uncertainties is None:
            raise ValueError("Coefficient covariance is required to report uncertainty")
        dg = np.asarray(differences.to_value(u.uGal))
        se_dg = np.asarray(uncertainties.to_value(u.uGal))
        separation = h2 - h1
        gradients = dg / separation
        se_gradients = se_dg / np.abs(separation)

        with open(path, "w") as f:
            f.write(
                "Results of the polynomial fit of the vertical gravity gradients at "
                + station
                + "\n\n"
            )
            f.write("Data:\n")
            f.write(5 * "*" + "\n")
            f.write("{}\n\n".format(data.to_string(index=False, na_rep="---")))
            f.write("Fit summary:\n" + 12 * "*" + "\n")
            f.write(f"{self.number_of_observations:2.0f} observations -\n")
            f.write(f"{self.number_of_unknowns:2.0f} parameters =\n")
            f.write(f"{self.degrees_of_freedom:2.0f} degrees of freedom\n\n")

            if hasattr(res, "ssr"):
                f.write(f"Sum of squared residuals: {res.ssr}\n\n")

            summary = res.summary2(alpha=0.05).tables[1].rename(index=offset_labels)
            f.write(summary.to_string(float_format="{:.2f}".format) + "\n\n")

            f.write("Covariance matrix\n")
            f.write(
                self.parameter_covariance.rename(
                    index=offset_labels, columns=offset_labels
                ).to_string(float_format="{:.6f}".format)
                + "\n\n"
            )

            f.write("Prediction table:\n")
            f.write(18 * "*" + "\n")
            f.write(
                "{:17}\t{:^}\t\t{:>}\n".format("Heights", "Gravity diffs", "Gradients")
            )
            for start, end, difference, sigma, gradient, gradient_sigma in zip(
                h1, h2, dg, se_dg, gradients, se_gradients, strict=True
            ):
                f.write(
                    f"{start:.3f} --> {end:.3f} m,\t{difference:6.1f} +/- {sigma:.1f} uGal,"
                    f"\t{gradient:6.1f} +/- {gradient_sigma:.1f} uGal/m\n"
                )
                if end == h_control[-1]:
                    f.write("\n")
            f.write("\n")
