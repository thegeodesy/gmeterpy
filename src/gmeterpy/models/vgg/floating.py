"""Floating gravity polynomial adjustment following Jaakko Mäkinen (2010)."""

import itertools
from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f as f_distribution
from statsmodels.stats.multitest import multipletests

import gmeterpy.units as u
from gmeterpy.numerics import polynomial

from .polynomial import PolynomialVerticalGravity, PolynomialVerticalGravityFitResults

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from statsmodels.iolib.summary2 import Summary
    from statsmodels.regression.linear_model import RegressionResultsWrapper


_UNIT_LEVERAGE = 1.0 - 1e-12


class _InsufficientFitData(ValueError):
    """The observations cannot support the requested fit."""


def _included_ties(
    set_id: NDArray[Any], exclude_sets: ArrayLike | Sequence[Hashable]
) -> NDArray[np.bool_]:
    """Select ties from all sets except the requested exclusions."""
    excluded = pd.Index(np.atleast_1d(np.asarray(exclude_sets, dtype=object)))
    if excluded.empty:
        return np.ones(set_id.shape, dtype=bool)
    unknown = excluded[~excluded.isin(set_id)].unique()
    if len(unknown):
        raise ValueError(f"Unknown measurement sets: {', '.join(map(str, unknown))}")
    included = ~pd.Index(set_id).isin(excluded)
    if not included.any():
        raise _InsufficientFitData("No measurement sets remain after exclusions")
    return included


def _held_out_p_value(refit: "FloatingGravityFitResults", ties: pd.DataFrame) -> float:
    """Test the omitted set against the refit using its joint prediction covariance."""
    model = refit.fitted_model
    coefficient_covariance = model.covariance
    variance = float(refit.regression.mse_resid)
    if coefficient_covariance is None or not np.isfinite(variance) or variance <= 0:
        return np.nan

    contrast = polynomial.polyvander(ties.h2.to_numpy(), model.degree) - (
        polynomial.polyvander(ties.h1.to_numpy(), model.degree)
    )
    errors = (ties.predicted_delta_g - ties.observed_delta_g).to_numpy()
    count = len(errors)
    # Each target contributes its own error plus the error of the shared reference.
    covariance = contrast @ coefficient_covariance @ contrast.T
    covariance += variance * (np.eye(count) + 1)
    if not np.isfinite(covariance).all() or not np.isfinite(errors).all():
        return np.nan
    try:
        standardized = np.linalg.solve(np.linalg.cholesky(covariance), errors)
    except np.linalg.LinAlgError:
        return np.nan
    statistic = standardized @ standardized / count
    return float(f_distribution.sf(statistic, count, refit.degrees_of_freedom))


def _holm_adjusted(p_values: pd.Series) -> pd.Series:
    """Adjust available p-values and preserve the original index."""
    available = p_values.dropna()
    adjusted = pd.Series(np.nan, index=p_values.index, dtype=float)
    if not available.empty:
        adjusted.loc[available.index] = multipletests(available, method="holm")[1]
    return adjusted


class FloatingGravityFit:
    """Fit a shared vertical gravity polynomial to floating gravity values.

    Parameters
    ----------
    degree : int, optional
        Positive polynomial degree. Default is 2.

    Attributes
    ----------
    method_name : str
        Display name of the fitting method.
    """

    method_name: ClassVar[str] = "Floating gravity OLS"

    def __init__(self, degree: int = 2) -> None:
        if degree < 1:
            raise ValueError("Polynomial degree must be at least one")
        self._degree = degree

    @property
    def degree(self) -> int:
        """Degree used for each fit."""
        return self._degree

    @u.quantity_input(h1=u.m, h2=u.m, delta_g=u.uGal)
    def fit(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        delta_g: u.Quantity,
        *,
        set_id: ArrayLike,
        exclude_sets: ArrayLike | Sequence[Hashable] = (),
        **kwargs: Any,
    ):
        r"""Fit a vertical gravity gradient from measured gravity differences.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Reference and target heights for each tie, in compatible length
            units. Supply ties from one reference height per set.
        delta_g : ~astropy.units.Quantity
            Measured differences :math:`\Delta g_{ij}=g(h_j)-g(h_i)`, where
            :math:`h_i` is the reference and :math:`h_j` is the target.
        set_id : array_like
            One identifier per tie. Use the same identifier for ties
            derived from the same set of adjusted gravity levels. Each
            identifier gets its own fitted offset. Arrays are paired by
            position.
        exclude_sets : scalar or array_like, optional
            Set identifiers to omit. Each identifier must occur in
            ``set_id``. All ties in those sets are excluded from the fit.
        **kwargs
            Arguments passed to statsmodels OLS.

        Returns
        -------
        gmeterpy.models.vgg.floating.FloatingGravityFitResults
            Fitted model and diagnostics for the reconstructed level values.

        Raises
        ------
        ValueError
            If a set has multiple reference heights, the coefficients cannot
            be estimated uniquely, or the fit has no
            residual degrees of freedom to estimate uncertainty. Also raised
            if an exclusion is unknown or leaves no measurement sets.

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

        if "auto_exclude" in kwargs:
            raise TypeError(
                "auto_exclude has been removed. Use exclude_sets to omit sets."
            )
        set_id = np.asarray(set_id)
        included = _included_ties(set_id, exclude_sets)
        degree = self.degree
        excluded_set_ids = tuple(pd.unique(set_id[~included]).tolist())
        if excluded_set_ids:
            h1 = cast(u.Quantity, h1[included])
            h2 = cast(u.Quantity, h2[included])
            delta_g = cast(u.Quantity, delta_g[included])
            set_id = set_id[included]
        references = pd.DataFrame(
            {"set_id": set_id, "h": np.asarray(h1.to_value(u.m), dtype=float)}
        )
        reference_heights = references.h.groupby(references.set_id, sort=False)
        counts = reference_heights.nunique()
        inconsistent = counts.index[counts > 1]
        if len(inconsistent):
            identifier = inconsistent[0]
            heights = reference_heights.get_group(identifier).dropna().unique()
            listed = ", ".join(f"{float(height)} m" for height in heights)
            raise ValueError(
                f"Set {identifier} has multiple reference heights: {listed}"
            )

        references = references.drop_duplicates(subset="set_id").assign(g=0.0)
        targets = pd.DataFrame(
            {
                "set_id": set_id,
                "h": np.asarray(h2.to_value(u.m), dtype=float),
                "g": np.asarray(delta_g.to_value(u.uGal), dtype=float),
            }
        )
        observations = (
            pd.concat([references, targets])
            .sort_values(["set_id", "g"], ascending=[True, False], ignore_index=True)
            .reindex(columns=["h", "g", "set_id"])
        )

        # design matrix
        height_powers = polynomial.polyvander(observations.h.to_numpy(), degree)[:, 1:]
        offsets = pd.get_dummies(observations.set_id)
        names = [f"offset[run={run}]" for run in offsets.columns]
        names.extend(f"a{k}" for k in range(1, degree + 1))
        exog = pd.DataFrame(
            np.hstack((offsets, height_powers)), columns=pd.Index(names)
        )

        adjustment = OLS(observations.g.to_numpy(copy=True), exog, **kwargs)
        design = adjustment.exog
        assert design is not None
        if np.linalg.matrix_rank(design) != design.shape[1]:
            raise _InsufficientFitData(
                f"The observations cannot identify a degree-{degree} polynomial "
                "and all run offsets"
            )
        if design.shape[0] <= design.shape[1]:
            raise _InsufficientFitData(
                "The fit needs residual degrees of freedom to estimate uncertainty"
            )
        regression = adjustment.fit()
        coefficient_slice = slice(len(offsets.columns), None)
        coefficients = regression.params.iloc[coefficient_slice].to_numpy()
        covariance = regression.cov_params().iloc[coefficient_slice, coefficient_slice]
        fitted_model = PolynomialVerticalGravity(
            np.concatenate(([0.0], coefficients)),
            np.pad(covariance.to_numpy(), ((1, 0), (1, 0))),
            units=(u.uGal, u.m),
        )
        return FloatingGravityFitResults(
            observations,
            regression,
            fitted_model,
            method_name=self.method_name,
            excluded_set_ids=excluded_set_ids,
        )

    @u.quantity_input(h1=u.m, h2=u.m, delta_g=u.uGal, transfer_h1=u.m, transfer_h2=u.m)
    def leave_one_set_out(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        delta_g: u.Quantity,
        *,
        set_id: ArrayLike,
        exclude_sets: ArrayLike | Sequence[Hashable] = (),
        transfer_h1: u.Quantity | None = None,
        transfer_h2: u.Quantity | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Refit without each set and predict its held-out ties.

        Parameters
        ----------
        h1, h2, delta_g : ~astropy.units.Quantity
            Tie reference heights, target heights, and measured gravity
            differences (target minus reference), as in ``fit()``.
        set_id : array_like
            Adjusted-level set identifiers as in ``fit()``. Supply one per
            tie with no missing identifiers.
        exclude_sets : scalar or array_like, optional
            Set identifiers to omit from the baseline, refits, and held-out
            ties. Each identifier must occur in ``set_id``.
        transfer_h1, transfer_h2 : ~astropy.units.Quantity, optional
            Initial and final heights for optional transfer comparisons.
            When supplied, arrays must broadcast and heights in each pair
            must differ.

        Returns
        -------
        transfers : pandas.DataFrame
            One row per omitted set and transfer pair. Includes the full-fit
            gravity difference and its standard uncertainty, the refitted
            difference and standard uncertainty (``refit_delta_g`` and
            ``refit_uncertainty``), and ``transfer_shift`` (refit minus full fit).
            ``full_fit_extrapolated`` and
            ``refit_extrapolated`` flag either endpoint outside the
            respective fitted height range.
            ``fit_error`` gives the reason when an omission leaves too little
            data to fit. The refitted difference, shift, and extrapolation
            flag are missing in that case. The table is empty when no
            transfer heights are supplied.
        held_out_ties : pandas.DataFrame
            One row per retained tie, in input order. ``source_row`` gives its
            1-based position before exclusions. Includes its observed
            difference, the prediction from the fit that omitted its set,
            and ``prediction_error`` (predicted minus observed). Predictions
            and errors are NaN when the refit is unavailable. The fitted
            height range after omission is given by ``refit_height_min`` and
            ``refit_height_max``. ``fit_error`` records why a refit was unavailable.
            ``set_p_value`` and ``set_p_value_adjusted`` give the raw and
            Holm-adjusted test p-values, repeated on all ties in a set.
            They are NaN when the test is unavailable.

        Raises
        ------
        ValueError
            If just one transfer height is supplied or exclusions are invalid.

        Notes
        -----
        Heights in the tables are in metres, and gravity values are in
        microgals. The full fit and refits use the same polynomial degree
        and equal weights. Transfer shifts measure sensitivity to set omission.
        The outlier test assumes independent Gaussian level errors of equal
        variance before differencing. The prediction covariance includes
        refit coefficient uncertainty and the covariance of ties that share
        a reference level. The joint F test reduces to a two-sided t test
        for one tie. Holm adjustment uses one p-value per testable set.
        """
        if (transfer_h1 is None) != (transfer_h2 is None):
            raise ValueError("Supply both transfer heights or omit both")
        set_id = np.asarray(set_id)
        included = _included_ties(set_id, exclude_sets)
        if pd.isna(set_id[included]).any():
            raise ValueError("Every tie must have a set identifier")
        full_fit = self.fit(
            h1,
            h2,
            delta_g,
            set_id=set_id,
            exclude_sets=exclude_sets,
        )
        if not included.all():
            h1 = cast(u.Quantity, h1[included])
            h2 = cast(u.Quantity, h2[included])
            delta_g = cast(u.Quantity, delta_g[included])
            set_id = set_id[included]
        held_out_ties = pd.DataFrame(
            {
                "source_row": np.flatnonzero(included) + 1,
                "set_id": set_id,
                "h1": h1.to_value(u.m),
                "h2": h2.to_value(u.m),
                "observed_delta_g": delta_g.to_value(u.uGal),
                "predicted_delta_g": np.nan,
                "refit_height_min": np.nan,
                "refit_height_max": np.nan,
                "fit_error": None,
            }
        )
        refits = {}
        fit_errors = {}
        p_values = {}
        for omitted_set_id in pd.unique(set_id):
            omitted = set_id == omitted_set_id
            try:
                if omitted.all():
                    raise _InsufficientFitData("No measurement sets remain")
                refit = self.fit(
                    cast(u.Quantity, h1[~omitted]),
                    cast(u.Quantity, h2[~omitted]),
                    cast(u.Quantity, delta_g[~omitted]),
                    set_id=set_id[~omitted],
                )
            except _InsufficientFitData as error:
                held_out_ties.loc[omitted, "fit_error"] = str(error)
                if transfer_h1 is not None:
                    fit_errors[omitted_set_id] = str(error)
                continue

            if transfer_h1 is not None:
                refits[omitted_set_id] = refit
            predicted, _ = refit.fitted_model.gravity_difference(
                cast(u.Quantity, h1[omitted]), cast(u.Quantity, h2[omitted])
            )
            held_out_ties.loc[omitted, "predicted_delta_g"] = predicted.to_value(u.uGal)
            held_out_ties.loc[omitted, ["refit_height_min", "refit_height_max"]] = (
                refit.fitted_height_range
            )
            p_value = _held_out_p_value(refit, held_out_ties.loc[omitted])
            if np.isfinite(p_value):
                p_values[omitted_set_id] = p_value

        held_out_ties["prediction_error"] = (
            held_out_ties.predicted_delta_g - held_out_ties.observed_delta_g
        )
        raw = pd.Series(p_values, dtype=float)
        held_out_ties["set_p_value"] = held_out_ties.set_id.map(raw)
        held_out_ties["set_p_value_adjusted"] = held_out_ties.set_id.map(
            _holm_adjusted(raw)
        )
        if transfer_h1 is None or transfer_h2 is None:
            return pd.DataFrame(
                columns=pd.Index(
                    [
                        "h1",
                        "h2",
                        "full_fit_delta_g",
                        "full_fit_uncertainty",
                        "full_fit_extrapolated",
                        "omitted_set_id",
                        "refit_delta_g",
                        "refit_uncertainty",
                        "transfer_shift",
                        "refit_extrapolated",
                        "fit_error",
                    ]
                )
            ), held_out_ties
        baseline = (
            full_fit.prediction_table(transfer_h1, transfer_h2)
            .filter(
                items=[
                    "h1",
                    "h2",
                    "gravity_difference",
                    "gravity_difference_uncertainty",
                    "extrapolated",
                ]
            )
            .rename(
                columns={
                    "gravity_difference": "full_fit_delta_g",
                    "gravity_difference_uncertainty": "full_fit_uncertainty",
                    "extrapolated": "full_fit_extrapolated",
                }
            )
        )
        transfers = []
        for omitted_set_id in pd.unique(set_id):
            refit = refits.get(omitted_set_id)
            refit_delta_g = np.full(len(baseline), np.nan)
            refit_uncertainty = np.full(len(baseline), np.nan)
            refit_extrapolated = None
            if refit is not None:
                predictions = refit.prediction_table(transfer_h1, transfer_h2)
                refit_delta_g = predictions.gravity_difference.to_numpy()
                refit_uncertainty = (
                    predictions.gravity_difference_uncertainty.to_numpy()
                )
                refit_extrapolated = predictions.extrapolated.to_numpy()

            transfers.append(
                baseline.assign(
                    omitted_set_id=omitted_set_id,
                    refit_delta_g=refit_delta_g,
                    refit_uncertainty=refit_uncertainty,
                    transfer_shift=refit_delta_g - baseline.full_fit_delta_g,
                    refit_extrapolated=refit_extrapolated,
                    fit_error=fit_errors.get(omitted_set_id),
                )
            )

        return pd.concat(transfers, ignore_index=True), held_out_ties


class FloatingGravityFitResults(PolynomialVerticalGravityFitResults):
    r"""Floating gravity observations, diagnostics and their fitted model.

    Parameters
    ----------
    observations : pandas.DataFrame
        Prepared level values: ``h`` in metres, ``g`` in microgals, and
        ``set_id`` identifying measurement sets. Copied at construction.
    regression : statsmodels.regression.linear_model.RegressionResultsWrapper
        Fitted equal-weight regression, including set offsets.
    fitted_model : PolynomialVerticalGravity
        Gravity model with coefficient covariance, constructed once by fit.
    method_name : str, optional
        Display name of the fitting method.
    excluded_set_ids : tuple, optional
        Set identifiers excluded before fitting.

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
    Plot returns a caller-owned figure.
    """

    def __init__(
        self,
        observations: pd.DataFrame,
        regression: "RegressionResultsWrapper",
        fitted_model: PolynomialVerticalGravity,
        *,
        method_name: str = FloatingGravityFit.method_name,
        excluded_set_ids: tuple[Hashable, ...] = (),
    ) -> None:
        """Retain the fitted model and snapshot the prepared observations."""
        super().__init__(regression, fitted_model)
        self._observations = observations.drop(columns="resid", errors="ignore").copy()
        self._method_name = method_name
        self._excluded_set_ids = excluded_set_ids

    @property
    def method_name(self) -> str:
        """Display name of the method used to fit the model."""
        return self._method_name

    @property
    def excluded_set_ids(self) -> tuple[Hashable, ...]:
        """Explicitly excluded set IDs, in input order."""
        return self._excluded_set_ids

    @property
    def observations(self) -> pd.DataFrame:
        """Prepared floating observations with geodetic residuals.

        Returns
        -------
        pandas.DataFrame
            Independent table with ``h`` in metres, ``g`` and ``resid`` in
            microgals, and set identifiers ``set_id``.
        """
        return self._observations.assign(resid=self.residuals)

    @property
    def fitted_height_range(self) -> tuple[float, float]:
        """Minimum and maximum heights included in the fit, in metres."""
        observations = self.observations
        heights = observations.loc[observations.resid.notna(), "h"]
        return float(heights.min()), float(heights.max())

    def prediction_table(self, h1: u.Quantity, h2: u.Quantity) -> pd.DataFrame:
        """Return transfer predictions and flag extrapolation.

        Parameters
        ----------
        h1, h2 : ~astropy.units.Quantity
            Distinct initial and final heights. Arrays must broadcast.

        Returns
        -------
        pandas.DataFrame
            Gravity differences, average gradients, and their uncertainties
            in microgals and microgals per metre, respectively. Heights are
            in metres. ``extrapolated`` marks pairs with either height
            outside the fitted height range.
        """
        predictions = super().prediction_table(h1, h2)
        lower, upper = self.fitted_height_range
        endpoints = predictions[["h1", "h2"]]
        predictions["extrapolated"] = ((endpoints < lower) | (endpoints > upper)).any(
            axis="columns"
        )
        return predictions

    def summary(self, *, alpha: float = 0.05) -> "Summary":
        """Return the statsmodels summary for the floating-gravity OLS fit.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the parameter confidence intervals.

        Returns
        -------
        statsmodels.iolib.summary2.Summary
            Regression summary with parameter estimates and fit statistics.
        """
        return self.regression.summary2(alpha=alpha)

    def influence_table(self) -> pd.DataFrame:
        """Return OLS influence diagnostics for the floating observations.

        Returns
        -------
        pandas.DataFrame
            The floating observations with leverage, internally studentized
            residuals (fitted minus observed), and Cook's distance. The three
            diagnostics are dimensionless. The ``unit_leverage`` flag marks
            leverage numerically equal to one. Studentized residuals and
            Cook's distance are NaN when undefined, including at unit leverage.
            The ``inspect`` flag marks observations that exceed an inspection
            threshold or have unit leverage.
        """
        influence = self.regression.get_influence()
        leverage = influence.hat_matrix_diag
        unit_leverage = leverage >= _UNIT_LEVERAGE
        with np.errstate(divide="ignore", invalid="ignore"):
            diagnostics = pd.DataFrame(
                {
                    "leverage": leverage,
                    "unit_leverage": unit_leverage,
                    "studentized_residual": -influence.resid_studentized_internal,
                    "cooks_distance": influence.cooks_distance[0],
                },
                index=self.regression.model.data.row_labels,
            )
        columns = ["studentized_residual", "cooks_distance"]
        values = diagnostics[columns]
        diagnostics[columns] = values.where(
            np.isfinite(values) & (leverage[:, None] < _UNIT_LEVERAGE)
        )
        table = self.observations.join(diagnostics)
        residual_limit, leverage_limit, cook_limit = self.inspection_limits
        table["inspect"] = (
            table.studentized_residual.abs().gt(residual_limit)
            | table.leverage.gt(leverage_limit)
            | table.cooks_distance.gt(cook_limit)
            | table.leverage.ge(_UNIT_LEVERAGE)
        )
        return table

    @property
    def inspection_limits(self) -> tuple[float, float, float]:
        """Cutoffs for absolute studentized residual, leverage, and Cook's distance."""
        n, p = self.number_of_observations, self.number_of_unknowns
        return 2, 2 * p / n, 4 / n

    def plot(
        self,
        station: str = "",
        *,
        h_ref: float = 0.710,
        subtract_gradient: float | None = None,
        xlim: tuple[float, float] | list[float] | None = None,
        ylim: tuple[float, float] | list[float] | None = None,
        height_range: tuple[float, float] | list[float] | None = None,
        highlight_set: str | int | None = None,
    ) -> "Figure":
        r"""Plot the detrended gravity fit and its transfer uncertainty.

        Parameters
        ----------
        station : str, optional
            Station name used in the title.
        h_ref : float, optional
            Uncertainty reference height in metres. Default is 0.710.
        subtract_gradient : float, optional
            Linear gradient to subtract in microgals per metre. Defaults
            to the secant between the lowest observed height and 1 m, or the
            local gradient at 1 m if those heights coincide.
        xlim : tuple or list of float, optional
            Displayed gravity limits in microgals. Default uses automatic limits.
        ylim : tuple or list of float, optional
            Height axis limits in metres. Default uses automatic limits.
        height_range : tuple or list of float, optional
            Height range in metres over which to sample the model curve.
            Default is the observed height range.
        highlight_set : str or int, optional
            Set identifier to highlight. Other sets are muted. Identifiers
            can be supplied as displayed in the report.

        Returns
        -------
        matplotlib.figure.Figure
            Figure with heights in metres and gravity in microgals.
            The caller is responsible for closing it.

        Raises
        ------
        ValueError
            If the specified set is absent or coefficient covariance is unavailable.

        Notes
        -----
        With display gradient :math:`s`, the center
        curve is :math:`g(h)-s h` and the observations are plotted at
        :math:`g(h)-s h-v`, where :math:`v` is fitted minus observed.
        The band is centered on the curve with half-width
        :math:`u\bigl(g(h)-g(h_{\mathrm{ref}})\bigr)`. The reference height
        is marked when it lies within the plotted curve's range. Dashed
        curve segments extend beyond the observed height range.
        """
        import matplotlib.pyplot as plt

        observations = self.observations
        sets = observations.groupby("set_id")
        highlighted_id = None if highlight_set is None else str(highlight_set)
        if highlighted_id is not None and highlighted_id not in map(str, sets.groups):
            raise ValueError(f"Unknown measurement set {highlighted_id!r}")

        model = self.fitted_model
        observed_min, observed_max = self.fitted_height_range
        height_min, height_max = (
            (observed_min, observed_max)
            if height_range is None
            else sorted(height_range)
        )
        reference_visible = height_min <= h_ref <= height_max
        heights = np.linspace(height_min, height_max, 100)
        # Include exact boundaries so the lines meet and the band closes at h_ref.
        boundaries = [observed_min, observed_max, h_ref]
        heights = np.union1d(
            heights,
            [height for height in boundaries if height_min <= height <= height_max],
        )
        _, uncertainty = model.gravity_difference(h_ref * u.m, heights * u.m)
        if uncertainty is None:
            raise ValueError("Coefficient covariance is required to plot uncertainty")
        sigma = np.asarray(uncertainty.to_value(u.uGal), dtype=float)

        gradient = subtract_gradient
        if gradient is None:
            if observed_min == 1.0:
                slope, _ = model.gradient(1.0 * u.m)
            else:
                slope, _ = model.mean_gradient(1.0 * u.m, observed_min * u.m)
            gradient = np.asarray(slope.to_value(u.uGal / u.m), dtype=float).item()

        def display_gravity(heights):
            return (
                model.gravity(np.asarray(heights) * u.m)[0].to_value(u.uGal)
                - gradient * heights
            )

        fitted_gravity = display_gravity(heights)

        fig, ax = plt.subplots(figsize=(6, 7))
        ax.set_ylabel("Height ($m$)", fontsize=12)
        ax.set_xlabel(r"Detrended gravity ($\mu$Gal)", fontsize=12)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        markers = itertools.cycle(("*", "s", "^", "p", "o", "D"))
        colors = itertools.cycle(("g", "r", "m", "c", "y"))

        for set_id, group in sets:
            group = group.sort_values("h", kind="stable")
            color = next(colors)
            highlighted = str(set_id) == highlighted_id
            muted = highlighted_id is not None and not highlighted
            ax.plot(
                display_gravity(group.h) - group.resid,
                group.h,
                "--",
                marker=next(markers),
                color="0.5" if muted else color,
                markersize=10,
                alpha=0.25 if muted else 0.8,
                linewidth=1.8 if highlighted else 1.0,
                zorder=4 if highlighted else 2,
            )

        observed = (observed_min <= heights) & (heights <= observed_max)
        if observed.any():
            ax.plot(fitted_gravity[observed], heights[observed], "b-", linewidth=2.0)
        if not observed.all():
            # Include the observed boundary in each extrapolated segment.
            extrapolated = ((height_min < observed_min) & (heights <= observed_min)) | (
                (height_max > observed_max) & (heights >= observed_max)
            )
            ax.plot(
                np.where(extrapolated, fitted_gravity, np.nan),
                heights,
                "--",
                color="0.4",
                linewidth=1.5,
                label="Extrapolated fit",
            )
        uncertainty_label = f"Standard uncertainty relative to {h_ref:.3f} m"
        ax.fill_betweenx(
            heights,
            fitted_gravity - sigma,
            fitted_gravity + sigma,
            color="royalblue",
            alpha=0.18,
            label=uncertainty_label,
        )
        if reference_visible:
            ax.axhline(h_ref, color="0.5", linestyle=":", linewidth=1, zorder=1)

        subtitle = (
            rf"Degree {model.degree} · ${gradient:.1f}\,\mu$Gal/m subtracted"
            rf" · Ref. height {h_ref:.3f} m"
        )
        title = station
        if highlighted_id is not None:
            title = (
                f"{station} · Set {highlighted_id}"
                if station
                else f"Set {highlighted_id}"
            )
        if title:
            fig.suptitle(title, fontsize=14, y=0.95)
        ax.set_title(subtitle, fontsize=11)

        return fig
