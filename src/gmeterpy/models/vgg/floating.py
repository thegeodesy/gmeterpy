"""Floating gravity polynomial adjustment following Jaakko Mäkinen (2010)."""

import itertools
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

import gmeterpy.units as u
from gmeterpy.numerics import polynomial

from .polynomial import PolynomialVerticalGravity, PolynomialVerticalGravityFitResults

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from statsmodels.regression.linear_model import RegressionResultsWrapper


_UNIT_LEVERAGE = 1.0 - 1e-12


class _InsufficientFitData(ValueError):
    """The observations cannot support the requested fit."""


class FloatingGravityFit:
    """Fit a shared vertical gravity polynomial to floating gravity values.

    Parameters
    ----------
    degree : int, optional
        Positive polynomial degree. Default is 2.
    """

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
        set_ids: ArrayLike,
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

        degree = self.degree

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
        height_powers = polynomial.polyvander(observations.h.to_numpy(), degree)[:, 1:]
        offsets = pd.get_dummies(observations.set_ids)
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
        return FloatingGravityFitResults(observations, regression, fitted_model)

    @u.quantity_input(h1=u.m, h2=u.m, delta_g=u.uGal, transfer_h1=u.m, transfer_h2=u.m)
    def leave_one_set_out(
        self,
        h1: u.Quantity,
        h2: u.Quantity,
        delta_g: u.Quantity,
        *,
        set_ids: ArrayLike,
        transfer_h1: u.Quantity | None = None,
        transfer_h2: u.Quantity | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Refit without each set and predict its held-out ties.

        Parameters
        ----------
        h1, h2, delta_g : ~astropy.units.Quantity
            Tie reference heights, target heights, and measured gravity
            differences (target minus reference), as in ``fit()``.
        set_ids : array_like
            One set identifier per tie, with no missing identifiers.
        transfer_h1, transfer_h2 : ~astropy.units.Quantity, optional
            Initial and final heights for optional transfer comparisons.
            When supplied, arrays must broadcast and heights in each pair
            must differ.

        Returns
        -------
        transfers : pandas.DataFrame
            One row per omitted set and transfer pair. Includes the full-fit
            gravity difference and its standard uncertainty, the refitted
            difference, and ``transfer_shift`` (refit minus full fit).
            ``is_largest_shift`` marks the largest absolute shift for each
            interval, including ties. ``full_fit_extrapolated`` and
            ``refit_extrapolated`` flag either endpoint outside the
            respective fit's observed height range.
            ``fit_error`` gives the reason when an omission leaves too little
            data to fit. The refitted difference, shift, and extrapolation
            flag are missing in that case. The table is empty when no
            transfer heights are supplied.
        held_out_ties : pandas.DataFrame
            One row per original tie, in input order. Includes its observed
            difference, the prediction from the fit that omitted its set,
            and ``prediction_error`` (predicted minus observed). Predictions
            and errors are NaN when the refit is unavailable.

        Raises
        ------
        ValueError
            If just one transfer height is supplied.

        Notes
        -----
        Heights in the tables are in metres, and gravity values are in
        microgals. The full fit and refits use the same polynomial degree
        and equal weights. Transfer shifts measure sensitivity to set omission.
        """
        if (transfer_h1 is None) != (transfer_h2 is None):
            raise ValueError("Supply both transfer heights or omit both")
        set_ids = np.asarray(set_ids)
        if pd.isna(set_ids).any():
            raise ValueError("Every tie must have a set identifier")

        full_fit = self.fit(h1, h2, delta_g, set_ids=set_ids)
        held_out_ties = pd.DataFrame(
            {
                "set_ids": set_ids,
                "h1": h1.to_value(u.m),
                "h2": h2.to_value(u.m),
                "observed_delta_g": delta_g.to_value(u.uGal),
                "predicted_delta_g": np.nan,
            }
        )
        refits = {}
        fit_errors = {}
        for set_id in pd.unique(set_ids):
            omitted = set_ids == set_id
            try:
                if omitted.all():
                    raise _InsufficientFitData("No measurement sets remain")
                refit = self.fit(
                    cast(u.Quantity, h1[~omitted]),
                    cast(u.Quantity, h2[~omitted]),
                    cast(u.Quantity, delta_g[~omitted]),
                    set_ids=set_ids[~omitted],
                )
            except _InsufficientFitData as error:
                if transfer_h1 is not None:
                    fit_errors[set_id] = str(error)
                continue

            if transfer_h1 is not None:
                refits[set_id] = refit
            predicted, _ = refit.fitted_model.gravity_difference(
                cast(u.Quantity, h1[omitted]), cast(u.Quantity, h2[omitted])
            )
            held_out_ties.loc[omitted, "predicted_delta_g"] = predicted.to_value(u.uGal)

        held_out_ties["prediction_error"] = (
            held_out_ties.predicted_delta_g - held_out_ties.observed_delta_g
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
                        "transfer_shift",
                        "refit_extrapolated",
                        "fit_error",
                        "is_largest_shift",
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
                ]
            )
            .rename(
                columns={
                    "gravity_difference": "full_fit_delta_g",
                    "gravity_difference_uncertainty": "full_fit_uncertainty",
                }
            )
        )
        endpoints = baseline.filter(items=["h1", "h2"])
        lower_endpoint = endpoints.min(axis="columns")
        upper_endpoint = endpoints.max(axis="columns")
        observed_heights = full_fit.observations.h
        baseline["full_fit_extrapolated"] = lower_endpoint.lt(
            observed_heights.min()
        ) | upper_endpoint.gt(observed_heights.max())

        transfers = []
        for set_id in pd.unique(set_ids):
            refit = refits.get(set_id)
            refit_delta_g = np.full(len(baseline), np.nan)
            refit_extrapolated = None
            if refit is not None:
                predicted, _ = refit.fitted_model.gravity_difference(
                    transfer_h1, transfer_h2
                )
                refit_delta_g = np.ravel(predicted.to_value(u.uGal))
                observed_heights = refit.observations.h
                refit_extrapolated = lower_endpoint.lt(
                    observed_heights.min()
                ) | upper_endpoint.gt(observed_heights.max())

            transfers.append(
                baseline.assign(
                    omitted_set_id=set_id,
                    refit_delta_g=refit_delta_g,
                    transfer_shift=refit_delta_g - baseline.full_fit_delta_g,
                    refit_extrapolated=refit_extrapolated,
                    fit_error=fit_errors.get(set_id),
                )
            )

        comparison = pd.concat(transfers, ignore_index=True)
        shifts = comparison.transfer_shift.abs()
        largest = shifts.groupby([comparison.h1, comparison.h2]).transform("max")
        comparison["is_largest_shift"] = shifts.eq(largest)
        return comparison, held_out_ties


class FloatingGravityFitResults(PolynomialVerticalGravityFitResults):
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
    Plot returns a caller-owned figure.
    """

    def __init__(
        self,
        observations: pd.DataFrame,
        regression: "RegressionResultsWrapper",
        fitted_model: PolynomialVerticalGravity,
    ) -> None:
        """Retain the fitted model and snapshot the prepared observations."""
        super().__init__(regression, fitted_model)
        self._observations = observations.drop(columns="resid", errors="ignore").copy()

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
        return self.observations.join(diagnostics)

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
        sets = observations.groupby("set_ids")
        highlighted_id = None if highlight_set is None else str(highlight_set)
        if highlighted_id is not None and highlighted_id not in map(str, sets.groups):
            raise ValueError(f"Unknown measurement set {highlighted_id!r}")

        model = self.fitted_model
        observed_min, observed_max = observations.h.min(), observations.h.max()
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
