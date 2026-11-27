"""Floating gravity polynomial adjustment following Jaakko Mäkinen (2010)."""

import itertools
from os import PathLike
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
        transfer_h1: u.Quantity = 1 * u.m,
        transfer_h2: u.Quantity = 0 * u.m,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Refit without each set and compare gravity transfers.

        Parameters
        ----------
        h1, h2, delta_g : ~astropy.units.Quantity
            Tie reference heights, target heights, and measured gravity
            differences (target minus reference), as in ``fit()``.
        set_ids : array_like
            One set identifier per tie, with no missing identifiers.
        transfer_h1, transfer_h2 : ~astropy.units.Quantity, optional
            Initial and final heights for the transfers to compare. Default
            is 1 m to 0 m. Heights in each pair must differ. Arrays must
            broadcast.

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
            flag are missing in that case.
        held_out_ties : pandas.DataFrame
            One row per original tie, in input order. Includes its observed
            difference, the prediction from the fit that omitted its set,
            and ``prediction_error`` (predicted minus observed). Predictions
            and errors are NaN when the refit is unavailable.

        Notes
        -----
        Heights in the tables are in metres, and gravity values are in
        microgals. The full fit and refits use the same polynomial degree
        and equal weights. Transfer shifts measure sensitivity to set omission.
        """
        set_ids = np.asarray(set_ids)
        if pd.isna(set_ids).any():
            raise ValueError("Every tie must have a set identifier")

        full_fit = self.fit(h1, h2, delta_g, set_ids=set_ids)
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

        held_out_ties = pd.DataFrame(
            {
                "set_ids": set_ids,
                "h1": h1.to_value(u.m),
                "h2": h2.to_value(u.m),
                "observed_delta_g": delta_g.to_value(u.uGal),
                "predicted_delta_g": np.nan,
            }
        )
        transfers = []
        for set_id in pd.unique(set_ids):
            omitted = set_ids == set_id
            refit_delta_g = np.full(len(baseline), np.nan)
            refit_extrapolated = None
            fit_error = None
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
                fit_error = str(error)
            else:
                predicted, _ = refit.fitted_model.gravity_difference(
                    transfer_h1, transfer_h2
                )
                refit_delta_g = np.ravel(predicted.to_value(u.uGal))
                observed_heights = refit.observations.h
                refit_extrapolated = lower_endpoint.lt(
                    observed_heights.min()
                ) | upper_endpoint.gt(observed_heights.max())

                predicted, _ = refit.fitted_model.gravity_difference(
                    cast(u.Quantity, h1[omitted]), cast(u.Quantity, h2[omitted])
                )
                held_out_ties.loc[omitted, "predicted_delta_g"] = predicted.to_value(
                    u.uGal
                )
            transfers.append(
                baseline.assign(
                    omitted_set_id=set_id,
                    refit_delta_g=refit_delta_g,
                    transfer_shift=refit_delta_g - baseline.full_fit_delta_g,
                    refit_extrapolated=refit_extrapolated,
                    fit_error=fit_error,
                )
            )

        held_out_ties["prediction_error"] = (
            held_out_ties.predicted_delta_g - held_out_ties.observed_delta_g
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
    Plot returns a caller-owned figure; report writes a text file.
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
            diagnostics are dimensionless. Studentized residuals and Cook's
            distance are NaN when undefined, including at unit leverage.
        """
        influence = self.regression.get_influence()
        leverage = influence.hat_matrix_diag
        with np.errstate(divide="ignore", invalid="ignore"):
            diagnostics = pd.DataFrame(
                {
                    "leverage": leverage,
                    "studentized_residual": -influence.resid_studentized_internal,
                    "cooks_distance": influence.cooks_distance[0],
                },
                index=self.regression.model.data.row_labels,
            )
        columns = ["studentized_residual", "cooks_distance"]
        values = diagnostics[columns]
        diagnostics[columns] = values.where(
            np.isfinite(values) & (leverage[:, None] < 1 - 1e-12)
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
        res = self.regression
        offset_labels = {
            f"offset[run={run}]": f"offset[run={run}, h_min={group.h.min():.3f} m]"
            for run, group in self._observations.groupby("set_ids")
        }

        h_control = np.array(
            [0.000, 0.05, 0.130, 0.270, 0.700, 0.720, 0.900, 1.000, 1.200, 1.300, 1.4]
        )
        initial, final = np.triu_indices(len(h_control), k=1)
        if self.fitted_model.covariance is None:
            raise ValueError("Coefficient covariance is required to report uncertainty")
        predictions = self.prediction_table(
            h_control[initial] * u.m, h_control[final] * u.m
        )

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
            for (
                start,
                end,
                difference,
                sigma,
                gradient,
                gradient_sigma,
            ) in predictions.itertuples(index=False, name=None):
                f.write(
                    f"{start:.3f} --> {end:.3f} m,\t{difference:6.1f} +/- {sigma:.1f} uGal,"
                    f"\t{gradient:6.1f} +/- {gradient_sigma:.1f} uGal/m\n"
                )
                if end == h_control[-1]:
                    f.write("\n")
            f.write("\n")
