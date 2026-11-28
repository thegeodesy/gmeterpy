"""Plain-text reporting for the vgfit command."""

from collections.abc import Hashable, Mapping
from datetime import UTC, datetime
from os import PathLike
from textwrap import fill

import numpy as np
import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg import FloatingGravityFitResults

from .input import resolve_columns

DEFAULT_TIE_TOLERANCE = 3.0


def write_report(
    results: FloatingGravityFitResults,
    path: str | PathLike[str],
    station: str = "",
    *,
    data: pd.DataFrame | None = None,
    input_source: str | None = None,
    columns: Mapping[str, str] | None = None,
    height_unit: u.UnitBase = u.mm,
    gravity_unit: u.UnitBase = u.uGal,
    held_out_ties: pd.DataFrame | None = None,
    held_out_transfers: pd.DataFrame | None = None,
    tie_tolerance: float = DEFAULT_TIE_TOLERANCE,
    review_note: str | None = None,
    exclusion_reasons: Mapping[Hashable, str] | None = None,
) -> None:
    r"""Write a vertical gravity fit report.

    Parameters
    ----------
    results : FloatingGravityFitResults
        Fitted gravity model and diagnostics.
    path : str or os.PathLike
        Destination file.
    station : str, optional
        Station name shown in the fit summary.
    data : pandas.DataFrame, optional
        Original tie table. Printed with its original column names, units,
        and row order when supplied.
    input_source : str, optional
        Source label accepted for compatibility with earlier callers.
    columns : mapping, optional
        Maps fit fields to headers in the original tie table.
    height_unit, gravity_unit : ~astropy.units.UnitBase, optional
        Source units for heights and gravity differences. Default to
        millimetres and microgals.
    held_out_ties : pandas.DataFrame, optional
        Tie prediction table returned as the second item by
        ``FloatingGravityFit.leave_one_set_out()``. The report summarizes
        the flagged ties and their signed prediction errors by set.
    held_out_transfers : pandas.DataFrame, optional
        First table returned by leave-one-set-out for a single transfer pair.
    tie_tolerance : float, optional
        Flag held-out tie errors whose magnitude exceeds this tolerance in
        microgals. Defaults to 3.
    review_note : str, optional
        Reviewer comments shown in the Review section.
    exclusion_reasons : mapping, optional
        Reasons keyed by manually excluded set ID, shown in the Review section.

    Notes
    -----
    The fit summary includes the report generation time in UTC. The
    Review section records exclusions and ties to check in the final fit. The transfer model
    section gives its full-precision
    coefficients, covariance, and base units. The full fit lists all
    parameters and their regression covariance. Each covariance entry has
    the product of the corresponding parameter units.
    Predictions use metres for heights, microgals for gravity differences,
    and microgals per metre for average gradients.
    Prediction uncertainties have coverage factor k = 1 and are shown
    with two significant digits when nonzero. A prediction is flagged
    as extrapolated if either height is outside the fitted height range.

    For each pair of control heights :math:`h_j>h_i`, the reported
    average gradient and its standard uncertainty are

    .. math::

       \bar\gamma = \frac{\Delta g_{ij}}{h_j-h_i},
       \qquad u(\bar\gamma)=\frac{u(\Delta g_{ij})}{|h_j-h_i|}.
    """

    observations = results.observations
    source_columns = None
    ties_by_set = None
    if data is not None:
        mapped = resolve_columns(data, columns)
        if all(column in data.columns for column in mapped.values()):
            source_columns = mapped
            ties_by_set = data.groupby(mapped["set_id"])
    res = results.regression
    number_of_sets = observations.set_id.nunique()
    excluded_sets = results.excluded_set_ids
    reasons = exclusion_reasons or {}
    for set_id in reasons:
        if set_id not in excluded_sets:
            raise ValueError(
                f"An exclusion reason requires a manually excluded set: {set_id!r}"
            )
    observed_min, observed_max = results.fitted_height_range
    offset_min_heights = observations.groupby("set_id").h.min()
    offset_names = {
        f"offset[run={run}]": f"offset[{run}]" for run in offset_min_heights.index
    }

    h_control = np.array(
        [0.000, 0.05, 0.130, 0.270, 0.700, 0.720, 0.900, 1.000, 1.200, 1.300, 1.4]
    )
    initial, final = np.triu_indices(len(h_control), k=1)
    model = results.fitted_model
    coefficients = model.coefficients
    covariance = model.covariance
    if covariance is None:
        raise ValueError("Coefficient covariance is required to report uncertainty")
    predictions = results.prediction_table(
        h_control[initial] * u.m, h_control[final] * u.m
    )
    inspection = []
    if held_out_ties is None:
        influence = results.influence_table()
        flagged = influence.loc[influence.inspect, ["set_id", "h"]]
        for set_id, group in flagged.groupby("set_id", sort=False):
            heights = group.h.unique()
            if (
                source_columns is None
                or ties_by_set is None
                or set_id not in ties_by_set.groups
            ):
                levels = ", ".join(f"{h:.6g}" for h in heights)
                inspection.append(
                    f"Set {set_id}: check ties involving the flagged heights "
                    f"({levels} m). Check every tie if the reference height is flagged."
                )
                continue
            ties = ties_by_set.get_group(set_id)
            first_tie = ties.iloc[0]
            reference_height = float(first_tie[source_columns["h1"]])
            if len(ties) == 1:
                target_height = float(first_tie[source_columns["h2"]])
                inspection.append(
                    f"Set {set_id}: check the {reference_height:.6g} to "
                    f"{target_height:.6g} {height_unit} tie."
                )
                continue
            if np.isclose(
                heights, reference_height * height_unit.to(u.m), rtol=0, atol=1e-12
            ).any():
                inspection.append(
                    f"Set {set_id}: check all ties. The shared reference height "
                    f"({reference_height:.6g} {height_unit}) is flagged."
                )
            else:
                for height in heights:
                    inspection.append(
                        f"Set {set_id}: check ties from {reference_height:.6g} {height_unit} "
                        f"to {height * u.m.to(height_unit):.6g} {height_unit}."
                    )
    held_out_check = ""
    if held_out_ties is not None:
        if not np.isfinite(tie_tolerance) or tie_tolerance <= 0:
            raise ValueError("Tie tolerance must be finite and greater than zero")
        inspected = held_out_ties.assign(
            inspect=held_out_ties.prediction_error.abs().gt(tie_tolerance)
        )
        held_out_check = _held_out_diagnostics(
            observations.set_id.unique(),
            (observed_min, observed_max),
            inspected,
            tie_tolerance,
            held_out_transfers,
        )

    note = (review_note or "").strip()
    with open(path, "w", encoding="utf-8") as f:
        f.write("Vertical gravity fit report\n\n")
        fit_summary = [
            ("Generated at", datetime.now(UTC).isoformat(timespec="seconds")),
            ("Station", station or "Unspecified"),
            ("Method", results.method_name),
            ("Polynomial degree", model.degree),
            (
                "Observed height range",
                f"{observed_min:.3f} to {observed_max:.3f} m",
            ),
            (
                "Ties used" if excluded_sets else "Input ties",
                len(observations) - number_of_sets,
            ),
            ("Sets used" if excluded_sets else "Measurement sets", number_of_sets),
            ("Fitted level observations", f"{results.number_of_observations:.0f}"),
            ("Fitted parameters", results.number_of_unknowns),
            ("Residual degrees of freedom", f"{results.degrees_of_freedom:.0f}"),
            ("Residual standard deviation", f"{results.s0:.3g} uGal"),
            ("Sum of squared residuals", f"{res.ssr:.6g} uGal^2"),
        ]
        exclusion_summary = str(len(excluded_sets))
        if excluded_sets:
            exclusion_summary += f" (IDs {', '.join(map(str, excluded_sets))})"
        fit_summary.append(("Excluded sets", exclusion_summary))
        if held_out_ties is not None:
            fit_summary.append(("Inspection threshold", f"{tie_tolerance:g} uGal"))
        f.write("Fit summary:\n" + 12 * "*" + "\n")
        for label, value in fit_summary:
            f.write(f"{label:<28} {value}\n")
        f.write("\n")

        if note or excluded_sets or held_out_check or inspection:
            f.write("Review:\n" + 7 * "*" + "\n")
            if note:
                f.write(f"Review note    {note}\n")
            if excluded_sets:
                if note:
                    f.write("\n")
                f.write("Manual exclusions\n")
                for set_id in excluded_sets:
                    reason = reasons.get(set_id, "").strip() or "Unspecified reason"
                    f.write(f"  Set {set_id}: {reason}\n")
            if held_out_check:
                if note or excluded_sets:
                    f.write("\n")
                f.write(held_out_check)
            if inspection:
                if note or excluded_sets:
                    f.write("\n")
                f.write("Final-fit inspection\n")
                f.write("\n".join(inspection) + "\n")
            f.write("\n")

        f.write("Transfer model:\n" + 15 * "*" + "\n")
        coefficient_names = [f"a{k}" for k in range(model.degree + 1)]
        polynomial_terms = ["a0"]
        polynomial_terms.extend(
            "a1*h" if k == 1 else f"a{k}*h^{k}" for k in range(1, model.degree + 1)
        )
        difference_terms = ["0"]
        difference_terms.extend(
            "h2 - h1" if k == 1 else f"h2^{k} - h1^{k}"
            for k in range(1, model.degree + 1)
        )
        covariance_rows = ",\n".join(f"    {row}" for row in covariance.tolist())
        f.write(
            "Use heights in metres. Gravity differences and uncertainties are in uGal.\n\n"
            "Gravity difference\n"
            f"  g(h) = {' + '.join(polynomial_terms)}\n"
            "  delta_g = g(h2) - g(h1)\n\n"
            "Standard uncertainty (k = 1)\n"
            f"  d = [{', '.join(difference_terms)}]\n"
            "  u(delta_g) = sqrt(d @ C @ d.T)\n"
            "  Transfer heights are treated as exact.\n\n"
            "Full-precision coefficients and covariance\n"
            f"a and the rows and columns of C follow the order [{', '.join(coefficient_names)}].\n"
            f"a = {coefficients.tolist()}\n"
            f"C = [\n{covariance_rows}\n]\n\n"
        )

        parameters = []
        for order, (coefficient, sigma) in enumerate(
            zip(coefficients, np.sqrt(covariance.diagonal()), strict=True)
        ):
            unit = "uGal"
            if order == 1:
                unit = "uGal/m"
            elif order > 1:
                unit = f"uGal/m^{order}"
            uncertainty = "fixed" if order == 0 and sigma == 0 else f"{sigma:.6g}"
            parameters.append(
                (f"a{order}", "", f"{coefficient:.6g}", uncertainty, unit)
            )

        estimates, uncertainties = results.params, results.sigma
        for run, height in offset_min_heights.items():
            name = f"offset[run={run}]"
            parameters.append(
                (
                    offset_names[name],
                    f"{height:.3f}",
                    f"{estimates[name]:.6g}",
                    f"{uncertainties[name]:.6g}",
                    "uGal",
                )
            )
        parameter_table = pd.DataFrame(
            parameters, columns=["Param.", "h_min (m)", "Value", "Std. unc.", "Unit"]
        )
        parameter_order = parameter_table["Param."]
        full_covariance = results.parameter_covariance.rename(
            index=offset_names, columns=offset_names
        )
        estimated_order = parameter_order[parameter_order.isin(full_covariance.columns)]
        covariance_table = (
            full_covariance.reindex(index=estimated_order, columns=estimated_order)
            .rename_axis("Param.")
            .reset_index()
        )
        f.write("Full fit:\n" + 9 * "*" + "\n")
        f.write(parameter_table.to_string(index=False) + "\n\n")
        f.write(
            "Full covariance matrix\n"
            "Entries use products of parameter units.\n"
            + covariance_table.to_string(index=False, float_format="{:.6g}".format)
            + "\n\n"
        )

        if data is not None:
            f.write("Data:\n")
            f.write(5 * "*" + "\n")
            if excluded_sets:
                f.write("Original input ties, including excluded sets.\n")
            if source_columns is not None:
                f.write(
                    "Input columns  "
                    + ", ".join(
                        f"{field}={name}" for field, name in source_columns.items()
                    )
                    + "\n"
                )
                f.write(
                    f"Input heights ({source_columns['h1']}, {source_columns['h2']}) in {height_unit}. "
                    f"Gravity differences ({source_columns['delta_g']}) in {gravity_unit}.\n"
                )
            f.write(data.to_string(index=False, na_rep="", float_format=str) + "\n\n")

        f.write("Prediction table:\n")
        f.write(18 * "*" + "\n")
        f.write(
            f"{'h1 --> h2 (m)':<15} {'Gravity diff. (uGal)':>21} "
            f"{'Mean grad. (uGal/m)':>21}  Extrap.\n"
        )
        for (
            start,
            end,
            difference,
            sigma,
            gradient,
            gradient_sigma,
            extrapolated,
        ) in predictions.itertuples(index=False, name=None):
            interval = f"{start:.3f} --> {end:.3f}"
            transfer = _format_uncertainty(difference, sigma)
            mean_gradient = _format_uncertainty(gradient, gradient_sigma)
            f.write(
                f"{interval:<15} {transfer:>21} {mean_gradient:>21}  "
                f"{'yes' if extrapolated else 'no'}\n"
            )
            if end == h_control[-1]:
                f.write("\n")
        f.write("\n")


def _held_out_diagnostics(
    set_ids: np.ndarray,
    fitted_height_range: tuple[float, float],
    held_out_ties: pd.DataFrame,
    tolerance: float,
    held_out_transfers: pd.DataFrame | None,
) -> str:
    """Show the evidence and consequences of omitting flagged sets."""
    ties_by_set = dict(iter(held_out_ties.groupby("set_id", sort=False)))
    transfers = {}
    if held_out_transfers is not None and not held_out_transfers.empty:
        if len(held_out_transfers[["h1", "h2"]].drop_duplicates()) != 1:
            raise ValueError(
                "The leave-one-set-out report requires a single transfer pair"
            )
        transfers = {row.omitted_set_id: row for row in held_out_transfers.itertuples()}

    shifts = {}
    for set_id, transfer in transfers.items():
        shift = "n/a"
        if pd.notna(transfer.transfer_shift):
            extrapolated = transfer.full_fit_extrapolated or transfer.refit_extrapolated
            shift = f"{transfer.transfer_shift:+.2f}{' E' if extrapolated else ''}"
        shifts[set_id] = shift

    details = []
    flagged_sets = set()
    unavailable_count = 0
    for set_id in set_ids:
        ties = ties_by_set.get(set_id)
        if ties is None:
            unavailable_count += 1
            details.extend(["", f"Set {set_id} has unavailable predictions."])
            continue
        flagged = ties.loc[ties.inspect]
        missing = int(ties.prediction_error.isna().sum())
        if not flagged.empty:
            flagged_sets.add(set_id)
            noun = "tie" if len(ties) == 1 else "ties"
            details.extend(
                [
                    "",
                    f"Set {set_id} retained ({len(flagged)} of {len(ties)} {noun} flagged)",
                ]
            )
            for tie in flagged.itertuples(index=False):
                source_row = getattr(tie, "source_row", None)
                row_label = f"Row {int(source_row)}  " if pd.notna(source_row) else ""
                error = f"{tie.prediction_error:+.2f}"
                if abs(float(error)) <= tolerance:
                    error = f"{tie.prediction_error:+}"
                details.append(
                    f"  {row_label}{tie.h1:.6g} to {tie.h2:.6g} m  {error} uGal"
                )
            reference = ties.iloc[0]
            for label, original, column in zip(
                ("Lowest", "Highest"),
                fitted_height_range,
                ("refit_height_min", "refit_height_max"),
                strict=True,
            ):
                refitted = reference.get(column, np.nan)
                if pd.notna(refitted) and refitted != original:
                    details.append(
                        f"  {label} fitted height on omission  {original:.6g} to {refitted:.6g} m"
                    )
            if set_id in transfers:
                transfer = transfers[set_id]
                details.append(f"  Transfer shift on omission (uGal)  {shifts[set_id]}")
                refit_uncertainty = getattr(transfer, "refit_uncertainty", np.nan)
                if pd.notna(refit_uncertainty) and pd.notna(
                    transfer.full_fit_uncertainty
                ):
                    details.append(
                        "  Standard uncertainty (uGal)  "
                        f"{transfer.full_fit_uncertainty:.3g} to {refit_uncertainty:.3g}"
                    )
        if missing:
            unavailable_count += 1
            noun = "tie" if missing == 1 else "ties"
            details.extend(
                ["", f"Set {set_id} has {missing} {noun} with unavailable predictions."]
            )
            fit_error = ties.iloc[0].get("fit_error")
            if pd.notna(fit_error):
                details.append(f"  {fit_error}.")

    lines = ["Leave-one-set-out check"]
    set_label = "set" if len(set_ids) == 1 else "sets"
    lines.append(
        f"{len(set_ids)} {set_label} checked, {len(flagged_sets)} flagged, "
        f"{unavailable_count} with unavailable predictions."
    )
    if flagged_sets:
        lines.extend(
            [
                "A tie is flagged when its absolute error exceeds the Inspection threshold in Fit summary.",
                "Errors are predicted minus observed.",
            ]
        )
    if transfers and flagged_sets:
        baseline = next(iter(transfers.values()))
        lines.append(
            f"Transfer from {baseline.h1!r} to {baseline.h2!r} m "
            "(shifts are refit minus final fit)."
        )
    lines.extend(details)
    other_shifts = [
        f"set {set_id} {shift}"
        for set_id, shift in shifts.items()
        if set_id not in flagged_sets
    ]
    if other_shifts:
        if flagged_sets:
            heading = "Other transfer shifts (uGal)"
        else:
            baseline = next(iter(transfers.values()))
            heading = (
                f"Transfer shifts (uGal) from {baseline.h1!r} to {baseline.h2!r} m, "
                "refit minus final fit."
            )
        lines.extend(["", heading, fill(", ".join(other_shifts), width=88)])
    if any(shift.endswith(" E") for shift in shifts.values()):
        lines.append("E marks extrapolation in either fit.")
    return "\n".join(lines) + "\n"


def _format_uncertainty(value: float, uncertainty: float) -> str:
    """Format a value and its standard uncertainty for the report."""
    if uncertainty == 0 or not np.isfinite(uncertainty):
        return f"{value:.1f} +/- {uncertainty:.1f}"

    scientific = f"{uncertainty:.1e}"
    decimals = 1 - int(scientific.split("e")[1])
    if decimals > 5:
        return f"{value:.6g} +/- {scientific}"

    places = max(decimals, 0)
    return f"{round(value, decimals):.{places}f} +/- {float(scientific):.{places}f}"
