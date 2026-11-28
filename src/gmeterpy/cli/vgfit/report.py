"""Plain-text reporting for the vgfit command."""

from collections.abc import Mapping
from datetime import UTC, datetime
from os import PathLike

import numpy as np
import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg import FloatingGravityFitResults

from .input import resolve_columns


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
    review_note: str | None = None,
) -> None:
    r"""Write a vertical gravity fit report.

    Parameters
    ----------
    results : FloatingGravityFitResults
        Fitted gravity model and diagnostics.
    path : str or os.PathLike
        Destination file.
    station : str, optional
        Station name used in the report.
    data : pandas.DataFrame, optional
        Original tie table. Printed with its original column names, units,
        and row order. Defaults to prepared floating observations.
    input_source : str, optional
        Filename or stdin label shown in the fit summary.
    columns : mapping, optional
        Maps fit fields to headers in the original tie table.
    height_unit, gravity_unit : ~astropy.units.UnitBase, optional
        Source units for heights and gravity differences. Default to
        millimetres and microgals.
    held_out_ties : pandas.DataFrame, optional
        Tie prediction table returned as the second item by
        ``FloatingGravityFit.leave_one_set_out()``. The report summarizes
        the mean and RMS prediction error by omitted set.
    review_note : str, optional
        Reviewer comments. A supplied note appears in the fit summary when
        sets were excluded or no inspection is needed. Otherwise it appears
        in Inspection, which shows "Pending" when no note is supplied.

    Notes
    -----
    The fit summary includes the report generation time in UTC. The
    inspection section identifies ties to check when diagnostics flag
    an issue. The transfer model section gives its full-precision
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
    influence_report, flagged = _influence_report(results, source_columns, height_unit)
    inspection = []
    if not flagged.empty:
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
    if held_out_ties is not None:
        missing = held_out_ties.loc[held_out_ties.prediction_error.isna(), "set_id"]
        unavailable_sets = missing.astype(str).unique()
        if len(unavailable_sets):
            label = "set" if len(unavailable_sets) == 1 else "sets"
            inspection.append(
                f"Missing held-out predictions for {label} {', '.join(unavailable_sets)}. "
                "Check the remaining levels and residual degrees of freedom for each omission."
            )

    note = (review_note or "").strip()
    note_in_summary = bool(note and (excluded_sets or not inspection))
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Results of the polynomial fit of the vertical gravity gradients at "
            + station
            + "\n\n"
        )
        fit_summary = [
            ("Generated at", datetime.now(UTC).isoformat(timespec="seconds")),
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
        if input_source is not None:
            fit_summary.insert(1, ("Input source", input_source))
        if excluded_sets:
            fit_summary.append(("Excluded sets", ", ".join(map(str, excluded_sets))))
        if results.auto_excluded_set_ids is not None:
            fit_summary.append(
                (
                    "Auto-excluded sets",
                    ", ".join(map(str, results.auto_excluded_set_ids)) or "None",
                )
            )
        if note_in_summary:
            fit_summary.append(("Review note", note))
        f.write("Fit summary:\n" + 12 * "*" + "\n")
        for label, value in fit_summary:
            f.write(f"{label:<28} {value}\n")
        f.write("\n")
        if inspection:
            f.write("Inspection:\n" + 11 * "*" + "\n")
            f.write("\n".join(inspection) + "\n")
            if not note_in_summary:
                f.write(f"Review note: {note or 'Pending'}\n")
            f.write("\n")

        f.write("Transfer model:\n" + 15 * "*" + "\n")
        coefficient_names = [f"a{k}" for k in range(model.degree + 1)]
        height_powers = ["1"]
        height_powers.extend(
            "h" if k == 1 else f"h^{k}" for k in range(1, model.degree + 1)
        )
        difference_terms = ["0"]
        difference_terms.extend(
            "h2 - h1" if k == 1 else f"h2^{k} - h1^{k}"
            for k in range(1, model.degree + 1)
        )
        covariance_rows = ",\n".join(f"    {row}" for row in covariance.tolist())
        f.write(
            "Units (gravity, height)  uGal, m\n"
            f"a = [{', '.join(coefficient_names)}]\n"
            f"g(h) = [{', '.join(height_powers)}] @ a\n"
            f"d = [{', '.join(difference_terms)}]\n"
            "delta_g(h1, h2) = g(h2) - g(h1) = d @ a\n"
            "u(delta_g) = sqrt(d @ C @ d.T)\n"
            "Standard uncertainty (k = 1) with transfer heights treated as exact.\n\n"
            "Coefficient vector (ascending powers) and covariance (same order)\n"
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
        parameter_width = max(len("Param."), max(len(name) for name, *_ in parameters))
        f.write("Full fit:\n" + 9 * "*" + "\n")
        f.write(
            f"{'Param.':<{parameter_width}} {'h_min (m)':>9} {'Value':>11} "
            f"{'Std. unc.':>11}  Unit\n"
        )
        for name, height, value, uncertainty, unit in parameters:
            f.write(
                f"{name:<{parameter_width}} {height:>9} {value:>11} "
                f"{uncertainty:>11}  {unit}\n"
            )
        f.write("\n")

        f.write("Full covariance matrix\n")
        f.write("Entries use products of parameter units.\n")
        f.write(
            results.parameter_covariance.rename(
                index=offset_names, columns=offset_names
            ).to_string(float_format="{:.6g}".format)
        )
        f.write("\n\n")

        f.write(influence_report)

        if held_out_ties is not None:
            f.write("Leave-one-set-out summary:\n" + 26 * "*" + "\n")
            f.write("Held-out tie errors (predicted minus observed)\n")
            f.write(f"{'Set':<6} {'Ties':>5} {'Mean (uGal)':>12} {'RMS (uGal)':>12}\n")
            for set_id, group in held_out_ties.groupby("set_id", sort=False):
                errors = group.prediction_error.dropna()
                if errors.empty:
                    mean = rms = "n/a"
                else:
                    mean = f"{errors.mean():+.2f}"
                    rms = f"{np.sqrt((errors**2).mean()):.2f}"
                f.write(f"{set_id!s:<6} {len(group):5d} {mean:>12} {rms:>12}\n")
            f.write("\n")
        f.write("Data:\n")
        f.write(5 * "*" + "\n")
        if data is None:
            display_data = observations.rename(
                columns={"h": "h (m)", "g": "g (uGal)", "resid": "resid (uGal)"}
            )
            f.write(display_data.to_string(index=False, na_rep="---") + "\n\n")
        else:
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


def _influence_report(
    results: FloatingGravityFitResults,
    columns: Mapping[str, str] | None,
    height_unit: u.UnitBase,
) -> tuple[str, pd.DataFrame]:
    """Format influence diagnostics and return the flagged heights and sets."""
    table = results.influence_table()
    n = results.number_of_observations
    p = results.number_of_unknowns
    residual_limit, leverage_limit, cook_limit = results.inspection_limits
    flagged_rows = table.pop("inspect")
    flagged = table.loc[flagged_rows].filter(items=["set_id", "h"])
    table["Inspect"] = np.where(flagged_rows, "*", "")
    table = table.drop(columns="unit_leverage").rename(
        columns={
            "set_id": "Set",
            "h": "h (m)",
            "g": "g (uGal)",
            "resid": "Resid. (uGal)",
            "leverage": "Lev.",
            "studentized_residual": "Stud. resid.",
            "cooks_distance": "Cook's D",
        }
    )
    lines = [
        "Influence diagnostics:",
        "*" * 22,
        "Floating observations. Residuals are fitted minus observed.",
        table.to_string(index=False, float_format="{:.3f}".format, na_rep="n/a"),
        "",
        "  *  Observation flagged for inspection. Any one rule can trigger the flag.",
        "",
        "     Criterion          Rule",
        f"     Large residual     |Stud. resid.| > {residual_limit}",
        f"     High leverage      Lev. > 2p/n ({leverage_limit:.3f} for this fit)",
        f"     Cook's distance    Cook's D > 4/n ({cook_limit:.3f} for this fit)",
        "     Unit leverage      Lev. is approximately 1",
        "",
        "     Stud. resid. is the internally studentized residual. At unit",
        "     leverage, Stud. resid. and Cook's D are undefined (n/a).",
        f"     n = {n:.0f} fitted observations. p = {p} fitted parameters, including offsets.",
    ]
    if leverage_limit >= 1:
        lines.append("     The high-leverage cutoff reaches or exceeds 1 for this fit.")
    if columns is not None:
        lines.extend(
            [
                "",
                f"     Match Set to {columns['set_id']} in the original ties.",
                f"     Check ties whose {columns['h1']} or {columns['h2']} matches",
                f"     the flagged height after conversion to {height_unit}.",
                "     For a flagged shared reference height, check all ties in that set.",
            ]
        )
    return "\n".join(lines) + "\n\n", flagged


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
