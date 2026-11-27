"""Plain-text reporting for the vgfit command."""

from os import PathLike

import numpy as np
import pandas as pd

import gmeterpy.units as u
from gmeterpy.models.vgg import FloatingGravityFitResults


def write_report(
    results: FloatingGravityFitResults,
    path: str | PathLike[str],
    station: str = "",
    *,
    data: pd.DataFrame | None = None,
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

    data = results.observations if data is None else data
    res = results.regression
    offset_labels = {
        f"offset[run={run}]": f"offset[run={run}, h_min={group.h.min():.3f} m]"
        for run, group in results.observations.groupby("set_ids")
    }

    h_control = np.array(
        [0.000, 0.05, 0.130, 0.270, 0.700, 0.720, 0.900, 1.000, 1.200, 1.300, 1.4]
    )
    initial, final = np.triu_indices(len(h_control), k=1)
    if results.fitted_model.covariance is None:
        raise ValueError("Coefficient covariance is required to report uncertainty")
    predictions = results.prediction_table(
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
        f.write(f"{results.number_of_observations:2.0f} observations -\n")
        f.write(f"{results.number_of_unknowns:2.0f} parameters =\n")
        f.write(f"{results.degrees_of_freedom:2.0f} degrees of freedom\n\n")

        if hasattr(res, "ssr"):
            f.write(f"Sum of squared residuals: {res.ssr}\n\n")

        summary = res.summary2(alpha=0.05).tables[1].rename(index=offset_labels)
        f.write(summary.to_string(float_format="{:.2f}".format) + "\n\n")

        f.write("Covariance matrix\n")
        f.write(
            results.parameter_covariance.rename(
                index=offset_labels, columns=offset_labels
            ).to_string(float_format="{:.6f}".format)
            + "\n\n"
        )

        f.write("Prediction table:\n")
        f.write(18 * "*" + "\n")
        f.write("{:17}\t{:^}\t\t{:>}\n".format("Heights", "Gravity diffs", "Gradients"))
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
