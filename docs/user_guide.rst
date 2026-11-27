User guide
==========

Units (``gmeterpy.units``)
--------------------------

gMeterPy uses ``astropy.units`` for calculations with physical units. Import
``gmeterpy.units`` to use Astropy's units together with the additional units
defined by gMeterPy:

>>> import gmeterpy.units as u

Express gravitational acceleration in SI units:

>>> gravity = 9.8 * u.m / u.s**2
>>> gravity
<Quantity 9.8 m / s2>

Convert it to the CGS unit gal:

>>> gravity.to(u.Gal)
<Quantity 980. Gal>

Use ``uGal`` for microgals:

>>> gravity.to(u.uGal)
<Quantity 9.8e+08 uGal>

gMeterPy also defines the eotvos (``E``), a unit of gravity gradient:

>>> gradient = 0.3086 * u.mGal / u.m
>>> gradient.to(u.E)
<Quantity 3086. E>


Constants (``gmeterpy.constants``)
----------------------------------

gMeterPy extends ``astropy.constants`` with constants used in gravimetry.

.. list-table::
   :header-rows: 1

   * - Constant
     - Description
   * - ``omega``
     - Mean angular rotation rate of the Earth
   * - ``atm_sens``
     - Default gravity/pressure sensitivity (0.3 uGal / mbar)

International Standard Atmosphere
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Standard Atmosphere (ISO 2533:1975) is used to calculate
the normal pressure in the atmospheric pressure correction.

.. list-table::
   :header-rows: 1

   * - Constant
     - Description
   * - ``p0``
     - Standard atmospheric pressure at sea level
   * - ``L``
     - Temperature lapse rate
   * - ``Tn``
     - Standard thermodynamic air temperature at sea level
   * - ``gn``
     - Standard acceleration of free-fall
   * - ``M``
     - Air molar mass at sea level
   * - ``R``
     - Universal gas constant


Vertical gravity fitting
------------------------

The reference method is the floating-gravity polynomial fit in Section 3,
Eq. (3-2), of Jaakko Mäkinen (2010) [Makinen2010]_, reproduced in Fig. 2a:

.. math::

   \hat g_{is} = b_s + g(h_{is}), \qquad v_{is} = \hat g_{is} - g_{is}.

Each measurement set has its own freely estimated offset ``b_s``. Polynomial
coefficients are shared across sets. OLS minimizes squared residuals of the
floating level values, not of all pairwise gravity differences. Sets containing
two or three levels enter as two or three level values, respectively; the
paper uses equal weighting without weights derived from the listed tie errors.

For degree :math:`n`, define the height-power vector
:math:`\mathbf h=(1,h,\ldots,h^n)^{\mathsf T}`. The polynomial, gravity
difference, and standard uncertainty use the following notation.

.. math::

   g(h) = \sum_{k=0}^{n} a_k h^k = \mathbf a^{\mathsf T}\mathbf h,
   \qquad a_0=0.

.. math::

   \Delta g_{ij} = g(h_j)-g(h_i) = \mathbf a^{\mathsf T}\mathbf h_{ij},
   \qquad \mathbf h_{ij}=\mathbf h_j-\mathbf h_i.

.. math::

   u(\Delta g_{ij}) = \sqrt{\mathbf h_{ij}^{\mathsf T} C_a \mathbf h_{ij}}.

Here :math:`C_a` denotes coefficient covariance. The fixed constant has zero
variance and covariance, so its row and column are zero in the model.

The input represents each set using a common reference level: supply differences
from ``level_1`` to the other levels. Do not include redundant pairwise
differences as additional observations. For the three-level sets in Table 1,
the two differences from the lowest level define all three floating values;
the difference between the upper levels is already implied.

This is the polynomial-fitting step only. Pier attraction modelling and the
remove–restore procedure discussed elsewhere in the paper are not implemented.
The regression test reproduces the published quadratic coefficients and the
1.200 m to 0.050 m correction, including their one-sigma uncertainties. Correction
uncertainty uses the full coefficient covariance, including off-diagonal terms,
as required by the paper.

.. [Makinen2010] Jaakko Mäkinen (2010). *Gravity gradients above piers in
   absolute gravimetry: use remove-restore for the pier attraction*. Manuscript submitted
   to the Proceedings of the IAG Symposium on Terrestrial Gravimetry
   (TG-SMM2010), Saint Petersburg, Russia, 22–25 June 2010. Sections 3 and 5,
   Table 1, and Fig. 2a.

The ``vgfit`` command fits a polynomial to gravity differences measured at
different heights. Its input table requires ``runn`` (measurement run),
``level_1`` and ``level_2`` (heights in millimetres), and ``delta_g``
(gravity at level_2 minus gravity at level_1, in microgals).

Each run uses a single ``level_1`` reference height. Additional columns are
preserved but do not affect the fit, including ``std``.

CSV units are fixed conventions: input heights are
converted from millimetres to metres once, and gravity remains in microgals.
Units are not inferred from values or table metadata. The input table is not
modified. Polynomial coefficient n is in microgals per metre to the power n;
covariance entries have the product of the corresponding coefficient units.

The fit uses statsmodels OLS with equal weights, following the paper.

An optional ``std`` column describes the uncertainty of each processed tie's
final gravity difference, in microgals. It does not describe correlations
between ties processed together. Run membership and individual uncertainties
alone do not determine those correlations, which may be unavailable for
historical data.

Equal weighting is the default because small differences in reported tie
uncertainties may not justify different weights, particularly across processing
methods or historical datasets. The floating formulation accounts for the
shared-level structure within each run. Additional correlations from prior
processing or systematic effects are
not modelled. Per-tie ``std`` values are retained for provenance and reporting,
but do not supply weights.

The :download:`full Table 1 transcription <../examples/vgfit/makinen2010_table1.csv>`
preserves all nine rows and all ten columns, including the reported mean
gradients and standard errors. The source table gives heights in metres
above the top of the station marker and one-sigma errors from statistical
scatter.

For the fit, select the seven ties from the lowest height in each set.
The 0.788 to 1.311 m differences in sets 4 and 5 (−214.1 and −212.5 µGal)
are already determined by the two differences from 0.049 m within their
respective sets.

.. code-block:: python

   import pandas as pd

   from gmeterpy import units as u
   from gmeterpy.models.vgg.floating import FloatingGravityFit

   table = pd.read_csv("makinen2010_table1.csv")
   lowest = table.groupby("set")["lower_height_m"].transform("min")
   ties = table.loc[table.lower_height_m == lowest]
   h1 = ties.lower_height_m.to_numpy() * u.m
   h2 = ties.upper_height_m.to_numpy() * u.m
   delta_g = ties.gravity_difference_ugal.to_numpy() * u.uGal

   fitter = FloatingGravityFit(degree=2)
   results = fitter.fit(h1, h2, delta_g, set_ids=ties["set"])
   model = results.fitted_model
   gravity_changes, gravity_uncertainty = model.gravity([0.5, 1.0] * u.m)
   difference, uncertainty = model.gravity_difference(1.200 * u.m, 0.050 * u.m)
   gradient, gradient_uncertainty = model.gradient(0.7 * u.m)

The transfer from 1.200 m to 0.050 m gives about 483.2 µGal with a standard
uncertainty of 1.8 µGal.

Model evaluation methods return a ``(value, uncertainty)`` tuple. Both
entries are quantities in the same units when coefficient covariance is
available. Models constructed without covariance return ``None`` for
uncertainty.

Arrays are paired by position. In each set, all ties must start at the same
reference height. Each difference is target gravity minus reference gravity.
The fit reconstructs one zero-valued reference observation per set and one
target observation per supplied tie. Equal target values remain separate
observations. Remove duplicated records before fitting. The requested degree
and run offsets must be identifiable, and the adjustment needs residual
degrees of freedom to estimate uncertainty.

The CLI reads the tie CSV and passes its heights and differences to
``FloatingGravityFit.fit()``. Reuse the fitter to apply the same degree to
several datasets.

The fitted polynomial, regression diagnostics, and ``prediction_table(h1, h2)``
are available on ``PolynomialVerticalGravityFitResults``. The Mäkinen fit
returns ``FloatingGravityFitResults``, which also provides the prepared floating
observations and the ``plot()`` method.

``results.observations`` contains heights in metres, gravity in microgals, set
identifiers in ``set_ids``, and geodetic residuals (fitted minus observed).
``results.parameter_covariance`` contains covariance of all fitted parameters,
including offsets named ``offset[run=1]``. Each offset is the fitted intercept
for that set at height zero. The report uses shorter labels such as
``offset[1]`` and shows the set's minimum measured height in a separate column.
Polynomial parameters are labeled ``a1``, ``a2``, and so on.
``model.covariance`` returns the polynomial coefficient block as a NumPy
array in ascending power order, including the off-diagonal terms.

``gravity()``, ``gradient()``, and ``curvature()`` accept
``height_uncertainty`` for their evaluation heights. For
``gravity_difference()``, ``mean_gravity()``, ``mean_gradient()``, and
``potential_difference()``, supply one standard uncertainty for both
interval endpoints or a pair ordered as ``(h1, h2)``.

``transfer_gravity()`` uses separate ``reference_height_uncertainty`` and
``target_height_uncertainty`` keywords. ``potential_difference()`` also
accepts ``reference_height_uncertainty`` alongside a supplied reference.
Omitted uncertainties are treated as zero. Height errors are assumed
uncorrelated and independent of the fit and reference gravity.

Observations, parameters, standard errors, residuals, and covariance tables
are returned as independent copies. The fitted model is constructed once;
its public coefficients and covariance are also copies. The underlying
``results.regression`` object is shared for read-only inspection and retains
statsmodels' native observed-minus-fitted residual convention.

Diagnostics are inherited from ``gmeterpy.numerics.statistics.FitResults``:
``params``, ``sigma``, ``residuals``, ``whitened_residuals``,
``parameter_covariance``, ``number_of_observations``, ``number_of_unknowns``,
``degrees_of_freedom`` and ``s0``. Parameter counts include constants, and
residual degrees of freedom are determined by the backend model rank.

Use ``results.influence_table()`` to see leverage, internally studentized
residuals, and Cook's distance alongside the floating observations.
Studentized residuals retain the fitted-minus-observed sign.
The report marks each observation that meets a screening criterion with
an asterisk in the ``Inspect`` column. The criteria use
``|studentized residual| > 2``, leverage greater than ``2p/n``, and
Cook's distance greater than ``4/n``, where ``p`` includes set offsets and
``n`` is the number of fitted observations. Unit leverage is flagged
separately because the studentized residual and Cook's distance are undefined
there. These rules identify observations for inspection.

To check agreement among measurement sets, predict the original ties in
each set using a fit to the remaining sets.

.. code-block:: python

   _, held_out = fitter.leave_one_set_out(h1, h2, delta_g, set_ids=ties["set"])

``held_out`` gives predicted-minus-observed differences for ties excluded
from each refit in microgals. To assess a specific transfer, supply
``transfer_h1`` and ``transfer_h2``. The first returned table then gives
each refit's shift from the full-fit transfer, flags extrapolation, and
records any refit failures in ``fit_error``.

Models can also be created directly from published coefficients:

.. code-block:: python

   from gmeterpy.models import PolynomialVerticalGravity

   model = PolynomialVerticalGravity(
       [0, -300, 12], covariance=[[0, 0, 0], [0, 4, 1], [0, 1, 9]]
   )

The default basis uses metres for height and microgals for gravity. Pass
``units=(gravity_unit, height_unit)`` when the supplied coefficients and
covariance use a different basis. The constructor converts them to metres
and microgals.

Supply coefficients in ascending order ``[a0, a1, ...]``, including the
constant, and any corresponding covariance. A fit to gravity differences
fixes the constant at zero and sets its covariance row and column to zero.
Coefficient n is in microgals per metre**n; covariance uses the same order.
Each evaluation returns a standard uncertainty when covariance is available.

``gravity(heights)`` evaluates the model in microgals. Use ``gravity_difference(h1, h2)`` for
:math:`g(h_2)-g(h_1)`. Use ``transfer_gravity()`` to carry a supplied gravity
value from a reference height to a target height, or ``mean_gravity()`` to
average the model over a height interval. ``mean_gradient()``
returns the gravity difference divided by the signed height separation.
Scalar and multidimensional array inputs are supported.
Measurement-set offsets remain in the fit
results and do not enter model evaluation.

To transfer a measured gravity value to several heights, pass the
measurement directly to ``transfer_gravity()``:

.. code-block:: python

   gravity, uncertainty = model.transfer_gravity(
       target_height=[0.71, 1.2] * u.m,
       reference_gravity=9.81 * u.m / u.s**2,
       reference_height=0.71 * u.m,
       reference_gravity_uncertainty=2 * u.uGal,
   )

``gravity()`` and ``mean_gravity(h1, h2)`` evaluate the original
polynomial. ``potential_difference()`` integrates that polynomial by
default. Supply a reference gravity and height to shift the gravity level
for that calculation:

.. code-block:: python

   delta_w, uncertainty = model.potential_difference(
       0 * u.m, 1 * u.m,
       reference_gravity=9.81 * u.m / u.s**2,
       reference_height=0.71 * u.m,
   )

A relative polynomial gives a potential difference in its relative gravity
datum. A reference on an absolute gravity datum supplies the offset needed
to integrate absolute gravity. A constant gravity shift :math:`c` changes
the potential difference by :math:`-c(h_2-h_1)`.

To evaluate gravity relative to a chosen height, use
``gravity_difference(h0, heights)``.

To run the command, write the selected ties with the CLI column names
and convert the heights from metres to millimetres.

.. code-block:: python

   from pathlib import Path

   data = pd.DataFrame(
       {
           "runn": ties["set"],
           "level_1": h1.to_value(u.mm),
           "level_2": h2.to_value(u.mm),
           "delta_g": ties.gravity_difference_ugal,
           "std": ties.gravity_standard_error_ugal,
       }
   )
   output = Path("output")
   output.mkdir(exist_ok=True)
   data.to_csv(output / "makinen2010_ties.csv", index=False)

.. code-block:: console

   vgfit -i output/makinen2010_ties.csv -n Makinen2010 -d 2 -o output

This writes ``output/Makinen2010_all.png`` and
``output/vg_Makinen2010_fit_all.txt``.

The command supports these presentation options.

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning and default
   * - ``--h-ref METRES``
     - Uncertainty reference height; 0.710 m
   * - ``--subtract-gradient GRADIENT``
     - Gradient to subtract in microgals/m; automatic secant by default
   * - ``--highlight-set ID``
     - Highlight one measurement set by its ``runn`` value and mute the others
   * - ``--leave-one-set-out``
     - Report held-out tie errors by measurement set
   * - ``--review-note TEXT``
     - Optional reviewer comments when the report flags an issue
   * - ``--xlim MIN MAX``
     - Detrended gravity limits in microgals. Defaults to -10 to 10.
   * - ``--ylim MIN MAX``
     - Height limits in metres; 0 to 1.4
   * - ``--auto-limits``
     - Use automatic limits for both axes, overriding ``--xlim`` and ``--ylim``
   * - ``--height-range MIN MAX``
     - Curve sampling range in metres; 0.0001 to 1.4
   * - ``-o DIR``, ``--output-dir DIR``
     - Existing output directory; current directory by default

To use automatic limits and a different uncertainty reference height, run
the command with these options.

.. code-block:: console

   vgfit -i output/makinen2010_ties.csv -n Makinen2010 -o output --auto-limits --h-ref 0.8

Use ``-i -`` to read CSV from standard input. The plot defaults to subtracting
the secant gradient between the lowest observed height and 1 m. Supply
``--subtract-gradient`` to use a different gradient.
If the lowest observation is exactly 1 m, automatic detrending uses the local
polynomial derivative there instead of a secant with coincident endpoints.

To generate individual outputs from Python, use the result's plot method
and the command's report writer.

.. code-block:: python

   from gmeterpy.cli.vgfit.report import write_report

   figure = results.plot(station="Makinen2010")
   write_report(results, "report.txt", station="Makinen2010", data=data)

Callers own figures returned by ``plot``; the command closes its own figure.
The report starts with the fit summary and its generation timestamp in UTC.
When diagnostics flag observations or held-out predictions are missing,
the **Inspection** section follows the fit summary and identifies what to
check. Its review note shows ``Pending`` until you supply
``--review-note TEXT`` or pass ``review_note="..."`` to ``write_report()``.
The transfer model gives the polynomial, gravity difference, and uncertainty
formulas, base units, and the full-precision coefficient vector and covariance
matrix. The full fit then lists
the polynomial coefficients and set offsets with their units, standard
uncertainties, and full regression covariance. Its entries use products of
parameter units.
The report labels input tie heights in millimetres and prepared observation
heights in metres. Pass the original CLI-format tie table as ``data`` to
include it, or omit ``data`` to show the prepared observations.
The prediction table flags pairs outside the observed height range and shows
nonzero standard uncertainties with two significant digits.

Pass the second table from ``fitter.leave_one_set_out()`` to
``write_report()`` as ``held_out_ties=held_out``. The leave-one-set-out section shows
the number of held-out ties and their mean and RMS prediction errors by set.

Library plots use the same detrending rule and automatic axis limits.
You can set the gradient, limits, and other presentation options by keyword.

.. code-block:: python

   figure = results.plot(
       station="Makinen2010",
       h_ref=0.710,
       subtract_gradient=-420,  # microgals per metre
       xlim=(-10, 10),         # microgals
       ylim=(0, 1.4),          # metres
       height_range=(0, 1.4),  # curve sampling range in metres
   )

``subtract_gradient`` shifts the displayed curve and observations.
``height_range`` defaults to the observed height range and controls where
the curve is drawn independently of the axis limits. Dashed curve segments
mark extrapolation beyond the observed heights. The shaded band shows one
standard uncertainty in the gravity difference from ``h_ref`` to each
plotted height. A dotted line marks ``h_ref`` when it lies within the curve
range.

Use ``results.plot(highlight_set=1)`` to inspect set 1. The same option is
available on the command line as ``--highlight-set 1``.

The command leaves other open figures intact and closes its own figure even if
saving fails. Report formatting does not change pandas display settings.

The CLI prepares ties and saves outputs through the results methods.
The model owns fitting and scientific evaluation; results own diagnostics,
plotting and reporting.

Scientific models are provided by ``gmeterpy.models`` (formerly
``gmeterpy.corrections``). This includes vertical gravity gradients, atmospheric
pressure, polar motion and tides. Instrument-family calculations remain under
``gmeterpy.meters``, such as free-fall finite-speed-of-light corrections.
