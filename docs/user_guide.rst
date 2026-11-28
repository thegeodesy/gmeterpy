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

The tie input uses one reference level per set. Supply the difference
from that reference to each target level. When all differences come from
the same adjusted level values, the two reference ties in a three-level
set already determine the third difference.

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
different heights. It uses the same field names as the Python fitter.
``set_id`` identifies the measurement set, ``h1`` and ``h2`` give the
reference and target heights, and ``delta_g`` is gravity at ``h2`` minus
gravity at ``h1``. Legacy CSV headers ``runn``, ``level_1``, and ``level_2``
are accepted for fields where the Python name is absent. Use ``--columns``
to specify other headers or override these defaults.

Choose ``set_id`` according to how the ties were reduced. Ties derived from
one set of adjusted gravity levels share an ID. For example, if levels
A, B, and C were adjusted together, give the A-to-B and A-to-C ties the
same ID and put A in ``h1`` for both. Those two ties already determine
the B-to-C difference. Give separately reduced ties different IDs, even
when they were collected during the same field visit.

The fit estimates a separate offset for each ID and assumes no correlation
between sets. Additional columns, including ``std``, are preserved for
reporting.

The CSV defaults to millimetres for both heights and microgals for gravity
differences. Set ``--height-unit`` and ``--gravity-unit`` to match the units
of your source data. The fitter converts to metres and microgals, and the
report prints the source table in its supplied units. Polynomial coefficient
n is in microgals per metre to the power n. Covariance entries use the
product of the corresponding coefficient units.

The fit uses statsmodels OLS with equal weights, following the paper.

An optional ``std`` column gives the standard uncertainty of each processed
tie's final gravity difference, in the source gravity unit. The set IDs describe which
ties share adjusted levels. Estimating a covariance matrix for the ties
would require more information about the original adjustment.

Equal weighting is the default because small differences in reported tie
uncertainties may not justify different weights, particularly across processing
methods or historical datasets. The floating formulation accounts for the
shared-level structure within each set. Additional correlations from prior
processing or systematic effects are
not modelled. Per-tie ``std`` values are retained for provenance and reporting,
but do not supply weights.

.. rubric:: Preparing the Aboa ties

The :download:`full Table 1 transcription <../examples/vgfit/makinen2010_table1.csv>`
preserves all nine rows and all ten columns, including the reported mean
gradients and standard errors. The source table gives heights in metres
above the top of the station marker and one-sigma errors from statistical
scatter.

Mäkinen fitted the adjusted gravity values at each level after estimating
instrument drift. The transcription supplies differences between those
levels. To reproduce the polynomial fit from the published differences:

1. Keep all five measurement sets. Use the ``set`` column as the set ID
   and select the ties starting at the lowest height in each set.
2. For sets 4 and 5, use the two ties from 0.049 m to 0.788 m and 1.311 m.
   All three levels in each of these sets were adjusted together. The
   0.788 to 1.311 m difference follows from the two ties sharing the
   reference height:

   .. math::

      \Delta g(0.788, 1.311)
      = \Delta g(0.049, 1.311) - \Delta g(0.049, 0.788).

   This gives −214.1 µGal for set 4 and −212.5 µGal for set 5. Those
   two rows remain in the source table.
3. Supply the resulting seven ties to the fitter. It sets the floating
   gravity at each reference level to zero and adds the tie differences
   to reconstruct the target values. The fit uses 12 floating level values
   and estimates one offset per set. The choice of zero does not affect the
   polynomial coefficients or predicted gravity differences.

This selection relies on how the Aboa ties were obtained. A separately
measured and reduced tie can carry additional information, even when it
shares the same endpoints as a difference derived from other ties.

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
   results = fitter.fit(h1, h2, delta_g, set_id=ties["set"])
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

After inspecting a fit, pass ``exclude_sets`` to refit using the same input
arrays. The fitter omits every tie in each selected set and records the IDs
in ``results.excluded_set_ids``. Pass the same IDs to
``leave_one_set_out()`` to run that diagnostic on the retained data.

.. code-block:: python

   reviewed = fitter.fit(h1, h2, delta_g, set_id=ties["set"], exclude_sets=[4])

The fitted polynomial, regression diagnostics, and ``prediction_table(h1, h2)``
are available on ``PolynomialVerticalGravityFitResults``. The Mäkinen fit
returns ``FloatingGravityFitResults``, which also provides the prepared floating
observations and the ``plot()`` method.

``results.fitted_height_range`` gives the minimum and maximum heights used
in the floating gravity fit, in metres. Its ``prediction_table(h1, h2)``
flags extrapolation when either endpoint falls outside that range.

``results.observations`` contains heights in metres, gravity in microgals, set
identifiers in ``set_id``, and geodetic residuals (fitted minus observed).
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

Use ``display(results.summary())`` in a notebook to see the statsmodels
regression summary. Parameter confidence intervals default to 95%.
Pass ``alpha=0.10`` for 90% intervals.

Use ``results.influence_table()`` to see leverage, internally studentized
residuals, and Cook's distance alongside the floating observations.
Studentized residuals retain the fitted-minus-observed sign.
The ``inspect`` column marks observations meeting the influence criteria.
The criteria use
``|studentized residual| > 2``, leverage greater than ``2p/n``, and
Cook's distance greater than ``4/n``, where ``p`` includes set offsets and
``n`` is the number of fitted observations. Unit leverage is flagged
separately because the studentized residual and Cook's distance are undefined
there. These rules identify observations for inspection.

To check agreement among entire measurement sets, predict the original
ties in each set using a fit to the remaining sets.

.. code-block:: python

   _, held_out = fitter.leave_one_set_out(h1, h2, delta_g, set_id=ties["set"])

``held_out`` gives predicted-minus-observed differences for ties excluded
from each refit in microgals, the fitted height range after omission, and
raw and Holm-adjusted p-values in ``set_p_value`` and ``set_p_value_adjusted``.
Each tie in the same set carries
the same test result. To assess a specific transfer, supply
``transfer_h1`` and ``transfer_h2``. The first returned table then gives
each refit's transfer, standard uncertainty, and shift from the full fit.
It marks extrapolation and records fit failures in ``fit_error``.

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

To run the command, write the selected ties with the source headers and
values. Pass a column mapping and the source height unit to ``vgfit``.

.. code-block:: python

   from pathlib import Path

   output = Path("output")
   output.mkdir(exist_ok=True)
   ties.to_csv(output / "makinen2010_ties.csv", index=False)

.. code-block:: console

   vgfit output/makinen2010_ties.csv -n Makinen2010 -d 2 -o output \
     --columns set_id=set h1=lower_height_m h2=upper_height_m delta_g=gravity_difference_ugal \
     --height-unit m --gravity-unit uGal

This writes ``output/makinen2010_ties_vgg_fit.png`` and
``output/makinen2010_ties_vgg_fit.txt``. The report's data section prints the
seven rows supplied in ``makinen2010_ties.csv``. The nine-row Table 1
transcription remains available as the source.

.. rubric:: Input and output

* ``FILE`` gives the tie CSV. Existing scripts can supply it via ``-i FILE``
  or ``--input FILE``. Use one form and pass ``-`` for standard input.
* ``--columns FIELD=COLUMN ...`` maps ``set_id``, ``h1``, ``h2``, and
  ``delta_g`` to source headers. Unspecified fields use the default lookup.
* ``--height-unit UNIT`` sets the input unit for both heights. Defaults to
  ``mm``.
* ``--gravity-unit UNIT`` sets the input unit for gravity differences.
  Defaults to ``uGal``. For example, use ``mGal`` for milligals.
* ``-n STATION`` or ``--name STATION`` labels the station in the plot and
  report. For standard input it also prefixes the output filenames.
* ``-o DIR`` or ``--output-dir DIR`` writes to an existing directory.
  Defaults to the current directory.
* ``--overwrite`` allows the command to replace existing outputs. By default,
  the command checks the requested output paths and stops if any already exist.

Output names are based on the input filename. For example,
``vgfit aboa_2004.csv -n Aboa`` writes ``aboa_2004_vgg_fit.png`` and
``aboa_2004_vgg_fit.txt``. With unnamed standard input, the filenames
are ``vgg_fit.png`` and ``vgg_fit.txt``.

The command prints the paths to the files it saves. Invalid input
values are identified by source column and data row, counting from 1
after the CSV header. If a set has multiple reference heights, the error
identifies the set and lists those heights in metres.

.. rubric:: Fitting

* ``-d DEGREE`` or ``--degree DEGREE`` sets the polynomial degree. Defaults to 2.
* ``--exclude-set ID`` omits all ties with the given set ID.
  Repeat for multiple sets.

.. rubric:: Diagnostics and reporting

* ``--leave-one-set-out`` refits without each retained set in turn and reports
  its worst held-out tie error and the number of held-out ties over tolerance.
* ``--tie-tolerance UGAL`` runs leave-one-set-out and overrides its 3 microgal
  tolerance for individual held-out errors.
* ``--loso-transfer H1 H2`` runs the same refits and adds gravity transfer
  shifts for the specified pair of heights in metres.
* ``--exclude-reason ID TEXT`` records a reason alongside a manual exclusion.
  Supply ``--exclude-set ID`` for the same set. Repeat for multiple sets.
* ``--review-note TEXT`` records reviewer comments in the Review section.

.. rubric:: Plotting

* ``--no-plot`` writes only the report and leaves any existing plot intact.
* ``--highlight-set ID`` highlights one measurement set in the plot and mutes
  the others.
* ``--h-ref METRES`` sets the reference height for the plot's uncertainty
  band. Defaults to 0.710 m.
* ``--subtract-gradient GRADIENT`` sets the gradient subtracted in the plot
  in microgals/m. Defaults to the automatic secant.
* ``--auto-limits`` uses automatic plot axis limits and draws the curve and band
  over the fitted height range. Overrides ``--xlim`` and ``--ylim``.
* ``--xlim MIN MAX`` sets plot limits for detrended gravity in microgals. Defaults to
  -10 to 10.
* ``--ylim MIN MAX`` sets height limits for the plot axis, fitted curve, and
  uncertainty band in metres. Defaults to 0 to 1.4.

To use automatic limits and a different uncertainty reference height, run
the command with these options.

.. code-block:: console

   vgfit output/makinen2010_ties.csv -n Makinen2010 -o output \
     --columns set_id=set h1=lower_height_m h2=upper_height_m delta_g=gravity_difference_ugal \
     --height-unit m --auto-limits --h-ref 0.8

After reviewing the initial report, rerun with the IDs you decide to exclude.
Use a separate output directory to keep both reports for comparison.

.. code-block:: console

   mkdir -p reviewed
   vgfit output/makinen2010_ties.csv -n Makinen2010 -o reviewed \
     --columns set_id=set h1=lower_height_m h2=upper_height_m delta_g=gravity_difference_ugal \
     --height-unit m --exclude-set 4 --exclude-reason 4 "Reference height could not be verified"

The command leaves the input CSV intact. The new report lists the excluded
sets and retains the original input ties in its data section.
``--leave-one-set-out`` uses only the retained sets when exclusions are supplied.

Use ``vgfit -`` to read CSV from standard input. The plot defaults to subtracting
the secant gradient between the lowest observed height and 1 m. Supply
``--subtract-gradient`` to use a different gradient.
If the lowest observation is exactly 1 m, automatic detrending uses the local
polynomial derivative there instead of a secant with coincident endpoints.

To generate individual outputs from Python, use the result's plot method
and the command's report writer.

.. code-block:: python

   from gmeterpy.cli.vgfit.report import write_report

   figure = results.plot(station="Makinen2010")
   write_report(
       results, "report.txt", station="Makinen2010", data=ties,
       columns={
           "set_id": "set", "h1": "lower_height_m", "h2": "upper_height_m",
           "delta_g": "gravity_difference_ugal",
       },
       height_unit=u.m,
   )

Callers own figures returned by ``plot``; the command closes its own figure.
The fit summary includes the station name, generation timestamp in UTC,
and the count and IDs of explicitly excluded sets.
When there are review details, the **Review** section follows the fit summary.
**Manual exclusions** lists the sets and their reasons. Review also lists ties
or missing predictions to check after the final fit.
The Review section includes a review note when supplied.
From Python, pass ``review_note="..."`` and ``exclusion_reasons={set_id: "reason"}``
to ``write_report()``.
The transfer model starts with the units and the formulas for the gravity
difference and its standard uncertainty. The full-precision coefficient
vector and covariance matrix follow, with their shared coefficient order.
The full fit lists the polynomial coefficients, set offsets, and their
standard uncertainties, followed by a separate full regression covariance
table. Its rows and columns follow the order of estimated parameters and
its entries use products of parameter units.
The data section shows the input column mapping and units and prints the
original tie table in source row order, including any excluded ties. Pass
the source table as ``data`` when writing a report from Python.
The prediction table flags pairs outside the fitted height range and shows
nonzero standard uncertainties with two significant digits.

With leave-one-set-out enabled, **Review** includes a **Leave-one-set-out
check** that counts all checked sets. Each flagged set has its own entry
with the original input rows, height pairs, and signed errors for its
flagged ties. The entry shows any change to the lowest or highest fitted
height when the set is omitted.
The **Inspection threshold** in **Fit summary** applies to absolute errors
before rounding. Review also identifies sets with unavailable predictions.
Each diagnostic refit omits every tie in the selected measurement set.
With a transfer pair, each flagged set entry gives its transfer shift and
change in standard uncertainty. The check also lists the shifts for the
other omissions and marks extrapolation.

The Review section uses held-out errors when requested and influence rules
otherwise. Both checks identify observations for inspection. Use
``--exclude-set ID`` to omit a set from the fitted model. Individual
floating-observation diagnostics are available through
``results.influence_table()`` in Python.
From Python, pass the second table from ``fitter.leave_one_set_out()`` as
``held_out_ties`` and the first as ``held_out_transfers`` if it contains one
transfer pair. Set ``tie_tolerance`` to override the default in microgals.

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
``height_range`` defaults to the fitted height range and controls where
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
