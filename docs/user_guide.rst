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

The model fits measured gravity differences between reference and target heights.

.. code-block:: python

   from gmeterpy import units as u
   from gmeterpy.models.vgg.floating import FloatingGravityFit

   fitter = FloatingGravityFit(degree=2)
   results = fitter.fit(
       h1=[0, 0, 0] * u.m,
       h2=[0.5, 1.0, 1.5] * u.m,
       delta_g=[-147, -288, -423] * u.uGal,
       set_ids=[1, 1, 1],
   )
   model = results.fitted_model
   gravity_changes, gravity_uncertainty = model.gravity([0.5, 1.0] * u.m)
   difference, uncertainty = model.gravity_difference(0.5 * u.m, 1.0 * u.m)
   gradient, gradient_uncertainty = model.gradient(0.7 * u.m)

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
observations and the ``plot()`` and ``report()`` methods.

``results.observations`` contains heights in metres, gravity in microgals, set
identifiers in ``set_ids``, and geodetic residuals (fitted minus observed).
``results.parameter_covariance`` contains covariance of all fitted parameters,
including offsets named ``offset[run=1]``. Each offset is the fitted intercept
for that set at height zero. The report also shows the set's minimum measured
height alongside its offset label.
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

The command connects input preparation, fitting and output methods:

.. code-block:: console

   vgfit -i measurements.csv -n Sochi -d 2 -o output

The destination directory must exist. This writes ``Sochi_all.png`` and
``vg_Sochi_fit_all.txt``. There is no workflow package or ``run`` wrapper.

Command-line presentation options preserve the existing appearance unless
overridden:

.. list-table::
   :header-rows: 1

   * - Option
     - Meaning and default
   * - ``--h-ref METRES``
     - Uncertainty reference height; 0.710 m
   * - ``--subtract-gradient GRADIENT``
     - Gradient to subtract in microgals/m; automatic secant by default
   * - ``--no-detrend``
     - Show the original polynomial; excludes ``--subtract-gradient``
   * - ``--xlim MIN MAX``
     - Gravity limits in microgals; -10 to 10
   * - ``--ylim MIN MAX``
     - Height limits in metres; 0 to 1.4
   * - ``--auto-limits``
     - Use automatic limits for both axes, overriding ``--xlim`` and ``--ylim``
   * - ``--height-range MIN MAX``
     - Curve sampling range in metres; 0.0001 to 1.4
   * - ``-o DIR``, ``--output-dir DIR``
     - Existing output directory; current directory by default

For example, show the original model with automatic limits and a different
uncertainty reference height:

.. code-block:: console

   vgfit -i measurements.csv -n Sochi --no-detrend --auto-limits --h-ref 0.8

Use ``-i -`` to read CSV from standard input. The command automatically
subtracts a secant gradient unless an explicit gradient or ``--no-detrend``
is selected.
If the lowest observation is exactly 1 m, automatic detrending uses the local
polynomial derivative there instead of a secant with coincident endpoints.

For individual outputs, use the result methods:

.. code-block:: python

   figure = results.plot(station="Sochi")
   results.report("report.txt", station="Sochi", data=data)

Callers own figures returned by ``plot``; the command closes its own figure.
The optional report ``data`` argument preserves the original tie table; if
omitted, the report includes prepared observations instead.

Library plots show the original polynomial without detrending and use automatic
axis limits. Presentation options are keyword-only:

.. code-block:: python

   figure = results.plot(
       station="Sochi",
       h_ref=0.710,
       subtract_gradient=-300,  # microgals per metre
       xlim=(-10, 10),         # microgals
       ylim=(0, 1.4),          # metres
       height_range=(0, 1.4),  # curve sampling range in metres
   )

``subtract_gradient`` shifts the displayed curve and observations without
changing the model or its uncertainty. ``height_range`` defaults to the observed
height range and is independent of the axis limits. the command
retains their established secant detrending, reference height, curve range, and
fixed axis limits.

The command leaves other open figures intact and closes its own figure even if
saving fails. Report formatting does not change pandas display settings.

The CLI prepares ties and saves outputs through the results methods.
The model owns fitting and scientific evaluation; results own diagnostics,
plotting and reporting.

Scientific models are provided by ``gmeterpy.models`` (formerly
``gmeterpy.corrections``). This includes vertical gravity gradients, atmospheric
pressure, polar motion and tides. Instrument-family calculations remain under
``gmeterpy.meters``, such as free-fall finite-speed-of-light corrections.
