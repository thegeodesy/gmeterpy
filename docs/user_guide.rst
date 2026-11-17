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
