# -*- coding: utf-8 -*-
# flake8: noqa
"""Units support for gMeterPy.

The `gMeterPy` is supporting units via `astropy.units` subpackage.
The Gal unit (1 Gal = 1 cm / s**2) is already there.

In addition, the Eotvos unit (1 E = 10**-9 s**-2) for the gravity gradient is
defined.

"""

from astropy.units import *
from astropy.units import (
    Quantity as Quantity,
    Unit as Unit,
    UnitBase as UnitBase,
    UnitsError as UnitsError,
    cds as cds,
    def_unit as def_unit,
    dimensionless_angles as dimensionless_angles,
    quantity_input as quantity_input,
)

# Astropy creates unit symbols dynamically. Declare the common ones for editors.
m = Unit("m")
mm = Unit("mm")
cm = Unit("cm")
km = Unit("km")
s = Unit("s")
Gal = Unit("Gal")
mGal = Unit("mGal")
uGal = Unit("uGal")
Pa = Unit("Pa")
mbar = Unit("mbar")
deg_C = Unit("deg_C")
rad = Unit("rad")
deg = Unit("deg")
arcsec = Unit("arcsec")

E = def_unit(
    ["E", "Eotvos"],
    represents=1e-9 * s**-2,
    doc="Eötvös-unit for the gravity gradient.",
    prefixes=True,
    namespace=globals(),
)
Eotvos = E
