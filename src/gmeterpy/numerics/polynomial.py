"""NumPy polynomial operations with coefficient covariance."""

import numpy as np
from numpy.polynomial import polynomial
from numpy.polynomial.polynomial import polyvander as polyvander
from numpy.typing import ArrayLike, NDArray


def _coefficient_inputs(c: ArrayLike, covariance: ArrayLike | None):
    """Normalize one real polynomial and its optional covariance."""
    coefficients = np.asarray(c, dtype=float)
    if coefficients.ndim != 1 or coefficients.size == 0:
        raise ValueError("Expected a nonempty one-dimensional coefficient array")
    if covariance is None:
        return coefficients, None
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (coefficients.size, coefficients.size):
        raise ValueError("Covariance must have one row and column per coefficient")
    return coefficients, matrix


def polyval(x: ArrayLike, c: ArrayLike, *, covariance: ArrayLike | None = None):
    """Evaluate a polynomial in ascending power order and return (value, standard uncertainty)."""
    c, matrix = _coefficient_inputs(c, covariance)
    x = np.asarray(x, dtype=float)
    if matrix is None:
        return polynomial.polyval(x, c), None
    basis = evaluation_basis(x, c.size - 1)
    return evaluate_basis(c, basis, matrix)


def polyder(
    c: ArrayLike, m: int = 1, scl: float = 1, *, covariance: ArrayLike | None = None
):
    """Differentiate a polynomial and return (coefficients, covariance)."""
    c, matrix = _coefficient_inputs(c, covariance)
    derivative = polynomial.polyder(c, m=m, scl=scl)
    if matrix is not None:
        transform = polynomial.polyder(np.eye(c.size), m=m, scl=scl, axis=0)
        matrix = transform @ matrix @ transform.T
    return derivative, matrix


def polyint(
    c: ArrayLike,
    m: int = 1,
    k: ArrayLike = (),
    lbnd: float = 0,
    scl: float = 1,
    *,
    covariance: ArrayLike | None = None,
):
    """Integrate a polynomial and return (coefficients, covariance); k and lbnd are exact."""
    c, matrix = _coefficient_inputs(c, covariance)
    integral = polynomial.polyint(
        c, m=m, k=np.asarray(k, dtype=float).tolist(), lbnd=lbnd, scl=scl
    )
    if matrix is not None:
        transform = polynomial.polyint(np.eye(c.size), m=m, lbnd=lbnd, scl=scl, axis=0)
        matrix = transform @ matrix @ transform.T
    return integral, matrix


def evaluation_basis(x: ArrayLike, degree: int):
    """Evaluate the monomial basis with the coefficient axis first."""
    return polynomial.polyval(np.asarray(x, dtype=float), np.eye(degree + 1))


def integral_basis(x1: ArrayLike, x2: ArrayLike, degree: int):
    """Integrate the monomial basis over the supplied endpoints."""
    x1, x2 = np.broadcast_arrays(
        np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    )
    antiderivative = polynomial.polyint(np.eye(degree + 1), axis=0)
    return polynomial.polyval(x2, antiderivative) - polynomial.polyval(
        x1, antiderivative
    )


def interval_mean_uncertainty(
    mean: ArrayLike,
    endpoints: tuple[ArrayLike, ArrayLike],
    separation: ArrayLike,
    endpoint_sigmas: tuple[ArrayLike, ArrayLike],
):
    """Return the contribution from independent endpoint errors to a mean."""
    start, end = endpoints
    u1, u2 = endpoint_sigmas
    mean = np.asarray(mean, dtype=float)
    return np.hypot(
        (mean - np.asarray(start)) * np.asarray(u1),
        (np.asarray(end) - mean) * np.asarray(u2),
    ) / np.abs(np.asarray(separation))


def evaluate_basis(
    coefficients: NDArray[np.float64],
    basis: NDArray[np.floating],
    covariance: NDArray[np.float64] | None = None,
):
    """Apply a coefficient sensitivity vector on axis zero and return (value, standard uncertainty)."""
    value = np.einsum("i,i...->...", coefficients, basis)
    uncertainty = None
    if covariance is not None:
        variance = np.einsum("i...,ij,j...->...", basis, covariance, basis)
        uncertainty = np.sqrt(variance)
    return value, uncertainty


class Polynomial:
    """A real power-series polynomial with optional coefficient covariance."""

    def __init__(self, coef: ArrayLike, *, covariance: ArrayLike | None = None):
        coefficients, matrix = _coefficient_inputs(coef, covariance)
        self._coef = coefficients.copy()
        self._covariance = None if matrix is None else matrix.copy()

    @property
    def coef(self) -> NDArray[np.float64]:
        """Copy of the coefficients, in ascending power order."""
        return self._coef.copy()

    @property
    def covariance(self) -> NDArray[np.float64] | None:
        """Copy of the coefficient covariance, if supplied."""
        return None if self._covariance is None else self._covariance.copy()

    def degree(self) -> int:
        """Return the highest stored power."""
        return self._coef.size - 1

    def __call__(self, x: ArrayLike):
        """Return the value and standard uncertainty at x."""
        return polyval(x, self._coef, covariance=self._covariance)

    def value(self, x: ArrayLike, *, derivative: int = 0):
        """Evaluate the nominal polynomial or its selected derivative."""
        coef = (
            self._coef
            if derivative == 0
            else polynomial.polyder(self._coef, m=derivative)
        )
        return polynomial.polyval(np.asarray(x, dtype=float), coef)

    def evaluate_basis(self, basis: NDArray[np.floating]):
        """Return a linear combination of the coefficients and its uncertainty."""
        return evaluate_basis(self._coef, basis, self._covariance)

    def difference(self, x1: ArrayLike, x2: ArrayLike):
        """Return p(x2) - p(x1) and its standard uncertainty."""
        x1, x2 = np.broadcast_arrays(
            np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
        )
        degree = self.degree()
        basis = evaluation_basis(x2, degree) - evaluation_basis(x1, degree)
        return self.evaluate_basis(basis)

    def interval_mean(
        self,
        x1: ArrayLike,
        x2: ArrayLike,
        *,
        x_uncertainty: ArrayLike | tuple[ArrayLike, ArrayLike] | None = None,
    ):
        """Return the interval mean and standard uncertainty.

        Supply one coordinate uncertainty for both endpoints or a tuple
        ordered as (x1, x2). The endpoint errors are independent of each
        other and of the coefficients. Uncertainty is None without covariance.
        """
        x1, x2 = np.broadcast_arrays(
            np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
        )
        sigmas = None
        if x_uncertainty is not None:
            u1, u2 = (
                x_uncertainty
                if isinstance(x_uncertainty, tuple)
                else (x_uncertainty, x_uncertainty)
            )
            x1, x2, u1, u2 = np.broadcast_arrays(
                x1, x2, np.asarray(u1, dtype=float), np.asarray(u2, dtype=float)
            )
            if np.any(~np.isfinite(u1) | (u1 < 0)) or np.any(
                ~np.isfinite(u2) | (u2 < 0)
            ):
                raise ValueError(
                    "Endpoint uncertainties must be finite and nonnegative"
                )
            sigmas = u1, u2
        separation = x2 - x1
        if np.any(separation == 0):
            raise ValueError(
                "An interval average requires distinct heights or coordinates"
            )
        mean, uncertainty = self.evaluate_basis(
            integral_basis(x1, x2, self.degree()) / separation
        )
        if sigmas is not None and uncertainty is not None:
            height_sigma = interval_mean_uncertainty(
                mean, (self.value(x1), self.value(x2)), separation, sigmas
            )
            uncertainty = np.hypot(uncertainty, height_sigma)
        return mean, uncertainty

    def deriv(self, m: int = 1) -> "Polynomial":
        """Return the derivative of order m."""
        coef, covariance = polyder(self._coef, m=m, covariance=self._covariance)
        return Polynomial(coef, covariance=covariance)

    def integ(self, m: int = 1, k: ArrayLike = (), lbnd: float = 0) -> "Polynomial":
        """Return the integral of order m with exact constants at lbnd."""
        coef, covariance = polyint(
            self._coef, m=m, k=k, lbnd=lbnd, covariance=self._covariance
        )
        return Polynomial(coef, covariance=covariance)
