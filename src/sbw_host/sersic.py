"""Sersic profile utilities.

Rubin's multiband Sersic fit (``sersic_*`` columns in the Object table) describes
each object as a single elliptical Sersic profile.  The ellipse is stored as a
covariance-like parameterisation ``(sersic_reff_x, sersic_reff_y, sersic_rho)``
in tract pixels (0.2 arcsec / pixel); the eigen-decomposition of

    Sigma = [[reff_x**2,          rho*reff_x*reff_y],
             [rho*reff_x*reff_y,  reff_y**2        ]]

gives ``sersic_reff_major`` / ``sersic_reff_minor`` (checked against DP2:
agreement at the 1e-5 level).  The pixel frame follows the LSST tract WCS, which
is a north-up TAN projection with ``flipX=True``: +x points West (decreasing RA)
and +y points North.  All functions here work in a local tangent plane with
offsets ``du`` (arcsec, West positive) and ``dv`` (arcsec, North positive).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaincc, gammaincinv, gammaln

PIXEL_SCALE_ARCSEC = 0.2  # LSST coadd / tract pixel scale

# Guard rails for degenerate fits.
MIN_REFF_ARCSEC = 1e-3
MAX_ABS_RHO = 0.999


def b_n(n):
    """Sersic b_n such that half the total flux is enclosed within R_eff (exact)."""
    n = np.asarray(n, dtype=float)
    return gammaincinv(2.0 * n, 0.5)


def sky_offsets(ra, dec, ra0, dec0):
    """Tangent-plane offsets (arcsec) of (ra, dec) from centre (ra0, dec0).

    Returns ``du`` (West positive) and ``dv`` (North positive) so that the frame
    matches the LSST tract pixel axes.  Small-angle approximation; fine for the
    arcminute scales relevant here.
    """
    ra, dec, ra0, dec0 = (np.asarray(v, dtype=float) for v in (ra, dec, ra0, dec0))
    dra = (ra - ra0 + 180.0) % 360.0 - 180.0
    du = -dra * np.cos(np.deg2rad(dec0)) * 3600.0
    dv = (dec - dec0) * 3600.0
    return du, dv


def ellipse_axes(reff_x, reff_y, rho):
    """Major/minor semi-axes and angle of the covariance ellipse.

    Returns ``(major, minor, phi_deg)`` in the same length units as the inputs,
    where ``phi_deg`` is measured counter-clockwise from +x (West) toward +y
    (North).
    """
    reff_x = np.maximum(np.asarray(reff_x, float), MIN_REFF_ARCSEC)
    reff_y = np.maximum(np.asarray(reff_y, float), MIN_REFF_ARCSEC)
    rho = np.clip(np.asarray(rho, float), -MAX_ABS_RHO, MAX_ABS_RHO)
    tr = reff_x**2 + reff_y**2
    det = (reff_x * reff_y) ** 2 * (1.0 - rho**2)
    disc = np.sqrt(np.maximum(tr**2 - 4.0 * det, 0.0))
    major = np.sqrt(0.5 * (tr + disc))
    minor = np.sqrt(np.maximum(0.5 * (tr - disc), 0.0))
    phi = 0.5 * np.degrees(np.arctan2(2.0 * rho * reff_x * reff_y, reff_x**2 - reff_y**2))
    return major, minor, phi


def elliptical_radius(du, dv, reff_x, reff_y, rho):
    """Elliptical radius ``m`` with ``m = 1`` on the effective-radius ellipse.

    ``m**2 = d^T Sigma^{-1} d`` for offset ``d = (du, dv)`` from the centre.
    """
    reff_x = np.maximum(np.asarray(reff_x, float), MIN_REFF_ARCSEC)
    reff_y = np.maximum(np.asarray(reff_y, float), MIN_REFF_ARCSEC)
    rho = np.clip(np.asarray(rho, float), -MAX_ABS_RHO, MAX_ABS_RHO)
    det = (reff_x * reff_y) ** 2 * (1.0 - rho**2)
    m2 = (reff_y**2 * du**2 - 2.0 * rho * reff_x * reff_y * du * dv + reff_x**2 * dv**2) / det
    return np.sqrt(np.maximum(m2, 0.0))


def log_effective_intensity(flux, reff_major, reff_minor, n):
    """``ln I_e`` (flux per arcsec^2 at R_eff) for a Sersic profile of total ``flux``.

    L = 2 pi n I_e R_maj R_min e^{b} b^{-2n} Gamma(2n)
    """
    n = np.asarray(n, float)
    b = b_n(n)
    reff_major = np.maximum(np.asarray(reff_major, float), MIN_REFF_ARCSEC)
    reff_minor = np.maximum(np.asarray(reff_minor, float), MIN_REFF_ARCSEC)
    log_norm = (np.log(2.0 * np.pi * n) + np.log(reff_major) + np.log(reff_minor)
                + b - 2.0 * n * np.log(b) + gammaln(2.0 * n))
    return np.log(np.asarray(flux, float)) - log_norm


def log_sersic_intensity(m, flux, reff_major, reff_minor, n):
    """``ln I(m)`` of a Sersic profile at elliptical radius ``m`` (units of R_eff)."""
    n = np.asarray(n, float)
    b = b_n(n)
    return log_effective_intensity(flux, reff_major, reff_minor, n) - b * (np.asarray(m, float) ** (1.0 / n) - 1.0)


def sersic_intensity(m, flux, reff_major, reff_minor, n):
    """``I(m)`` in flux units per arcsec^2."""
    return np.exp(log_sersic_intensity(m, flux, reff_major, reff_minor, n))


def sersic_intensity_at(du, dv, flux, reff_x, reff_y, rho, n):
    """Surface brightness of one Sersic model at tangent-plane offset (du, dv).

    All lengths in arcsec; ``flux`` in any unit (result is per arcsec^2).
    """
    m = elliptical_radius(du, dv, reff_x, reff_y, rho)
    major, minor, _ = ellipse_axes(reff_x, reff_y, rho)
    return sersic_intensity(m, flux, major, minor, n)


def frac_light_beyond(m, n):
    """Fraction of a Sersic profile's total light at elliptical radius > ``m`` (units of R_eff).

    L(<m) / L_tot = gamma(2n, b_n m^{1/n}) / Gamma(2n), so the fraction outside is
    the regularised upper incomplete gamma function Q(2n, b_n m^{1/n}).  By
    construction it is 0.5 at ``m = 1``.  Used as a p-value-like "hostless"
    indicator: a transient far outside every candidate gets tiny values everywhere.
    """
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    return gammaincc(2.0 * n, b_n(n) * m ** (1.0 / n))
