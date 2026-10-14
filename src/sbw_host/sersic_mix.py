"""Gaussian-mixture Sersic profiles as used by MultiProFit / gauss2d_fit.

Rubin's ``sersic_*`` Object columns come from MultiProFit, which does not fit an
analytic Sersic profile.  It represents the profile as a sum of ``order`` (4 by
default) concentric Gaussians whose relative weights and sizes are tabulated as a
function of Sersic index (``lsst/gauss2d_fit/src/sersicmix.cc``, knots every
0.001 in n from 0.5 to 8).  Component *k* has

    sigma_k = knot_sigma_k(n) * sersic_reff      (gauss2d_fit SersicEllipseData::get_sigma_x)

so the catalogue ``sersic_reff_x/y`` are *scale parameters* of that mixture, not
half-light radii: the mixture's actual half-light radius is ~1.39 x sersic_reff
(for n = 0.5 the single component has sigma = 1.1774 reff, i.e. R_half = 1.386 reff).
Both facts were confirmed against DP2 deep_coadd cutouts (see README): the
PSF-convolved mixture with this scaling reproduces isolated galaxies with
chi^2/pixel ~ 1-3, while an analytic Sersic with R_eff = sersic_reff is far too
compact (chi^2/pixel ~ 40).

Because the mixture's outermost Gaussian has sigma of only a few reff, the model's
wings fall off far faster than an analytic Sersic (dramatically so for n >~ 2),
which matters for both the intensity at a transient position and the fraction of
light beyond it.

All functions below take the elliptical radius ``m`` in units of the catalogue
``sersic_reff`` (``m**2 = d^T Sigma^{-1} d`` with Sigma built from reff_x, reff_y,
rho), and lengths in arcsec.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

DEFAULT_ORDER = 4
_KNOTS_FILE = Path(__file__).resolve().parent / "data" / "sersic_mix_knots.json"


@lru_cache(maxsize=None)
def _knots(order: int):
    data = json.loads(_KNOTS_FILE.read_text())
    try:
        knots = data[str(order)]
    except KeyError as e:
        raise ValueError(f"No Sersic mixture knots for order={order}; have {sorted(data)}") from e
    n = np.array([k["n"] for k in knots])
    w = np.array([k["integral"] for k in knots])
    s = np.array([k["sigma"] for k in knots])
    o = np.argsort(n)
    return n[o], w[o], s[o]


def mixture(n, order: int = DEFAULT_ORDER):
    """Weights and sigma ratios of the Gaussian mixture for Sersic index ``n``.

    Returns ``(weights, sigma_ratios)`` with trailing axis of length ``order``;
    weights sum to one and ``sigma_k = sigma_ratios_k * sersic_reff``.  Linear
    interpolation between knots (gauss2d_fit uses a GSL spline on the same 0.001
    grid, so the difference is negligible).  ``n`` is clipped to the knot range.
    """
    kn, kw, ks = _knots(order)
    n = np.clip(np.asarray(n, dtype=float), kn[0], kn[-1])
    w = np.stack([np.interp(n, kn, kw[:, i]) for i in range(kw.shape[1])], axis=-1)
    s = np.stack([np.interp(n, kn, ks[:, i]) for i in range(ks.shape[1])], axis=-1)
    return w, s


def log_intensity(m, flux, reff_major, reff_minor, n, order: int = DEFAULT_ORDER):
    """``ln I(m)`` (flux per arcsec^2) of the mixture at elliptical radius ``m``.

    Each component is a 2-D Gaussian sharing the catalogue ellipse scaled by its
    sigma ratio: ``I_k = w_k F / (2 pi s_k^2 a b) exp(-m^2 / (2 s_k^2))`` with
    ``a, b`` the catalogue reff major/minor axes in arcsec.
    """
    m = np.asarray(m, dtype=float)
    w, s = mixture(n, order)                       # (..., K)
    m_ = m[..., None]
    a = np.asarray(reff_major, float)[..., None]
    b = np.asarray(reff_minor, float)[..., None]
    with np.errstate(divide="ignore"):
        logw = np.log(w)
    log_terms = (logw + np.log(np.asarray(flux, float))[..., None]
                 - np.log(2.0 * np.pi) - 2.0 * np.log(s) - np.log(a) - np.log(b)
                 - 0.5 * (m_ / s) ** 2)
    return logsumexp(log_terms, axis=-1)


def frac_light_beyond(m, n, order: int = DEFAULT_ORDER):
    """Fraction of the mixture's light outside elliptical radius ``m``.

    For a 2-D Gaussian the fraction outside radius ``m`` is ``exp(-m^2/(2 s^2))``,
    so the mixture gives ``sum_k w_k exp(-m^2 / (2 s_k^2))``.
    """
    m = np.asarray(m, dtype=float)
    w, s = mixture(n, order)
    return np.sum(w * np.exp(-0.5 * (m[..., None] / s) ** 2), axis=-1)


def half_light_factor(n, order: int = DEFAULT_ORDER):
    """``R_half / sersic_reff`` for the mixture (~1.39; depends weakly on n).

    Solved by bisection on ``frac_light_beyond(m, n) = 0.5``.
    """
    n = np.asarray(n, dtype=float)
    lo = np.full(n.shape, 0.1)
    hi = np.full(n.shape, 10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = frac_light_beyond(mid, n, order)
        hi = np.where(f < 0.5, mid, hi)   # too far out -> move inward
        lo = np.where(f >= 0.5, mid, lo)
    return 0.5 * (lo + hi)
