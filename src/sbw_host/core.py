"""Table-free SBW core: surface-brightness-weighted host association from model parameters.

Everything in this module works on plain NumPy arrays in a flat 2-D frame, with
no reference to the Rubin Object table.  Use it to run the SBW idea on another
catalogue, on mock data, or to experiment with the algorithm::

    >>> from sbw_host.core import sbw_exponential
    >>> res = sbw_exponential(0.0, 0.0, x=[3.0, -8.0], y=[1.0, 5.0],
    ...                       r_eff=[4.0, 12.0], flux=[2e4, 8e4], q=[0.5, 0.9], phi=[30.0, -10.0])
    >>> res.p_sbw.round(3)
    array([0.792, 0.208])

Conventions
-----------
* Positions ``(tx, ty)`` of the transient and ``(x, y)`` of the hosts share one
  flat frame in one length unit (arcsec, pixels, kpc ...).  Radii use the same
  unit.  :func:`sky_to_plane` converts RA/Dec to such a frame.
* ``phi`` is the position angle of the major axis in degrees, measured
  counter-clockwise from +x toward +y.  With :func:`sky_to_plane` (+x West,
  +y North) the usual astronomical PA (North through East) is ``phi - 90``.
* ``q`` is the axis ratio (minor / major), between 0 and 1.
* Exactly one size (``r_eff`` half-light radius, or ``r_scale`` exponential /
  Sérsic scale length) and exactly one amplitude (``flux`` total, ``i_eff``
  intensity at the half-light radius, or ``i_0`` central intensity) per call.
* Probabilities are the intensities normalised over the hosts supplied, so the
  caller chooses the candidate list.  To drop a candidate, set its ``flux`` (or
  ``i_eff`` / ``i_0``) to zero; it then gets ``p_sbw = 0``.
* ``profile="analytic"`` is a true Sérsic profile.  ``profile="mixture"`` is the
  4-Gaussian approximation that MultiProFit fits in the Rubin pipeline; ``r_eff``
  is still the real half-light radius (Rubin's catalogue ``reff`` is a scale
  parameter equal to ``r_eff / 1.386``; see the package README).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from . import sersic as S
from . import sersic_mix as SM

PROFILES = ("analytic", "mixture")
MIN_AXIS_RATIO = 1e-6


@dataclass
class SBWResult:
    """Per-host SBW quantities for one transient.

    Attributes
    ----------
    intensity : ndarray
        Model surface brightness of each host at the transient position, in
        ``flux / length**2`` of the inputs.
    log_intensity : ndarray
        ``ln`` of ``intensity`` (``-inf`` where the intensity is zero).
    p_sbw : ndarray
        SBW host probability ``I_i / sum_j I_j``; all NaN if every intensity is zero.
    frac_light_beyond : ndarray
        Fraction of each host's light fainter than (outside the isophote through)
        the transient position.  Small everywhere means hostless.
    m : ndarray or None
        Elliptical radius of the transient in units of each host's half-light
        radius (None for the stamp-based version).
    """

    intensity: np.ndarray
    log_intensity: np.ndarray
    p_sbw: np.ndarray
    frac_light_beyond: np.ndarray
    m: np.ndarray | None = None

    @property
    def best(self) -> int | None:
        """Index of the most probable host, or None if no host has flux there."""
        if not np.any(np.isfinite(self.p_sbw)):
            return None
        return int(np.nanargmax(self.p_sbw))


def sky_to_plane(ra, dec, ra0, dec0):
    """Flat-frame coordinates (arcsec) of ``(ra, dec)`` about the tangent point ``(ra0, dec0)``.

    Returns ``(x, y)`` with +x toward West (decreasing RA) and +y toward North,
    the orientation of LSST tract pixel axes.  Small-angle approximation.
    """
    return S.sky_offsets(ra, dec, ra0, dec0)


def sbw_probabilities(log_intensity):
    """Normalise log intensities to SBW probabilities ``I_i / sum_j I_j``.

    Non-finite entries (``-inf`` or NaN) get probability zero.  If nothing is
    finite the result is all NaN.  Add log-priors to ``log_intensity`` before
    calling this to experiment with weighted variants of SBW.
    """
    log_i = np.asarray(log_intensity, dtype=float)
    ok = np.isfinite(log_i)
    if not np.any(ok):
        return np.full(log_i.shape, np.nan)
    p = np.zeros(log_i.shape)
    p[ok] = np.exp(log_i[ok] - logsumexp(log_i[ok]))
    return p


def elliptical_radius(tx, ty, x, y, r_major, q, phi_deg):
    """Elliptical radius of ``(tx, ty)`` about hosts at ``(x, y)``, in units of ``r_major``.

    ``m = 1`` on the ellipse with semi-major axis ``r_major``, axis ratio ``q`` and
    position angle ``phi_deg`` (degrees, counter-clockwise from +x toward +y).
    """
    dx = np.asarray(tx, float) - np.asarray(x, float)
    dy = np.asarray(ty, float) - np.asarray(y, float)
    ph = np.deg2rad(np.asarray(phi_deg, float))
    u = dx * np.cos(ph) + dy * np.sin(ph)      # along the major axis
    v = -dx * np.sin(ph) + dy * np.cos(ph)     # along the minor axis
    a = np.asarray(r_major, float)
    b = a * np.maximum(np.asarray(q, float), MIN_AXIS_RATIO)
    return np.sqrt((u / a) ** 2 + (v / b) ** 2)


def _one_of(**kw):
    given = [k for k, v in kw.items() if v is not None]
    if len(given) != 1:
        raise ValueError(f"give exactly one of {', '.join(kw)} (got {given or 'none'})")
    return given[0], np.asarray(kw[given[0]], dtype=float)


def _unit_log_intensity(m_half, a_half, b_half, n, profile):
    """``ln I`` at half-light-unit radius ``m_half`` for unit total flux, plus ``f_beyond``."""
    if profile == "analytic":
        log_i = S.log_sersic_intensity(m_half, 1.0, a_half, b_half, n)
        f_beyond = S.frac_light_beyond(m_half, n)
    elif profile == "mixture":
        hl = SM.half_light_factor(n)              # R_half / mixture scale radius
        a_s, b_s, m_s = a_half / hl, b_half / hl, m_half * hl
        log_i = SM.log_intensity(m_s, 1.0, a_s, b_s, n)
        f_beyond = SM.frac_light_beyond(m_s, n)
    else:
        raise ValueError(f"profile must be one of {PROFILES}")
    return log_i, f_beyond


def sbw_model(tx, ty, x, y, *, r_eff=None, r_scale=None, flux=None, i_eff=None, i_0=None,
              n=1.0, q=1.0, phi=0.0, profile="analytic") -> SBWResult:
    """SBW association of one transient against elliptical Sérsic-type hosts.

    Parameters
    ----------
    tx, ty : float
        Transient position in the flat frame.
    x, y : array_like, shape (N,)
        Host centres in the same frame and unit.
    r_eff, r_scale : array_like
        Host size (one of the two): half-light semi-major axis ``r_eff``, or the
        scale length ``r_scale`` with ``I ∝ exp(-(r / r_scale)**(1/n))``
        (``r_eff = b_n**n r_scale``; 1.678 ``r_scale`` for an exponential).
    flux, i_eff, i_0 : array_like
        Host amplitude (one of the three): total flux, surface brightness at
        ``r_eff`` on the major axis, or central surface brightness, in
        ``flux / length**2``.  Zero excludes a host.
    n : array_like, optional
        Sérsic index (default 1, exponential).
    q, phi : array_like, optional
        Axis ratio and major-axis position angle in degrees (see module notes).
    profile : {"analytic", "mixture"}
        True Sérsic profile, or Rubin's 4-Gaussian approximation to it.

    Returns
    -------
    SBWResult
    """
    tx, ty = float(np.squeeze(tx)), float(np.squeeze(ty))
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length (one entry per host)")
    size_kind, size = _one_of(r_eff=r_eff, r_scale=r_scale)
    amp_kind, amp = _one_of(flux=flux, i_eff=i_eff, i_0=i_0)
    n = np.asarray(n, float)
    size, amp, n, q, phi = np.broadcast_arrays(size, amp, n, np.asarray(q, float), np.asarray(phi, float))
    size, amp, n, q, phi = (np.broadcast_to(v, x.shape).astype(float) for v in (size, amp, n, q, phi))
    if np.any(n <= 0):
        raise ValueError("Sérsic index n must be positive")
    if np.any(size <= 0):
        raise ValueError("host sizes must be positive")

    a_half = size * S.b_n(n) ** n if size_kind == "r_scale" else size
    b_half = a_half * np.maximum(q, MIN_AXIS_RATIO)
    m = elliptical_radius(tx, ty, x, y, a_half, q, phi)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_i_unit, f_beyond = _unit_log_intensity(m, a_half, b_half, n, profile)
        if amp_kind == "flux":
            log_amp = np.log(amp)
        else:
            m_ref = np.ones_like(m) if amp_kind == "i_eff" else np.zeros_like(m)
            log_i_ref, _ = _unit_log_intensity(m_ref, a_half, b_half, n, profile)
            log_amp = np.log(amp) - log_i_ref          # ln(total flux) that gives ``amp`` at m_ref
        log_i = log_i_unit + log_amp
        log_i = np.where(amp > 0, log_i, -np.inf)
        intensity = np.exp(log_i)
    return SBWResult(intensity=intensity, log_intensity=log_i, p_sbw=sbw_probabilities(log_i),
                     frac_light_beyond=f_beyond, m=m)


def sbw_exponential(tx, ty, x, y, *, r_eff=None, r_scale=None, flux=None, i_eff=None, i_0=None,
                    q=1.0, phi=0.0, profile="analytic") -> SBWResult:
    """SBW association against exponential (Sérsic n = 1) hosts; see :func:`sbw_model`.

    ``r_scale`` is the exponential scale length (``r_eff = 1.678 r_scale``) and
    ``i_0`` the central surface brightness, the parameters of
    ``I(r) = i_0 exp(-r / r_scale)``.
    """
    return sbw_model(tx, ty, x, y, r_eff=r_eff, r_scale=r_scale, flux=flux, i_eff=i_eff, i_0=i_0,
                     n=1.0, q=q, phi=phi, profile=profile)


def sbw_sersic(tx, ty, x, y, n, *, r_eff=None, r_scale=None, flux=None, i_eff=None, i_0=None,
               q=1.0, phi=0.0, profile="analytic") -> SBWResult:
    """SBW association against Sérsic hosts of index ``n``; see :func:`sbw_model`."""
    return sbw_model(tx, ty, x, y, r_eff=r_eff, r_scale=r_scale, flux=flux, i_eff=i_eff, i_0=i_0,
                     n=n, q=q, phi=phi, profile=profile)
