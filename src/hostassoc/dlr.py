"""Directional light radius (DLR) host matching, for comparison.

Following Sullivan et al. (2006), Gupta et al. (2016) and Qu et al. (2024, ApJ,
DES-SN host mismatch): for a galaxy with semi-major axis ``a``, semi-minor axis
``b`` and the transient at angle ``phi`` from the major axis,

    DLR   = a b / sqrt( (b cos phi)^2 + (a sin phi)^2 )      (ellipse radius toward the SN)
    d_DLR = separation / DLR

The galaxy with the smallest d_DLR is the host; candidates with d_DLR > 4 are
discarded, and DLR assigns no probabilities.  Here ``a``, ``b`` are the model
half-light radii.  Note d_DLR is exactly the elliptical radius ``m`` of the
transient in units of the half-light ellipse (``m_rhalf`` in the association
table), so DLR is "nearest galaxy in units of its own half-light radius",
independent of brightness and Sersic index.
"""

from __future__ import annotations

import numpy as np

DLR_MAX = 4.0


def d_dlr(du, dv, a, b, phi_deg):
    """d_DLR for offsets (du, dv) [arcsec] from a galaxy with axes a, b and angle phi.

    ``phi_deg`` is the major-axis angle measured counter-clockwise from +u toward
    +v (the same convention as :func:`hostassoc.sersic.ellipse_axes`).
    """
    du, dv = np.asarray(du, float), np.asarray(dv, float)
    a = np.maximum(np.asarray(a, float), 1e-6)
    b = np.maximum(np.asarray(b, float), 1e-6)
    sep = np.hypot(du, dv)
    theta = np.arctan2(dv, du) - np.deg2rad(phi_deg)   # angle from the major axis
    dlr = a * b / np.sqrt((b * np.cos(theta)) ** 2 + (a * np.sin(theta)) ** 2)
    return sep / dlr


def dlr_assignment(d, dlr_max=DLR_MAX):
    """Hard DLR host choice: 1 for the smallest d_DLR if it is <= dlr_max, else 0."""
    d = np.asarray(d, float)
    out = np.zeros(d.shape)
    if d.size and np.any(np.isfinite(d)):
        k = int(np.nanargmin(d))
        if d[k] <= dlr_max:
            out[k] = 1.0
    return out
