"""Model-free SBW from deblended host stamps.

Instead of a parametric profile, each candidate host is described by a 2-D array
of its (deblended) light.  The transient's SBW weight for host ``i`` is the
stamp's surface brightness at the transient position, and the hostless indicator
is the fraction of the stamp's flux in pixels fainter than the transient's
neighbourhood.

Placement convention
--------------------
Stamps may have different sizes.  Each is placed in a common pixel frame by the
coordinate of its ``[0, 0]`` element (``origins[i] = (x0, y0)``), so array element
``stamp[j, k]`` sits at ``x = x0 + k``, ``y = y0 + j`` and the transient is at
``(tx, ty)`` in the same frame.  This is the bounding-box corner convention of
LSST footprints.  Alternatively pass one astropy ``WCS`` per stamp and give the
transient in RA/Dec; intensities are then per arcsec² so stamps with different
pixel scales compare correctly.  All stamps must share one band and flux zero point.

A transient outside a stamp's footprint gets zero intensity from that host.
Negative pixels are clipped to zero.  Note that on noisy *data* stamps this
clipping adds positive flux to the faint pixels, which biases
``frac_light_beyond`` upward; the method is intended for deblender model stamps,
or data stamps that have been thresholded or smoothed first.
"""

from __future__ import annotations

import numpy as np

from .core import SBWResult, sbw_probabilities

INTERPOLATIONS = ("bilinear", "nearest", "box")


def _clean(stamp):
    s = np.asarray(stamp, dtype=float)
    if s.ndim != 2:
        raise ValueError("each stamp must be a 2-D array")
    return np.maximum(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def _inside(s, px, py):
    ny, nx = s.shape
    return (-0.5 <= px <= nx - 0.5) and (-0.5 <= py <= ny - 0.5)


def _bilinear(s, px, py):
    ny, nx = s.shape
    px, py = min(max(px, 0.0), nx - 1.0), min(max(py, 0.0), ny - 1.0)   # clamp to pixel centres
    x0, y0 = int(np.floor(px)), int(np.floor(py))
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    fx, fy = px - x0, py - y0
    return ((1 - fx) * (1 - fy) * s[y0, x0] + fx * (1 - fy) * s[y0, x1]
            + (1 - fx) * fy * s[y1, x0] + fx * fy * s[y1, x1])


def _box_mean(s, px, py, box):
    """Mean of the ``box x box`` pixels centred on the pixel containing ``(px, py)``.

    Only pixels inside the stamp count; returns 0 if none are.
    """
    ny, nx = s.shape
    cx, cy = int(np.rint(px)), int(np.rint(py))
    h = box // 2
    x0, x1 = max(cx - h, 0), min(cx + h + 1, nx)
    y0, y1 = max(cy - h, 0), min(cy + h + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(np.mean(s[y0:y1, x0:x1]))


def _transient_pixel(tx, ty, i, origins, wcs):
    if wcs is not None:
        px, py = wcs[i].all_world2pix(tx, ty, 0)
        return float(px), float(py)
    x0, y0 = origins[i]
    return float(tx) - float(x0), float(ty) - float(y0)


def _pixel_area_arcsec2(w):
    from astropy.wcs.utils import proj_plane_pixel_area
    return float(proj_plane_pixel_area(w)) * 3600.0**2


def sbw_stamps(tx, ty, stamps, origins=None, *, wcs=None, interp="bilinear", box=3) -> SBWResult:
    """SBW association of one transient against deblended host stamps.

    Parameters
    ----------
    tx, ty : float
        Transient position: pixel coordinates in the common frame when
        ``origins`` is given, or RA/Dec in degrees when ``wcs`` is given.
    stamps : sequence of 2-D arrays
        One deblended image per candidate host (any sizes).
    origins : sequence of (x0, y0), optional
        Common-frame pixel coordinate of each stamp's ``[0, 0]`` element.
    wcs : sequence of astropy.wcs.WCS, optional
        One WCS per stamp, as an alternative to ``origins``.
    interp : {"bilinear", "nearest", "box"}
        How to read the surface brightness at the transient position:
        bilinear interpolation between pixel centres (default), the nearest
        pixel, or the same ``box x box`` mean used for the hostless indicator.
    box : int
        Odd side length of the pixel box averaged around the transient for the
        hostless indicator (default 3: a 3x3 mean, to be robust to faint noisy
        pixels).  Use 1 for the single pixel.

    Returns
    -------
    SBWResult
        ``intensity`` per pixel (or per arcsec² with ``wcs``), ``p_sbw``,
        ``frac_light_beyond`` (fraction of each stamp's flux in pixels fainter than
        the box mean at the transient; NaN for an empty stamp) and ``m=None``.
    """
    if (origins is None) == (wcs is None):
        raise ValueError("give exactly one of origins or wcs")
    if interp not in INTERPOLATIONS:
        raise ValueError(f"interp must be one of {INTERPOLATIONS}")
    box = int(box)
    if box < 1 or box % 2 == 0:
        raise ValueError("box must be a positive odd integer (1, 3, 5, ...)")
    n_host = len(stamps)
    ref = origins if wcs is None else wcs
    if len(ref) != n_host:
        raise ValueError("need one origin (or WCS) per stamp")

    intensity = np.zeros(n_host)
    f_beyond = np.full(n_host, np.nan)
    for i, stamp in enumerate(stamps):
        s = _clean(stamp)
        px, py = _transient_pixel(tx, ty, i, origins, wcs)
        total = float(s.sum())
        thr = _box_mean(s, px, py, box) if _inside(s, px, py) else 0.0
        if _inside(s, px, py):
            if interp == "bilinear":
                val = _bilinear(s, px, py)
            elif interp == "nearest":
                val = float(s[int(np.rint(py)), int(np.rint(px))])
            else:
                val = thr
        else:
            val = 0.0
        if wcs is not None:
            area = _pixel_area_arcsec2(wcs[i])
            val, thr_cmp = val / area, thr
        else:
            thr_cmp = thr
        intensity[i] = val
        if total > 0:
            f_beyond[i] = float(s[s < thr_cmp].sum()) / total
    with np.errstate(divide="ignore"):
        log_i = np.log(intensity)
    return SBWResult(intensity=intensity, log_intensity=log_i, p_sbw=sbw_probabilities(log_i),
                     frac_light_beyond=f_beyond, m=None)
