"""Diagnostic figure: image of the field with candidates and probabilities marked."""

from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

from . import sersic as S
from .images import ImageAccessError, fetch_cutout, render_model_image, tangent_wcs

CAND_COLORS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
               "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", "#1f77b4"]


def _ellipse_sky(ra0, dec0, reff_x, reff_y, rho, scale=1.0, npts=181):
    """RA/Dec points along the ``scale`` x R_eff ellipse of a Sersic model."""
    major, minor, phi = S.ellipse_axes(reff_x, reff_y, rho)
    t = np.linspace(0, 2 * np.pi, npts)
    c, s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    u = scale * (major * np.cos(t) * c - minor * np.sin(t) * s)   # West+
    v = scale * (major * np.cos(t) * s + minor * np.sin(t) * c)   # North+
    ra = ra0 - u / 3600.0 / np.cos(np.deg2rad(dec0))
    dec = dec0 + v / 3600.0
    return ra, dec


def _place_labels_outside_ring(ax, wcs, labels, ra0, dec0, radius_arcsec, shape, min_gap_frac=0.075):
    """Stack candidate labels in two columns just outside the search ring.

    Each label goes to the left or right column according to which side of the
    transient its candidate lies, at the candidate's own y where possible; labels
    in a column are then pushed apart vertically so their boxes never overlap, and
    a leader line connects each box to its candidate.
    """
    if not labels:
        return
    cx, cy = wcs.world_to_pixel_values(ra0, dec0)
    scale = np.mean(np.abs(np.diag(wcs.pixel_scale_matrix))) * 3600.0  # arcsec / pixel
    ny, nx = shape
    x_off = min(1.08 * radius_arcsec / scale, 0.46 * nx)
    gap = min_gap_frac * ny
    for side in (-1, 1):
        col = [lab for lab in labels if (lab[0] - cx) * side >= 0]
        if not col:
            continue
        col.sort(key=lambda lab: lab[1])
        ys = np.array([lab[1] for lab in col], dtype=float)
        # enforce minimum spacing, then keep the stack inside the image
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        shift = min(0.0, (ny - 1 - 0.5 * gap) - ys[-1])
        ys += shift
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
        ys = np.maximum(ys, 0.5 * gap)
        lx = cx + side * x_off
        for (px, py, txt, color), ly in zip(col, ys):
            ax.annotate(txt, (px, py), xytext=(lx, ly), textcoords="data", color=color,
                        fontsize=8.5, fontweight="bold", ha="left" if side > 0 else "right", va="center", linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.8, shrinkA=0, shrinkB=2),
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9))


def method_summary(result):
    """One line comparing the SBW and DLR choices, e.g. for figure titles."""
    if len(result) == 0:
        return "SBW: no candidates      DLR: no candidates"
    k = int(np.nanargmax(result["p_sbw"]))
    sbw = f"SBW host: #{k + 1} (P_SBW={result['p_sbw'][k]:.2f}, max f>={np.nanmax(result['frac_light_beyond']):.2g})"
    if "p_dlr" in result.colnames and np.any(result["p_dlr"] > 0):
        j = int(np.argmax(result["p_dlr"]))
        dlr = f"DLR host: #{j + 1} (d_DLR={result['d_dlr'][j]:.2f})"
    else:
        dmin = np.nanmin(result["d_dlr"]) if "d_dlr" in result.colnames else np.nan
        dlr = f"DLR host: none (min d_DLR={dmin:.1f} > 4)"
    agree = "agree" if ("p_dlr" in result.colnames and np.any(result["p_dlr"] > 0) and j == k) else "DISAGREE"
    return f"{sbw}      {dlr}      [{agree}]"


def plot_association(result, image=None, wcs=None, size_arcsec=None, ax=None,
                     label_top=6, reff_scales=(1.0, 2.0), title=None, show_search_radius=True,
                     raw=None):
    """Draw the field with the SN marked and each candidate's ellipse + probability.

    Parameters
    ----------
    result : Table from :func:`hostassoc.associate_host`.
    image, wcs : optional image array and matching celestial WCS.  If omitted the
        candidates' Sersic models are rendered instead.
    size_arcsec : field of view for the model render (default 2.4 x search radius).
    label_top : annotate only the N most probable candidates (others get a marker).
    reff_scales : draw ellipses at these multiples of the model half-light radius
        (first solid, rest dashed).
    raw : optional unfiltered cone-search table (``associate_host(..., return_raw=True)``);
        objects in it that were rejected (point sources, failed fits) are shown as
        small grey circles so you can see what was *not* considered a host.
    """
    meta = result.meta
    ra0, dec0 = meta["sn_ra"], meta["sn_dec"]
    radius = meta.get("radius_arcsec", 30.0)
    source = "deep_coadd cutout"
    if image is None:
        size_arcsec = size_arcsec or 2.7 * radius
        wcs, shape = tangent_wcs(ra0, dec0, size_arcsec)
        image = render_model_image(result, wcs, shape)
        source = "Sersic model render"

    if ax is None:
        fig = plt.figure(figsize=(8.5, 8))
        ax = fig.add_subplot(111, projection=wcs)
    else:
        fig = ax.figure

    finite = image[np.isfinite(image)]
    vmin = np.percentile(finite, 1) if finite.size else 0.0
    norm = ImageNormalize(image, interval=PercentileInterval(99.7), stretch=AsinhStretch(0.05), vmin=vmin)
    ax.imshow(image, origin="lower", cmap="gray_r", norm=norm)
    ny, nx = image.shape
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    tr = ax.get_transform("world")

    if show_search_radius:
        t = np.linspace(0, 2 * np.pi, 361)
        ax.plot(ra0 - radius / 3600.0 * np.cos(t) / np.cos(np.deg2rad(dec0)),
                dec0 + radius / 3600.0 * np.sin(t), transform=tr, color="cyan", lw=0.8, ls=":",
                label=f'search radius {radius:g}"')

    if raw is not None and len(raw):
        cand_ids = np.asarray(result.meta.get("candidate_ids", list(result["objectId"])), dtype=np.int64)
        rejected = ~np.isin(np.asarray(raw["objectId"], dtype=np.int64), cand_ids)
        if rejected.any():
            ax.plot(np.asarray(raw["coord_ra"])[rejected], np.asarray(raw["coord_dec"])[rejected],
                    transform=tr, marker="o", ms=7, mfc="none", mec="0.35", mew=0.8, ls="none",
                    label=f"rejected (point-like / no fit): {int(rejected.sum())}")

    # SN marker
    ax.plot(ra0, dec0, transform=tr, marker="+", ms=18, mew=2.2, color="red", ls="none",
            label="transient")

    n_suspect_drawn = 0
    labels = []  # (pixel x, pixel y, text, color) for the top candidates
    for k, row in enumerate(result):
        col = CAND_COLORS[k % len(CAND_COLORS)]
        is_suspect = bool(row["suspect"]) if "suspect" in result.colnames else False
        n_suspect_drawn += is_suspect
        for j, sc in enumerate(reff_scales):
            era, edec = _ellipse_sky(row["ra"], row["dec"], row["reff_x_arcsec"],
                                     row["reff_y_arcsec"], row["rho"], scale=sc * row["rhalf_major_arcsec"] / row["reff_major_arcsec"])
            ax.plot(era, edec, transform=tr, color=col, lw=1.6 if j == 0 else 1.0,
                    ls=(":" if is_suspect else ("-" if j == 0 else "--")),
                    alpha=0.95 if j == 0 else 0.6)
        ax.plot(row["ra"], row["dec"], transform=tr, marker="x", ms=7, mew=1.5, color=col, ls="none")
        if k < label_top:
            px, py = wcs.world_to_pixel_values(row["ra"], row["dec"])
            p = row["p_sbw"]
            fb = row["frac_light_beyond"]
            dlr_tag = (" host" if row["p_dlr"] > 0 else "") if "p_dlr" in result.colnames else ""
            d = row["d_dlr"] if "d_dlr" in result.colnames else np.nan
            line1 = f"#{k + 1}  P_SBW={p:.3f}  f>={fb:.2g}" if np.isfinite(p) else f"#{k + 1}  P_SBW=nan"
            if is_suspect:
                line1 += "  [suspect]"
            txt = line1 + f"\nd_DLR={d:.2f}{dlr_tag}"
            labels.append((float(px), float(py), txt, col))

    _place_labels_outside_ring(ax, wcs, labels, ra0, dec0, radius, image.shape)

    # Legend table of the top candidates
    lines = []
    for k, row in enumerate(result[:label_top]):
        flag = "?" if ("suspect" in result.colnames and row["suspect"]) else " "
        dlr_col = (f"d_DLR={row['d_dlr']:5.2f}{'*' if row['p_dlr'] > 0 else ' '} "
                   if "d_dlr" in result.colnames else "")
        lines.append(f"#{k + 1}{flag}id={row['objectId']}  P_SBW={row['p_sbw']:.3f}  f>={row['frac_light_beyond']:8.2g}  "
                     f"{dlr_col} sep={row['sep_arcsec']:4.1f}\"  "
                     f"{meta.get('band_used', meta['band'])}={row['mag']:.1f}  n={row['sersic_index']:.1f}  SB={row['sb_eff']:.1f}")
    notes = ["* = DLR host (min d_DLR <= 4)"]
    if n_suspect_drawn:
        notes.append(f"dotted ellipses / '?' = suspect Sersic fit ({n_suspect_drawn})")
    lines.append("   ".join(notes))
    if lines:
        ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes, fontsize=8.5,
                family="monospace", va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))

    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_major_formatter("hh:mm:ss.s")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    if title is None:
        title = (f"({ra0:.5f}, {dec0:.5f})  band={meta.get('band_used', meta['band'])}  {meta.get('dataset', '')}  "
                 f"fit={meta.get('model', 'sersic')}  {meta['n_candidates']} candidates  [{source}]\n{method_summary(result)}")
    ax.set_title(title, fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    return fig, ax


def plot_with_best_image(result, band=None, dataset=None, size_arcsec=None, raw=None, token=None, **kwargs):
    """Try a real deep_coadd cutout; fall back to the model render.

    Returns ``(fig, ax, used_cutout)``.
    """
    meta = result.meta
    band = band or meta.get("band_used", meta["band"])
    dataset = dataset or meta.get("dataset", "dp2")
    size_arcsec = size_arcsec or 2.7 * meta.get("radius_arcsec", 30.0)
    try:
        img, wcs = fetch_cutout(meta["sn_ra"], meta["sn_dec"], size_arcsec, band=band, dataset=dataset,
                                token=token, image_token=token)
        fig, ax = plot_association(result, image=img, wcs=wcs, raw=raw, **kwargs)
        return fig, ax, True
    except ImageAccessError as e:
        kwargs.setdefault("title", None)
        fig, ax = plot_association(result, size_arcsec=size_arcsec, raw=raw, **kwargs)
        ax.text(0.5, -0.13, f"cutout unavailable: {str(e)[:90]}", transform=ax.transAxes,
                ha="center", fontsize=8, color="0.4")
        return fig, ax, False
