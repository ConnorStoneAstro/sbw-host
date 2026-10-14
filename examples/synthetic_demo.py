"""Synthetic three-galaxy scenarios comparing light-fraction association with DLR.

Each panel is a synthetic image with three analytic Sersic galaxies (convolved with
a Gaussian PSF) and a transient at the image centre.  For every galaxy we report

  P_SBW  - this package's probability, I_i(SN) / sum_j I_j(SN)
  f>       - fraction of the galaxy's light beyond the transient (hostless indicator)
  d_DLR    - separation / directional light radius (Sullivan+06, Gupta+16, Qu+24);
             the DLR host is the smallest d_DLR if it is <= 4 (hard assignment, no probability)

Galaxies are specified by true half-light radius (arcsec), Sersic index, total flux,
axis ratio and position angle, so both algorithms are evaluated with
``profile="analytic"`` and the DLR ellipse is the half-light ellipse.

Usage: python synthetic_demo.py [--out examples/synthetic_scenarios.png]
"""

from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize
from scipy.ndimage import gaussian_filter

from sbw_host import sersic as S
from sbw_host.associate import compute_association

PIX = 0.2          # arcsec / pixel
FOV = 40.0         # arcsec on a side
PSF_FWHM = 0.8     # arcsec
RA0, DEC0 = 150.0, 2.0   # arbitrary tangent point for the fake catalogue
COLORS = ["#ff7f0e", "#2ca02c", "#9467bd"]


def galaxy(du, dv, re, n, mag, q=1.0, pa=0.0):
    """Galaxy at offset (du West+, dv North+) arcsec from the SN; re = half-light radius (major axis)."""
    return dict(du=du, dv=dv, re=re, n=n, mag=mag, q=q, pa=pa)


# --- scenarios -------------------------------------------------------------------
# (title, [galaxy A, galaxy B, galaxy C]).  The transient is always at (0, 0).
SCENARIOS = [
    ("1. SN on the centre of A", [
        galaxy(0.0, 0.3, 2.0, 1.0, 20.0, 0.7, 30),
        galaxy(9.0, -4.0, 2.0, 1.0, 20.0, 0.7, -40),
        galaxy(-10.0, 8.0, 1.5, 2.0, 21.0)]),
    ("2. Closer to A than B (twins)", [
        galaxy(-2.5, 0.0, 2.0, 1.0, 20.0, 0.8, 90),
        galaxy(5.0, 0.0, 2.0, 1.0, 20.0, 0.8, 90),
        galaxy(2.0, 12.0, 1.5, 1.0, 21.0)]),
    ("3. Exactly between twins A and B", [
        galaxy(-4.0, 0.0, 2.0, 1.0, 20.0),
        galaxy(4.0, 0.0, 2.0, 1.0, 20.0),
        galaxy(0.0, -11.0, 2.5, 1.0, 20.5, 0.6, 0)]),
    ("4. Between big faint disk A and small bright bulge B", [
        galaxy(-5.0, 0.0, 4.0, 1.0, 20.5, 0.8, 60),
        galaxy(3.6, 0.0, 1.0, 4.0, 18.5),
        galaxy(3.0, 12.0, 1.5, 1.0, 21.0)]),
    ("5. Same d_DLR, different brightness", [
        galaxy(-4.0, 0.0, 2.0, 1.0, 19.0),
        galaxy(4.0, 0.0, 2.0, 1.0, 22.0),
        galaxy(0.0, 10.0, 1.0, 1.0, 21.0)]),
    ("6. Outskirts of big elliptical A, faint dwarf B in projection", [
        galaxy(-9.0, 0.0, 5.0, 4.0, 17.5, 0.8, 20),
        galaxy(1.2, 0.8, 0.6, 1.0, 23.0),
        galaxy(8.0, -9.0, 1.5, 1.0, 21.0)]),
    ("7. SN on the major axis of edge-on disk A; round B equidistant", [
        galaxy(-5.0, 0.0, 3.0, 1.0, 20.0, 0.3, 0),
        galaxy(5.0, 0.0, 3.0, 1.0, 20.0, 1.0, 0),
        galaxy(0.0, 11.0, 1.5, 2.0, 21.0)]),
    ("8. SN on the minor axis of edge-on disk A; round B equidistant", [
        galaxy(0.0, -5.0, 3.0, 1.0, 20.0, 0.3, 0),
        galaxy(0.0, 5.0, 3.0, 1.0, 20.0, 1.0, 0),
        galaxy(11.0, 0.0, 1.5, 2.0, 21.0)]),
    ("9. Same distance, n=1 disk A vs n=4 bulge B (same R_e, flux)", [
        galaxy(-6.0, 0.0, 2.0, 1.0, 20.0),
        galaxy(6.0, 0.0, 2.0, 4.0, 20.0),
        galaxy(0.0, 12.0, 1.5, 1.0, 21.0)]),
    ("10. Hostless: everything > 5 R_e away", [
        galaxy(-10.0, 3.0, 1.5, 1.0, 20.0, 0.7, 45),
        galaxy(9.0, -6.0, 1.2, 2.0, 20.5),
        galaxy(2.0, 13.0, 1.0, 1.0, 21.5)]),
]


def cands_table(gals):
    """Fake Object-table rows (sersic_reff = true half-light radius, in pixels)."""
    rows = {k: [] for k in ("objectId", "coord_ra", "coord_dec", "sersic_ra", "sersic_dec", "sersic_reff_x",
                            "sersic_reff_y", "sersic_rho", "sersic_index", "refExtendedness", "parentObjectId",
                            "sersic_no_data_flag", "sersic_unknown_flag", "r_sersicFlux", "r_sersicFluxErr",
                            "r_cModelFlux", "r_extendedness")}
    for i, g in enumerate(gals):
        ra = RA0 - g["du"] / 3600.0 / np.cos(np.deg2rad(DEC0))
        dec = DEC0 + g["dv"] / 3600.0
        a, b = g["re"] / PIX, g["q"] * g["re"] / PIX
        c, s = np.cos(np.deg2rad(g["pa"])), np.sin(np.deg2rad(g["pa"]))
        # covariance ellipse with semi-axes a (along pa) and b
        sxx, syy, sxy = a * a * c * c + b * b * s * s, a * a * s * s + b * b * c * c, (a * a - b * b) * s * c
        rx, ry = np.sqrt(sxx), np.sqrt(syy)
        flux = 10 ** (-0.4 * (g["mag"] - 31.4))
        for k, v in (("objectId", i + 1), ("coord_ra", ra), ("coord_dec", dec), ("sersic_ra", ra), ("sersic_dec", dec),
                     ("sersic_reff_x", rx), ("sersic_reff_y", ry), ("sersic_rho", sxy / (rx * ry)), ("sersic_index", g["n"]),
                     ("refExtendedness", 1.0), ("parentObjectId", 0), ("sersic_no_data_flag", False), ("sersic_unknown_flag", False),
                     ("r_sersicFlux", flux), ("r_sersicFluxErr", 1.0), ("r_cModelFlux", flux), ("r_extendedness", 1.0)):
            rows[k].append(v)
    t = Table(rows)
    t.meta["model"] = "sersic"
    return t


def render(gals, noise_nJy=2.0, seed=0):
    npix = int(round(FOV / PIX))
    yy, xx = np.mgrid[0:npix, 0:npix]
    cx = cy = (npix - 1) / 2.0
    du_pix, dv_pix = (xx - cx) * PIX, (yy - cy) * PIX
    img = np.zeros((npix, npix))
    for g in gals:
        a, b = g["re"], g["q"] * g["re"]
        c, s = np.cos(np.deg2rad(g["pa"])), np.sin(np.deg2rad(g["pa"]))
        rx, ry = np.sqrt(a * a * c * c + b * b * s * s), np.sqrt(a * a * s * s + b * b * c * c)
        rho = (a * a - b * b) * s * c / (rx * ry)
        flux = 10 ** (-0.4 * (g["mag"] - 31.4))
        img += S.sersic_intensity_at(du_pix - g["du"], dv_pix - g["dv"], flux, rx, ry, rho, g["n"]) * PIX**2
    img = gaussian_filter(img, PSF_FWHM / 2.3548 / PIX)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, noise_nJy, img.shape)
    return img, (cx, cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/synthetic_scenarios.png")
    args = ap.parse_args()
    ncol = 2
    nrow = int(np.ceil(len(SCENARIOS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.2 * ncol, 7.6 * nrow))
    axes = axes.ravel()
    lines_md = ["| scenario | galaxy | R_e (\") | n | mag | sep (\") | P_SBW | f> | d_DLR | DLR host |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, (title, gals) in enumerate(SCENARIOS):
        res = compute_association(cands_table(gals), RA0, DEC0, "r", profile="analytic")
        res.sort("objectId")
        img, (cx, cy) = render(gals, seed=k)
        ax = axes[k]
        norm = ImageNormalize(img, vmin=-4.0, vmax=np.percentile(img, 99.8), stretch=AsinhStretch(0.03))
        ax.imshow(img, origin="lower", cmap="gray_r", norm=norm, extent=[(-cx - 0.5) * PIX, (cx + 0.5) * PIX, (-cy - 0.5) * PIX, (cy + 0.5) * PIX])
        ax.invert_xaxis()  # West (positive du) to the right -> show East left
        ax.plot(0, 0, "+", color="red", ms=16, mew=2.2)
        table_lines = []
        d = np.asarray(res["d_dlr"])
        order = np.argsort(d)
        dlr_tie = len(d) > 1 and abs(d[order[0]] - d[order[1]]) < 1e-6 and d[order[0]] <= 4
        for i, (g, row) in enumerate(zip(gals, res)):
            col = COLORS[i]
            name = "ABC"[i]
            # half-light ellipse
            t = np.linspace(0, 2 * np.pi, 181)
            a, b = g["re"], g["q"] * g["re"]
            c, s = np.cos(np.deg2rad(g["pa"])), np.sin(np.deg2rad(g["pa"]))
            eu = g["du"] + a * np.cos(t) * c - b * np.sin(t) * s
            ev = g["dv"] + a * np.cos(t) * s + b * np.sin(t) * c
            ax.plot(eu, ev, color=col, lw=1.6)
            ax.plot(g["du"], g["dv"], "x", color=col, ms=7, mew=1.5)
            ax.annotate(name, (g["du"], g["dv"]), xytext=(6, 6), textcoords="offset points", color=col, fontsize=12, fontweight="bold")
            dlr_tag = ("tie" if dlr_tie and abs(row["d_dlr"] - d[order[0]]) < 1e-6
                       else "host" if row["p_dlr"] > 0 else ("none" if row["d_dlr"] > 4 else "-"))
            table_lines.append(f"{name}: P_SBW={row['p_sbw']:.3f}  f>={row['frac_light_beyond']:.2g}   d_DLR={row['d_dlr']:.2f} ({dlr_tag})")
            lines_md.append(f"| {title} | {name} | {g['re']:.1f} | {g['n']:.0f} | {g['mag']:.1f} | {row['sep_arcsec']:.1f} | {row['p_sbw']:.3f} | {row['frac_light_beyond']:.2g} | {row['d_dlr']:.2f} | {dlr_tag} |")
        if dlr_tie:
            dlr_pick = "tie " + "/".join("ABC"[i] for i in range(len(d)) if abs(d[i] - d[order[0]]) < 1e-6)
        else:
            dlr_pick = "ABC"[int(np.argmax(res["p_dlr"]))] if res["p_dlr"].max() > 0 else "none (all d_DLR > 4)"
        light_pick = "ABC"[int(np.argmax(res["p_sbw"]))]
        ax.set_title(f"{title}\nSBW host: {light_pick} (P={res['p_sbw'].max():.2f}, max f>={res['frac_light_beyond'].max():.2g})"
                     f"      DLR host: {dlr_pick}", fontsize=10.5)
        ax.text(0.02, 0.02, "\n".join(table_lines), transform=ax.transAxes, fontsize=9.2, family="monospace", va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.92))
        ax.set_xlabel("ΔRA·cosδ [arcsec]  (East left)")
        ax.set_ylabel("ΔDec [arcsec]")
    for ax in axes[len(SCENARIOS):]:
        ax.axis("off")
    fig.suptitle(f"Synthetic 3-galaxy scenarios (Sérsic models, PSF FWHM {PSF_FWHM}\", noise 2 nJy/pix). Red + = transient; "
                 "ellipses = half-light radii.\nP_SBW: Sérsic surface-brightness fraction at the transient. f>: fraction of that galaxy's light beyond the transient. "
                 "d_DLR = separation / directional half-light radius; DLR host = min d_DLR if <= 4.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(args.out, dpi=100)
    md = args.out.rsplit(".", 1)[0] + ".md"
    open(md, "w").write("\n".join(lines_md) + "\n")
    print("wrote", args.out, "and", md)


if __name__ == "__main__":
    main()
