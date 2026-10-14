"""Residual test of the Object-table Sersic models against deep_coadd cutouts.

For a handful of fairly isolated bright galaxies, build the PSF-convolved model
from the catalogue parameters (using the two-Gaussian PSF model columns), subtract
it from the real r-band cutout and report chi^2 per pixel inside 3 catalogue reff.
Models compared:

* ``mixture``  - MultiProFit's 4-Gaussian Sersic with sigma_k = knot_k * sersic_reff
                 (what ``sbw_host`` uses)
* ``analytic`` - a true Sersic profile with R_eff = sersic_reff (what the column
                 names suggest; it is too compact)

Usage: python check_sersic_model.py [--n 8] [--field 53.13 -28.10] [--out examples/residuals.png]
"""

from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from sbw_host import fetch_cutout, get_tap_service
from sbw_host import sersic as S
from sbw_host import sersic_mix as SM


def gauss2d(dx, dy, cov):
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]]) / det
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


def psf_components(row, band):
    comps = []
    f1 = float(row[f"{band}_psfModel_TwoGaussian_gauss1_fluxfrac"])
    for k, frac in ((1, f1), (2, 1 - f1)):
        sx, sy, rho = (float(row[f"{band}_psfModel_TwoGaussian_gauss{k}_{c}"]) for c in ("sigma_x", "sigma_y", "rho"))
        comps.append((frac, np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])))
    return comps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--field", type=float, nargs=2, default=(53.13, -28.10))
    ap.add_argument("--band", default="r")
    ap.add_argument("--out", default="examples/residuals.png")
    ap.add_argument("--model", default="sersic", choices=["sersic", "exponential"],
                    help="which MultiProFit fit to test (exponential has n fixed at 1)")
    args = ap.parse_args()
    b = args.band
    M = args.model
    svc = get_tap_service()
    tab = svc.search(
        f"SELECT objectId, {M}_ra, {M}_dec, {M}_reff_x, {M}_reff_y, {M}_rho, "
        + ("sersic_index, " if M == "sersic" else "")
        + f"{b}_{M}Flux, {b}_psfModel_TwoGaussian_gauss1_sigma_x, {b}_psfModel_TwoGaussian_gauss1_sigma_y, "
        f"{b}_psfModel_TwoGaussian_gauss1_rho, {b}_psfModel_TwoGaussian_gauss1_fluxfrac, "
        f"{b}_psfModel_TwoGaussian_gauss2_sigma_x, {b}_psfModel_TwoGaussian_gauss2_sigma_y, {b}_psfModel_TwoGaussian_gauss2_rho "
        f"FROM dp2.Object WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', {args.field[0]}, {args.field[1]}, 0.1)) = 1 "
        f"AND refExtendedness > 0.9 AND {b}_cModelFlux > 15000 AND {M}_reff_major > 0.7 AND {b}_blendedness < 0.05 "
        f"AND deblend_parentNChild <= 6").to_table()
    tab = tab[np.argsort(-np.asarray(tab[f"{b}_{M}Flux"]))][: args.n]
    fig, axes = plt.subplots(len(tab), 3, figsize=(10, 3.2 * len(tab)), squeeze=False)
    chi = {"mixture": [], "analytic": []}
    for i, r in enumerate(tab):
        ra, dec = float(r[f"{M}_ra"]), float(r[f"{M}_dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img, wcs = fetch_cutout(ra, dec, 16.0, band=b, service=svc)
        x0, y0 = wcs.world_to_pixel_values(ra, dec)
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        dx, dy = xx - x0, yy - y0
        bg = np.median(img)
        sig = 1.4826 * np.median(np.abs(img - bg))
        data = img - bg
        F = float(r[f"{b}_{M}Flux"])
        n = float(r["sersic_index"]) if M == "sersic" else 1.0
        rx, ry, rho = (float(r[c]) for c in (f"{M}_reff_x", f"{M}_reff_y", f"{M}_rho"))  # pixels
        cov = np.array([[rx * rx, rho * rx * ry], [rho * rx * ry, ry * ry]])
        psf = psf_components(r, b)
        # mixture: Gaussian (x) Gaussian is analytic
        w, s = SM.mixture(n)
        mix = np.zeros_like(data)
        for wk, sk in zip(w, s):
            for fj, cp in psf:
                mix += wk * fj * F * gauss2d(dx, dy, sk**2 * cov + cp)
        # analytic Sersic, sub-sampled, FFT-convolved with the PSF image
        a_px, b_px, _ = S.ellipse_axes(rx, ry, rho)
        sub = 5
        off = (np.arange(sub) + 0.5) / sub - 0.5
        ana = np.zeros_like(data)
        for ox in off:
            for oy in off:
                m = S.elliptical_radius(dx + ox, dy + oy, rx, ry, rho)
                ana += S.sersic_intensity(m, F, a_px, b_px, n)
        ana /= sub * sub
        pyy, pxx = np.mgrid[0:41, 0:41] - 20.0
        kern = sum(fj * gauss2d(pxx, pyy, cp) for fj, cp in psf)
        ana = fftconvolve(ana, kern, mode="same")
        apert = S.elliptical_radius(dx, dy, rx, ry, rho) < 3.0
        for name, model in (("mixture", mix), ("analytic", ana)):
            c2 = np.sum(((data - model) / sig) ** 2 * apert) / apert.sum()
            chi[name].append(c2)
        print(f"{int(r['objectId'])} n={n:.2f} reff={0.2 * a_px:.2f}\" mag={-2.5 * np.log10(F) + 31.4:.2f}: "
              f"chi2/pix mixture={chi['mixture'][-1]:.2f} analytic={chi['analytic'][-1]:.2f}")
        vmax = np.percentile(data, 99.9)
        axes[i, 0].imshow(data, origin="lower", cmap="gray_r", vmin=-3 * sig, vmax=vmax)
        axes[i, 0].set_title(f"{int(r['objectId'])}  n={n:.2f}  reff={0.2 * a_px:.2f}\"", fontsize=9)
        for j, (name, model) in enumerate((("mixture", mix), ("analytic", ana))):
            axes[i, j + 1].imshow(data - model, origin="lower", cmap="RdBu_r", vmin=-8 * sig, vmax=8 * sig)
            axes[i, j + 1].set_title(f"data - {name}   chi2/pix={chi[name][-1]:.2f}", fontsize=9)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    print(f"[{M}] mean chi2/pix: mixture={np.mean(chi['mixture']):.2f}  analytic={np.mean(chi['analytic']):.2f}  -> {args.out}")


if __name__ == "__main__":
    main()
