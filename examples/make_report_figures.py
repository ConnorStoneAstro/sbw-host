"""Figures for the report on the Object-table Sersic size convention.

For isolated bright DP2 galaxies, compare three PSF-convolved models built from the
catalogue parameters against the real r-band deep_coadd cutout:

  A. AstroPhot Sersic, taken naively: R_e = sersic_reff_major, n = sersic_index,
     q = reff_minor/reff_major, flux = r_sersicFlux
  B. AstroPhot Sersic with R_e = (mixture half-light factor ~1.39) x sersic_reff_major
  C. The Gaussian-mixture "Sersic" actually used by MultiProFit / gauss2d_fit:
     4 Gaussians, sigma_k = knot_k(n) * sersic_reff  (sbw_host.sersic_mix)

The PSF is the two-Gaussian PSF model stored in the Object table
({band}_psfModel_TwoGaussian_*), i.e. the PSF MultiProFit convolved with.

Outputs (in report/): residual_grid.png, radial_profiles.png, profile_shapes.png,
chi2_summary.json
"""

from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gammaincinv, gammaln

import astrophot as ap
from sbw_host import fetch_cutout, get_tap_service
from sbw_host import sersic as S
from sbw_host import sersic_mix as SM

OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "sersic_reff_report"
PIX = S.PIXEL_SCALE_ARCSEC
GALAXIES = [755369573523391944, 755369573523418740, 755369573523414208,
            755369642242894509, 755369573523420748, 755369573523416335]
BAND = "r"


def gauss2d(dx, dy, cov):
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] ** 2
    inv = np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]]) / det
    q = inv[0, 0] * dx * dx + 2 * inv[0, 1] * dx * dy + inv[1, 1] * dy * dy
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det))


def psf_components(row, band):
    f1 = float(row[f"{band}_psfModel_TwoGaussian_gauss1_fluxfrac"])
    comps = []
    for k, frac in ((1, f1), (2, 1 - f1)):
        sx, sy, rho = (float(row[f"{band}_psfModel_TwoGaussian_gauss{k}_{c}"]) for c in ("sigma_x", "sigma_y", "rho"))
        comps.append((frac, np.array([[sx * sx, rho * sx * sy], [rho * sx * sy, sy * sy]])))
    return comps


def psf_image(psf, size=41):
    yy, xx = np.mgrid[0:size, 0:size] - (size - 1) / 2.0
    k = sum(f * gauss2d(xx, yy, c) for f, c in psf)
    return k / k.sum()


def sersic_Ie(flux, re_arcsec, q, n):
    b = gammaincinv(2 * n, 0.5)
    return flux / np.exp(np.log(2 * np.pi * n) + 2 * np.log(re_arcsec) + np.log(q) + b - 2 * n * np.log(b) + gammaln(2 * n))


def astrophot_sersic(data, psf_img, x0, y0, re_arcsec, q, phi_deg, n, flux):
    """PSF-convolved AstroPhot Sersic image. phi is CCW from +x(col) toward +y(row)."""
    target = ap.TargetImage(data=data, pixelscale=PIX, psf=ap.PSFImage(data=psf_img, upsample=1))
    pa = np.deg2rad((phi_deg - 90.0) % 180.0)   # AstroPhot PA: from +y toward -x
    model = ap.Model(model_type="sersic galaxy model", target=target,
                     center=[x0 * PIX, y0 * PIX], q=q, PA=pa, n=n, Re=re_arcsec,
                     Ie=sersic_Ie(flux, re_arcsec, q, n), psf_convolve=True)
    model.initialize()
    d = model().data
    d = d.value if hasattr(d, "value") else d
    d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    return np.array(d, dtype=float)


def mixture_image(dx, dy, cov_pix, n, flux, psf):
    w, s = SM.mixture(n)
    img = np.zeros_like(dx, dtype=float)
    for wk, sk in zip(w, s):
        for fj, cp in psf:
            img += wk * fj * flux * gauss2d(dx, dy, sk**2 * cov_pix + cp)
    return img


def moments_angle(img, x0, y0, rmax):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    w = np.where(np.hypot(xx - x0, yy - y0) < rmax, np.clip(img, 0, None), 0)
    dx, dy = xx - x0, yy - y0
    mxx, myy, mxy = (w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum()
    return 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))


def elliptical_profile(img, m, edges):
    prof = np.zeros(len(edges) - 1)
    err = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        prof[i] = img[sel].mean() if sel.any() else np.nan
        err[i] = img[sel].std() / np.sqrt(max(sel.sum(), 1)) if sel.any() else np.nan
    return prof, err


def main():
    ap.config.DTYPE = torch.float64
    svc = get_tap_service()
    b = BAND
    tab = svc.search(
        f"SELECT objectId, sersic_ra, sersic_dec, sersic_reff_x, sersic_reff_y, sersic_rho, sersic_index, "
        f"sersic_reff_major, sersic_reff_minor, {b}_sersicFlux, {b}_cModelFlux, "
        f"{b}_psfModel_TwoGaussian_gauss1_sigma_x, {b}_psfModel_TwoGaussian_gauss1_sigma_y, "
        f"{b}_psfModel_TwoGaussian_gauss1_rho, {b}_psfModel_TwoGaussian_gauss1_fluxfrac, "
        f"{b}_psfModel_TwoGaussian_gauss2_sigma_x, {b}_psfModel_TwoGaussian_gauss2_sigma_y, {b}_psfModel_TwoGaussian_gauss2_rho "
        f"FROM dp2.Object WHERE objectId IN ({', '.join(map(str, GALAXIES))})").to_table()
    rows = [tab[tab["objectId"] == g][0] for g in GALAXIES]

    labels = ["A: AstroPhot Sérsic\nR_e = sersic_reff (naive)",
              "B: AstroPhot Sérsic\nR_e = 1.39 × sersic_reff",
              "C: gauss2d_fit mixture\nσ_k = knot_k(n) × sersic_reff"]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.6 * len(rows)))
    summary = []
    profiles = []
    for i, r in enumerate(rows):
        oid = int(r["objectId"])
        ra, dec = float(r["sersic_ra"]), float(r["sersic_dec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img, wcs = fetch_cutout(ra, dec, 16.0, band=b, service=svc)
        assert wcs.pixel_scale_matrix[0, 0] < 0 and wcs.pixel_scale_matrix[1, 1] > 0  # +x West, +y North
        x0, y0 = (float(v) for v in wcs.world_to_pixel_values(ra, dec))
        ny, nx = img.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        dx, dy = xx - x0, yy - y0
        bg = np.median(img)
        sig = 1.4826 * np.median(np.abs(img - bg))
        data = img - bg
        F, n = float(r[f"{b}_sersicFlux"]), float(r["sersic_index"])
        rx, ry, rho = (float(r[c]) for c in ("sersic_reff_x", "sersic_reff_y", "sersic_rho"))   # pixels
        cov = np.array([[rx * rx, rho * rx * ry], [rho * rx * ry, ry * ry]])
        a_px, b_px, phi = S.ellipse_axes(rx, ry, rho)
        q = b_px / a_px
        re_cat = a_px * PIX                      # catalogue "effective radius", arcsec
        hl = float(SM.half_light_factor(n))      # mixture half-light radius / sersic_reff
        psf = psf_components(r, b)
        pimg = psf_image(psf)

        models = {
            "A": astrophot_sersic(data, pimg, x0, y0, re_cat, q, phi, n, F),
            "B": astrophot_sersic(data, pimg, x0, y0, hl * re_cat, q, phi, n, F),
            "C": mixture_image(dx, dy, cov, n, F, psf),
        }
        # orientation sanity check: AstroPhot model vs mixture model must share the major-axis angle
        angA, angC = moments_angle(models["A"], x0, y0, 3 * a_px), moments_angle(models["C"], x0, y0, 3 * a_px)
        dang = abs((angA - angC + 90) % 180 - 90)
        m_ell = S.elliptical_radius(dx, dy, rx, ry, rho)
        apert = m_ell < 3.0
        chis = {k: float(np.sum(((data - v) / sig) ** 2 * apert) / apert.sum()) for k, v in models.items()}
        fluxes = {k: float(v[apert].sum() / data[apert].sum()) for k, v in models.items()}
        summary.append({"objectId": oid, "n": n, "reff_cat_arcsec": re_cat, "half_light_factor": hl,
                        "q": q, "mag": -2.5 * np.log10(F) + 31.4, "sersic_over_cmodel": F / float(r[f"{b}_cModelFlux"]),
                        "chi2_per_pix": chis, "model_over_data_flux_3reff": fluxes, "pa_check_deg": dang,
                        "sigma_pix_nJy": sig})
        print(f"{oid} n={n:.2f} reff_cat={re_cat:.2f}\" hl={hl:.3f} mag={summary[-1]['mag']:.2f} | chi2/pix "
              f"A={chis['A']:.1f} B={chis['B']:.1f} C={chis['C']:.1f} | PA check {dang:.1f} deg")

        vmax = np.percentile(data, 99.9)
        axes[i, 0].imshow(data, origin="lower", cmap="gray_r", vmin=-3 * sig, vmax=vmax)
        axes[i, 0].set_title(f"objectId {oid}\nn={n:.2f}  sersic_reff={re_cat:.2f}\"  r={summary[-1]['mag']:.1f}", fontsize=9)
        for j, k in enumerate("ABC"):
            axes[i, j + 1].imshow((data - models[k]) / sig, origin="lower", cmap="RdBu_r", vmin=-8, vmax=8)
            axes[i, j + 1].set_title(f"{labels[j]}\nχ²/pix = {chis[k]:.1f}", fontsize=9)
        for a in axes[i]:
            a.set_xticks([]); a.set_yticks([])

        # radial (elliptical-annulus) profiles in the catalogue-ellipse frame
        edges = np.concatenate([[0], np.geomspace(0.35, 8.0, 18)])
        prof = {"data": elliptical_profile(data, m_ell, edges)}
        for k, v in models.items():
            prof[k] = elliptical_profile(v, m_ell, edges)
        profiles.append((oid, n, re_cat, hl, edges, prof, sig))

    fig.suptitle("DP2 r-band deep_coadd minus PSF-convolved model built from Object-table Sérsic parameters "
                 "(residuals in units of pixel noise, ±8σ)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / "residual_grid.png", dpi=90)
    plt.close(fig)

    # radial profile figure
    ncol = 3
    nrow = int(np.ceil(len(profiles) / ncol))
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(3.9 * ncol, 4.6 * nrow), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.3] * nrow})
    cols = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c"}
    for j, (oid, n, re_cat, hl, edges, prof, sig) in enumerate(profiles):
        mid = 0.5 * (edges[1:] + edges[:-1]) * re_cat   # arcsec along the catalogue ellipse
        rr, cc = divmod(j, ncol)
        ax, axr = axes[2 * rr, cc], axes[2 * rr + 1, cc]
        d, de = prof["data"]
        ax.errorbar(mid, d / PIX**2, yerr=de / PIX**2, fmt="ko", ms=3, label="data")
        for k in "ABC":
            ax.plot(mid, prof[k][0] / PIX**2, color=cols[k], lw=1.6, label={"A": "A: Sérsic, R_e = reff", "B": "B: Sérsic, R_e = 1.39 reff", "C": "C: gauss2d mixture"}[k])
            axr.plot(mid, (d - prof[k][0]) / de, color=cols[k], lw=1.4)
        ax.axvline(re_cat, color="0.5", ls=":", lw=1)
        ax.axvline(hl * re_cat, color="0.5", ls="--", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(max(0.5 * sig / PIX**2, 1e-3 * np.nanmax(d) / PIX**2), 2 * np.nanmax(d) / PIX**2)
        ax.set_title(f"{oid}\nn={n:.2f}  reff={re_cat:.2f}\"", fontsize=9)
        axr.axhline(0, color="k", lw=0.8)
        axr.set_ylim(-12, 12)
        axr.set_xlabel("major-axis radius [arcsec]")
        if cc == 0:
            ax.set_ylabel("mean surface brightness [nJy / arcsec²]")
            axr.set_ylabel("(data − model) / σ")
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper right")
    fig.suptitle("Elliptical-annulus profiles: data vs. PSF-convolved models from the catalogue parameters\n"
                 "(vertical dotted line: sersic_reff;  dashed line: 1.39 × sersic_reff)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "radial_profiles.png", dpi=90)
    plt.close(fig)

    # unconvolved profile shapes: analytic Sersic vs mixture
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    m = np.geomspace(0.02, 40, 400)
    for n, c in ((1.0, "C0"), (4.0, "C1"), (6.0, "C3")):
        hl = float(SM.half_light_factor(n))
        I_an = S.sersic_intensity(m, 1.0, 1.0, 1.0, n)
        I_mix = np.exp(SM.log_intensity(m, 1.0, 1.0, 1.0, n))
        axes[0].plot(m, I_an, color=c, ls="--", lw=1.4, label=f"analytic Sérsic n={n:g}, R_e = reff")
        axes[0].plot(m, I_mix, color=c, lw=1.8, label=f"gauss2d mixture n={n:g} (R_half = {hl:.2f} reff)")
        axes[1].plot(m, S.frac_light_beyond(m, n), color=c, ls="--", lw=1.4)
        axes[1].plot(m, SM.frac_light_beyond(m, n), color=c, lw=1.8)
    for a in axes:
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("elliptical radius m  [units of catalogue sersic_reff]")
        a.axvline(1.0, color="0.6", ls=":", lw=1)
    axes[0].set_ylabel("I(m) for unit total flux  [1/reff²]"); axes[0].set_ylim(1e-12, 1e3)
    axes[0].legend(fontsize=7.5)
    axes[1].set_ylabel("fraction of total light beyond m"); axes[1].set_ylim(1e-12, 1.5)
    axes[1].set_title("dashed: analytic Sérsic   solid: 4-Gaussian mixture (as fitted)", fontsize=10)
    axes[0].set_title("Unconvolved profiles for the same catalogue parameters", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "profile_shapes.png", dpi=100)
    plt.close(fig)

    json.dump(summary, open(OUT / "chi2_summary.json", "w"), indent=1)
    mean = {k: float(np.mean([s["chi2_per_pix"][k] for s in summary])) for k in "ABC"}
    print("mean chi2/pix:", mean)


if __name__ == "__main__":
    main()
