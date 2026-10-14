"""Run host association on a handful of random positions and save diagnostic figures.

Usage:  python run_examples.py [--band r] [--radius 30] [--n-random 2] [--seed 42]

For each DP2 field it (a) draws random bright galaxies and places a fake SN at a
random offset of 0.3-2.5 R_eff, and (b) draws purely random positions.  Figures go
to ``examples/`` and a summary table to ``examples/summary.md``.
"""

from __future__ import annotations

import argparse
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np

from hostassoc import associate_host, get_tap_service
from hostassoc.plot import method_summary, plot_with_best_image

FIELDS = {"ECDFS": (53.13, -28.10), "COSMOS": (150.12, 2.21), "EDFS": (59.10, -48.73)}


def random_galaxies(service, ra, dec, n, rng, dataset="dp2", r_deg=0.03):
    tab = service.search(
        "SELECT objectId, sersic_ra, sersic_dec, sersic_reff_x, sersic_reff_y, sersic_rho, r_cModelFlux "
        f"FROM {dataset}.Object WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        f"CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1 AND refExtendedness > 0.9 AND r_cModelFlux > 5000 "
        "AND sersic_reff_major > 0.7").to_table()
    idx = rng.choice(len(tab), size=min(n, len(tab)), replace=False)
    return tab[idx]


def offset_position(row, rng):
    """Random point at 0.3-2.5 R_eff from the galaxy along a random direction."""
    from hostassoc import sersic as S
    rx, ry, rho = (float(row[c]) * S.PIXEL_SCALE_ARCSEC for c in ("sersic_reff_x", "sersic_reff_y", "sersic_rho"))
    theta = rng.uniform(0, 2 * np.pi)
    m_target = rng.uniform(0.3, 2.5)
    # scale a unit direction so its elliptical radius equals m_target
    du, dv = np.cos(theta), np.sin(theta)
    m_unit = S.elliptical_radius(du, dv, rx, ry, rho)
    du, dv = du * m_target / m_unit, dv * m_target / m_unit
    ra0, dec0 = float(row["sersic_ra"]), float(row["sersic_dec"])
    return ra0 - du / 3600 / np.cos(np.deg2rad(dec0)), dec0 + dv / 3600, m_target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="r")
    p.add_argument("--radius", type=float, default=30.0)
    p.add_argument("--n-galaxy", type=int, default=2, help="offset-from-galaxy cases per field")
    p.add_argument("--n-random", type=int, default=1, help="uniform random cases per field")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", default="dp2")
    p.add_argument("--outdir", default="examples")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = pathlib.Path(args.outdir)
    out.mkdir(exist_ok=True)
    service = get_tap_service(dataset=args.dataset)
    rows = []
    case = 0
    for field, (fra, fdec) in FIELDS.items():
        cases = []
        for g in random_galaxies(service, fra, fdec, args.n_galaxy, rng, args.dataset):
            ra, dec, m = offset_position(g, rng)
            cases.append((ra, dec, f"{m:.2f} Re from {int(g['objectId'])}", int(g["objectId"])))
        for _ in range(args.n_random):
            ra = fra + rng.uniform(-0.02, 0.02) / np.cos(np.deg2rad(fdec))
            dec = fdec + rng.uniform(-0.02, 0.02)
            cases.append((ra, dec, "uniform random", None))
        for ra, dec, how, truth in cases:
            case += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res, raw = associate_host(ra, dec, band=args.band, radius_arcsec=args.radius,
                                          dataset=args.dataset, service=service, return_raw=True)
                fig, ax, used_cutout = plot_with_best_image(res, raw=raw)
                ax.set_title(f"case {case:02d}  {field}  ({ra:.5f}, {dec:.5f})  {how}\n"
                             f"band={args.band}  fit={res.meta['model']}  {res.meta['n_candidates']} candidates  "
                             f"[{'deep_coadd cutout' if used_cutout else 'Sersic model render'}]\n"
                             f"{method_summary(res)}", fontsize=10.5)
            fname = out / f"case_{case:02d}_{field}.png"
            fig.savefig(fname, dpi=110, bbox_inches="tight")
            matplotlib.pyplot.close(fig)
            top = res[0] if len(res) else None
            recovered = (truth is not None and top is not None and int(top["objectId"]) == truth)
            rows.append((case, field, ra, dec, how, res.meta["n_candidates"],
                         int(top["objectId"]) if top is not None else None,
                         float(top["p_sbw"]) if top is not None else np.nan,
                         float(top["sep_arcsec"]) if top is not None else np.nan,
                         "yes" if recovered else ("n/a" if truth is None else "NO"), fname.name))
            print(f"case {case:02d} {field:7s} ({ra:.5f},{dec:.5f}) {how:28s} n={res.meta['n_candidates']:3d} "
                  f"top={rows[-1][6]} P={rows[-1][7]:.3f} sep={rows[-1][8]:.1f}\" recovered={rows[-1][9]} "
                  f"{'cutout' if used_cutout else 'model'}")
            res.write(out / f"case_{case:02d}_{field}.ecsv", overwrite=True)

    with open(out / "summary.md", "w") as f:
        f.write("| case | field | RA | Dec | setup | n cand | top objectId | P(top) | sep (\") | recovered | figure |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]:02d} | {r[1]} | {r[2]:.5f} | {r[3]:.5f} | {r[4]} | {r[5]} | {r[6]} | {r[7]:.3f} | {r[8]:.1f} | {r[9]} | {r[10]} |\n")
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()
