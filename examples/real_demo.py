"""Figures for hypothetical transients at interesting positions in real DP2 fields.

Reads the positions selected by ``disagreement_search.py``
(``examples/real_disagreements.json``), runs the full association with a real
deep_coadd cutout for each and writes ``examples/real_XX_<category>_<field>.png``
plus ``examples/real_disagreements.md``.

Usage: python real_demo.py [--json examples/real_disagreements.json] [--only 1,4,7] [--fov 60]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hostassoc import associate_host, get_tap_service
from hostassoc.plot import method_summary, plot_with_best_image

CATEGORY_TEXT = {
    "twins": "two similar galaxies tie in d_DLR (within 20%); SBW splits the light",
    "bright_small": "SBW host >= 1 mag brighter but smaller than the DLR host",
    "elongated": "methods disagree; an elongated galaxy (q < 0.5) is involved",
    "dwarf_proj": "transient in the outskirts of a bright galaxy with a faint dwarf in projection",
    "split": "SBW split 40-60%; DLR decisive",
    "dlr_none": "all d_DLR > 4 (DLR: no host); SBW host has f> > 0.05",
    "swap": "SBW and DLR pick different hosts",
    "weak_dlr": "DLR host gets P_SBW < 0.3",
    "dlr_tie": "two galaxies tie in d_DLR; SBW decisive",
    "sbw_split": "SBW split ~50/50; DLR decisive",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="examples/real_disagreements.json")
    ap.add_argument("--only", default=None, help="comma-separated 1-based indices into the chosen list")
    ap.add_argument("--fov", type=float, default=60.0, help="figure field of view in arcsec")
    ap.add_argument("--radius", type=float, default=30.0)
    ap.add_argument("--outdir", default="examples")
    ap.add_argument("--renumber", action="store_true", help="number the output figures 1..N in --only order")
    a = ap.parse_args()
    chosen = json.load(open(a.json))["chosen"]
    idx = [int(x) - 1 for x in a.only.split(",")] if a.only else list(range(len(chosen)))
    svc = get_tap_service()
    out = pathlib.Path(a.outdir)
    rows = ["| # | category | field | RA | Dec | SBW host | P_SBW | f> | d_DLR(SBW host) | DLR host | d_DLR | P_SBW(DLR host) | figure |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for num, i in enumerate(idx, start=1):
        c = chosen[i]
        if not a.renumber:
            num = i + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, raw = associate_host(c["ra"], c["dec"], band="r", radius_arcsec=a.radius, service=svc, return_raw=True)
            fig, ax, used = plot_with_best_image(res, raw=raw, size_arcsec=a.fov)
        ax.set_title(f"real {num:02d}  {c['field']}  ({c['ra']:.5f}, {c['dec']:.5f})  fit={res.meta['model']}   [{c['category']}: {CATEGORY_TEXT[c['category']]}]\n"
                     f"{method_summary(res)}", fontsize=10)
        fname = out / f"real_{num:02d}_{c['category']}_{c['field']}.png"
        fig.savefig(fname, dpi=110, bbox_inches="tight"); plt.close(fig)
        res.write(out / (fname.stem + ".ecsv"), overwrite=True)
        k = int(np.argmax(res["p_sbw"])); j = int(np.argmin(res["d_dlr"]))
        dlr_host = int(res["objectId"][j]) if res["d_dlr"][j] <= 4 else "none"
        rows.append(f"| {num:02d} | {c['category']} | {c['field']} | {c['ra']:.5f} | {c['dec']:.5f} | {int(res['objectId'][k])} | "
                    f"{res['p_sbw'][k]:.2f} | {res['frac_light_beyond'][k]:.2g} | {res['d_dlr'][k]:.2f} | {dlr_host} | {res['d_dlr'][j]:.2f} | "
                    f"{res['p_sbw'][j]:.2f} | {fname.name} |")
        print(f"real {num:02d} {c['category']:9s} {c['field']:6s} -> {method_summary(res)}  [{'cutout' if used else 'model'}]")
    open(out / "real_disagreements.md", "w").write("\n".join(rows) + "\n")
    print("wrote", out / "real_disagreements.md")


if __name__ == "__main__":
    main()
