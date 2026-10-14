"""Visual review of SBW / DLR disagreements in a batch output table.

    python -m sbw_host.review hosts.csv --token .rsp_image_tap_token --n 20 --out review/

Reads the wide table written by ``associate_many`` / ``python -m sbw_host.batch``
(csv, ecsv, fits, parquet, ...), selects random rows where the DLR candidate
(``dlr_host_objectId``) differs from the SBW top host (``host_1_objectId``),
downloads a deep_coadd cutout around each transient and saves one figure per
transient with the transient position, the SBW-ranked hosts (numbered, with
P_SBW / f> / d_DLR) and the DLR host (square marker) highlighted.  A summary table
of the reviewed rows is written next to the figures.

Only the positions and numbers in the table are used, so the figures reflect
exactly the catalogue the user has in hand; ``--full`` re-runs the association for
each selected transient and draws the complete diagnostic figure (all candidates,
half-light ellipses) instead.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

import numpy as np
from astropy.table import Table
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

from .images import ImageAccessError, fetch_cutout
from .plot import CAND_COLORS, _place_labels_outside_ring, plot_with_best_image
from .tap import get_tap_service

DLR_MAX = 4.0


def _set_token(token: str | None):
    """Accept a token string or a path to a token file; export it for the loaders."""
    if not token:
        return
    p = pathlib.Path(token).expanduser()
    tok = p.read_text().strip() if p.is_file() else token.strip()
    os.environ["RSP_TOKEN"] = tok
    os.environ["RSP_IMAGE_TOKEN"] = tok


def host_columns(tab: Table) -> list[int]:
    """Host ranks i present as ``host_i_objectId`` columns."""
    ranks = []
    for c in tab.colnames:
        if c.startswith("host_") and c.endswith("_objectId"):
            try:
                ranks.append(int(c.split("_")[1]))
            except ValueError:
                pass
    return sorted(ranks)


def select_disagreements(tab: Table, n: int = 20, seed: int = 0, include_dlr_none: bool = False,
                         dlr_max: float = DLR_MAX) -> Table:
    """Random rows where the DLR candidate is not the SBW top host.

    By default both methods must have a host: ``host_1_objectId > 0`` and
    ``dlr_host_d_dlr <= dlr_max``.  With ``include_dlr_none`` rows where DLR finds
    nothing but SBW has a host are included too.
    """
    sbw = np.asarray(tab["host_1_objectId"])
    dlr = np.asarray(tab["dlr_host_objectId"])
    d = np.asarray(tab["dlr_host_d_dlr"], float)
    has_sbw = sbw > 0
    dlr_ok = np.isfinite(d) & (d <= dlr_max) & (dlr > 0)
    mask = has_sbw & ((dlr_ok & (dlr != sbw)) | (include_dlr_none & ~dlr_ok))
    idx = np.flatnonzero(mask)
    rng = np.random.default_rng(seed)
    if len(idx) > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    out = tab[idx]
    out.meta["n_disagreements_total"] = int(mask.sum())
    return out


def review_figure(row, ranks, fov_arcsec=60.0, band="r", dataset="dp2", service=None, dlr_max=DLR_MAX):
    """Cutout with the transient, SBW hosts and DLR host marked (from table values only)."""
    import matplotlib.pyplot as plt

    ra0, dec0 = float(row["transient_ra"]), float(row["transient_dec"])
    img, wcs = fetch_cutout(ra0, dec0, fov_arcsec, band=band, dataset=dataset, service=service)
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection=wcs)
    finite = img[np.isfinite(img)]
    norm = ImageNormalize(img, interval=PercentileInterval(99.7), stretch=AsinhStretch(0.05),
                          vmin=np.percentile(finite, 1) if finite.size else 0.0)
    ax.imshow(img, origin="lower", cmap="gray_r", norm=norm)
    ny, nx = img.shape
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
    tr = ax.get_transform("world")
    ax.plot(ra0, dec0, transform=tr, marker="+", ms=18, mew=2.2, color="red", ls="none", label="transient")

    labels, table_lines = [], []
    dlr_id = int(row["dlr_host_objectId"])
    d_dlr = float(row["dlr_host_d_dlr"])
    dlr_has = dlr_id > 0 and np.isfinite(d_dlr) and d_dlr <= dlr_max
    for k, i in enumerate(ranks):
        oid = int(row[f"host_{i}_objectId"])
        if oid <= 0:
            continue
        col = CAND_COLORS[k % len(CAND_COLORS)]
        hra, hdec = float(row[f"host_{i}_ra"]), float(row[f"host_{i}_dec"])
        ax.plot(hra, hdec, transform=tr, marker="x", ms=10, mew=2, color=col, ls="none")
        tag = "  = DLR host" if oid == dlr_id and dlr_has else ""
        sus = "  [suspect]" if bool(row[f"host_{i}_suspect"]) else ""
        txt = (f"#{i}  P_SBW={row[f'host_{i}_p_sbw']:.3f}  f>={row[f'host_{i}_frac_light_beyond']:.2g}{sus}\n"
               f"d_DLR={row[f'host_{i}_d_dlr']:.2f}{tag}")
        px, py = wcs.world_to_pixel_values(hra, hdec)
        labels.append((float(px), float(py), txt, col))
        table_lines.append(f"#{i} id={oid}  P_SBW={row[f'host_{i}_p_sbw']:.3f}  f>={row[f'host_{i}_frac_light_beyond']:8.2g}  "
                           f"d_DLR={row[f'host_{i}_d_dlr']:5.2f}  sep={row[f'host_{i}_sep_arcsec']:4.1f}\"  {band}={row[f'host_{i}_mag']:.1f}{sus}")
    if dlr_has:
        dra, ddec = float(row["dlr_host_ra"]), float(row["dlr_host_dec"])
        ax.plot(dra, ddec, transform=tr, marker="s", ms=14, mfc="none", mec="deepskyblue", mew=2, ls="none",
                label=f"DLR host (d_DLR={d_dlr:.2f})")
        if dlr_id not in [int(row[f"host_{i}_objectId"]) for i in ranks]:
            px, py = wcs.world_to_pixel_values(dra, ddec)
            labels.append((float(px), float(py), f"DLR host\nd_DLR={d_dlr:.2f}  (not in top {len(ranks)} SBW)", "deepskyblue"))
            table_lines.append(f"DLR id={dlr_id}  d_DLR={d_dlr:.2f}  (outside the {len(ranks)} SBW hosts in the table)")
    else:
        table_lines.append(f"DLR: no host (min d_DLR={d_dlr:.1f} > {dlr_max:g})" if np.isfinite(d_dlr) else "DLR: no candidates")
    _place_labels_outside_ring(ax, wcs, labels, ra0, dec0, 0.37 * fov_arcsec, img.shape)
    if table_lines:
        ax.text(0.01, 0.01, "\n".join(table_lines), transform=ax.transAxes, fontsize=8.5, family="monospace",
                va="bottom", ha="left", bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))
    ax.coords[0].set_axislabel("RA"); ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_major_formatter("hh:mm:ss.s"); ax.coords[1].set_major_formatter("dd:mm:ss")
    sbw_txt = f"SBW host: #1 (P_SBW={row['host_1_p_sbw']:.2f})"
    dlr_txt = f"DLR host: id {dlr_id} (d_DLR={d_dlr:.2f})" if dlr_has else "DLR host: none"
    ax.set_title(f"{row['transient_key']}  ({ra0:.5f}, {dec0:.5f})  {row['n_candidates']} candidates\n{sbw_txt}      {dlr_txt}", fontsize=10.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(prog="sbw-host-review", description=__doc__.split("\n\n")[1])
    p.add_argument("table", help="wide output table from sbw_host.batch (csv, ecsv, fits, parquet, ...)")
    p.add_argument("--token", default=None, help="RSP token string or path to a token file (needs read:tap and read:image)")
    p.add_argument("--n", type=int, default=20, help="number of random disagreements to review")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="review", help="output directory")
    p.add_argument("--fov", type=float, default=60.0, help="cutout size in arcsec")
    p.add_argument("--band", default=None, help="band for the cutout (default: the table's band, else r)")
    p.add_argument("--dataset", default=None, help="default: the table's dataset, else dp2")
    p.add_argument("--dlr-max", type=float, default=DLR_MAX)
    p.add_argument("--include-dlr-none", action="store_true", help="also review rows where DLR finds no host")
    p.add_argument("--full", action="store_true", help="re-run the association and draw the full diagnostic figure")
    a = p.parse_args(argv)
    _set_token(a.token)
    tab = Table.read(a.table)
    band = a.band or tab.meta.get("band", "r")
    dataset = a.dataset or tab.meta.get("dataset", "dp2")
    ranks = host_columns(tab)
    if not ranks or "dlr_host_objectId" not in tab.colnames:
        sys.exit("input does not look like a sbw_host.batch wide table (needs host_i_objectId and dlr_host_objectId)")
    sel = select_disagreements(tab, n=a.n, seed=a.seed, include_dlr_none=a.include_dlr_none, dlr_max=a.dlr_max)
    print(f"{sel.meta['n_disagreements_total']} disagreements in {len(tab)} rows; reviewing {len(sel)}")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    svc = get_tap_service(dataset=dataset)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = []
    for row in sel:
        key = str(row["transient_key"]).replace("/", "_")
        fname = out / f"disagree_{key}.png"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if a.full:
                    from .associate import associate_host
                    res, raw = associate_host(float(row["transient_ra"]), float(row["transient_dec"]), band=band,
                                              radius_arcsec=float(tab.meta.get("radius_arcsec", 30.0)), dataset=dataset,
                                              model=tab.meta.get("model", "exponential"), service=svc, return_raw=True)
                    fig, ax, _ = plot_with_best_image(res, raw=raw, size_arcsec=a.fov)
                    ax.set_title(f"{row['transient_key']}\n" + ax.get_title(), fontsize=10)
                else:
                    fig = review_figure(row, ranks, fov_arcsec=a.fov, band=band, dataset=dataset, service=svc, dlr_max=a.dlr_max)
            fig.savefig(fname, dpi=110, bbox_inches="tight"); plt.close(fig)
            names.append(fname.name)
            print("wrote", fname)
        except ImageAccessError as e:
            names.append("")
            print(f"{row['transient_key']}: cutout failed: {e}")
    sel["figure"] = names
    sel.write(out / "reviewed.ecsv", overwrite=True)
    print("summary:", out / "reviewed.ecsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
