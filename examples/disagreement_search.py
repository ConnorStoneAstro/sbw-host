"""Search real DP2 fields for transient positions where SBW and DLR disagree.

For each field, every extended object within 0.04 deg is fetched once.  Trial
transient positions are generated around real galaxies (between pairs, along the
minor axes of elongated galaxies, in the outskirts of bright ones) and both
methods are evaluated locally with ``compute_association`` on the objects within
30".  Disagreements are categorised and a diverse selection is written to
``examples/real_disagreements.json`` for ``real_demo.py``.

Categories (both the SBW host and the DLR host must be non-suspect, n <= 3 and
brighter than r = 24.5, so the picks are driven by geometry, not by the wings of
n ~ 6 fits):
  twins        two galaxies of similar brightness (|dmag| < 0.75) and size (ratio < 1.5)
               whose d_DLR agree within 20% -- DLR cannot decide; SBW reports how the light splits
  bright_small SBW host is >= 1 mag brighter and smaller than the DLR host (size vs brightness)
  elongated    the methods disagree and one of the two is elongated (q < 0.5); the
               transient's angle to its major axis is recorded
  dwarf_proj   transient in the outskirts (1.5-4 R_half) of a bright galaxy with a faint
               dwarf (> 2.5 mag fainter) within 3" -- DLR picks the dwarf, SBW the big one
  dlr_none     all d_DLR > 4 but SBW's host has f> > 0.05
  split        SBW is split (0.4-0.6) while DLR is decisive (second d_DLR > 1.4 x first)
"""

from __future__ import annotations

import json
import warnings

import numpy as np
from astropy.table import Table

from sbw_host import sersic as S
from sbw_host.associate import DEFAULT_MODEL, build_query, compute_association, available_columns, _OPTIONAL_COLUMNS
from sbw_host.tap import get_tap_service

FIELDS = {"ECDFS": (53.13, -28.10), "COSMOS": (150.12, 2.21), "EDFS": (59.10, -48.73)}
RADIUS = 30.0
BAND = "r"
MODEL = DEFAULT_MODEL   # "exponential" or "sersic"
RNG = np.random.default_rng(7)


def fetch_field(service, ra, dec, r_deg=0.04):
    extra = tuple(c for c in _OPTIONAL_COLUMNS if c in available_columns(service, "dp2"))
    q = build_query(ra, dec, r_deg * 3600, BAND, "dp2", extra, model=MODEL)
    tab = service.search(q).to_table()
    out = Table()
    for name in tab.colnames:
        col = tab[name]
        if hasattr(col, "filled"):
            fill = np.nan if col.dtype.kind == "f" else (0 if col.dtype.kind in "iu" else False)
            col = col.filled(fill)
        out[name] = np.asarray(col)
    out.meta["model"] = MODEL
    return out


def local(cands, ra, dec):
    du, dv = S.sky_offsets(ra, dec, cands["coord_ra"], cands["coord_dec"])
    sel = np.hypot(du, dv) < RADIUS
    return compute_association(cands[sel], ra, dec, BAND)


def classify(res):
    """Return (category, features) or (None, None)."""
    if len(res) < 2:
        return None, None
    p = np.asarray(res["p_sbw"]); d = np.asarray(res["d_dlr"]); f = np.asarray(res["frac_light_beyond"])
    mag = np.asarray(res["mag"]); n = np.asarray(res["sersic_index"]); rh = np.asarray(res["rhalf_major_arcsec"])
    q = np.asarray(res["rhalf_minor_arcsec"]) / rh; sep = np.asarray(res["sep_arcsec"])
    phi = np.asarray(res["phi_deg"]); du = np.asarray(res["du_arcsec"]); dv = np.asarray(res["dv_arcsec"])
    k = int(np.argmax(p)); order = np.argsort(d); j = int(order[0]); j2 = int(order[1])
    for i in (k, j):
        if res["suspect"][i] or n[i] > 3.0 or mag[i] > 24.5 or not np.isfinite(d[i]):
            return None, None
    # angle between the SN direction and the galaxy's major axis, folded to [0, 90]
    ang = lambda i: abs((np.degrees(np.arctan2(dv[i], du[i])) - phi[i] + 90) % 180 - 90)
    feat = dict(sbw_n=float(n[k]), dlr_n=float(n[j]), sbw_mag=float(mag[k]), dlr_mag=float(mag[j]),
                sbw_rh=float(rh[k]), dlr_rh=float(rh[j]), sbw_q=float(q[k]), dlr_q=float(q[j]),
                sbw_axis_angle=float(ang(k)), dlr_axis_angle=float(ang(j)), dlr_ratio=float(d[j2] / d[j]))
    dlr_has = d[j] <= 4
    if not dlr_has:
        return ("dlr_none", feat) if (p[k] > 0.9 and f[k] > 0.05 and d[k] < 6) else (None, None)
    ratio = d[j2] / d[j]
    # twins: compare the two DLR-closest galaxies
    a, b = j, j2
    if (abs(mag[a] - mag[b]) < 0.75 and 1 / 1.5 < rh[a] / rh[b] < 1.5 and ratio < 1.2
            and not res["suspect"][b] and n[b] <= 3 and mag[b] < 24.5 and k in (a, b)):
        return "twins", feat
    if j != k:
        if d[k] > 5:          # SBW host far outside its own half-light ellipse: wing-driven, skip
            return None, None
        if mag[k] < 21.5 and rh[k] > 1.2 and 1.5 <= d[k] <= 4 and mag[j] > mag[k] + 2.5 and sep[j] < 3.0:
            return "dwarf_proj", feat
        if mag[k] < mag[j] - 1.0 and rh[k] < rh[j]:
            return "bright_small", feat
        if (q[k] < 0.5 or q[j] < 0.5) and p[k] > 0.7:
            return "elongated", feat
    if 0.4 < p[k] < 0.6 and ratio > 1.4 and d[k] <= 5:
        return "split", feat
    return None, None


def trial_positions(gal):
    """Candidate transient positions (ra, dec, description) built from the galaxy table."""
    good = (gal["refExtendedness"] > 0.5) & np.isfinite(gal[f"{MODEL}_reff_x"]) & (gal[f"{BAND}_{MODEL}Flux"] > 0)
    g = gal[good]
    flux = np.asarray(g[f"{BAND}_{MODEL}Flux"]); mag = -2.5 * np.log10(flux) + 31.4
    rx, ry, rho = (np.asarray(g[c], float) * S.PIXEL_SCALE_ARCSEC for c in (f"{MODEL}_reff_x", f"{MODEL}_reff_y", f"{MODEL}_rho"))
    major, minor, phi = S.ellipse_axes(rx, ry, rho)
    rh = 1.386 * major
    ra, dec = np.asarray(g[f"{MODEL}_ra"]), np.asarray(g[f"{MODEL}_dec"])
    bright = np.where((mag < 23.0) & (rh > 0.3))[0]
    second = np.where((mag < 24.5) & (rh > 0.2))[0]
    pos = []
    cosd = np.cos(np.deg2rad(dec.mean()))
    # pairs
    for a in bright:
        for b in second:
            if b == a or (b in bright and b < a):
                continue
            du = (ra[b] - ra[a]) * cosd * 3600; dv = (dec[b] - dec[a]) * 3600
            sep = np.hypot(du, dv)
            if not (2.0 < sep < 20.0):
                continue
            for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                for perp in (0.0, 1.5, 3.0, -1.5):
                    pu = -dv / sep * perp; pv = du / sep * perp
                    pos.append((ra[a] + (frac * du + pu) / 3600 / cosd, dec[a] + (frac * dv + pv) / 3600,
                                f"pair {int(g['objectId'][a])}-{int(g['objectId'][b])} frac={frac} perp={perp}"))
    # elongated galaxies: along the minor axis at 1.5 and 2.5 R_half; bright big galaxies: outskirts
    for a in bright:
        if minor[a] / major[a] < 0.5 and rh[a] > 0.8:
            for mult in (1.5, 2.5):
                for sign in (1, -1):
                    r = mult * 1.386 * minor[a]
                    du = -sign * r * np.sin(np.deg2rad(phi[a])); dv = sign * r * np.cos(np.deg2rad(phi[a]))
                    pos.append((ra[a] + du / 3600 / cosd, dec[a] + dv / 3600, f"minor-axis {int(g['objectId'][a])} x{mult}"))
        if mag[a] < 21.0 and rh[a] > 1.0:
            for mult in (1.5, 2.0, 3.0, 4.0):
                th = RNG.uniform(0, 2 * np.pi)
                du, dv = mult * rh[a] * np.cos(th), mult * rh[a] * np.sin(th)
                pos.append((ra[a] + du / 3600 / cosd, dec[a] + dv / 3600, f"outskirts {int(g['objectId'][a])} x{mult}"))
    return pos


def main():
    svc = get_tap_service()
    found = {}
    for field, (fra, fdec) in FIELDS.items():
        gal = fetch_field(svc, fra, fdec)
        pos = trial_positions(gal)
        print(f"{field}: {len(gal)} objects, {len(pos)} trial positions")
        n_cat = {}
        for ra, dec, how in pos:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = local(gal, float(ra), float(dec))
            cat, feat = classify(res)
            if cat is None:
                continue
            n_cat[cat] = n_cat.get(cat, 0) + 1
            p = np.asarray(res["p_sbw"]); d = np.asarray(res["d_dlr"]); k = int(np.argmax(p)); j = int(np.argmin(d))
            rec = dict(field=field, ra=float(ra), dec=float(dec), how=how, category=cat,
                       sbw_id=int(res["objectId"][k]), p_sbw=float(p[k]), f_beyond=float(res["frac_light_beyond"][k]),
                       sbw_dlr=float(d[k]), dlr_id=int(res["objectId"][j]), d_dlr=float(d[j]), p_sbw_of_dlr=float(p[j]),
                       n_cand=len(res), mags=[float(res["mag"][k]), float(res["mag"][j])],
                       n_ser=[float(res["sersic_index"][k]), float(res["sersic_index"][j])],
                       rhalf=[float(res["rhalf_major_arcsec"][k]), float(res["rhalf_major_arcsec"][j])], **feat)
            found.setdefault(cat, []).append(rec)
        print("   ", n_cat)
    # diversify: up to 2 per category, different fields / galaxies where possible
    chosen = []
    used_ids = set()
    for cat in ("twins", "bright_small", "elongated", "dwarf_proj", "split", "dlr_none"):
        recs = found.get(cat, [])
        used_fields = []
        # prefer the clearest cases
        key = {"twins": lambda r: -abs(r["p_sbw"] - 0.5),          # most decisive SBW split of a DLR tie
               "bright_small": lambda r: -r["p_sbw"],
               "elongated": lambda r: -r["p_sbw"],
               "dwarf_proj": lambda r: -r["p_sbw"],
               "split": lambda r: abs(r["p_sbw"] - 0.5),
               "dlr_none": lambda r: -r["f_beyond"]}[cat]
        for r in sorted(recs, key=key):
            ids = {r["sbw_id"], r["dlr_id"]}
            if ids & used_ids or (used_fields.count(r["field"]) >= 2 and len(set(x["field"] for x in recs)) > 1):
                continue
            chosen.append(r); used_ids |= ids; used_fields.append(r["field"])
            if len([c for c in chosen if c["category"] == cat]) >= 3:
                break
    json.dump({"model": MODEL, "found_counts": {k: len(v) for k, v in found.items()}, "chosen": chosen, "all": found},
              open("examples/real_disagreements.json", "w"), indent=1)
    print("chosen", len(chosen))
    for r in chosen:
        print(f"  {r['category']:12s} {r['field']:6s} ({r['ra']:.5f},{r['dec']:.5f}) {r['how'][:40]:40s} "
              f"SBW->{r['sbw_id']} P={r['p_sbw']:.2f} f>={r['f_beyond']:.2g} d={r['sbw_dlr']:.2f} "
              f"[r={r['sbw_mag']:.1f} n={r['sbw_n']:.1f} Rh={r['sbw_rh']:.1f} q={r['sbw_q']:.2f} ang={r['sbw_axis_angle']:.0f}] | "
              f"DLR->{r['dlr_id']} d={r['d_dlr']:.2f} P_SBW={r['p_sbw_of_dlr']:.2f} "
              f"[r={r['dlr_mag']:.1f} n={r['dlr_n']:.1f} Rh={r['dlr_rh']:.1f} q={r['dlr_q']:.2f} ang={r['dlr_axis_angle']:.0f}] ratio={r['dlr_ratio']:.2f}")


if __name__ == "__main__":
    main()
