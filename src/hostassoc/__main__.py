"""Command-line host association.

    python -m hostassoc RA DEC [--band r] [--radius 30] [--max-hosts N] [--dataset dp2]
                              [--plot out.png] [--csv out.ecsv]

Example:
    python -m hostassoc 53.12934 -28.12088 --band r --radius 30 --plot host.png
"""

from __future__ import annotations

import argparse
import sys
import warnings

from .associate import associate_host


def main(argv=None):
    p = argparse.ArgumentParser(prog="hostassoc", description=__doc__.split("\n\n")[0])
    p.add_argument("ra", type=float, help="transient RA (deg, ICRS)")
    p.add_argument("dec", type=float, help="transient Dec (deg, ICRS)")
    p.add_argument("--band", default="r", choices=list("ugrizy"))
    p.add_argument("--radius", type=float, default=30.0, help="search radius in arcsec")
    p.add_argument("--max-hosts", type=int, default=None, help="keep only the N most probable candidates")
    p.add_argument("--dataset", default="dp2", help="TAP schema: dp2 (default) or dp1")
    p.add_argument("--model", default="exponential", choices=["exponential", "sersic"],
                   help="MultiProFit fit to use: exponential (default, n=1) or sersic (free n)")
    p.add_argument("--exclude-suspect", action="store_true", help="drop candidates with unreliable Sersic fits")
    p.add_argument("--no-band-fallback", action="store_true", help="do not try other bands (r,i,g,z,u,y) when --band has no usable fluxes")
    p.add_argument("--plot", metavar="FILE", default=None, help="save the diagnostic figure to FILE (png/pdf)")
    p.add_argument("--csv", metavar="FILE", default=None, help="save the candidate table (ECSV)")
    p.add_argument("--token", default=None, help="RSP token (else RSP_TOKEN or a .rsp_image_tap_token / .rsp_token file)")
    a = p.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = associate_host(a.ra, a.dec, band=a.band, radius_arcsec=a.radius, max_hosts=a.max_hosts,
                             dataset=a.dataset, exclude_suspect=a.exclude_suspect, token=a.token, model=a.model,
                             band_fallback=not a.no_band_fallback,
                             plot=a.plot or False)
    m = res.meta
    print(f"# ({a.ra:.6f}, {a.dec:.6f}) band={m.get('band_used', a.band)} (requested {a.band}) radius={a.radius}\" {a.dataset} fit={m['model']}: "
          f"{m['n_in_radius']} objects in radius, {m['n_candidates']} candidates, "
          f"{m['n_dropped_pointlike']} point-like dropped, {m['n_dropped_bad_fit']} without fit, "
          f"{m['n_suspect']} suspect; max frac_light_beyond={m['max_frac_light_beyond']:.3g}")
    if len(res):
        cols = ["objectId", "p_sbw", "frac_light_beyond", "sep_arcsec", "m_rhalf", "d_dlr", "mag",
                "rhalf_major_arcsec", "sersic_index", "suspect"]
        res[cols].pprint(max_lines=-1, max_width=-1)
    else:
        print("no usable candidates")
    if a.csv:
        res.write(a.csv, overwrite=True)
        print("wrote", a.csv)
    if a.plot:
        print("wrote", a.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
