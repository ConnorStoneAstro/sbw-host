import numpy as np
from astropy.table import Table


def make_cands(rows, band="r"):
    cols = {
        "objectId": [], "coord_ra": [], "coord_dec": [], "sersic_ra": [], "sersic_dec": [],
        "sersic_reff_x": [], "sersic_reff_y": [], "sersic_rho": [], "sersic_index": [],
        "refExtendedness": [], "parentObjectId": [], "sersic_no_data_flag": [],
        "sersic_unknown_flag": [], f"{band}_sersicFlux": [], f"{band}_sersicFluxErr": [],
        f"{band}_cModelFlux": [], f"{band}_extendedness": [],
    }
    for i, r in enumerate(rows):
        cols["objectId"].append(1000 + i)
        cols["coord_ra"].append(r["ra"]); cols["coord_dec"].append(r["dec"])
        cols["sersic_ra"].append(r.get("sra", r["ra"])); cols["sersic_dec"].append(r.get("sdec", r["dec"]))
        cols["sersic_reff_x"].append(r.get("rx", 5.0)); cols["sersic_reff_y"].append(r.get("ry", 5.0))
        cols["sersic_rho"].append(r.get("rho", 0.0)); cols["sersic_index"].append(r.get("n", 1.0))
        cols["refExtendedness"].append(r.get("ext", 1.0)); cols["parentObjectId"].append(0)
        cols["sersic_no_data_flag"].append(r.get("nodata", False)); cols["sersic_unknown_flag"].append(False)
        cols[f"{band}_sersicFlux"].append(r.get("flux", 1e4)); cols[f"{band}_sersicFluxErr"].append(10.0)
        cols[f"{band}_cModelFlux"].append(r.get("flux", 1e4)); cols[f"{band}_extendedness"].append(r.get("ext", 1.0))
    t = Table(cols)
    t.meta["model"] = "sersic"
    return t


RA0, DEC0 = 53.1, -28.1
ASEC = 1 / 3600.0
