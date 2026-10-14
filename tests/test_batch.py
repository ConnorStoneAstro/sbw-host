import numpy as np
import pytest
from astropy.table import Table

from sbw_host import batch as B
from sbw_host.associate import compute_association
from .helpers import ASEC, DEC0, RA0, make_cands


def test_group_positions_cells_and_singletons():
    rng = np.random.default_rng(0)
    ra = np.r_[150.0 + rng.uniform(-0.05, 0.05, 20), 53.1, 59.1, 59.1 + 0.001]
    dec = np.r_[2.0 + rng.uniform(-0.05, 0.05, 20), -28.1, -48.7, -48.7]
    groups = B.group_positions(ra, dec, cell_deg=0.3, radius_arcsec=30)
    regional = [g for g in groups if g[3] is not None]
    singles = [g for g in groups if g[3] is None]
    assert sum(len(g[0]) for g in groups) == len(ra)
    assert all(len(g[0]) >= 3 for g in regional)
    assert len(singles) == 3            # 1 + 2 isolated transients -> cone searches
    for idx, cra, cdec, rq in regional:
        # every member lies inside the query circle minus the search radius
        u = B._unit(ra[idx], dec[idx]); c = B._unit(cra, cdec)[0]
        sep = np.degrees(np.arccos(np.clip(u @ c, -1, 1)))
        assert np.all(sep <= rq - 30 / 3600 + 1e-9)
        assert rq < 0.3


def test_group_positions_ra_wrap():
    ra = np.array([359.99, 0.01, 0.02, 359.98])
    dec = np.zeros(4)
    groups = B.group_positions(ra, dec, cell_deg=0.3)
    assert sum(len(g[0]) for g in groups) == 4   # nothing lost at the RA = 0 seam


def test_wide_row_and_dlr_host():
    cands = make_cands([
        {"ra": RA0, "dec": DEC0 + 1 * ASEC, "flux": 1e5},                    # SBW host
        {"ra": RA0, "dec": DEC0 + 8 * ASEC, "rx": 40.0, "ry": 40.0, "flux": 1e3},  # huge faint: DLR winner
        {"ra": RA0, "dec": DEC0 - 20 * ASEC, "ext": 0.0},                     # star, rejected
    ])
    res = compute_association(cands, RA0, DEC0, "r")
    row = B.wide_row(res, n_hosts=3)
    assert row["host_1_objectId"] == 1000 and row["host_1_p_sbw"] > 0.5
    assert row["dlr_host_objectId"] == 1001 and row["dlr_host_d_dlr"] == pytest.approx(res["d_dlr"].min())
    assert row["host_3_objectId"] == 0 and np.isnan(row["host_3_p_sbw"]) and row["host_3_suspect"] is np.False_ or row["host_3_suspect"] == False  # noqa: E712
    assert row["n_candidates"] == 2 and row["n_in_radius"] == 3
    assert row["host_1_ra"] == pytest.approx(RA0) and row["host_1_dec"] == pytest.approx(DEC0 + 1 * ASEC)
    assert row["dlr_host_ra"] == pytest.approx(RA0) and row["dlr_host_dec"] == pytest.approx(DEC0 + 8 * ASEC)
    assert "max_frac_light_beyond" not in row
    empty = B.wide_row(res[:0], n_hosts=2)
    assert empty["host_1_objectId"] == 0 and np.isnan(empty["dlr_host_ra"]) and np.isnan(empty["host_2_ra"])


def test_associate_many_validates_keys(monkeypatch):
    with pytest.raises(ValueError, match="unique"):
        B.associate_many([1.0, 2.0], [0.0, 0.0], keys=["a", "a"], service=object())
    with pytest.raises(ValueError, match="same length"):
        B.associate_many([1.0, 2.0], [0.0], service=object())


def test_associate_many_offline(monkeypatch):
    """Drive associate_many with fake query functions and check the wide/long assembly."""
    gal = make_cands([{"ra": RA0 + k * 20 * ASEC, "dec": DEC0, "flux": 1e4 * (k + 1)} for k in range(5)])
    gal.meta["model"] = "sersic"
    monkeypatch.setattr(B, "resolve_model", lambda m, s, d: "sersic")
    monkeypatch.setattr(B, "query_region", lambda service, ra, dec, r, band, dataset, model, **kw: gal)
    def fake_cone(ra, dec, rad, band, dataset, service=None, model=None, **kw):
        du, dv = __import__("sbw_host.sersic", fromlist=["x"]).sky_offsets(ra, dec, gal["coord_ra"], gal["coord_dec"])
        return gal[np.hypot(du, dv) < rad]
    monkeypatch.setattr(B, "query_candidates", fake_cone)
    ra = RA0 + np.array([0, 20, 40, 2000]) * ASEC      # three near the galaxies, one far away
    dec = np.full(4, DEC0)
    keys = np.array(["dia1", "dia2", "dia3", "dia4"])
    wide, long = B.associate_many(ra, dec, keys=keys, n_hosts=2, service=object(), return_long=True, verbose=False)
    assert list(wide["transient_key"]) == list(keys) and wide["transient_key"].dtype.kind in "US"
    assert wide["host_1_objectId"][0] == 1000 and wide["host_1_objectId"][1] == 1001
    assert wide["n_candidates"][3] == 0 and wide["host_1_objectId"][3] == 0
    assert set(long["transient_key"]) == {"dia1", "dia2", "dia3"}
    assert wide.meta["n_hosts"] == 2


def test_lsdb_backend_matches_local_computation():
    lsdb = pytest.importorskip("lsdb")
    import pandas as pd
    from sbw_host import sersic as S
    # synthetic "Object table" of 5 galaxies, exponential-fit columns
    base = make_cands([{"ra": RA0 + k * 20 * ASEC, "dec": DEC0, "flux": 1e4 * (k + 1)} for k in range(5)])
    gal = Table({c.replace("sersic", "exponential"): base[c] for c in base.colnames if c != "sersic_index"})
    df = gal.to_pandas()
    cat = lsdb.from_dataframe(df, ra_column="coord_ra", dec_column="coord_dec", lowest_order=2, highest_order=6)
    ra = RA0 + np.array([0, 20, 40, 2000]) * ASEC
    dec = np.full(4, DEC0)
    keys = np.array([101, 202, 303, 404])
    wide, long = B.associate_many(ra, dec, keys=keys, n_hosts=2, backend="lsdb", lsdb_catalog=cat,
                                  return_long=True, verbose=False)
    assert wide.meta["backend"] == "lsdb" and wide.meta["model"] == "exponential"
    assert list(wide["transient_key"]) == [101, 202, 303, 404]
    assert wide["host_1_objectId"][0] == 1000 and wide["host_1_objectId"][1] == 1001
    assert wide["n_candidates"][3] == 0 and wide["host_1_objectId"][3] == 0
    # same numbers as compute_association on the hand-selected candidates
    gal.meta["model"] = "exponential"
    du, dv = S.sky_offsets(ra[1], dec[1], gal["coord_ra"], gal["coord_dec"])
    ref = compute_association(gal[np.hypot(du, dv) < 30], ra[1], dec[1], "r")
    assert wide["host_1_p_sbw"][1] == pytest.approx(float(ref["p_sbw"][0]))
    assert wide["n_candidates"][1] == ref.meta["n_candidates"]
    assert set(long["transient_key"]) == {101, 202, 303}


def test_lsdb_backend_rejects_missing_columns():
    lsdb = pytest.importorskip("lsdb")
    import pandas as pd
    cat = lsdb.from_dataframe(pd.DataFrame({"objectId": [1], "coord_ra": [RA0], "coord_dec": [DEC0]}),
                              ra_column="coord_ra", dec_column="coord_dec", lowest_order=2, highest_order=4)
    with pytest.raises(RuntimeError, match="lacks columns"):
        B.associate_many([RA0], [DEC0], backend="lsdb", lsdb_catalog=cat, verbose=False)


def test_wide_row_has_drop_counts_and_band_used():
    cands = make_cands([{"ra": RA0, "dec": DEC0 + 2 * ASEC}, {"ra": RA0, "dec": DEC0 - 2 * ASEC, "ext": 0.0},
                        {"ra": RA0, "dec": DEC0 + 5 * ASEC, "nodata": True}])
    res = compute_association(cands, RA0, DEC0, "r")
    row = B.wide_row(res, 1)
    assert row["n_in_radius"] == 3 and row["n_candidates"] == 1
    assert row["n_dropped_pointlike"] == 1 and row["n_dropped_bad_fit"] == 1 and row["band_used"] == "r"
    empty = B._empty_wide_row(1)
    assert empty["band_used"] == "" and empty["n_dropped_bad_fit"] == 0
