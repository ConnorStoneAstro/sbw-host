import numpy as np
import pytest
from astropy.table import Table

from sbw_host import associate as A
from sbw_host import sersic as S

from .helpers import ASEC, DEC0, RA0, make_cands


def test_probabilities_normalise_and_rank():
    cands = make_cands([
        {"ra": RA0, "dec": DEC0},                                  # SN at its centre
        {"ra": RA0 + 10 * ASEC / np.cos(np.deg2rad(DEC0)), "dec": DEC0},  # 10" away
        {"ra": RA0, "dec": DEC0 + 25 * ASEC, "flux": 5e5},          # bright but 25" away
    ])
    res = A.compute_association(cands, RA0, DEC0, "r")
    assert np.isclose(res["p_sbw"].sum(), 1.0)
    assert res["objectId"][0] == 1000
    assert res["p_sbw"][0] > 0.9
    assert np.all(np.diff(res["p_sbw"]) <= 0)
    assert res.meta["n_candidates"] == 3


def test_point_sources_and_bad_fits_dropped():
    cands = make_cands([
        {"ra": RA0, "dec": DEC0, "ext": 0.0},            # star at the SN position
        {"ra": RA0, "dec": DEC0 + 3 * ASEC},             # galaxy
        {"ra": RA0, "dec": DEC0 - 3 * ASEC, "nodata": True},
        {"ra": RA0, "dec": DEC0 - 4 * ASEC, "flux": np.nan},
    ])
    res = A.compute_association(cands, RA0, DEC0, "r")
    assert list(res["objectId"]) == [1001]
    assert res["p_sbw"][0] == pytest.approx(1.0)
    assert res.meta["n_dropped_pointlike"] == 1
    assert res.meta["n_dropped_bad_fit"] == 2
    res_all = A.compute_association(cands, RA0, DEC0, "r", min_extendedness=None)
    assert set(res_all["objectId"]) == {1000, 1001}


def test_max_hosts_truncates_without_renormalising():
    cands = make_cands([{"ra": RA0, "dec": DEC0 + k * 4 * ASEC} for k in range(6)])
    full = A.compute_association(cands, RA0, DEC0, "r")
    top = A.compute_association(cands, RA0, DEC0, "r", max_hosts=2)
    assert len(top) == 2
    assert np.allclose(top["p_sbw"], full["p_sbw"][:2])
    assert top.meta["prob_in_table"] == pytest.approx(full["p_sbw"][:2].sum())
    assert top.meta["n_candidates"] == 6


def test_far_compact_candidates_stay_finite():
    # Intensities underflow to exp(-huge); log-domain normalisation must cope.
    cands = make_cands([
        {"ra": RA0, "dec": DEC0 + 20 * ASEC, "rx": 0.5, "ry": 0.5, "n": 0.5},
        {"ra": RA0, "dec": DEC0 - 25 * ASEC, "rx": 0.5, "ry": 0.5, "n": 0.5},
    ])
    res = A.compute_association(cands, RA0, DEC0, "r")
    assert np.all(np.isfinite(res["p_sbw"]))
    assert np.isclose(res["p_sbw"].sum(), 1.0)
    assert res["objectId"][0] == 1000  # the nearer one wins


def test_elliptical_geometry_matters():
    # Galaxy elongated along North; SN 6" North vs 6" East of a twin elongated along East.
    cands = make_cands([
        {"ra": RA0, "dec": DEC0 - 6 * ASEC, "rx": 2.0, "ry": 10.0},   # major axis N-S, SN on it
        {"ra": RA0 - 6 * ASEC / np.cos(np.deg2rad(DEC0)), "dec": DEC0, "rx": 2.0, "ry": 10.0},  # SN off-axis
    ])
    res = A.compute_association(cands, RA0, DEC0, "r")
    assert res["objectId"][0] == 1000
    assert res["m_reff"][0] < res["m_reff"][1]


def test_build_query_validates_band_and_units():
    q = A.build_query(10.0, -5.0, 36.0, "i", dataset="dp2")
    assert "dp2.Object" in q and "i_exponentialFlux" in q and "exponential_reff_x" in q and "0.01000000" in q
    assert "sersic_index" not in q
    qs = A.build_query(10.0, -5.0, 36.0, "i", dataset="dp1", model="sersic")
    assert "dp1.Object" in qs and "i_sersicFlux" in qs and "sersic_index" in qs
    with pytest.raises(ValueError):
        A.build_query(10.0, -5.0, 36.0, "k")
    with pytest.raises(ValueError):
        A.build_query(10.0, -5.0, 36.0, "i", model="devaucouleurs")


def test_exponential_columns_and_default_model():
    # an exponential-fit table (no sersic_index) is evaluated with n = 1
    cands = make_cands([{"ra": RA0, "dec": DEC0 + 2 * ASEC, "n": 3.0}])
    exp = Table({c.replace("sersic", "exponential"): cands[c] for c in cands.colnames if c != "sersic_index"})
    exp.rename_column("r_exponentialFlux", "r_exponentialFlux")
    res = A.compute_association(exp, RA0, DEC0, "r")
    assert res.meta["model"] == "exponential" and res["sersic_index"][0] == 1.0
    res_s = A.compute_association(cands, RA0, DEC0, "r")
    assert res_s.meta["model"] == "sersic" and res_s["sersic_index"][0] == 3.0
    assert res["frac_light_beyond"][0] != res_s["frac_light_beyond"][0]


def test_frac_light_beyond_column_flags_hostless():
    on_host = make_cands([{"ra": RA0, "dec": DEC0}])
    far = make_cands([{"ra": RA0, "dec": DEC0 + 25 * ASEC, "rx": 2.0, "ry": 2.0}])
    r1 = A.compute_association(on_host, RA0, DEC0, "r")
    r2 = A.compute_association(far, RA0, DEC0, "r")
    assert r1.meta["profile"] == "mixture"
    assert r1["frac_light_beyond"][0] == pytest.approx(1.0)
    assert r1["p_sbw"][0] == pytest.approx(1.0) and r2["p_sbw"][0] == pytest.approx(1.0)
    assert r2["frac_light_beyond"][0] < 1e-6          # P=1 but clearly hostless
    assert r2.meta["max_frac_light_beyond"] < 1e-6


def test_suspect_flag_and_exclusion():
    cands = make_cands([
        {"ra": RA0, "dec": DEC0 + 2 * ASEC},                                           # fine
        {"ra": RA0, "dec": DEC0 + 3 * ASEC, "sra": RA0, "sdec": DEC0 + 5 * ASEC},        # centroid shifted 2"
        {"ra": RA0, "dec": DEC0 - 3 * ASEC, "flux": 1e4},                               # ratio 1 unless cModel differs
    ])
    cands["r_cModelFlux"][2] = 500.0  # Sersic flux 20x cModel
    res = A.compute_association(cands, RA0, DEC0, "r")
    by_id = {int(r["objectId"]): r for r in res}
    assert not by_id[1000]["suspect"] and by_id[1001]["suspect"] and by_id[1002]["suspect"]
    assert by_id[1001]["centroid_shift_arcsec"] == pytest.approx(2.0, rel=1e-3)
    assert by_id[1002]["flux_ratio_cmodel"] == pytest.approx(20.0)
    assert res.meta["n_suspect"] == 2
    kept = A.compute_association(cands, RA0, DEC0, "r", exclude_suspect=True)
    assert list(kept["objectId"]) == [1000] and kept["p_sbw"][0] == pytest.approx(1.0)
    # sb_eff: mag + 0.753 + 2.5 log10(pi a b); reff 5 px = 1" circular
    assert by_id[1000]["sb_eff"] == pytest.approx(by_id[1000]["mag"] + 0.753 + 2.5 * np.log10(np.pi), abs=1e-6)


def test_bright_diffuse_phantom_is_suspect():
    # r = 22.6 with R_eff 3.35" x 1.20" (case 09 phantom): SB_eff ~ 26.1 -> suspect
    cands = make_cands([
        {"ra": RA0, "dec": DEC0 + 10 * ASEC, "rx": 3.35 / 0.2, "ry": 1.20 / 0.2, "flux": 3300.0},
        {"ra": RA0, "dec": DEC0 + 3 * ASEC, "rx": 3.35 / 0.2, "ry": 1.20 / 0.2, "flux": 3.3e5},  # same size, 5 mag brighter
        {"ra": RA0, "dec": DEC0 - 3 * ASEC, "rx": 0.3 / 0.2, "ry": 0.3 / 0.2, "flux": 50.0},     # faint small galaxy, SB 27: fine
    ])
    res = A.compute_association(cands, RA0, DEC0, "r")
    by_id = {int(r["objectId"]): r for r in res}
    assert by_id[1000]["suspect"] and by_id[1000]["sb_eff"] > 25.5
    assert not by_id[1001]["suspect"]
    assert not by_id[1002]["suspect"] and by_id[1002]["sb_eff"] > 25.5


def test_profile_option_and_half_light_columns():
    cands = make_cands([{"ra": RA0, "dec": DEC0 + 2 * ASEC, "n": 1.0}])
    mix = A.compute_association(cands, RA0, DEC0, "r")
    ana = A.compute_association(cands, RA0, DEC0, "r", profile="analytic")
    assert mix["rhalf_major_arcsec"][0] == pytest.approx(1.386 * mix["reff_major_arcsec"][0], rel=0.01)
    assert ana["rhalf_major_arcsec"][0] == pytest.approx(ana["reff_major_arcsec"][0])
    assert mix["m_rhalf"][0] == pytest.approx(mix["m_reff"][0] / 1.386, rel=0.01)
    assert mix["frac_light_beyond"][0] != ana["frac_light_beyond"][0]
    with pytest.raises(ValueError):
        A.compute_association(cands, RA0, DEC0, "r", profile="bogus")


def test_candidate_ids_survive_truncation():
    cands = make_cands([{"ra": RA0, "dec": DEC0 + k * 4 * ASEC} for k in range(5)] + [{"ra": RA0, "dec": DEC0, "ext": 0.0}])
    top = A.compute_association(cands, RA0, DEC0, "r", max_hosts=2)
    assert len(top) == 2 and len(top.meta["candidate_ids"]) == 5
    assert 1005 not in top.meta["candidate_ids"]   # the star was rejected, not truncated


def test_band_fallback_order_and_use():
    assert A.fallback_order("r") == ("r", "i", "g", "z", "u", "y")
    assert A.fallback_order("i") == ("i", "r", "g", "z", "u", "y")
    assert A.fallback_order("g", fallback=False) == ("g",)
    with pytest.raises(ValueError):
        A.fallback_order("k")
    # r-band fluxes missing (no coverage), i-band fine -> fallback picks i
    cands = make_cands([{"ra": RA0, "dec": DEC0 + 2 * ASEC}, {"ra": RA0, "dec": DEC0 - 4 * ASEC}])
    for b in ("i", "g"):
        cands[f"{b}_sersicFlux"] = cands["r_sersicFlux"] * (2.0 if b == "i" else 3.0)
        cands[f"{b}_sersicFluxErr"] = 1.0; cands[f"{b}_cModelFlux"] = cands[f"{b}_sersicFlux"]; cands[f"{b}_extendedness"] = 1.0
    cands["r_sersicFlux"] = np.nan
    res = A.associate_bands(cands, RA0, DEC0, "r")
    assert res.meta["band"] == "r" and res.meta["band_used"] == "i" and res.meta["n_candidates"] == 2
    assert res["mag"][0] == pytest.approx(-2.5 * np.log10(2e4) + 31.4)
    res_nf = A.associate_bands(cands, RA0, DEC0, "r", band_fallback=False)
    assert res_nf.meta["band_used"] == "r" and res_nf.meta["n_candidates"] == 0 and res_nf.meta["n_dropped_bad_fit"] == 2
    # nothing usable anywhere -> requested band's empty result
    for b in ("i", "g"):
        cands[f"{b}_sersicFlux"] = np.nan
    res0 = A.associate_bands(cands, RA0, DEC0, "r")
    assert res0.meta["band_used"] == "r" and res0.meta["n_candidates"] == 0
    # query carries the fallback bands' columns
    q = A.build_query(10.0, -5.0, 36.0, "r", bands=A.fallback_order("r"))
    assert all(f"{b}_exponentialFlux" in q for b in "ugrizy")


def test_compute_association_regression_against_pre_core_values():
    """Outputs recorded before compute_association was refactored onto sbw_host.core."""
    rows = [
        {"ra": RA0 + 3 * ASEC, "dec": DEC0 + 1 * ASEC, "rx": 6.0, "ry": 3.0, "rho": 0.4, "n": 1.0, "flux": 2e4},
        {"ra": RA0 - 8 * ASEC, "dec": DEC0 + 5 * ASEC, "rx": 12.0, "ry": 10.0, "rho": -0.2, "n": 2.5, "flux": 8e4},
        {"ra": RA0 + 1 * ASEC, "dec": DEC0 - 12 * ASEC, "rx": 4.0, "ry": 4.5, "rho": 0.1, "n": 0.7, "flux": 5e3},
        {"ra": RA0 + 20 * ASEC, "dec": DEC0 + 20 * ASEC, "rx": 30.0, "ry": 15.0, "rho": 0.0, "n": 4.0, "flux": 3e5},
    ]
    expected = {
        "mixture": {
            "p_sbw": [0.6671202539752984, 0.26166282060979446, 0.07121692541490734, 2.65668617256209e-16],
            "log_intensity": [4.566912481481912, 3.6309988916989457, 2.3296726676685333, -30.892584501572145],
            "frac_light_beyond": [0.07163715925955165, 0.1342420219244714, 0.05791604213910863, 8.497949630178678e-17],
            "m_rhalf": [2.5614774242529035, 3.1123483866471715, 5.303467907041009, 9.620430613438675],
            "m_reff": [3.549254994621314, 4.310448970241789, 7.286386375403463, 13.335349341731202],
        },
        "analytic": {
            "p_sbw": [0.5537497144507229, 0.36265670776412734, 0.08359357778515004, 2.7507384579572297e-18],
            "log_intensity": [3.5600285586868274, 3.136772431153756, 1.669282449380221, -36.28359123612149],
            "frac_light_beyond": [0.018004201485103923, 0.07987672528197443, 0.06636620651366582, 4.500541368885045e-19],
            "m_rhalf": [3.549254994621314, 4.310448970241789, 7.286386375403463, 13.335349341731202],
            "m_reff": [3.549254994621314, 4.310448970241789, 7.286386375403463, 13.335349341731202],
        },
    }
    for profile, cols in expected.items():
        t = A.compute_association(make_cands(rows), RA0, DEC0, "r", profile=profile)
        assert list(t["objectId"]) == [1000, 1001, 1003, 1002]
        for k, v in cols.items():
            assert np.allclose(np.asarray(t[k], float), v, rtol=1e-9, atol=0), (profile, k)
