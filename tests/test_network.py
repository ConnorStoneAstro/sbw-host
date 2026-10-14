"""Live queries against the Rubin TAP service (run with ``pytest --run-network``)."""

import numpy as np
import pytest

pytestmark = pytest.mark.network

from hostassoc import associate_host, get_tap_service, load_token  # noqa: E402
from hostassoc.plot import plot_association  # noqa: E402


@pytest.fixture(scope="module")
def service():
    try:
        load_token()
    except RuntimeError as e:
        pytest.skip(str(e))
    return get_tap_service()


# A few random ECDFS / COSMOS / EDFS positions (DP2 footprint).
POSITIONS = [(53.1312, -28.0987), (150.1178, 2.2143), (59.0963, -48.7261)]


@pytest.mark.parametrize("ra,dec", POSITIONS)
def test_random_positions_return_normalised_probabilities(service, ra, dec):
    res = associate_host(ra, dec, band="r", radius_arcsec=30, service=service)
    assert res.meta["n_in_radius"] > 0
    if len(res):
        assert np.isclose(res["p_sbw"].sum(), 1.0)
        assert np.all(np.diff(res["p_sbw"]) <= 0)
        assert np.all(res["sep_arcsec"] <= 30 + 1.0)


def test_sn_on_galaxy_centre_recovers_that_galaxy(service):
    # pick a bright, isolated galaxy and put the "SN" on its Sersic centroid
    tab = service.search(
        "SELECT objectId, sersic_ra, sersic_dec, r_cModelFlux FROM dp2.Object "
        "WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', 53.13, -28.10, 0.03)) = 1 "
        "AND refExtendedness > 0.9 AND r_cModelFlux > 20000 AND sersic_reff_major > 0.8").to_table()
    assert len(tab) >= 1
    k = int(np.argmax(tab["r_cModelFlux"]))  # brightest; Qserv rejects TOP ... ORDER BY
    oid, ra, dec = int(tab["objectId"][k]), float(tab["sersic_ra"][k]), float(tab["sersic_dec"][k])
    res = associate_host(ra, dec, band="r", radius_arcsec=30, service=service)
    assert int(res["objectId"][0]) == oid
    assert res["p_sbw"][0] > 0.5


def test_band_choice_and_max_hosts(service):
    ra, dec = POSITIONS[0]
    res_i = associate_host(ra, dec, band="i", radius_arcsec=20, max_hosts=2, service=service)
    assert len(res_i) <= 2
    assert res_i.meta["band"] == "i"


def test_plot_runs(service, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    res = associate_host(*POSITIONS[0], radius_arcsec=30, service=service)
    fig, _ = plot_association(res)
    fig.savefig(tmp_path / "x.png", dpi=60)


def test_cutout_matches_catalogue_geometry(service):
    """Real deep_coadd cutout: WCS is north-up/east-left and the field centre is right."""
    from hostassoc import fetch_cutout, load_image_token
    try:
        load_image_token()
    except RuntimeError as e:
        pytest.skip(str(e))
    ra, dec = POSITIONS[0]
    img, wcs = fetch_cutout(ra, dec, 40.0, band="r", service=service)
    assert img.ndim == 2 and min(img.shape) >= 150
    cd = wcs.pixel_scale_matrix
    assert cd[0, 0] < 0 and cd[1, 1] > 0 and abs(cd[0, 1]) < 1e-9  # +x West, +y North
    px, py = wcs.world_to_pixel_values(ra, dec)
    assert abs(px - (img.shape[1] - 1) / 2) < 3 and abs(py - (img.shape[0] - 1) / 2) < 3


def test_ellipse_orientation_matches_image_moments(service):
    """Light-weighted moments on a real cutout agree with the (reff_x, reff_y, rho) angle.

    This pins down the pixel-frame convention (+x West, +y North) and shows that
    ``sersic_theta`` has the opposite sign.
    """
    from hostassoc import fetch_cutout, load_image_token
    from hostassoc import sersic as S
    try:
        load_image_token()
    except RuntimeError as e:
        pytest.skip(str(e))
    tab = service.search(
        "SELECT objectId, sersic_ra, sersic_dec, sersic_reff_x, sersic_reff_y, sersic_rho, "
        "sersic_theta, sersic_reff_major FROM dp2.Object "
        "WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), CIRCLE('ICRS', 53.13, -28.10, 0.05)) = 1 "
        "AND refExtendedness > 0.9 AND r_cModelFlux > 30000 AND sersic_reff_major > 1.0 "
        "AND sersic_reff_minor < 0.6 * sersic_reff_major").to_table()
    assert len(tab) >= 2
    for row in tab[:2]:
        ra, dec = float(row["sersic_ra"]), float(row["sersic_dec"])
        img, wcs = fetch_cutout(ra, dec, 16.0, band="r", service=service)
        x0, y0 = wcs.world_to_pixel_values(ra, dec)
        yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
        dx, dy = xx - x0, yy - y0
        rmax = 2.5 * float(row["sersic_reff_major"]) / S.PIXEL_SCALE_ARCSEC
        w = np.where(np.hypot(dx, dy) < rmax, np.clip(img - np.median(img), 0, None), 0.0)
        mxx, myy, mxy = ((w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum())
        ang_img = 0.5 * np.degrees(np.arctan2(2 * mxy, mxx - myy))
        _, _, phi = S.ellipse_axes(float(row["sersic_reff_x"]), float(row["sersic_reff_y"]), float(row["sersic_rho"]))
        diff = lambda a, b: abs((a - b + 90) % 180 - 90)
        assert diff(ang_img, phi) < 8.0
        assert diff(float(row["sersic_theta"]), -phi) < 1.0


def test_plot_option_returns_figure(service, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    res, fig = associate_host(*POSITIONS[0], radius_arcsec=20, service=service, plot=True)
    assert len(res) and fig is not None
    fig.savefig(tmp_path / "a.png", dpi=50)
    out = tmp_path / "b.png"
    res2, raw = associate_host(*POSITIONS[0], radius_arcsec=20, service=service, plot=str(out), return_raw=True)
    assert out.exists() and len(raw) >= len(res2)


def test_batch_matches_single_object_path(service):
    from hostassoc import associate_many
    rng = np.random.default_rng(21)
    ra = np.r_[53.13 + rng.uniform(-0.03, 0.03, 5), 150.12]        # one regional cell + one single cone
    dec = np.r_[-28.10 + rng.uniform(-0.03, 0.03, 5), 2.21]
    keys = np.array([f"k{i}" for i in range(6)])
    wide, long = associate_many(ra, dec, keys=keys, n_hosts=2, service=service, return_long=True, verbose=False)
    assert list(wide["transient_key"]) == list(keys) and len(wide) == 6
    for i in (0, 5):
        r = associate_host(ra[i], dec[i], service=service)
        assert wide["n_candidates"][i] == r.meta["n_candidates"]
        if len(r):
            assert wide["host_1_objectId"][i] == int(r["objectId"][0])
            assert wide["host_1_p_sbw"][i] == pytest.approx(float(r["p_sbw"][0]), abs=1e-9)
            assert wide["dlr_host_d_dlr"][i] == pytest.approx(float(np.nanmin(r["d_dlr"])), abs=1e-9)
    assert set(long["transient_key"]) <= set(keys)
