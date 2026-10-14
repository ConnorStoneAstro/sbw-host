import numpy as np
import pytest

from sbw_host import sersic as S


def test_bn_known_values():
    assert S.b_n(1.0) == pytest.approx(1.67835, abs=1e-4)
    assert S.b_n(4.0) == pytest.approx(7.66925, abs=1e-4)


@pytest.mark.parametrize("n", [0.6, 1.0, 2.5, 4.0, 6.0])
@pytest.mark.parametrize("reff_x,reff_y,rho", [(1.0, 1.0, 0.0), (2.0, 0.8, 0.0), (1.5, 1.0, 0.6)])
def test_profile_integrates_to_total_flux(n, reff_x, reff_y, rho):
    """Integrate I(du, dv) over the plane in elliptical-polar coordinates.

    (du, dv) = m * M (cos t, sin t) with M the matrix square root of Sigma, so the
    area element is |det M| m dm dt = major * minor * m dm dt.  A log-spaced m grid
    resolves the central cusp of high-n profiles.
    """
    flux = 1234.0
    major, minor, _ = S.ellipse_axes(reff_x, reff_y, rho)
    sigma = np.array([[reff_x**2, rho * reff_x * reff_y], [rho * reff_x * reff_y, reff_y**2]])
    w, v = np.linalg.eigh(sigma)
    M = v @ np.diag(np.sqrt(w)) @ v.T
    m_edges = np.logspace(-6, np.log10(400.0), 4001)
    m_mid = np.sqrt(m_edges[1:] * m_edges[:-1])
    dm = np.diff(m_edges)
    t = np.linspace(0, 2 * np.pi, 361)[:-1]
    dt = t[1] - t[0]
    mm, tt = np.meshgrid(m_mid, t, indexing="ij")
    unit = np.stack([np.cos(tt), np.sin(tt)])            # (2, nm, nt)
    du, dv = np.tensordot(M, unit, axes=(1, 0)) * mm       # (2, nm, nt)
    img = S.sersic_intensity_at(du, dv, flux, reff_x, reff_y, rho, n)
    total = np.sum(img * major * minor * mm * dm[:, None] * dt)
    assert total == pytest.approx(flux, rel=2e-3)
    # and the elliptical radius we fed in is what the code recovers
    assert np.allclose(S.elliptical_radius(du, dv, reff_x, reff_y, rho), mm, rtol=1e-6)


def test_half_light_within_effective_ellipse():
    flux, n = 1.0, 1.5
    reff_x, reff_y, rho = 1.2, 0.7, -0.4
    step = 0.02
    g = np.arange(-20, 20, step) + step / 2
    du, dv = np.meshgrid(g, g)
    img = S.sersic_intensity_at(du, dv, flux, reff_x, reff_y, rho, n)
    m = S.elliptical_radius(du, dv, reff_x, reff_y, rho)
    inside = img[m <= 1].sum() * step**2
    assert inside == pytest.approx(0.5, abs=0.01)


def test_elliptical_radius_unity_on_axes():
    reff_x, reff_y, rho = 2.0, 1.0, 0.3
    major, minor, phi = S.ellipse_axes(reff_x, reff_y, rho)
    c, s = np.cos(np.deg2rad(phi)), np.sin(np.deg2rad(phi))
    assert S.elliptical_radius(major * c, major * s, reff_x, reff_y, rho) == pytest.approx(1.0)
    assert S.elliptical_radius(-minor * s, minor * c, reff_x, reff_y, rho) == pytest.approx(1.0)
    assert major >= minor


def test_axes_match_dp2_relation():
    # DP2 publishes major/minor as the eigen-decomposition of (reff_x, reff_y, rho)
    major, minor, _ = S.ellipse_axes(1.285, 1.301, 0.0194)
    tr, det = 1.285**2 + 1.301**2, (1.285 * 1.301) ** 2 * (1 - 0.0194**2)
    assert major**2 + minor**2 == pytest.approx(tr)
    assert (major * minor) ** 2 == pytest.approx(det)


def test_sky_offsets_orientation():
    du, dv = S.sky_offsets(10.0 - 1 / 3600.0 / np.cos(np.deg2rad(-30)), -30.0 + 2 / 3600.0, 10.0, -30.0)
    # smaller RA is West -> positive du; larger Dec -> positive dv
    assert du == pytest.approx(1.0, rel=1e-6)
    assert dv == pytest.approx(2.0, rel=1e-6)
    du, _ = S.sky_offsets(359.9999, 0.0, 0.0001, 0.0)
    assert du == pytest.approx(0.72, rel=1e-3)  # wraps across RA = 0


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0])
def test_frac_light_beyond(n):
    assert S.frac_light_beyond(1.0, n) == pytest.approx(0.5, abs=1e-6)
    assert S.frac_light_beyond(0.0, n) == pytest.approx(1.0)
    f = S.frac_light_beyond(np.array([0.5, 1.0, 3.0, 20.0]), n)
    assert np.all(np.diff(f) < 0) and f[-1] < 0.02  # de Vaucouleurs keeps ~1% beyond 20 Re
    # consistency with direct integration of the 1-D profile: L(>m) = int_m^inf I(m') 2 pi m' dm'
    mm = np.logspace(-4, 3, 20001)
    I = S.sersic_intensity(mm, 1.0, 1.0, 1.0, n)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] * mm[1:] + I[:-1] * mm[:-1]) * np.diff(mm))]) * 2 * np.pi
    outside = 1.0 - cum / cum[-1]
    for m0 in (0.3, 1.0, 2.5):
        assert S.frac_light_beyond(m0, n) == pytest.approx(np.interp(m0, mm, outside), abs=2e-3)
