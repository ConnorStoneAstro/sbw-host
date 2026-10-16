import numpy as np
import pytest

from sbw_host.core import (sbw_exponential, sbw_model, sbw_probabilities, sbw_sersic, sky_to_plane,
                           elliptical_radius)
from sbw_host import sersic as S
from sbw_host import sersic_mix as SM


def test_single_host_gets_everything():
    r = sbw_exponential(1.0, 2.0, x=[0.0], y=[0.0], r_eff=[3.0], flux=[10.0])
    assert r.p_sbw == pytest.approx([1.0])
    assert r.best == 0
    assert 0 < r.frac_light_beyond[0] < 1


def test_two_identical_hosts_equidistant_split_evenly():
    r = sbw_exponential(0.0, 0.0, x=[-5.0, 5.0], y=[0.0, 0.0], r_eff=[3.0, 3.0], flux=[1e4, 1e4])
    assert r.p_sbw == pytest.approx([0.5, 0.5])
    assert r.frac_light_beyond[0] == pytest.approx(r.frac_light_beyond[1])


def test_frac_light_beyond_is_half_at_half_light_radius():
    r = sbw_exponential(4.0, 0.0, x=[0.0], y=[0.0], r_eff=[4.0], flux=[1.0])
    assert r.frac_light_beyond[0] == pytest.approx(0.5, rel=1e-6)
    r = sbw_sersic(4.0, 0.0, x=[0.0], y=[0.0], n=3.0, r_eff=[4.0], flux=[1.0], profile="mixture")
    assert r.frac_light_beyond[0] == pytest.approx(0.5, rel=1e-6)
    assert r.m == pytest.approx([1.0])


def test_zero_flux_excludes_a_host():
    r = sbw_exponential(0.0, 0.0, x=[1.0, 2.0], y=[0.0, 0.0], r_eff=[3.0, 3.0], flux=[1e4, 0.0])
    assert r.p_sbw == pytest.approx([1.0, 0.0])
    assert r.log_intensity[1] == -np.inf
    r = sbw_exponential(0.0, 0.0, x=[1.0], y=[0.0], r_eff=[3.0], flux=[0.0])
    assert np.isnan(r.p_sbw).all() and r.best is None


def test_ellipse_orientation_matters():
    # transient 6 units along +x; host elongated along x (phi=0) vs along y (phi=90)
    along = sbw_exponential(6.0, 0.0, x=[0.0], y=[0.0], r_eff=[3.0], flux=[1.0], q=[0.3], phi=[0.0])
    across = sbw_exponential(6.0, 0.0, x=[0.0], y=[0.0], r_eff=[3.0], flux=[1.0], q=[0.3], phi=[90.0])
    assert along.m[0] == pytest.approx(2.0)
    assert across.m[0] == pytest.approx(2.0 / 0.3)
    assert along.intensity[0] > across.intensity[0]


def test_elliptical_radius_rotation_invariance():
    m0 = elliptical_radius(3.0, 4.0, 0.0, 0.0, 2.0, 0.5, 0.0)
    # rotate both the point and the ellipse by 37 degrees
    th = np.deg2rad(37.0)
    px, py = 3.0 * np.cos(th) - 4.0 * np.sin(th), 3.0 * np.sin(th) + 4.0 * np.cos(th)
    m1 = elliptical_radius(px, py, 0.0, 0.0, 2.0, 0.5, 37.0)
    assert m1 == pytest.approx(m0)


def test_scale_length_and_effective_radius_are_equivalent():
    a = sbw_exponential(2.0, 1.0, x=[0.0, 7.0], y=[0.0, 0.0], r_eff=[5.0, 3.0], flux=[1.0, 2.0])
    b = sbw_exponential(2.0, 1.0, x=[0.0, 7.0], y=[0.0, 0.0], r_scale=np.array([5.0, 3.0]) / S.b_n(1.0),
                        flux=[1.0, 2.0])
    assert b.log_intensity == pytest.approx(a.log_intensity)
    assert S.b_n(1.0) == pytest.approx(1.678, abs=1e-3)
    # general n: r_eff = b_n^n r_scale
    n = 2.5
    c = sbw_sersic(2.0, 1.0, x=[0.0], y=[0.0], n=n, r_eff=[5.0], flux=[1.0])
    d = sbw_sersic(2.0, 1.0, x=[0.0], y=[0.0], n=n, r_scale=[5.0 / S.b_n(n) ** n], flux=[1.0])
    assert d.log_intensity == pytest.approx(c.log_intensity)


@pytest.mark.parametrize("profile", ["analytic", "mixture"])
def test_amplitude_conventions_agree(profile):
    kw = dict(x=[0.0], y=[0.0], r_eff=[4.0], q=[0.6], phi=[20.0], profile=profile)
    ref = sbw_sersic(3.0, -2.0, n=1.5, flux=[1e4], **kw)
    # intensity at the half-light radius on the major axis
    at_re = sbw_sersic(4.0 * np.cos(np.deg2rad(20.0)), 4.0 * np.sin(np.deg2rad(20.0)), n=1.5, flux=[1e4], **kw)
    via_ie = sbw_sersic(3.0, -2.0, n=1.5, i_eff=at_re.intensity, **kw)
    assert via_ie.intensity == pytest.approx(ref.intensity, rel=1e-9)
    at_0 = sbw_sersic(0.0, 0.0, n=1.5, flux=[1e4], **kw)
    via_i0 = sbw_sersic(3.0, -2.0, n=1.5, i_0=at_0.intensity, **kw)
    assert via_i0.intensity == pytest.approx(ref.intensity, rel=1e-9)


def test_total_flux_normalisation_analytic():
    # numerically integrate the analytic exponential over a big grid: should recover ``flux``
    xs = np.linspace(-200, 200, 2001)
    X, Y = np.meshgrid(xs, xs)
    dA = (xs[1] - xs[0]) ** 2
    tot = 0.0
    for row in range(0, len(xs), 1):
        r = sbw_exponential(0.0, 0.0, x=-X[row], y=-Y[row], r_eff=5.0, flux=123.0, q=0.7, phi=40.0)
        tot += r.intensity.sum() * dA
    assert tot == pytest.approx(123.0, rel=2e-3)


def test_mixture_vs_analytic_differ_but_agree_in_half_light():
    for n in (1.0, 4.0):
        a = sbw_sersic(3.0, 0.0, x=[0.0], y=[0.0], n=n, r_eff=[3.0], flux=[1.0], profile="analytic")
        m = sbw_sersic(3.0, 0.0, x=[0.0], y=[0.0], n=n, r_eff=[3.0], flux=[1.0], profile="mixture")
        assert a.frac_light_beyond[0] == pytest.approx(0.5, rel=1e-6)
        assert m.frac_light_beyond[0] == pytest.approx(0.5, rel=1e-6)
    assert SM.half_light_factor(1.0) == pytest.approx(1.386, abs=0.01)


def test_probabilities_helper_and_priors():
    log_i = np.array([0.0, np.log(3.0), -np.inf, np.nan])
    p = sbw_probabilities(log_i)
    assert p == pytest.approx([0.25, 0.75, 0.0, 0.0])
    assert np.isnan(sbw_probabilities([-np.inf, np.nan])).all()
    # a log-prior tilts the result
    p2 = sbw_probabilities(log_i + np.array([np.log(3.0), 0.0, 0.0, 0.0]))
    assert p2 == pytest.approx([0.5, 0.5, 0.0, 0.0])


def test_sky_to_plane_orientation():
    x, y = sky_to_plane(10.0 - 1 / 3600.0, 20.0 + 2 / 3600.0, 10.0, 20.0)
    assert x == pytest.approx(np.cos(np.deg2rad(20.0)), rel=1e-6)   # West is +x
    assert y == pytest.approx(2.0, rel=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0], [0.0], r_eff=[1.0], r_scale=[1.0], flux=[1.0])
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0], [0.0], r_eff=[1.0])
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0], [0.0], r_eff=[1.0], flux=[1.0], i_0=[1.0])
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0, 1.0], [0.0], r_eff=[1.0], flux=[1.0])
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0], [0.0], r_eff=[1.0], flux=[1.0], profile="gaussian")
    with pytest.raises(ValueError):
        sbw_model(0, 0, [0.0], [0.0], r_eff=[-1.0], flux=[1.0])
