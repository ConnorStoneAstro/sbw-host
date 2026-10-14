import numpy as np
import pytest

from sbw_host import sersic as S
from sbw_host import sersic_mix as SM


@pytest.mark.parametrize("n", [0.5, 0.73, 1.0, 2.5, 4.0, 6.0])
def test_mixture_weights_sum_to_one(n):
    w, s = SM.mixture(n)
    assert w.shape == (4,) and s.shape == (4,)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.all(w >= 0) and np.all(s > 0)


def test_gaussian_limit_half_light_factor():
    # n = 0.5 is a single Gaussian with sigma = 1.1774 reff -> R_half = 1.1774^2 reff
    w, s = SM.mixture(0.5)
    assert w[0] == pytest.approx(1.0) and s[0] == pytest.approx(1.17741, abs=1e-4)
    assert SM.half_light_factor(0.5) == pytest.approx(1.17741**2, rel=1e-4)
    for n in (1.0, 4.0):
        assert 1.3 < SM.half_light_factor(n) < 1.45


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0])
def test_frac_light_beyond_mixture(n):
    assert SM.frac_light_beyond(0.0, n) == pytest.approx(1.0)
    hl = SM.half_light_factor(n)
    assert SM.frac_light_beyond(hl, n) == pytest.approx(0.5, abs=1e-6)
    f = SM.frac_light_beyond(np.array([1.0, 3.0, 10.0, 25.0]), n)
    assert np.all(np.diff(f) < 0)
    # the mixture wings are Gaussian: far less light outside than an analytic Sersic
    assert SM.frac_light_beyond(25.0, n) < 1e-5
    if n >= 1.0:
        assert SM.frac_light_beyond(25.0, n) < 1e-3 * S.frac_light_beyond(25.0 / hl, n)


@pytest.mark.parametrize("n", [0.6, 1.0, 4.0])
@pytest.mark.parametrize("reff_x,reff_y,rho", [(1.0, 1.0, 0.0), (2.0, 0.8, 0.0), (1.5, 1.0, 0.6)])
def test_mixture_intensity_integrates_to_flux(n, reff_x, reff_y, rho):
    flux = 321.0
    major, minor, _ = S.ellipse_axes(reff_x, reff_y, rho)
    m_edges = np.logspace(-5, np.log10(60.0), 3001)
    m_mid = np.sqrt(m_edges[1:] * m_edges[:-1])
    dm = np.diff(m_edges)
    I = np.exp(SM.log_intensity(m_mid, flux, major, minor, n))
    total = np.sum(I * 2 * np.pi * major * minor * m_mid * dm)  # elliptical annuli
    assert total == pytest.approx(flux, rel=2e-3)


def test_mixture_matches_single_gaussian_at_half():
    # n = 0.5: I(m) = F / (2 pi s^2 a b) exp(-m^2/(2 s^2)), s = 1.1774
    a, b, F = 1.3, 0.7, 10.0
    s = SM.mixture(0.5)[1][0]
    for m in (0.0, 0.7, 2.0):
        expect = F / (2 * np.pi * s**2 * a * b) * np.exp(-0.5 * (m / s) ** 2)
        assert np.exp(SM.log_intensity(m, F, a, b, 0.5)) == pytest.approx(expect, rel=1e-9)


def test_case5_phantom_far_candidate_has_tiny_fraction():
    # case 05 #2: n = 6 candidate 22.7 catalogue-reff away had f_beyond = 0.026 analytically
    assert S.frac_light_beyond(22.7, 6.0) > 0.02
    assert SM.frac_light_beyond(22.7, 6.0) < 1e-4
