import numpy as np
import pytest

from sbw_host.stamps import sbw_stamps


def gaussian_stamp(size, cx, cy, sigma, amp=1.0):
    yy, xx = np.mgrid[:size, :size]
    return amp * np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) / sigma**2)


def test_two_stamps_closer_one_wins():
    s1 = gaussian_stamp(21, 10, 10, 3.0)   # centred at common-frame (10, 10)
    s2 = gaussian_stamp(31, 15, 15, 3.0)   # centred at (40 + 15, 15) = (55, 15)
    r = sbw_stamps(20.0, 10.0, [s1, s2], origins=[(0, 0), (40, 0)])
    assert r.p_sbw[0] > 0.99
    assert r.best == 0
    assert r.m is None


def test_outside_footprint_is_zero_and_hostless():
    s = gaussian_stamp(21, 10, 10, 3.0)
    r = sbw_stamps(100.0, 100.0, [s], origins=[(0, 0)])
    assert r.intensity[0] == 0.0
    assert np.isnan(r.p_sbw).all()
    assert r.frac_light_beyond[0] == 0.0


def test_frac_light_beyond_at_centre_and_edge():
    s = gaussian_stamp(41, 20, 20, 4.0)
    centre = sbw_stamps(20.0, 20.0, [s], origins=[(0, 0)], box=1)
    edge = sbw_stamps(32.0, 20.0, [s], origins=[(0, 0)], box=1)
    # everything but the peak pixel itself (1 / (2 pi sigma^2) of the flux) is fainter than the peak
    assert centre.frac_light_beyond[0] == pytest.approx(1.0 - 1.0 / (2 * np.pi * 16.0), rel=1e-3)
    assert edge.frac_light_beyond[0] < 0.05
    # 3 sigma out: fraction outside is exp(-9/2) for a 2-D Gaussian
    three_sig = sbw_stamps(32.0, 20.0, [s], origins=[(0, 0)], box=1)
    assert three_sig.frac_light_beyond[0] == pytest.approx(np.exp(-4.5), rel=0.15)


def test_box_mean_reduces_noise_scatter():
    rng = np.random.default_rng(1)
    s = gaussian_stamp(41, 20, 20, 5.0, amp=100.0)
    clean = {b: sbw_stamps(27.0, 20.0, [s], origins=[(0, 0)], box=b).frac_light_beyond[0] for b in (1, 5)}
    est = {1: [], 5: []}
    for _ in range(60):
        noisy = s + rng.normal(0, 8.0, s.shape)
        for b in (1, 5):
            est[b].append(sbw_stamps(27.0, 20.0, [noisy], origins=[(0, 0)], box=b).frac_light_beyond[0])
    # the box mean is a far less noisy threshold than a single pixel
    assert np.std(est[5]) < 0.5 * np.std(est[1])
    # (both are biased high on a noisy stamp because clipped noise adds flux to the faint pixels;
    # that is a property of noisy data stamps, see the module notes)
    assert np.mean(est[1]) > clean[1] and np.mean(est[5]) > clean[5]


def test_interpolation_modes():
    s = np.zeros((5, 5))
    s[2, 2] = 4.0
    s[2, 3] = 2.0
    bil = sbw_stamps(2.5, 2.0, [s], origins=[(0, 0)], interp="bilinear").intensity[0]
    near = sbw_stamps(2.4, 2.0, [s], origins=[(0, 0)], interp="nearest").intensity[0]
    box = sbw_stamps(2.0, 2.0, [s], origins=[(0, 0)], interp="box", box=3).intensity[0]
    assert bil == pytest.approx(3.0)
    assert near == pytest.approx(4.0)
    assert box == pytest.approx(6.0 / 9.0)


def test_negative_and_nan_pixels_are_clipped():
    s = np.full((5, 5), -1.0)
    s[2, 2] = np.nan
    s[1, 1] = 3.0
    r = sbw_stamps(1.0, 1.0, [s], origins=[(0, 0)], box=1)
    assert r.intensity[0] == 3.0
    assert r.frac_light_beyond[0] == 0.0


def test_wcs_placement_and_pixel_area():
    from astropy.wcs import WCS
    def make_wcs(ra0, dec0, scale_arcsec, n):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra0, dec0]
        w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
        w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
        return w
    s1 = gaussian_stamp(41, 20, 20, 4.0)
    s2 = gaussian_stamp(41, 20, 20, 2.0)                # same galaxy sampled at twice the pixel size
    w1 = make_wcs(10.0, -5.0, 0.2, 41)
    w2 = make_wcs(10.0, -5.0, 0.4, 41)
    s2 *= 4.0                                            # flux per (bigger) pixel
    r = sbw_stamps(10.0, -5.0 + 0.8 / 3600.0, [s1, s2], wcs=[w1, w2], interp="bilinear")
    assert r.p_sbw == pytest.approx([0.5, 0.5], abs=0.02)   # per-arcsec^2 intensities match


def test_validation():
    s = np.ones((3, 3))
    with pytest.raises(ValueError):
        sbw_stamps(1, 1, [s])
    with pytest.raises(ValueError):
        sbw_stamps(1, 1, [s], origins=[(0, 0)], box=2)
    with pytest.raises(ValueError):
        sbw_stamps(1, 1, [s], origins=[(0, 0)], interp="cubic")
    with pytest.raises(ValueError):
        sbw_stamps(1, 1, [s, s], origins=[(0, 0)])
