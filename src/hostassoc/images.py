"""Images for diagnostics: Sersic-model renders and real deep_coadd cutouts.

Two sources are supported:

* :func:`render_model_image` draws every candidate's Sersic model onto a
  north-up tangent-plane grid.  Needs only the catalogue (``read:tap`` token).
* :func:`fetch_cutout` pulls a real ``deep_coadd`` cutout through ObsCore ->
  DataLink -> SODA.  This needs an RSP token with the ``read:image`` scope; with a
  ``read:tap``-only token the image services answer 403 and
  :class:`ImageAccessError` is raised so callers can fall back to the model render.
"""

from __future__ import annotations

import io

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

from . import sersic as S
from . import sersic_mix as SM
from .tap import RSP_BASE, get_tap_service, load_image_token, make_session


class ImageAccessError(RuntimeError):
    """Raised when the RSP image services refuse or fail a request."""


def tangent_wcs(ra: float, dec: float, size_arcsec: float,
                pixel_scale: float = S.PIXEL_SCALE_ARCSEC) -> tuple[WCS, tuple[int, int]]:
    """North-up, east-left TAN WCS centred on (ra, dec)."""
    npix = int(np.ceil(size_arcsec / pixel_scale))
    npix += npix % 2 == 0  # odd so the centre lands on a pixel
    w = WCS(naxis=2)
    w.wcs.crpix = [(npix + 1) / 2.0, (npix + 1) / 2.0]  # FITS 1-based
    w.wcs.crval = [ra, dec]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = [-pixel_scale / 3600.0, pixel_scale / 3600.0]
    w.wcs.cunit = ["deg", "deg"]
    return w, (npix, npix)


def render_model_image(result, wcs: WCS, shape: tuple[int, int],
                       psf_fwhm_arcsec: float | None = 0.8) -> np.ndarray:
    """Render the candidates' Sersic models (nJy per pixel) onto ``wcs``.

    ``result`` is the table from :func:`hostassoc.associate_host`.  An optional
    Gaussian blur mimics seeing so the render resembles a coadd.
    """
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    ra_pix, dec_pix = wcs.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
    ra_pix = ra_pix.reshape(shape)
    dec_pix = dec_pix.reshape(shape)
    pix_area = S.PIXEL_SCALE_ARCSEC**2
    img = np.zeros(shape, dtype=float)
    profile = result.meta.get("profile", "mixture")
    for row in result:
        du, dv = S.sky_offsets(ra_pix, dec_pix, float(row["ra"]), float(row["dec"]))
        rx, ry, rho = float(row["reff_x_arcsec"]), float(row["reff_y_arcsec"]), float(row["rho"])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if profile == "mixture":
                m = S.elliptical_radius(du, dv, rx, ry, rho)
                a, b, _ = S.ellipse_axes(rx, ry, rho)
                img += np.nan_to_num(np.exp(SM.log_intensity(m, float(row["flux_nJy"]), a, b, float(row["sersic_index"]))))
            else:
                img += np.nan_to_num(S.sersic_intensity_at(
                    du, dv, float(row["flux_nJy"]), rx, ry, rho, float(row["sersic_index"])))
    img *= pix_area
    if psf_fwhm_arcsec:
        sigma_pix = psf_fwhm_arcsec / 2.3548 / S.PIXEL_SCALE_ARCSEC
        img = gaussian_filter(img, sigma_pix)
    return img


def find_coadd(ra: float, dec: float, band: str, dataset: str = "dp2", service=None, token=None):
    """ObsCore row for the deep_coadd covering (ra, dec) in ``band``."""
    service = service or get_tap_service(token=token, dataset=dataset)
    q = (
        "SELECT access_url, obs_publisher_did, lsst_tract, lsst_patch, obs_collection "
        "FROM ivoa.ObsCore WHERE dataproduct_subtype = 'lsst.deep_coadd' "
        f"AND lsst_band = '{band}' AND obs_collection = 'LSST.{dataset.upper()}' "
        f"AND CONTAINS(POINT('ICRS', {ra:.8f}, {dec:.8f}), s_region) = 1"
    )
    tab = service.search(q).to_table()
    if len(tab) == 0:
        raise ImageAccessError(f"No {dataset} {band}-band deep_coadd covers ({ra:.5f}, {dec:.5f}).")
    return tab[0]


def _first_image_hdu(hdul):
    for hdu in hdul:
        if hdu.data is not None and np.ndim(hdu.data) == 2:
            return hdu
    raise ImageAccessError("Cutout response contained no 2-D image HDU.")


def fetch_cutout(ra: float, dec: float, size_arcsec: float, band: str = "r",
                 dataset: str = "dp2", service=None, token=None, image_token=None, session=None):
    """Return ``(data, wcs)`` for a deep_coadd cutout of side ``size_arcsec``.

    The ObsCore lookup uses the TAP service (``read:tap``); DataLink and the SODA
    cutout use ``image_token`` (``read:image``, see :func:`hostassoc.tap.load_image_token`).
    Tries the DataLink SODA descriptor first, then the direct ``/api/cutout/sync`` URL.
    """
    import requests
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery

    session = session or make_session(load_image_token(image_token))
    service = service or get_tap_service(token=token, dataset=dataset)
    row = find_coadd(ra, dec, band, dataset, service=service)
    radius_deg = (size_arcsec / 2.0) * np.sqrt(2.0) / 3600.0  # circle enclosing the square

    content = None
    errors = []
    try:
        dl = DatalinkResults.from_result_url(str(row["access_url"]), session=session)
        desc = dl.get_adhocservice_by_id("cutout-sync")
        sq = SodaQuery.from_resource(dl, desc, session=session)
        sq.circle = (ra, dec, radius_deg)
        content = sq.execute_stream().read()
    except Exception as e:  # noqa: BLE001 - want to fall through to direct URL
        errors.append(f"datalink/SODA: {e!r}")
    if content is None:
        try:
            r = session.get(f"{RSP_BASE}/api/cutout/sync",
                            params={"ID": str(row["obs_publisher_did"]),
                                    "POS": f"CIRCLE {ra:.8f} {dec:.8f} {radius_deg:.8f}"},
                            timeout=120)
            if r.status_code == 403:
                raise ImageAccessError(
                    "Cutout service returned 403 Forbidden. An RSP token with the "
                    "'read:image' scope is needed (.rsp_image_token or RSP_IMAGE_TOKEN; "
                    "create one at https://data.lsst.cloud, Account -> Security tokens)."
                )
            r.raise_for_status()
            content = r.content
        except ImageAccessError:
            raise
        except requests.RequestException as e:
            errors.append(f"direct cutout: {e!r}")
    if content is None:
        raise ImageAccessError("Cutout failed: " + "; ".join(errors))

    with fits.open(io.BytesIO(content)) as hdul:
        hdu = _first_image_hdu(hdul)
        data = np.array(hdu.data, dtype=float)
        wcs = WCS(hdu.header).celestial
    return data, wcs
