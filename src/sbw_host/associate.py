"""Host-galaxy association for Rubin transients using Object-table Sersic fits.

Given a transient position, query the Rubin Object table for extended sources
within a search radius, evaluate every candidate's Sersic model surface
brightness at the transient position in the requested band, and define

    P(host = i) = I_i(SN) / sum_j I_j(SN)

over all candidates returned by the cone search.
"""

from __future__ import annotations

import datetime as _dt
import warnings

import numpy as np
from astropy.table import Table
from scipy.special import logsumexp

from . import core as CORE
from . import sersic as S
from . import sersic_mix as SM
from . import dlr as DLR
from .tap import get_tap_service

BANDS = ("u", "g", "r", "i", "z", "y")
FALLBACK_BANDS = ("r", "i", "g", "z", "u", "y")   # tried in this order when the requested band has no usable fluxes


def fallback_order(band: str, fallback: bool = True) -> tuple[str, ...]:
    """Bands to try: the requested band first, then the fallback order (or just the band)."""
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    if not fallback:
        return (band,)
    return (band,) + tuple(b for b in FALLBACK_BANDS if b != band)

MODELS = ("exponential", "sersic")
DEFAULT_MODEL = "exponential"

# Columns pulled from the Object table.  ``{m}`` is the MultiProFit model prefix:
# ``exponential`` (n fixed at 1; DP2 only) or ``sersic`` (free n; DP1 and DP2).
_COMMON_COLUMNS = ("objectId", "coord_ra", "coord_dec", "refExtendedness", "parentObjectId")
_MODEL_COLUMNS = ("{m}_ra", "{m}_dec", "{m}_reff_x", "{m}_reff_y", "{m}_rho",
                  "{m}_no_data_flag", "{m}_unknown_flag")
_SERSIC_ONLY = ("sersic_index",)


# Quality-diagnostic columns; included when the release publishes them.
_OPTIONAL_COLUMNS = ("detect_fromBlend", "deblend_parentNChild", "shape_flag",
                     "sersic_chi2_reduced", "sersic_chisq_reduced")

# Criteria for the ``suspect`` flag (see compute_association).
SUSPECT_MAX_CENTROID_SHIFT_ARCSEC = 1.0
SUSPECT_FLUX_RATIO_RANGE = (0.5, 2.0)
# "bright but diffuse": mean SB inside R_eff fainter than this while the total
# magnitude is brighter than this.  Calibrated on DP2 r band (see README).
SUSPECT_DIFFUSE_SB_MAG_ARCSEC2 = 25.5
SUSPECT_DIFFUSE_MAX_MAG = 24.0

_available_cache: dict[str, frozenset[str]] = {}


def _check_model(model: str) -> str:
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}, got {model!r}")
    return model


def model_columns(band: str, model: str = DEFAULT_MODEL, bands=None) -> tuple[str, ...]:
    """Object-table columns needed for ``model`` in ``band`` (and any extra ``bands`` for fallback)."""
    m = _check_model(model)
    cols = _COMMON_COLUMNS + tuple(c.format(m=m) for c in _MODEL_COLUMNS)
    if m == "sersic":
        cols += _SERSIC_ONLY
    for b in dict.fromkeys((band,) + tuple(bands or ())):
        cols += (f"{b}_{m}Flux", f"{b}_{m}FluxErr", f"{b}_cModelFlux", f"{b}_extendedness")
    return cols


def _band_columns(band: str) -> tuple[str, ...]:  # kept for backwards compatibility
    return (f"{band}_sersicFlux", f"{band}_sersicFluxErr", f"{band}_cModelFlux",
            f"{band}_extendedness")


def available_columns(service, dataset: str = "dp2") -> frozenset[str]:
    """Column names of ``{dataset}.Object`` (cached per dataset for the session)."""
    if dataset not in _available_cache:
        tab = service.search(
            f"SELECT column_name FROM tap_schema.columns WHERE table_name = '{dataset}.Object'"
        ).to_table()
        _available_cache[dataset] = frozenset(str(c) for c in tab["column_name"])
    return _available_cache[dataset]


def build_query(ra: float, dec: float, radius_arcsec: float, band: str,
                dataset: str = "dp2", extra_columns=(), model: str = DEFAULT_MODEL, bands=None) -> str:
    """ADQL cone search on the Object table returning the model-fit columns."""
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, got {band!r}")
    cols = ", ".join(model_columns(band, model, bands) + tuple(extra_columns))
    return (
        f"SELECT {cols} FROM {dataset}.Object "
        f"WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec), "
        f"CIRCLE('ICRS', {ra:.8f}, {dec:.8f}, {radius_arcsec / 3600.0:.8f})) = 1"
    )


def resolve_model(model: str, service, dataset: str) -> str:
    """Fall back to the Sersic fit when the release lacks the requested model (e.g. DP1)."""
    m = _check_model(model)
    try:
        cols = available_columns(service, dataset)
    except Exception:
        return m
    if f"{m}_reff_x" not in cols:
        if m != "sersic" and "sersic_reff_x" in cols:
            warnings.warn(f"{dataset} has no '{m}' fit columns; using the Sersic fit instead.")
            return "sersic"
        raise RuntimeError(f"{dataset}.Object has no '{m}' model columns.")
    return m


def query_candidates(ra, dec, radius_arcsec=30.0, band="r", dataset="dp2",
                     service=None, token=None, model: str = DEFAULT_MODEL, bands=None) -> Table:
    """Run the cone search and return the raw Object rows as an astropy Table.

    The table's ``meta['model']`` records which fit was actually used.  ``bands``
    adds the flux columns of further bands (for the band fallback).
    """
    service = service or get_tap_service(token=token, dataset=dataset)
    model = resolve_model(model, service, dataset)
    try:
        extra = tuple(c for c in _OPTIONAL_COLUMNS if c in available_columns(service, dataset))
    except Exception:  # schema lookup failed; proceed without diagnostics
        extra = ()
    tab = service.search(build_query(ra, dec, radius_arcsec, band, dataset, extra, model, bands)).to_table()
    return plain_table(tab, model)


def optional_columns(service, dataset: str = "dp2") -> tuple[str, ...]:
    """The quality-diagnostic columns this release publishes (empty if the lookup fails)."""
    try:
        return tuple(c for c in _OPTIONAL_COLUMNS if c in available_columns(service, dataset))
    except Exception:
        return ()


def plain_table(tab: Table, model: str) -> Table:
    """Unmask a TAP result (masked -> NaN / 0 / False) and record the model used."""
    out = Table()
    for name in tab.colnames:
        col = tab[name]
        if hasattr(col, "filled"):
            fill = np.nan if col.dtype.kind == "f" else (0 if col.dtype.kind in "iu" else False)
            col = col.filled(fill)
        out[name] = np.asarray(col)
    out.meta["model"] = model
    return out


def query_region(service, ra: float, dec: float, radius_deg: float, band: str = "r", dataset: str = "dp2",
                 model: str = DEFAULT_MODEL, sync: bool = False, maxrec: int | None = None, bands=None) -> Table:
    """Cone search over a large region (async by default) returning the model-fit columns."""
    model = resolve_model(model, service, dataset)
    q = build_query(ra, dec, radius_deg * 3600.0, band, dataset, optional_columns(service, dataset), model, bands)
    if sync:
        tab = service.search(q, maxrec=maxrec).to_table()
    else:
        job = service.submit_job(q, maxrec=maxrec)
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
        job.raise_if_error()
        tab = job.fetch_result().to_table()
        try:
            job.delete()
        except Exception:
            pass
    return plain_table(tab, model)


def _as_float(tab: Table, name: str) -> np.ndarray:
    return np.asarray(tab[name], dtype=float)


def compute_association(cands: Table, ra: float, dec: float, band: str,
                        min_extendedness: float | None = 0.5,
                        max_hosts: int | None = None,
                        exclude_suspect: bool = False,
                        profile: str = "mixture", model: str | None = None) -> Table:
    """Evaluate Sersic surface brightness at (ra, dec) and build probabilities.

    Parameters
    ----------
    cands : Table
        Rows from :func:`query_candidates` (or any table with the same columns).
    ra, dec : float
        Transient position (deg, ICRS).
    band : str
        Band whose ``{band}_sersicFlux`` sets each model's normalisation.
    min_extendedness : float or None
        Keep only objects with ``refExtendedness > min_extendedness``.  ``None``
        keeps everything (stars included).
    max_hosts : int or None
        Return only the N most probable hosts.

    exclude_suspect : bool
        Drop candidates flagged ``suspect`` (below) before computing probabilities.
    profile : {'mixture', 'analytic'}
        ``'mixture'`` (default) evaluates the Gaussian-mixture profile actually
        fitted by MultiProFit (see :mod:`sbw_host.sersic_mix`), with the catalogue
        ``reff`` as the mixture scale parameter.  ``'analytic'`` evaluates a true
        Sersic profile with R_eff = ``reff``; it is too compact in the core and far
        too extended in the wings, and is kept only for comparison.
    model : {'exponential', 'sersic'}, optional
        Which MultiProFit fit the columns of ``cands`` come from.  Defaults to
        ``cands.meta['model']`` (set by :func:`query_candidates`), else
        ``'exponential'``.  The exponential fit has n fixed at 1, so its Gaussian
        mixture is the same for every object and it is immune to the n -> 6 wing
        problem of the free-n Sersic fit.

    Quality diagnostics.  Sersic fits are made to the *deblended* model of each
    child, so a faint peak inside a big blend (typically in the halo of a bright
    star) can be handed some of its neighbour's wings and come out as a large,
    very low surface brightness model with no visible counterpart.  The table
    therefore carries ``centroid_shift_arcsec`` (Sersic centroid minus detection
    centroid), ``flux_ratio_cmodel`` (Sersic / cModel flux), ``sb_eff`` (mean
    surface brightness inside R_eff, mag/arcsec^2), ``from_blend``,
    ``parent_nchild`` and ``shape_flag`` where the release provides them, and a
    boolean ``suspect`` that is True when the centroid shift exceeds
    ``SUSPECT_MAX_CENTROID_SHIFT_ARCSEC``, the flux ratio lies outside
    ``SUSPECT_FLUX_RATIO_RANGE``, or the object is bright yet very diffuse
    (``sb_eff > SUSPECT_DIFFUSE_SB_MAG_ARCSEC2`` and ``mag < SUSPECT_DIFFUSE_MAX_MAG``,
    thresholds calibrated on DP2 r band).  In DP2 ~4% of extended objects are
    suspect overall, but most of those with R_eff > 1.5".

    Besides ``p_sbw`` the table carries ``frac_light_beyond``: the fraction of each
    candidate's model light lying outside the elliptical isophote through the
    transient.  It is 0.5 at R_eff and plays the role of a p-value for "this galaxy
    could plausibly host a transient this far out"; when it is tiny for *every*
    candidate the transient is probably hostless (or its host is undetected), even
    though ``p_sbw`` still sums to one.  Probabilities are *not*
        renormalised after truncation, so they remain the probability relative to
        every candidate inside the search radius; ``meta['prob_in_table']`` gives
        the retained total.
    """
    n_in = len(cands)
    m_ = _check_model(model or cands.meta.get("model", DEFAULT_MODEL))
    flux = _as_float(cands, f"{band}_{m_}Flux")
    reff_x = _as_float(cands, f"{m_}_reff_x") * S.PIXEL_SCALE_ARCSEC
    reff_y = _as_float(cands, f"{m_}_reff_y") * S.PIXEL_SCALE_ARCSEC
    rho = _as_float(cands, f"{m_}_rho")
    nser = _as_float(cands, "sersic_index") if m_ == "sersic" else np.ones(n_in)
    ext = _as_float(cands, "refExtendedness")

    # model centroid, falling back to the reference centroid if the fit has none.
    cra = _as_float(cands, f"{m_}_ra")
    cdec = _as_float(cands, f"{m_}_dec")
    bad_c = ~np.isfinite(cra) | ~np.isfinite(cdec)
    cra[bad_c] = _as_float(cands, "coord_ra")[bad_c]
    cdec[bad_c] = _as_float(cands, "coord_dec")[bad_c]

    good = (np.isfinite(flux) & (flux > 0) & np.isfinite(reff_x) & np.isfinite(reff_y)
            & np.isfinite(rho) & np.isfinite(nser) & (nser > 0))
    for flag in (f"{m_}_no_data_flag", f"{m_}_unknown_flag"):
        if flag in cands.colnames:
            good &= ~np.asarray(cands[flag], bool)
    n_bad_fit = int(np.sum(~good))
    if min_extendedness is not None:
        good &= np.isfinite(ext) & (ext > min_extendedness)

    major, minor, phi = S.ellipse_axes(reff_x, reff_y, rho)

    # --- quality diagnostics -------------------------------------------------
    sh_du, sh_dv = S.sky_offsets(cra, cdec, _as_float(cands, "coord_ra"), _as_float(cands, "coord_dec"))
    centroid_shift = np.hypot(sh_du, sh_dv)
    cmodel = _as_float(cands, f"{band}_cModelFlux") if f"{band}_cModelFlux" in cands.colnames else np.full(len(cands), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_ratio = flux / cmodel
    lo, hi = SUSPECT_FLUX_RATIO_RANGE
    with np.errstate(divide="ignore", invalid="ignore"):
        mag_tot = -2.5 * np.log10(flux) + 31.4
        sb_eff = mag_tot + 0.753 + 2.5 * np.log10(np.pi * major * minor)
    suspect = ((centroid_shift > SUSPECT_MAX_CENTROID_SHIFT_ARCSEC)
               | (np.isfinite(flux_ratio) & ((flux_ratio < lo) | (flux_ratio > hi)))
               | ((sb_eff > SUSPECT_DIFFUSE_SB_MAG_ARCSEC2) & (mag_tot < SUSPECT_DIFFUSE_MAX_MAG)))
    n_suspect = int(np.sum(suspect & good))
    if exclude_suspect:
        good &= ~suspect

    du, dv = S.sky_offsets(ra, dec, cra, cdec)
    if profile not in CORE.PROFILES:
        raise ValueError("profile must be 'mixture' or 'analytic'")
    hl = SM.half_light_factor(nser) if profile == "mixture" else np.ones_like(nser)   # R_half / sersic_reff
    # The table-free core does the SBW evaluation; it takes true half-light radii, so
    # scale the catalogue reff (a mixture scale parameter) by ``hl``.  Hosts failing the
    # cuts get zero flux, which the core turns into zero probability.
    core = CORE.sbw_model(0.0, 0.0, -du, -dv, r_eff=np.where(good, major * hl, 1.0),
                          flux=np.where(good, flux, 0.0), n=np.where(good, nser, 1.0),
                          q=minor / major, phi=phi, profile=profile)
    m = core.m * hl            # elliptical radius in units of catalogue reff
    log_i, f_beyond, p_sbw = core.log_intensity, core.frac_light_beyond, core.p_sbw

    out = Table()
    out["objectId"] = np.asarray(cands["objectId"])
    out["ra"] = cra
    out["dec"] = cdec
    out["sep_arcsec"] = np.hypot(du, dv)
    out["du_arcsec"] = du
    out["dv_arcsec"] = dv
    out["rhalf_major_arcsec"] = major * hl   # model half-light radius (major axis)
    out["rhalf_minor_arcsec"] = minor * hl
    out["reff_major_arcsec"] = major         # catalogue sersic_reff (mixture scale parameter)
    out["reff_minor_arcsec"] = minor
    out["phi_deg"] = phi
    out["reff_x_arcsec"] = reff_x
    out["reff_y_arcsec"] = reff_y
    out["rho"] = rho
    out["sersic_index"] = nser
    out["flux_nJy"] = flux
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mag"] = -2.5 * np.log10(flux) + 31.4
    out["refExtendedness"] = ext
    out["sb_eff"] = sb_eff  # mean surface brightness within R_eff, mag/arcsec^2
    out["centroid_shift_arcsec"] = centroid_shift
    out["flux_ratio_cmodel"] = flux_ratio
    out["suspect"] = suspect
    if "detect_fromBlend" in cands.colnames:
        out["from_blend"] = np.asarray(cands["detect_fromBlend"], bool)
    if "deblend_parentNChild" in cands.colnames:
        out["parent_nchild"] = np.asarray(cands["deblend_parentNChild"], int)
    if "shape_flag" in cands.colnames:
        out["shape_flag"] = np.asarray(cands["shape_flag"], bool)
    out["m_rhalf"] = m / hl        # elliptical radius of the SN in units of the model half-light radius
    out["d_dlr"] = DLR.d_dlr(du, dv, major * hl, minor * hl, phi)   # == m_rhalf; kept for the DLR comparison
    out["m_reff"] = m              # ... in units of the catalogue sersic_reff
    # floor at 1e-300 so tiny values survive CSV round trips (no subnormal floats)
    out["frac_light_beyond"] = np.maximum(f_beyond, 1e-300)  # fraction of the host's light outside the SN isophote
    out["log_intensity"] = log_i   # ln(nJy / arcsec^2) at the SN position
    out["intensity_nJy_arcsec2"] = np.maximum(np.exp(log_i), 1e-300)
    out["p_sbw"] = p_sbw

    keep = good
    out = out[keep]
    out["p_dlr"] = DLR.dlr_assignment(out["d_dlr"])   # hard DLR host (d_DLR <= 4), for comparison
    order = np.argsort(-np.nan_to_num(out["p_sbw"], nan=-1.0), kind="stable")
    out = out[order]
    n_cands = len(out)
    candidate_ids = [int(x) for x in out["objectId"]]   # before any max_hosts truncation
    if max_hosts is not None:
        out = out[: int(max_hosts)]

    out.meta.update({
        "sn_ra": float(ra), "sn_dec": float(dec), "band": band, "band_used": band,
        "n_in_radius": n_in, "n_candidates": n_cands,
        "n_dropped_bad_fit": n_bad_fit,
        "n_suspect": n_suspect, "exclude_suspect": exclude_suspect,
        "n_dropped_pointlike": int(n_in - n_bad_fit - n_cands) if min_extendedness is not None else 0,
        "prob_in_table": float(np.nansum(out["p_sbw"])) if len(out) else 0.0,
        "max_frac_light_beyond": float(np.nanmax(out["frac_light_beyond"])) if len(out) else 0.0,
        "min_extendedness": min_extendedness, "max_hosts": max_hosts, "profile": profile, "model": m_,
        "candidate_ids": candidate_ids,
    })
    for c, fmt in (("ra", "%.6f"), ("dec", "%.6f"), ("sep_arcsec", "%.2f"),
                   ("reff_major_arcsec", "%.2f"), ("reff_minor_arcsec", "%.2f"),
                   ("phi_deg", "%.1f"), ("sersic_index", "%.2f"), ("flux_nJy", "%.1f"),
                   ("mag", "%.2f"), ("m_reff", "%.2f"), ("m_rhalf", "%.2f"), ("p_sbw", "%.4f"), ("frac_light_beyond", "%.3g"),
                   ("rhalf_major_arcsec", "%.2f"), ("rhalf_minor_arcsec", "%.2f"),
                   ("intensity_nJy_arcsec2", "%.3g"), ("log_intensity", "%.2f"),
                   ("du_arcsec", "%.2f"), ("dv_arcsec", "%.2f"), ("refExtendedness", "%.2f"),
                   ("reff_x_arcsec", "%.2f"), ("reff_y_arcsec", "%.2f"), ("rho", "%.2f"),
                   ("sb_eff", "%.2f"), ("centroid_shift_arcsec", "%.2f"), ("flux_ratio_cmodel", "%.2f"),
                   ("d_dlr", "%.2f"), ("p_dlr", "%.0f")):
        out[c].format = fmt
    return out


def associate_bands(cands: Table, ra: float, dec: float, band: str = "r", band_fallback: bool = True, **kw) -> Table:
    """:func:`compute_association` with band fallback.

    Tries ``band`` first and then the remaining bands in :data:`FALLBACK_BANDS`
    order (r, i, g, z, u, y) until one yields at least one candidate; the Object
    table has a row wherever *any* band was coadded, but the per-band model fluxes
    are NaN where that band has no coverage.  ``meta['band_used']`` records the band
    the probabilities refer to and ``meta['band']`` the one requested; if no band
    works the requested band's (empty) result is returned.  The flux columns of
    every band to be tried must be present in ``cands``; bands whose columns are
    missing are skipped.
    """
    first = None
    for b in fallback_order(band, band_fallback):
        m = cands.meta.get("model", DEFAULT_MODEL)
        if f"{b}_{m}Flux" not in cands.colnames:
            continue
        res = compute_association(cands, ra, dec, b, **kw)
        res.meta["band"] = band
        res.meta["band_used"] = b
        if first is None:
            first = res
        if res.meta["n_candidates"] > 0:
            return res
    if first is None:
        raise KeyError(f"no flux columns for band {band!r} in the candidate table")
    return first


def associate_host(ra: float, dec: float, band: str = "r", radius_arcsec: float = 30.0,
                   max_hosts: int | None = None, dataset: str = "dp2",
                   min_extendedness: float | None = 0.5, exclude_suspect: bool = False,
                   profile: str = "mixture", model: str = DEFAULT_MODEL, band_fallback: bool = True,
                   service=None, token=None, return_raw: bool = False, plot: bool | str = False,
                   plot_size_arcsec: float | None = None):
    """Associate a transient at (ra, dec) with Rubin Object-table galaxies.

    Parameters
    ----------
    ra, dec : float
        Transient position in degrees (ICRS).
    band : {'u','g','r','i','z','y'}
        Band in which the Sersic model intensity is evaluated.
    radius_arcsec : float
        Cone-search radius for candidate hosts.
    max_hosts : int, optional
        Keep only the N most probable candidates (see :func:`compute_association`).
    dataset : {'dp2', 'dp1'}
        TAP schema to query.
    min_extendedness : float or None
        ``refExtendedness`` threshold used to drop point sources (``None`` keeps all).
    exclude_suspect : bool
        Drop candidates whose Sersic fit looks unreliable (see
        :func:`compute_association`) before computing probabilities.
    profile : {'mixture', 'analytic'}
        Light profile used to evaluate the models (see :func:`compute_association`).
    model : {'exponential', 'sersic'}
        Which MultiProFit fit to use: ``'exponential'`` (default; n fixed at 1,
        DP2 only, falls back to Sersic on DP1) or ``'sersic'`` (free n).
    band_fallback : bool
        If the requested band yields no candidate (typically no coverage in that
        band), try the other bands in the order r, i, g, z, u, y (see
        :func:`associate_bands`); ``meta['band_used']`` records the outcome.
    service : pyvo.dal.TAPService, optional
        Reuse an existing authenticated service.
    token : str, optional
        RSP token (otherwise resolved by :func:`sbw_host.tap.load_token`).
    return_raw : bool
        Also return the unfiltered cone-search rows (useful for plotting the
        point sources and failed fits that were rejected).
    plot : bool or str
        Make the diagnostic figure (real ``deep_coadd`` cutout when the token has
        ``read:image``, otherwise a render of the Sersic models) showing the
        transient, every candidate's half-light ellipse, probabilities and the
        rejected objects.  A string is a filename to save to; ``True`` returns the
        matplotlib figure.
    plot_size_arcsec : float, optional
        Field of view of the figure (default 2.7 x ``radius_arcsec``).

    Returns
    -------
    astropy.table.Table
        One row per candidate, sorted by descending ``p_sbw``; search metadata in
        ``.meta``.  Empty table if nothing usable lies within the radius.
        With ``return_raw=True`` the raw cone-search table is appended to the
        return value, and with ``plot=True`` the figure is appended after that,
        e.g. ``result, fig = associate_host(..., plot=True)``.
    """
    service = service or get_tap_service(token=token, dataset=dataset)
    bands = fallback_order(band, band_fallback)
    cands = query_candidates(ra, dec, radius_arcsec, band, dataset, service=service, token=token, model=model,
                             bands=bands)
    result = associate_bands(cands, ra, dec, band, band_fallback, min_extendedness=min_extendedness,
                             max_hosts=max_hosts, exclude_suspect=exclude_suspect, profile=profile)
    result.meta.update({"radius_arcsec": float(radius_arcsec), "dataset": dataset,
                        "queried_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")})
    if len(result) == 0:
        warnings.warn(f"No usable host candidates within {radius_arcsec}\" of ({ra:.5f}, {dec:.5f}).")
    out = [result]
    if return_raw:
        out.append(cands)
    if plot:
        from .plot import plot_with_best_image
        fig, _, _ = plot_with_best_image(result, raw=cands, size_arcsec=plot_size_arcsec, token=token)
        if isinstance(plot, str):
            fig.savefig(plot, dpi=110, bbox_inches="tight")
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            out.append(fig)
    return out[0] if len(out) == 1 else tuple(out)
