"""Batch SBW host association for many transients.

Instead of one TAP cone search per transient (~3.5 s each), transients are grouped
into sky cells, one asynchronous regional query is issued per occupied cell (a few
run concurrently), and each transient's candidates are then selected from the
downloaded rows with a KD-tree and passed to the same
:func:`hostassoc.associate.compute_association` as the single-object path, so all
columns, flags and conventions are identical.  Cells holding only one or two
transients use ordinary small cone searches instead.

Python::

    from hostassoc.batch import associate_many
    wide = associate_many(ra, dec, keys=dia_object_ids, n_hosts=3)
    wide, long = associate_many(ra, dec, keys=..., n_hosts=3, return_long=True)

Command line::

    python -m hostassoc.batch transients.csv --key-col diaObjectId --ra-col ra --dec-col dec \\
        --n-hosts 3 --out hosts.ecsv --long hosts_long.ecsv

Backends: ``backend="tap"`` (default; the RSP TAP service, grouped regional
queries) or ``backend="lsdb"`` (``--lsdb``): the DP2 Object table as a HATS
catalogue read with LSDB, e.g. the NERSC copy at
``/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/object_collection``.  Both backends
stream: candidates are fetched for one group / chunk of transients, their
associations are computed and flattened into output rows, the candidate rows are
discarded, and the next group is processed.  Memory therefore stays bounded for
lists of any length, and the two backends give the same results for the same
catalogue contents.

Wide output: one row per transient with ``transient_key``, ``transient_ra``,
``transient_dec``, ``n_in_radius``, ``n_candidates``, ``n_dropped_bad_fit``,
``n_dropped_pointlike``, ``band_used``, ``dlr_host_objectId`` /
``dlr_host_ra`` / ``dlr_host_dec`` / ``dlr_host_d_dlr`` (the smallest-d_DLR
candidate, whatever its value, so the d_DLR < 4 cut can be applied downstream)
and, for i = 1..N in order of decreasing ``p_sbw``: ``host_i_objectId``,
``host_i_ra``, ``host_i_dec``, ``host_i_p_sbw``, ``host_i_frac_light_beyond``,
``host_i_d_dlr``, ``host_i_sep_arcsec``, ``host_i_mag``, ``host_i_suspect``.
Missing hosts have objectId 0 and NaN values.
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

from .associate import DEFAULT_MODEL, associate_bands, compute_association, fallback_order, query_candidates, query_region, resolve_model
from .tap import get_tap_service

HOST_FIELDS = ("objectId", "ra", "dec", "p_sbw", "frac_light_beyond", "d_dlr", "sep_arcsec", "mag", "suspect")
LSDB_DP2_OBJECT_PATH = "/global/cfs/cdirs/lsst/shared/rubin/DP2/HATS/object_collection"
LSDB_MAX_NEIGHBORS = 1000   # upper bound on transients per galaxy in the crossmatch (galaxies are the left side)
LSDB_WORKERS = 4            # Dask threads used for the crossmatch when no distributed Client is active
LSDB_CHUNK = 2000           # transients per crossmatch pass (bounds peak memory independently of list size)


# --------------------------------------------------------------------------- geometry
def _unit(ra, dec):
    ra, dec = np.radians(np.asarray(ra, float)), np.radians(np.asarray(dec, float))
    return np.c_[np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]


def _chord(deg):
    return 2.0 * np.sin(np.radians(deg) / 2.0)


def group_positions(ra, dec, cell_deg=0.3, radius_arcsec=30.0, min_group=3):
    """Group transients into sky cells; return a list of (indices, center_ra, center_dec, query_radius_deg).

    Cells are a grid in (RA cos Dec, Dec) of side ``cell_deg``; the query circle for
    a cell is the smallest circle around the cell's transients plus the search
    radius.  Groups smaller than ``min_group`` are returned as single-transient
    groups with ``query_radius_deg = None`` (meaning: use a normal cone search).
    """
    ra = np.asarray(ra, float) % 360.0
    dec = np.asarray(dec, float)
    cell = max(float(cell_deg), 1e-3)
    idec = np.floor((dec + 90.0) / cell).astype(int)
    dec_cell_mid = (idec + 0.5) * cell - 90.0
    cosd = np.maximum(np.cos(np.radians(dec_cell_mid)), 1e-6)
    ira = np.floor(ra * cosd / cell).astype(int)
    groups = {}
    for i, key in enumerate(zip(idec, ira)):
        groups.setdefault(key, []).append(i)
    out = []
    for idx in groups.values():
        idx = np.array(idx)
        if len(idx) < min_group:
            for i in idx:
                out.append((np.array([i]), ra[i], dec[i], None))
            continue
        u = _unit(ra[idx], dec[idx])
        c = u.mean(axis=0)
        c /= np.linalg.norm(c)
        cra = np.degrees(np.arctan2(c[1], c[0])) % 360.0
        cdec = np.degrees(np.arcsin(np.clip(c[2], -1, 1)))
        max_sep = np.degrees(np.arccos(np.clip(u @ c, -1, 1))).max()
        out.append((idx, cra, cdec, max_sep + radius_arcsec / 3600.0 + 2.0 / 3600.0))
    return out


# --------------------------------------------------------------------------- output rows
def _empty_wide_row(n_hosts):
    row = {"n_in_radius": 0, "n_candidates": 0, "n_dropped_bad_fit": 0, "n_dropped_pointlike": 0, "band_used": "",
           "dlr_host_objectId": 0, "dlr_host_ra": np.nan, "dlr_host_dec": np.nan, "dlr_host_d_dlr": np.nan}
    for i in range(1, n_hosts + 1):
        for f in HOST_FIELDS:
            row[f"host_{i}_{f}"] = 0 if f == "objectId" else (False if f == "suspect" else np.nan)
    return row


def wide_row(res: Table, n_hosts: int) -> dict:
    """Flatten one association result into the wide-format row (without the key/position)."""
    row = _empty_wide_row(n_hosts)
    row["n_in_radius"] = int(res.meta.get("n_in_radius", 0))
    row["n_candidates"] = int(res.meta.get("n_candidates", len(res)))
    row["n_dropped_bad_fit"] = int(res.meta.get("n_dropped_bad_fit", 0))
    row["n_dropped_pointlike"] = int(res.meta.get("n_dropped_pointlike", 0))
    row["band_used"] = str(res.meta.get("band_used", res.meta.get("band", "")))
    if len(res) == 0:
        return row
    d = np.asarray(res["d_dlr"], float)
    if np.any(np.isfinite(d)):
        j = int(np.nanargmin(d))
        row["dlr_host_objectId"] = int(res["objectId"][j])
        row["dlr_host_ra"] = float(res["ra"][j])
        row["dlr_host_dec"] = float(res["dec"][j])
        row["dlr_host_d_dlr"] = float(d[j])
    for i, r in enumerate(res[:n_hosts], start=1):
        row[f"host_{i}_objectId"] = int(r["objectId"])
        row[f"host_{i}_suspect"] = bool(r["suspect"])
        for f in ("ra", "dec", "p_sbw", "frac_light_beyond", "d_dlr", "sep_arcsec", "mag"):
            row[f"host_{i}_{f}"] = float(r[f])
    return row


class _Collector:
    """Flattens results as they arrive so only the output rows are kept in memory."""

    def __init__(self, n, n_hosts, return_long):
        self.rows = [None] * n
        self.n_hosts = n_hosts
        self.return_long = return_long
        self.long_parts = []
        self.done = 0

    def add(self, i, key, res):
        self.rows[i] = wide_row(res, self.n_hosts)
        if self.return_long and len(res):
            r = res.copy()
            r.meta.clear()
            r.add_column(np.repeat(np.asarray([key]), len(r)), index=0, name="transient_key")
            self.long_parts.append(r)
        self.done += 1


def _assemble(col, keys, ra, dec, n_hosts, band, radius_arcsec, dataset, model,
              min_extendedness, exclude_suspect, profile, return_long, backend):
    rows = [r if r is not None else _empty_wide_row(n_hosts) for r in col.rows]
    wide = Table()
    wide["transient_key"] = keys
    wide["transient_ra"] = ra
    wide["transient_dec"] = dec
    for c in rows[0]:
        wide[c] = np.array([r[c] for r in rows])
    wide.meta.update({"band": band, "radius_arcsec": float(radius_arcsec), "dataset": dataset, "model": model,
                      "n_hosts": int(n_hosts), "min_extendedness": min_extendedness,
                      "exclude_suspect": exclude_suspect, "profile": profile, "backend": backend,
                      "note": "hosts ordered by p_sbw; dlr_host_* is the min-d_dlr candidate regardless of the d_DLR<4 cut"})
    for c in wide.colnames:
        if wide[c].dtype.kind == "f":
            wide[c].format = "%.6f" if (c.endswith("_ra") or c.endswith("_dec")) else "%.4g"
    if not return_long:
        return wide
    long = vstack(col.long_parts, metadata_conflicts="silent") if col.long_parts else Table()
    long.meta.update(wide.meta)
    return wide, long


# --------------------------------------------------------------------------- LSDB backend
def lsdb_all_columns(lsdb_path=LSDB_DP2_OBJECT_PATH) -> list[str]:
    """Every column available in a HATS catalogue/collection (not just the collection defaults)."""
    import lsdb
    probe = lsdb.open_catalog(lsdb_path)
    cols = getattr(probe, "all_columns", None)
    return list(cols) if cols is not None else list(probe.columns)


def lsdb_open(lsdb_path=LSDB_DP2_OBJECT_PATH, band: str = "r", model: str = DEFAULT_MODEL, catalog=None, bands=None):
    """Open (or validate) an LSDB catalogue with the columns ``model`` needs; returns ``(catalog, model)``."""
    import lsdb
    from .associate import _OPTIONAL_COLUMNS, model_columns, _check_model

    model = _check_model(model)
    if catalog is None:
        # A HATS *collection* may define default columns; ``open_catalog(path)`` then
        # exposes only those, so look at ``all_columns`` before deciding what exists.
        have = set(lsdb_all_columns(lsdb_path))
        if f"{model}_reff_x" not in have:
            if "sersic_reff_x" in have and model != "sersic":
                warnings.warn(f"{lsdb_path} has no '{model}' fit columns; using the Sersic fit instead.")
                model = "sersic"
            else:
                near = sorted(c for c in have if any(k in c.lower() for k in ("sersic", "exponential", "reff", "exp_")))
                raise RuntimeError(
                    f"{lsdb_path} has no '{model}' model columns ({model}_reff_x etc.). "
                    f"Columns with similar names: {near[:30] or 'none'}. "
                    "Use `hostassoc-batch --lsdb-list-columns [--lsdb-path ...]` to list all columns.")
        cols = list(model_columns(band, model, bands)) + [c for c in _OPTIONAL_COLUMNS if c in have]
        missing = [c for c in cols if c not in have]
        if missing:
            raise RuntimeError(f"columns missing from {lsdb_path}: {missing}")
        catalog = lsdb.open_catalog(lsdb_path, columns=cols)
    else:
        have = set(catalog.columns)
        need = [c for c in model_columns(band, model) if c not in have]   # fallback bands are optional here
        if need:
            hint = ""
            try:
                if all(c in set(catalog.all_columns) for c in need):
                    hint = (" They exist in the catalogue but were not loaded; open it with "
                            "lsdb.open_catalog(path, columns=[...]) including them, or pass lsdb_path instead.")
            except Exception:
                pass
            raise RuntimeError(f"lsdb catalog lacks columns {need}.{hint}")
    return catalog, model


def _dask_context(n_workers: int):
    """Limit Dask's threaded scheduler unless the user already runs a distributed Client."""
    import dask
    try:
        from dask.distributed import default_client
        default_client()
        return dask.config.set({})          # a Client exists: leave scheduling to it
    except Exception:
        return dask.config.set(scheduler="threads", num_workers=max(1, int(n_workers)))


def _plain_dtypes(frame):
    """pyarrow-backed dtypes -> plain numpy so astropy/numpy code downstream behaves."""
    frame = frame.convert_dtypes(dtype_backend="numpy_nullable")
    for c in frame.columns:
        if str(frame[c].dtype) in ("Int64", "Int32"):
            frame[c] = frame[c].astype("float64" if frame[c].isna().any() else "int64")
        elif str(frame[c].dtype) in ("Float64", "Float32"):
            frame[c] = frame[c].astype("float64")
        elif str(frame[c].dtype) == "boolean":
            frame[c] = frame[c].fillna(False).astype(bool)
        elif str(frame[c].dtype) == "string":
            frame[c] = frame[c].astype(object)
    return frame


def lsdb_candidate_chunks(ra, dec, keys, radius_arcsec: float = 30.0, band: str = "r", model: str = DEFAULT_MODEL,
                          lsdb_path=LSDB_DP2_OBJECT_PATH, catalog=None, max_neighbors: int = LSDB_MAX_NEIGHBORS,
                          verbose: bool = True, n_workers: int = LSDB_WORKERS, chunk: int = LSDB_CHUNK, bands=None):
    """Generator: crossmatch the transients ``chunk`` at a time against a HATS/LSDB Object table.

    Yields ``(indices, frame, model)`` per chunk, where ``indices`` index the input
    arrays and ``frame`` is a pandas DataFrame with one row per (transient,
    candidate) pair: the transient key in ``_tkey`` plus the Object columns (TAP
    names, unsuffixed).  A chunk is read only when the consumer asks for it, so the
    caller can compute and flatten one chunk's associations before the next is
    read.  ``catalog`` may be an already-open ``lsdb.Catalog``; otherwise
    ``lsdb_path`` is opened with just the needed columns.

    Memory: every Object partition touched is read into memory while it is
    processed, so the crossmatch runs with at most ``n_workers`` Dask threads (unless
    a ``dask.distributed`` Client is active, in which case its workers and memory
    limits apply).  Raise ``n_workers`` on a compute node with plenty of memory; lower
    it (or use a Client with ``memory_limit``) if the process gets killed.
    """
    import lsdb
    import pandas as pd

    catalog, model = lsdb_open(lsdb_path, band, model, catalog, bands)
    # The transients go on the RIGHT side of the crossmatch: lsdb requires the right
    # catalogue's margin cache to be at least as wide as the search radius, and we
    # build the transient catalogue ourselves so we can give it any margin.
    tr_all = pd.DataFrame({"_tkey": np.asarray(keys), "_tra": np.asarray(ra, float), "_tdec": np.asarray(dec, float)})
    # sort by position so each chunk touches few Object partitions
    order = np.lexsort((tr_all["_tra"].values, np.round(tr_all["_tdec"].values / 2.0)))
    chunk = max(1, int(chunk))
    t0 = time.time()
    with _dask_context(n_workers):
        for start in range(0, len(tr_all), chunk):
            idx = order[start:start + chunk]
            tr = tr_all.iloc[idx].reset_index(drop=True)
            transients = lsdb.from_dataframe(tr, ra_column="_tra", dec_column="_tdec", lowest_order=3, highest_order=9,
                                             margin_threshold=float(radius_arcsec) + 15.0)
            if verbose:
                print(f"[batch/lsdb] crossmatching transients {start + 1}-{start + len(tr)} of {len(tr_all)} "
                      f"(radius {radius_arcsec}\", {n_workers} threads)", file=sys.stderr, flush=True)
            try:
                xm = catalog.crossmatch(transients, n_neighbors=int(max_neighbors), radius_arcsec=float(radius_arcsec),
                                        suffixes=("", "_t"), suffix_method="all_columns")
            except TypeError:   # older lsdb without suffix_method
                xm = catalog.crossmatch(transients, n_neighbors=int(max_neighbors), radius_arcsec=float(radius_arcsec),
                                        suffixes=("", "_t"))
            part = xm.compute().reset_index(drop=True)
            part = part.rename(columns={"_tkey_t": "_tkey"})
            drop = [c for c in part.columns if c in ("_tra_t", "_tdec_t", "_dist_arcsec")
                    or c.startswith("Norder") or c.startswith("Dir") or c.startswith("Npix")]
            part = _plain_dtypes(part.drop(columns=drop))
            if verbose:
                print(f"[batch/lsdb]   {len(part)} pairs ({time.time() - t0:.0f} s elapsed)", file=sys.stderr, flush=True)
            yield idx, part, model
            del part, xm, transients


def lsdb_candidates(ra, dec, keys, **kwargs):
    """All candidate pairs at once (concatenation of :func:`lsdb_candidate_chunks`); returns ``(frame, model)``."""
    import pandas as pd
    frames, model = [], kwargs.get("model", DEFAULT_MODEL)
    for _, part, model in lsdb_candidate_chunks(ra, dec, keys, **kwargs):
        frames.append(part)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), model


def _empty_cands(band, model, bands=()):
    cands = Table()
    cands.meta["model"] = model
    for c in ("objectId", "coord_ra", "coord_dec", f"{model}_ra", f"{model}_dec", f"{model}_reff_x",
              f"{model}_reff_y", f"{model}_rho", "refExtendedness"):
        cands[c] = np.zeros(0)
    for b in dict.fromkeys((band,) + tuple(bands)):
        cands[f"{b}_{model}Flux"] = np.zeros(0)
        cands[f"{b}_cModelFlux"] = np.zeros(0)
    if model == "sersic":
        cands["sersic_index"] = np.zeros(0)
    return cands


def _iter_results_from_frame(frame, idx, keys, ra, dec, band, model, kw, band_fallback=True):
    """Yield ``(i, result)`` for each transient index in ``idx`` from its crossmatched rows."""
    from .associate import plain_table
    groups = {k: g for k, g in frame.groupby("_tkey", sort=False)} if len(frame) else {}
    for i in idx:
        g = groups.get(keys[i])
        cands = (_empty_cands(band, model, fallback_order(band, band_fallback)) if g is None
                 else plain_table(Table.from_pandas(g.drop(columns=["_tkey"])), model))
        yield int(i), associate_bands(cands, float(ra[i]), float(dec[i]), band, band_fallback, **kw)


# --------------------------------------------------------------------------- main entry point
def associate_many(ra, dec, keys=None, n_hosts: int = 3, band: str = "r", radius_arcsec: float = 30.0,
                   dataset: str = "dp2", model: str = DEFAULT_MODEL, min_extendedness: float | None = 0.5,
                   exclude_suspect: bool = False, profile: str = "mixture", cell_deg: float = 0.3,
                   max_concurrent: int = 6, service=None, token=None, return_long: bool = False,
                   verbose: bool = True, backend: str = "tap", lsdb_path=LSDB_DP2_OBJECT_PATH,
                   lsdb_catalog=None, lsdb_workers: int = LSDB_WORKERS, lsdb_chunk: int = LSDB_CHUNK,
                   band_fallback: bool = True):
    """Associate many transients at once.

    Parameters
    ----------
    ra, dec : array-like
        Transient positions in degrees (ICRS).
    keys : array-like, optional
        A unique identifier per transient (any dtype: ints, strings, ...).  Carried
        through unchanged as ``transient_key``.  Defaults to the row index.
    n_hosts : int
        Number of SBW-ranked candidates to flatten into the wide table.  All
        candidates within the radius still enter the probabilities (and the long
        table).
    cell_deg : float
        TAP backend: sky-cell size for grouping; each occupied cell with >= 3
        transients gets one asynchronous regional query (0.3 deg -> ~45k Object
        rows, ~15-20 s).
    max_concurrent : int
        TAP backend: regional queries / cone searches run concurrently (the RSP
        allows ~12).
    return_long : bool
        Also return the stacked per-candidate table with a ``transient_key`` column.
    backend : {'tap', 'lsdb'}
        ``'tap'`` queries the RSP TAP service; ``'lsdb'`` reads a HATS copy of the
        Object table with LSDB (``lsdb_path``, default the NERSC DP2 collection) or
        an already-open ``lsdb_catalog``.  ``lsdb_workers`` Dask threads are used
        (ignored when a ``dask.distributed`` Client is active) and the transients are
        crossmatched ``lsdb_chunk`` at a time; each chunk's associations are computed
        and flattened before the next chunk is read.
    band_fallback : bool
        If a transient has no candidate in ``band`` (usually no coverage in that
        band), try the other bands in the order r, i, g, z, u, y and record the band
        used in ``band_used`` (see :func:`hostassoc.associate.associate_bands`).
    Other arguments are passed to :func:`hostassoc.associate.compute_association`.

    Returns
    -------
    Table, or (Table, Table) with ``return_long=True``.
    """
    ra = np.atleast_1d(np.asarray(ra, float))
    dec = np.atleast_1d(np.asarray(dec, float))
    n = len(ra)
    if len(dec) != n:
        raise ValueError("ra and dec must have the same length")
    if keys is None:
        keys = np.arange(n)
    keys = np.atleast_1d(np.asarray(keys))
    if len(keys) != n:
        raise ValueError("keys must have one entry per transient")
    if len(np.unique(keys)) != n:
        raise ValueError("transient keys must be unique")
    if n_hosts < 1:
        raise ValueError("n_hosts must be >= 1")
    kw = dict(min_extendedness=min_extendedness, exclude_suspect=exclude_suspect, profile=profile)
    bands = fallback_order(band, band_fallback)
    col = _Collector(n, n_hosts, return_long)

    if backend == "lsdb":
        t0 = time.time()
        model_used = model
        for idx, frame, model_used in lsdb_candidate_chunks(ra, dec, keys, radius_arcsec=radius_arcsec, band=band,
                                                            model=model, lsdb_path=lsdb_path, catalog=lsdb_catalog,
                                                            verbose=verbose, n_workers=lsdb_workers, chunk=lsdb_chunk,
                                                            bands=bands):
            for i, res in _iter_results_from_frame(frame, idx, keys, ra, dec, band, model_used, kw, band_fallback):
                col.add(i, keys[i], res)
            del frame
            if verbose:
                print(f"[batch/lsdb]   associations done for {col.done}/{n} transients ({time.time() - t0:.0f} s)",
                      file=sys.stderr, flush=True)
        return _assemble(col, keys, ra, dec, n_hosts, band, radius_arcsec, dataset, model_used,
                         min_extendedness, exclude_suspect, profile, return_long, backend)
    if backend != "tap":
        raise ValueError("backend must be 'tap' or 'lsdb'")

    service = service or get_tap_service(token=token, dataset=dataset)
    model = resolve_model(model, service, dataset)
    groups = group_positions(ra, dec, cell_deg=cell_deg, radius_arcsec=radius_arcsec)
    n_regional = sum(1 for g in groups if g[3] is not None)
    if verbose:
        print(f"[batch] {n} transients -> {n_regional} regional queries + {len(groups) - n_regional} single cone searches",
              file=sys.stderr, flush=True)
    chord = _chord(radius_arcsec / 3600.0)

    def do_group(g):
        idx, cra, cdec, rq = g
        if rq is None:
            i = int(idx[0])
            cands = query_candidates(ra[i], dec[i], radius_arcsec, band, dataset, service=service, model=model, bands=bands)
            return [(i, associate_bands(cands, ra[i], dec[i], band, band_fallback, **kw))]
        try:
            region = query_region(service, cra, cdec, rq, band, dataset, model, bands=bands)
        except Exception as e:  # fall back to individual cones for this cell
            warnings.warn(f"regional query at ({cra:.4f}, {cdec:.4f}) r={rq:.3f} failed ({e!s:.80}); using cone searches")
            out = []
            for i in idx:
                cands = query_candidates(ra[i], dec[i], radius_arcsec, band, dataset, service=service, model=model, bands=bands)
                out.append((int(i), associate_bands(cands, ra[i], dec[i], band, band_fallback, **kw)))
            return out
        out = []
        if len(region) == 0:
            for i in idx:
                out.append((int(i), associate_bands(region, ra[i], dec[i], band, band_fallback, **kw)))
            return out
        tree = cKDTree(_unit(region["coord_ra"], region["coord_dec"]))
        hits = tree.query_ball_point(_unit(ra[idx], dec[idx]), chord)
        for i, h in zip(idx, hits):
            sub = region[np.asarray(h, dtype=int)] if len(h) else region[:0]
            sub.meta["model"] = model
            out.append((int(i), associate_bands(sub, ra[i], dec[i], band, band_fallback, **kw)))
        return out

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrent))) as ex:
        futures = [ex.submit(do_group, g) for g in groups]
        for fut in as_completed(futures):
            for i, res in fut.result():
                col.add(i, keys[i], res)
            if verbose:
                print(f"[batch] {col.done}/{n} transients done ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    return _assemble(col, keys, ra, dec, n_hosts, band, radius_arcsec, dataset, model,
                     min_extendedness, exclude_suspect, profile, return_long, backend)


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(prog="hostassoc-batch",
                                description="Batch SBW host association for a table of transient positions.")
    p.add_argument("positions", nargs="?", default=None,
                   help="input table (csv, ecsv, fits, parquet, ... anything astropy Table.read understands)")
    p.add_argument("--ra-col", default="ra"); p.add_argument("--dec-col", default="dec")
    p.add_argument("--key-col", default=None, help="column with a unique transient key (default: row index)")
    p.add_argument("--n-hosts", type=int, default=3)
    p.add_argument("--band", default="r", choices=list("ugrizy"))
    p.add_argument("--radius", type=float, default=30.0)
    p.add_argument("--dataset", default="dp2")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=["exponential", "sersic"])
    p.add_argument("--exclude-suspect", action="store_true")
    p.add_argument("--no-band-fallback", action="store_true", help="do not try other bands (r,i,g,z,u,y) when --band has no usable fluxes")
    p.add_argument("--cell-deg", type=float, default=0.3)
    p.add_argument("--max-concurrent", type=int, default=6)
    p.add_argument("--lsdb", action="store_true", help="read the Object table as a HATS catalogue with LSDB instead of TAP")
    p.add_argument("--lsdb-path", default=LSDB_DP2_OBJECT_PATH, help="HATS collection path (default: NERSC DP2 object_collection)")
    p.add_argument("--lsdb-workers", type=int, default=LSDB_WORKERS,
                   help=f"with --lsdb: Dask threads for the crossmatch (default {LSDB_WORKERS}; each thread holds one Object partition in memory)")
    p.add_argument("--lsdb-chunk", type=int, default=LSDB_CHUNK, help=f"with --lsdb: transients per crossmatch pass (default {LSDB_CHUNK})")
    p.add_argument("--dask-workers", type=int, default=0,
                   help="with --lsdb: instead start a dask.distributed Client with this many single-threaded workers")
    p.add_argument("--dask-memory", default=None, help="with --dask-workers: memory limit per worker, e.g. 16GiB")
    p.add_argument("--lsdb-list-columns", metavar="PATTERN", nargs="?", const="", default=None,
                   help="list the columns of the HATS catalogue at --lsdb-path (optionally only those containing PATTERN) and exit")
    p.add_argument("--out", default=None, help="wide output table (format from extension, e.g. .ecsv .csv .fits .parquet)")
    p.add_argument("--long", default=None, help="optional long (per-candidate) output table")
    p.add_argument("--token", default=None)
    a = p.parse_args(argv)
    if a.lsdb_list_columns is not None:
        cols = lsdb_all_columns(a.lsdb_path)
        pat = a.lsdb_list_columns.lower()
        for c in cols:
            if pat in c.lower():
                print(c)
        print(f"# {sum(pat in c.lower() for c in cols)} of {len(cols)} columns", file=sys.stderr)
        return 0
    if not a.positions or not a.out:
        p.error("positions and --out are required")
    tab = Table.read(a.positions)
    keys = tab[a.key_col] if a.key_col else None
    client = None
    if a.lsdb and a.dask_workers > 0:
        from dask.distributed import Client
        client = Client(n_workers=a.dask_workers, threads_per_worker=1, memory_limit=a.dask_memory)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = associate_many(tab[a.ra_col], tab[a.dec_col], keys=keys, n_hosts=a.n_hosts, band=a.band,
                             radius_arcsec=a.radius, dataset=a.dataset, model=a.model,
                             exclude_suspect=a.exclude_suspect, cell_deg=a.cell_deg,
                             max_concurrent=a.max_concurrent, token=a.token, return_long=bool(a.long),
                             backend="lsdb" if a.lsdb else "tap", lsdb_path=a.lsdb_path,
                             lsdb_workers=a.lsdb_workers, lsdb_chunk=a.lsdb_chunk,
                             band_fallback=not a.no_band_fallback)
    if client is not None:
        client.close()
    wide, long = res if a.long else (res, None)
    wide.write(a.out, overwrite=True)
    print("wrote", a.out, f"({len(wide)} transients)")
    if long is not None:
        long.write(a.long, overwrite=True)
        print("wrote", a.long, f"({len(long)} candidate rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
