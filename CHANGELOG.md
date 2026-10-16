# Changelog

## 0.2.0 (2026-10-16)

- New table-free core, `sbw_host.core`: `sbw_exponential`, `sbw_sersic` and
  `sbw_model` evaluate SBW for one transient against hosts given as plain arrays
  (positions, half-light radius or scale length, total flux or `i_eff` / `i_0`,
  axis ratio, position angle), with `sbw_probabilities` separated out so priors can
  be added before normalising.  `compute_association` now calls this core; its
  outputs are unchanged (regression test to 1e-9).
- New model-free variant, `sbw_host.stamps.sbw_stamps`: SBW from deblended host
  stamps of any sizes, placed by their `[0, 0]`-pixel origins or by per-stamp WCS,
  with a `box` x `box` mean around the transient for the hostless indicator.
- Version bumped to 0.2.0; Python >= 3.12.

## 0.1.0 (2026-10-14)

First packaged release, collecting the work done against Rubin DP2 in October 2026.

- SBW host association from the Object table's MultiProFit fits (`associate_host`),
  with the hostless indicator `frac_light_beyond` and the DLR comparison
  (`d_dlr`, `p_dlr`).
- Exponential fit by default, free-n Sérsic fit as an option; both evaluated as the
  Gaussian mixture the pipeline actually fits, with the half-light radius
  `1.386 x reff` convention (see `docs/sersic_reff_report/`).
- `suspect` flag for deblend-halo phantom fits; band fallback (r, i, g, z, u, y).
- Batch mode (`associate_many`, `sbw-host-batch`) with grouped asynchronous TAP
  queries or an LSDB/HATS backend, streaming chunk by chunk.
- Diagnostic figures on real deep_coadd cutouts; `sbw-host-review` for random
  SBW/DLR disagreements in a batch table.
