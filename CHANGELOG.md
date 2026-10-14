# Changelog

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
