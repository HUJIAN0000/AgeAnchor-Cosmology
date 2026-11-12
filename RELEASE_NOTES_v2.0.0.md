# v2.0.0 — Joint stellar-age calibration with Pantheon+ and DESI DR2

This release replaces the PRL-era numbered scripts with a common full-covariance
likelihood, frozen production configurations, posterior summaries and
reproducible validation/figure entry points.

## Scientific implementation

- GC25 is the primary published stellar calibration; GC26 is an alternative.
- DESI DR2 uses the 13-component vector and full covariance; the ruler is free.
- Stellar formation redshift is marginalized within each cosmology.
- Flat and curved LambdaCDM and flat CPL retain the covariance with supernova
  magnitude and the acoustic ruler.
- CPL prior sensitivity uses GC25 with broad bounds, Omega_m > 0.25 and
  w0 + wa < 0. Each restriction has two independently sampled ensembles.
- Matched no-age controls, age-error changes and formation assumptions expose
  the conditional interpretation of the dimensional scale constraints.

## Reproduction material

- One portable launcher for verification, likelihood validation, figure/table
  regeneration and independent sampling.
- Twelve figures, five joint triangles and six numerical tables are reproducible
  after restoring the separate plotting-samples archive. Prebuilt outputs are included in the compact source distribution.
- Four observational inputs are bundled with hashes and pinned upstream URLs.
- Four additional GC25 CPL runs pass the specified stopping criteria; minimum
  recomputed ESS exceeds 4000. Independent-seed median differences are below
  0.034 central-interval half-widths. Forty-eight higher-order checks differ by
  less than 1.3e-12 in log posterior on the evaluated points.
- Original MIT license retained. Git history preserves the previous pipeline.

The large complete sampler chains are retained separately; configurations allow
independent reruns. This release makes no claim of journal acceptance or a
unique physical localization of a Hubble-scale discrepancy. A Zenodo DOI can
be added after an actual archive is created.
