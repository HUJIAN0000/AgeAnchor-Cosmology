# AgeAnchor-Cosmology

Version **2.0.0**: a stellar-age-anchored inverse distance ladder with Pantheon+
and **DESI DR2**. The likelihood compares flat/curved LambdaCDM and flat CPL,
marginalizes stellar formation redshift in each cosmology, and fits a free
acoustic ruler. GC25 is the primary published stellar calibration; GC26 is an
alternative analysis of overlapping stellar observations.

This compact version supplies code, frozen configurations and result summaries for the study
*A stellar-age-anchored inverse distance ladder: Joint constraints and
stellar-calibration dependence with Pantheon+ and DESI DR2*. The manuscript is
not included in the code distribution. The paper's six authors are recorded in CITATION.cff.

## What the results mean

The primary flat SN+BAO+GC25 analysis gives
`H0 = 69.08 (+1.48/-1.40) km/s/Mpc` and `rd = 146.2 (+2.9/-2.9) Mpc`, conditional
on the adopted age summary, Gaussian systematic treatment and uniform formation
redshift over 11--30. GC26 provides a calibration alternative, not an independent
age factor to multiply with GC25. The inference is a consistency test within the
fitted backgrounds. Shared SN calibration information and stellar systematics
limit causal interpretations of a magnitude or Hubble-scale offset.

For the GC25 stellar input we use the estimator explicitly selected in Sec. 5.2
and the abstract of [JCAP 10 (2025) 030](https://doi.org/10.1088/1475-7516/2025/10/030).
GC26 uses Eq. 10 of [arXiv:2603.04872](https://arxiv.org/abs/2603.04872), with its
intrinsic-width/statistical-error wording treated as an input-definition limit.

## Install and reproduce

Tested environment: Windows with Python 3.11. Exact dependency versions are listed in
`requirements-current.txt`. Run from the repository root in an isolated environment:

```text
python -m pip install -r requirements-current.txt
python RUN_REPRO.py verify
python RUN_REPRO.py validate
```

The four observational inputs are bundled in `data/observational_inputs.zip`.
They are unpacked and checked against SHA-256 hashes before calculations. If the
archive is absent, the downloader uses the fixed upstream commits recorded in
`input_sources.json`. Data retain their upstream provenance and terms.

`validate` checks the full-covariance SN calculation against direct integration,
the age/formation calculations, the selected 1580 light curves, and the 13 BAO
components. Prebuilt figures and numerical tables are included in
`reference_outputs/` for inspection without sampling or plotting.

### Rebuild figures with optional plotting samples

Posterior sample arrays and full raw MCMC chains are omitted from this compact
source distribution. A separate version-matched plotting archive can be restored:

```text
python RUN_REPRO.py figures --samples-archive ../AgeAnchor-Cosmology-v2.0.0-plotting-samples.zip
```

The launcher verifies its file list and SHA-256 hashes before restoring arrays.
If the archive is distributed through a release, download it alongside this
source directory. Once restored, `python RUN_REPRO.py figures` regenerates all
12 figures and 6 tables, including the 5 joint triangles made with the unchanged
`basetools/cosmo_tools.py`. Outputs are written under `aa_revision/`; no manuscript
source is included. Alternatively, run independent sampling as described below
and point the source resolver to those new results.

## File map

| Location | Contents |
| --- | --- |
| `fresh_results/` | Principal 29-configuration sampler, frozen settings and result products |
| `release_results/` | GC25 CPL restricted-prior products and independent-seed checks |
| `release_configurations.json` | Four additional GC25 CPL configurations and seeds |
| `reanalysis/` | Likelihood, numerical checks and complementary sensitivity configurations |
| `integration/` | Source resolver and publication plotting/table generation |
| `basetools/` | Author-designated triangle renderer |
| `audit/figure_sources.json` | Figure-to-posterior source hashes and parameter lists |
| `audit/gc25_cpl_validation.json` | Reopened-chain ESS, high-order checks and seed agreement |
| `RELEASE_MANIFEST.json` | Frozen file hashes, excluding generated outputs and the manifest itself |

The optional archive contains 60000 derived plotting draws per included case,
which is distinct from full-chain effective sample size. Its exact contents are
listed in `plotting_samples_manifest.json`. The compact source upload retains
frozen summaries, seeds, configurations and the numerical validation records.
Large complete sampler chains remain in the author's separate project archive.

## Independent sampling

Use a separate output directory to preserve the supplied results. The launcher
sets numerical DLL paths for its Python environment on Windows and limits BLAS
threads before spawning numerical processes.

```text
python RUN_REPRO.py sample --suite gc25-priors --output reruns/gc25_priors --workers 2 --threads 4
python RUN_REPRO.py sample --suite principal --output reruns/principal --workers 2 --threads 4
python RUN_REPRO.py sample --suite sensitivity --output reruns/sensitivity --workers 2 --threads 4
python RUN_REPRO.py sample --suite formation --output reruns/formation --workers 2 --threads 4
```

Reduce workers/threads if needed. To plot independent chains, change the result
directories in `integration/posterior_sources.py` to those new locations. The
drivers reuse a converged case when rerun in its existing output directory and
resume compatible incomplete cases. Different machines need not generate
bitwise-identical chains; compare estimates with their Monte Carlo uncertainty.

## Changes from the PRL-era pipeline

Version 2 replaces the old top-level numbered scripts with the common covariance
likelihood and portable launcher. It uses DESI DR2, distinguishes stellar from
already formation-corrected cosmic ages, retains parameter covariance and CPL
tails, and includes matched no-age controls. GC25 CPL prior sensitivity now uses
the same stellar-formation likelihood as the primary analysis and has two
independent seeds per restriction.

The PRL-era source remains accessible through Git history, including
[commit d2dcc86173fd9932b6af8245b28c71e3bf6027af](https://github.com/HUJIAN0000/AgeAnchor-Cosmology/tree/d2dcc86173fd9932b6af8245b28c71e3bf6027af).
See `MIGRATION_FROM_PRL.md` and `RELEASE_NOTES_v2.0.0.md`. This replacement does
not require force-pushing or rewriting history.

## License and citation

The existing **MIT License** and copyright notice are retained in `LICENSE`.
Observational inputs remain subject to their original release terms. Please cite this
software using `CITATION.cff`.

Repository: https://github.com/HUJIAN0000/AgeAnchor-Cosmology

Contact: Jian Hu (`dg1626002@smail.nju.edu.cn`).
