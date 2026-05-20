# Stellar ages localize the Hubble tension to the supernova absolute magnitude

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the analysis pipeline and figure-generation scripts for the manuscript:

> **Stellar ages localize the Hubble tension to the supernova absolute magnitude**
> Jian Hu (Dali University), 2026.

The pipeline uses the absolute age of the oldest stars (Gaia-calibrated globular clusters; HD 140283 asteroseismology) as an independent geometric calibration anchor to break the SN Ia $M_B$–$H_0$ degeneracy, demonstrating that under flat $\Lambda$CDM the Hubble tension is localised within the SN Ia absolute-magnitude calibration step rather than between cosmic epochs.

---

## Repository contents

| File | Purpose | Output |
|------|---------|--------|
| `cosmoc.py` | Shared physics: cosmology integrals (Numba-jitted), Pantheon+ and DESI Y1 BAO loaders, reference values, scenario definitions. | (imported by all scripts) |
| `cosmo_tools.py` | Post-processing utilities: posterior statistics, GetDist-based triangle plotting, multi-chain overlay. | (imported by 06) |
| `01_concept_figure.py` | Figure 1 — analytic illustration of how $t_0$ contours break the $H_0$–$\Omega_m$ degeneracy. No MCMC. | `Figure1_Concept.pdf` |
| `02_main_NoBAOf2.py` | Figure 2 — core result: Pantheon+ + stellar age, flat $\Lambda$CDM, three age scenarios. | `Figure2_Main_NoBAO.pdf` |
| `03_with_BAO2f.py` | Figure 3 — adds DESI Y1 BAO; tests whether the inferred $r_d$ matches BBN. | `Figure3A_Triangle_BAO.pdf`, `Figure3B_rd_distribution.pdf` |
| `04_sensitivity_curvef.py` | Figure 4 — continuous $H_0(t_0)$ sensitivity curve over $12$–$15$ Gyr. | `Figure4_Sensitivity.pdf` |
| `05_dt_form_testf.py` | Extended Data Fig. 1 — robustness to formation-time delay $\Delta t_{\rm form}$. | `EDFig1_dt_form.pdf` |
| `06_curvature_tests1p4s.py` | Extended Data Fig. 2 — robustness to spatial curvature ($\Omega_k$ free); three age scenarios overlaid. | `EDFig2_Curvature_AllScenarios.pdf` |
| `07_w0wa_testf2.py` | Extended Data Fig. 3 — robustness under Chevallier–Polarski–Linder dynamical dark energy ($w_0 w_a$CDM). | `EDFig3_w0wa.pdf` |
| `08_age_systematics_scan2.py` | Extended Data Fig. 4 — sensitivity to assumed stellar-age systematic $\sigma_{\rm age}$. | `EDFig4_age_sigma_scan.pdf` |
| `09_joint_age_priors2.py` | Extended Data Fig. 5 — joint inference using Valcin et al. globular clusters + Lundkvist et al. HD 140283. | `EDFig5_joint_ages.pdf` |

---

## Installation

We recommend Conda for environment reproducibility.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
conda env create -f environment.yml
conda activate hubble-stellar-ages
```

This installs Python 3.11 with the exact package versions used in the published analysis (`emcee 3.1`, `numba 0.59`, `numpy 1.26`, `scipy 1.13`, `pandas 2.2`, `getdist 1.5`, `matplotlib`, `tqdm`).

If you prefer `pip`:

```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install emcee==3.1.* numba==0.59.* numpy==1.26.* scipy==1.13.* \
            pandas==2.2.* getdist==1.5.* matplotlib tqdm
```

---

## Data files (download separately)

The following data files are **not redistributed** in this repository; please download them from their official releases and place them in the repository root (same directory as the `.py` scripts).

### Pantheon+SH0ES

From <https://github.com/PantheonPlusSH0ES/DataRelease>:

- `Pantheon+SH0ES.dat`  — SN Ia data table (`Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat`)
- `Pantheon+SH0ES_STAT+SYS.cov` — full $1701\times1701$ stat+sys covariance matrix (`Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov`)

`cosmoc.py` will automatically apply the standard Pantheon+ Hubble-flow selection (exclude `IS_CALIBRATOR == 1` and `zHD < 0.01`), leaving 1580 SNe Ia and constructing the corresponding $1580\times 1580$ sub-covariance matrix.

### DESI Year 1 BAO

From the DESI Y1 public data release (<https://data.desi.lbl.gov/public/papers/c3/y1-kp7a/>):

- `desi_gaussian_bao_ALL_GCcomb_mean.txt` — 13 BAO measurements
- `desi_gaussian_bao_ALL_GCcomb_cov.txt`  — $13\times 13$ covariance matrix

Scripts that don't use BAO (`02`, `04`, `05`, `08`, `09`) will skip the BAO loader gracefully if these two files are absent.

### Stellar-age priors (no download needed)

The Gaia-calibrated globular-cluster age $t_{\rm GC} = 13.5 \pm 0.27$ Gyr (Valcin et al. 2021) and the HD 140283 asteroseismic age $t = 14.2 \pm 0.4$ Gyr (Lundkvist et al. 2025) are hard-coded as Gaussian priors in the scripts. To use a different age prior, edit `SCENARIOS` in `cosmoc.py` or the `age_specs` argument in `09_joint_age_priors2.py`.

---

## Running the pipeline

Each script is self-contained and writes its figure(s) to the working directory. Scripts 02–09 also save the MCMC chains as compressed `.npz` files for downstream re-plotting.

**Recommended order** (only `02→03→07` carry numerical dependencies through the hard-coded `H0_LCDM_GC_BASELINE` constants in `07_w0wa_testf2.py`; if you rerun `03` you should update those constants at the top of `07` to keep the reported $|\Delta H_0|$ consistent):

```bash
python 01_concept_figure.py              # ~2  s   (no MCMC; analytic)
python 02_main_NoBAOf2.py                # ~15 s   (3 MCMC × ~3000 steps)
python 03_with_BAO2f.py                  # ~30 s   (3 MCMC × ~3000 steps + BAO)
python 04_sensitivity_curvef.py          # ~45 s   (15-point H0(t0) scan)
python 05_dt_form_testf.py               # ~35 s   (13-point dt_form scan)
python 06_curvature_tests1p4s.py         # ~65 s   (3 MCMC × 10000 steps, 200×200 grid)
python 07_w0wa_testf2.py                 # ~65 s   (3 MCMC × ~6500 steps, 30×30×30 grid)
python 08_age_systematics_scan2.py       # ~25 s   (4-point σ_age scan)
python 09_joint_age_priors2.py           # ~25 s   (3 MCMC × 5000 steps)
```

Total wall-clock time: **about 5–6 minutes** on a single modern CPU core (Intel i7-class or equivalent). Numba JIT compilation accounts for ~1–3 s of overhead per script on first run; subsequent calls within the same Python session reuse the compiled kernels.

Each script prints a convergence diagnostic block of the form

```
[Convergence <label>] tau_max: ..., Chain/tau: ..., N_eff: ..., acceptance: ...
```

These are the numbers reproduced in Table 3 of the manuscript.

---

## Reproducing manuscript figures and numbers

All figures and parameter constraints reported in the paper can be reproduced by running scripts 01–09 once on a fresh checkout. The MCMC seeds are not fixed, so individual posterior medians fluctuate by $\lesssim 0.1$ km s$^{-1}$ Mpc$^{-1}$ in $H_0$ (well below the $\sim 1.5$ km s$^{-1}$ Mpc$^{-1}$ statistical uncertainty) between independent runs; all qualitative conclusions are run-invariant.

| Manuscript element | Script |
|--------------------|--------|
| Figure 1 (concept) | `01_concept_figure.py` |
| Figure 2 (core result) | `02_main_NoBAOf2.py` |
| Figure 3 (+ BAO) | `03_with_BAO2f.py` |
| Figure 4 (sensitivity curve) | `04_sensitivity_curvef.py` |
| Table 1 row "SN + age (no BAO)" | `02_main_NoBAOf2.py` (GC scenario) |
| Table 1 row "+ DESI Y1 BAO (flat)" | `03_with_BAO2f.py` (GC scenario) |
| Table 1 row "+ free $\Omega_k$" | `06_curvature_tests1p4s.py` (GC scenario) |
| Table 1 row "+ $w_0 w_a$CDM" | `07_w0wa_testf2.py` (GC scenario) |
| Table 1 row "Joint GC + HD 140283" | `09_joint_age_priors2.py` (joint config) |
| Table 3 (convergence diagnostics) | each script's `[Convergence ...]` printout |
| Extended Data Fig. 1 | `05_dt_form_testf.py` |
| Extended Data Fig. 2 | `06_curvature_tests1p4s.py` |
| Extended Data Fig. 3 | `07_w0wa_testf2.py` |
| Extended Data Fig. 4 | `08_age_systematics_scan2.py` |
| Extended Data Fig. 5 | `09_joint_age_priors2.py` |

---

## Implementation notes

- **Grid precomputation.** All cosmological-distance and age integrals are precomputed once on a parameter grid using Numba-jitted trapezoidal quadrature, then interpolated at MCMC evaluation time via `scipy.interpolate.RegularGridInterpolator`. Grid resolution is configuration-dependent:
  - flat $\Lambda$CDM (1D): 100 nodes in $\Omega_m$, 1000 redshift sub-steps
  - non-flat (2D): $200\times 200$ in $(\Omega_m, \Omega_k)$, 2000 redshift sub-steps
  - CPL (3D): $30\times 30\times 30$ in $(\Omega_m, w_0, w_a)$, 1000 redshift sub-steps

  Cross-validation against direct (non-interpolated) likelihood evaluation at the posterior median agrees at the $10^{-5}$ level in $\ln \mathcal{L}$.

- **Vectorised likelihoods.** All `emcee` runs use `vectorize=True`, evaluating the full walker batch in one NumPy matrix call per step (~500 likelihood evaluations per second per CPU core).

- **Adaptive convergence.** Scripts 02–03 monitor $\tau_{\rm max}$ every 100–200 steps and stop early once `chain_length > 50 τ_max` and `|Δτ|/τ < 0.01`. Scripts 06–07 use fixed-length chains (10000 / 6000 steps respectively) sized to comfortably exceed the same criterion.

---

## Citation

If you use this code or build on the analysis, please cite the paper:

```bibtex
@article{Hu2026StellarAges,
  author       = {Hu, Jian},
  title        = {Stellar ages localize the Hubble tension to the supernova absolute magnitude},
  journal      = {Nature},
  year         = {2026},
  note         = {Submitted}
}
```

A permanent Zenodo DOI for this code will be issued at the time of publication and inserted here.

---

## License

This code is released under the MIT License (see `LICENSE`).

Third-party data files (Pantheon+SH0ES, DESI Y1 BAO) are governed by their respective collaboration licenses; please consult those releases directly.

---

## Contact

Jian Hu — `dg1626002@smail.nju.edu.cn`
Institute of Astronomy and Information, Dali University, Yunnan 671003, P. R. China

Bug reports and questions are welcome via GitHub Issues.