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
| `cosmoc.py` | Shared physics: cosmology integrals (Numba-jitted), Pantheon+ and DESI Y1 BAO loaders, reference values. | (imported by all scripts) |
| `cosmo_tools.py` | Post-processing utilities: posterior statistics, GetDist-based triangle plotting, multi-chain overlay. | (imported by scripts) |
| `01_concept_figure.py` | Figure 1 — analytic illustration of how $t_0$ contours break the $H_0$–$\Omega_m$ degeneracy. No MCMC. | `Figure1_Concept.pdf` |
| `02_main_NoBAOf2.py` | Figure 2 — core result: Pantheon+ + stellar age, flat $\Lambda$CDM, three age scenarios. | `Figure2_Main_NoBAO.pdf` |
| `03_with_BAO2f.py` | Figure 3 — adds DESI Y1 BAO; tests whether the inferred $r_d$ matches BBN. | `Figure3A_Triangle_BAO.pdf`, `Figure3B_rd_distribution.pdf` |
| `04_sensitivity_curvef.py` | Figure 4 — continuous $H_0(t_0)$ sensitivity curve over $12$–$15$ Gyr. | `Figure4_Sensitivity.pdf` |
| `05_dt_form_testf.py` | Extended Data Fig. 1 — robustness to formation-time delay $\Delta t_{\rm form}$. | `EDFig1_dt_form.pdf` |
| `06_curvature_tests1p4s.py` | Extended Data Fig. 2 — robustness to spatial curvature ($\Omega_k$ free); three age scenarios. | `EDFig2_Curvature_AllScenarios.pdf` |
| `07_w0wa_testf2.py` | Extended Data Fig. 3 — robustness under Chevallier–Polarski–Linder dynamical dark energy ($w_0 w_a$CDM). | `EDFig3_w0wa.pdf` |
| `08_age_systematics_scan2.py` | Extended Data Fig. 4 — sensitivity to assumed stellar-age systematic $\sigma_{\rm age}$. | `EDFig4_age_sigma_scan.pdf` |
| `09_joint_age_priors2.py` | Extended Data Fig. 5 — joint inference using Valcin et al. globular clusters + Lundkvist et al. HD 140283. | `EDFig5_joint_ages.pdf` |

---

## Installation

We recommend Conda for environment reproducibility. Windows, Linux, and macOS are fully supported.

```bash
git clone [https://github.com/HUJIAN0000/AgeAnchor-Cosmology.git](https://github.com/HUJIAN0000/AgeAnchor-Cosmology.git)
cd AgeAnchor-Cosmology
conda env create -f environment.yml
conda activate hubble-stellar-ages
```

This installs Python 3.11 with the exact package versions used in the published analysis (`emcee 3.1`, `numba 0.59`, `numpy 1.26`, `scipy 1.13`, `pandas 2.2`, `getdist 1.5`, `matplotlib`, `tqdm`).

If you prefer `pip` (e.g., on Windows Command Prompt):

```cmd
python -m venv venv
venv\Scripts\activate
pip install emcee==3.1.* numba==0.59.* numpy==1.26.* scipy==1.13.* pandas==2.2.* getdist==1.5.* matplotlib tqdm
```

---

## ⚠️ Data files (Crucial Step before running!)

The required astrophysical datasets are included directly in this repository, but **you must unzip the covariance matrix before running the scripts**.

### 1. Pantheon+SH0ES Covariance Matrix

Due to GitHub file size limits, the $1580 \times 1580$ covariance matrix is compressed.

* **ACTION REQUIRED:** Extract the `Pantheon+SH0ES_STAT+SYS.zip` archive directly into the repository root folder.
* Ensure the extracted file is named `Pantheon+SH0ES_STAT+SYS.cov` and sits alongside the Python scripts. If it is not extracted, the scripts will crash with a `FileNotFoundError`.

### 2. DESI Year 1 BAO

The repository also includes the consensus BAO measurements from the DESI Y1 public data release:

* `desi_gaussian_bao_ALL_GCcomb_mean.txt`
* `desi_gaussian_bao_ALL_GCcomb_cov.txt`

*(No action required for these, they are ready to use).*

---

## Running the pipeline

Each script is self-contained and writes its figure(s) to the working directory. Scripts 02–09 also save the MCMC chains as compressed `.npz` files for downstream re-plotting.

**Recommended order** (only `02→03→07` carry numerical dependencies through the hard-coded `H0_LCDM_GC_BASELINE` constants in `07`):

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

Total wall-clock time: **about 5–6 minutes** on a single modern CPU core. Numba JIT compilation accounts for ~1–3 s of overhead per script on the first run.

Each script prints a convergence diagnostic block (reproduced in Table 3 of the manuscript):

```text
[Convergence <label>] tau_max: ..., Chain/tau: ..., N_eff: ..., acceptance: ...
```

---

## Implementation notes (Cross-Platform Optimization)

* **Grid precomputation.** All cosmological integrals are precomputed once on a parameter grid using Numba-jitted trapezoidal quadrature, then interpolated at MCMC evaluation time via `scipy.interpolate.RegularGridInterpolator`.
* **Pure CPU Vectorized Likelihoods.** All `emcee` runs use `vectorize=True`, evaluating the full walker batch in one NumPy matrix call per step (~500 likelihood evaluations per second). **This architecture completely bypasses Python's native `multiprocessing` module, ensuring ultra-fast and stable execution on Windows, macOS, and Linux alike without memory pickling overhead.**
* **Adaptive convergence.** MCMC drivers dynamically monitor the integrated autocorrelation time $\tau_{\rm max}$ and stop early once `chain_length > 50 τ_max` and `|Δτ|/τ < 0.01`.

---

## Citation

If you use this code or build on the analysis, please cite the paper:

```bibtex
@article{Hu2026StellarAges,
  author       = {Hu, Jian},
  title        = {Stellar ages localize the Hubble tension to the supernova absolute magnitude},
  journal      = {XXXXXX},
  year         = {2026},
  note         = {Submitted}
}
```

A permanent Zenodo DOI for this code will be issued at the time of publication and inserted here.

---

## License

This code is released under the MIT License (see `LICENSE`).

Third-party data files (Pantheon+SH0ES, DESI Y1 BAO) are governed by their respective collaboration licenses.

---

## Contact

Jian Hu — `dg1626002@smail.nju.edu.cn`  
Institute of Astronomy and Information, Dali University, Yunnan 671003, P. R. China

