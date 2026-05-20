# -*- coding: utf-8 -*-
"""
ED Fig. 2 — Three-scenario overlay (non-flat LambdaCDM robustness)

Runs the curvature-test MCMC for all three age scenarios and overlays
them on a single corner plot via cosmo_tools.plot_getdist_comparison
(same calling convention as the JLA three-scenario figure).

Output
------
- Console: full 6-parameter posterior for each scenario + convergence
- EDFig2_Curvature_AllScenarios.pdf
- EDFig2_Curvature_AllScenarios_summary.txt   (machine-readable)


%runfile F:/work/code/age/newtest01/06_curvature_tests1p4.py --wdir
Reloaded modules: cosmoc, cosmo_tools
============================================================
Non-flat ΛCDM robustness — three age scenarios overlaid
Pantheon+ + age + DESI Y1 BAO, free Ω_k
============================================================

[Step 1] Loading shared data...
  Pantheon+ loaded: 1580 SNe (removed 121: 77 calibrators + 44 low-z)
  DESI BAO loaded: 13 points
  Precomputing (Om, Ok) grid (200 x 200)...
  Grid built in 1.07 s
  Shared data ready.

[Step 2] Running MCMC for three scenarios...

============================================================
Scenario: $t_0 = 14.2 \pm 0.4$ Gyr  (HD 140283)  (t = 14.2 ± 0.4 Gyr)
============================================================
100%|██████████| 10000/10000 [00:21<00:00, 474.83it/s]
  MCMC done in 21.1 s

  --- 6-parameter posterior (median + 68% CI) ---
    Om       = 0.2932  +0.0116 / -0.0114
    Ok       = 0.0480  +0.0371 / -0.0361
    H0       = 65.3903  +1.9131 / -1.5922
    MB       = -19.5040  +0.0619 / -0.0526
    rd       = 153.8266  +3.6053 / -4.2590
    dt_form  = 0.1488  +0.0495 / -0.0493
  τ_max = 89.8,  chain/τ = 111.4
  N_eff median = 1898,  N_eff min = 1671
  Acceptance fraction = 0.499

============================================================
Scenario: $t_0 = 13.5 \pm 0.27$ Gyr  (Globular Clusters)  (t = 13.5 ± 0.27 Gyr)
============================================================
100%|██████████| 10000/10000 [00:21<00:00, 468.39it/s]
  MCMC done in 21.4 s

  --- 6-parameter posterior (median + 68% CI) ---
    Om       = 0.2933  +0.0114 / -0.0112
    Ok       = 0.0484  +0.0367 / -0.0359
    H0       = 68.4445  +1.5223 / -1.4574
    MB       = -19.4044  +0.0464 / -0.0458
    rd       = 146.8788  +2.9865 / -2.9600
    dt_form  = 0.1493  +0.0494 / -0.0489
  τ_max = 68.6,  chain/τ = 145.8
  N_eff median = 2204,  N_eff min = 2186
  Acceptance fraction = 0.512

============================================================
Scenario: $t_0 = 12.6 \pm 0.27$ Gyr  (SH0ES Implied)  (t = 12.6 ± 0.27 Gyr)
============================================================
100%|██████████| 10000/10000 [00:21<00:00, 473.19it/s]
  MCMC done in 21.1 s

  --- 6-parameter posterior (median + 68% CI) ---
    Om       = 0.2932  +0.0115 / -0.0115
    Ok       = 0.0488  +0.0371 / -0.0364
    H0       = 73.2756  +1.7005 / -1.6414
    MB       = -19.2567  +0.0490 / -0.0479
    rd       = 137.2175  +2.9364 / -2.9722
    dt_form  = 0.1494  +0.0496 / -0.0499
  τ_max = 95.9,  chain/τ = 104.3
  N_eff median = 1909,  N_eff min = 1564
  Acceptance fraction = 0.512

========================================================================
TENSION ANALYSIS
========================================================================
  [HD140283]
    H₀  = 65.39 ± 1.91   (vs Planck 1.02σ, vs SH0ES 3.51σ)
    M_B = -19.504 ± 0.062   (vs SH0ES M_B 3.72σ)
    r_d = 153.83 ± 4.26   (vs BBN 1.62σ)
  [GC]
    H₀  = 68.44 ± 1.52   (vs Planck 0.65σ, vs SH0ES 2.49σ)
    M_B = -19.404 ± 0.046   (vs SH0ES M_B 2.82σ)
    r_d = 146.88 ± 2.99   (vs BBN 0.03σ)
  [SH0ES]
    H₀  = 73.28 ± 1.70   (vs Planck 3.31σ, vs SH0ES 0.12σ)
    M_B = -19.257 ± 0.049   (vs SH0ES M_B 0.07σ)
    r_d = 137.22 ± 2.97   (vs BBN 3.11σ)

Wrote: EDFig2_Curvature_AllScenarios_summary.txt

[Step 3] Per-scenario stats (5 physical params, no dt_form)...

--- HD140283 ---

========================================================
MCMC constraints (median and +upper / -lower)
========================================================
\Omega_m   = 0.2932  +0.0116  -0.0114
\Omega_k   = 0.0480  +0.0371  -0.0361
H_0        = 65.3903  +1.9131  -1.5922
M_B        = -19.5040  +0.0619  -0.0526
r_d        = 153.8266  +3.6053  -4.2590
========================================================


--- GC ---

========================================================
MCMC constraints (median and +upper / -lower)
========================================================
\Omega_m   = 0.2933  +0.0114  -0.0112
\Omega_k   = 0.0484  +0.0367  -0.0359
H_0        = 68.4445  +1.5223  -1.4574
M_B        = -19.4044  +0.0464  -0.0458
r_d        = 146.8788  +2.9865  -2.9600
========================================================


--- SH0ES ---

========================================================
MCMC constraints (median and +upper / -lower)
========================================================
\Omega_m   = 0.2932  +0.0115  -0.0115
\Omega_k   = 0.0488  +0.0371  -0.0364
H_0        = 73.2756  +1.7005  -1.6414
M_B        = -19.2567  +0.0490  -0.0479
r_d        = 137.2175  +2.9364  -2.9722
========================================================


[Step 4] Building overlay corner plot...
Removed no burn in
Removed no burn in
Removed no burn in
✅ Comparison plot saved: EDFig2_Curvature_AllScenarios.pdf

============================================================
All done.
============================================================
"""

import os
import time
import numpy as np
import emcee
from numba import jit
from scipy.interpolate import RegularGridInterpolator

from cosmoc import load_pantheon_plus, load_bao
import cosmo_tools

c_light = 299792.458
AGE_CONVERSION_FACTOR = 977.8


# =====================================================================
# Numba grid precomputation (shared across scenarios)
# =====================================================================
@jit(nopython=True, fastmath=True)
def compute_grids_ultra_fast_cpu(z_sn, z_bao, Om_grid, Ok_grid):
    z_max = 0.0
    if len(z_sn) > 0:
        z_max = max(z_max, np.max(z_sn))
    if len(z_bao) > 0:
        z_max = max(z_max, np.max(z_bao))

    n_z_steps = 2000
    dz = z_max / n_z_steps
    z_array = np.linspace(0, z_max, n_z_steps + 1)

    N_Om, N_Ok = len(Om_grid), len(Ok_grid)
    N_sn, N_bao = len(z_sn), len(z_bao)

    grid_sn = np.zeros((N_Om, N_Ok, N_sn), dtype=np.float64)
    grid_bao = np.zeros((N_Om, N_Ok, N_bao), dtype=np.float64)
    grid_age = np.zeros((N_Om, N_Ok), dtype=np.float64)

    for i in range(N_Om):
        for j in range(N_Ok):
            Om = Om_grid[i]
            Ok = Ok_grid[j]
            Ol = 1.0 - Om - Ok

            # Age integral
            age_int = 0.0
            n_steps_age = 1000
            da = 1.0 / n_steps_age
            for k in range(n_steps_age):
                a = (k + 0.5) * da
                val = Om / a + Ok + Ol * a * a
                if val > 0:
                    age_int += 1.0 / np.sqrt(val)
            grid_age[i, j] = age_int * da

            # Cumulative distance integral
            cum_I = np.zeros(n_z_steps + 1, dtype=np.float64)
            prev_val = Om + Ok + Ol
            prev_inv = 1.0 / np.sqrt(prev_val) if prev_val > 0 else 0.0
            for k in range(1, n_z_steps + 1):
                z = z_array[k]
                val = Om * (1.0 + z) ** 3 + Ok * (1.0 + z) ** 2 + Ol
                curr_inv = 1.0 / np.sqrt(val) if val > 0 else 0.0
                cum_I[k] = cum_I[k - 1] + 0.5 * (prev_inv + curr_inv) * dz
                prev_inv = curr_inv

            for s in range(N_sn):
                z = z_sn[s]
                idx = int(z / dz)
                if idx >= n_z_steps:
                    idx = n_z_steps - 1
                w = (z - z_array[idx]) / dz
                I_z = cum_I[idx] * (1.0 - w) + cum_I[idx + 1] * w
                if np.abs(Ok) < 1e-5:
                    grid_sn[i, j, s] = I_z
                elif Ok > 0:
                    grid_sn[i, j, s] = (1.0 / np.sqrt(Ok)) * np.sinh(np.sqrt(Ok) * I_z)
                else:
                    grid_sn[i, j, s] = (1.0 / np.sqrt(-Ok)) * np.sin(np.sqrt(-Ok) * I_z)

            for b in range(N_bao):
                z = z_bao[b]
                idx = int(z / dz)
                if idx >= n_z_steps:
                    idx = n_z_steps - 1
                w = (z - z_array[idx]) / dz
                I_z = cum_I[idx] * (1.0 - w) + cum_I[idx + 1] * w
                if np.abs(Ok) < 1e-5:
                    grid_bao[i, j, b] = I_z
                elif Ok > 0:
                    grid_bao[i, j, b] = (1.0 / np.sqrt(Ok)) * np.sinh(np.sqrt(Ok) * I_z)
                else:
                    grid_bao[i, j, b] = (1.0 / np.sqrt(-Ok)) * np.sin(np.sqrt(-Ok) * I_z)

    return grid_sn, grid_bao, grid_age


# =====================================================================
# Shared likelihood
# =====================================================================
class LikelihoodNonFlatVectorizedCPU:
    _shared = {'loaded': False}

    def __init__(self, age_obs, age_err):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.dt_prior_mu = 0.15
        self.dt_prior_sig = 0.05

    @classmethod
    def load_shared(cls):
        if cls._shared.get('loaded'):
            return

        z_sn, mb_sn, inv_cov_sn = load_pantheon_plus()
        bao_data = load_bao()

        z_sn = z_sn.astype(np.float64)
        mb_sn = mb_sn.astype(np.float64)
        inv_cov_sn = inv_cov_sn.astype(np.float64)

        bao_z_list = []
        for d in bao_data:
            d['z'] = d['z'].astype(np.float64)
            d['y'] = d['y'].astype(np.float64)
            d['icov'] = d['icov'].astype(np.float64)
            bao_z_list.extend(d['z'])
        z_bao = np.array(bao_z_list, dtype=np.float64)

        print("  Precomputing (Om, Ok) grid (200 x 200)...")
        Om_grid = np.linspace(0.05, 0.65, 200, dtype=np.float64)
        Ok_grid = np.linspace(-0.6, 0.6, 200, dtype=np.float64)
        t0 = time.time()
        grid_sn, grid_bao, grid_age = compute_grids_ultra_fast_cpu(
            z_sn, z_bao, Om_grid, Ok_grid
        )
        print(f"  Grid built in {time.time() - t0:.2f} s")

        interp_sn = RegularGridInterpolator(
            (Om_grid, Ok_grid), grid_sn, bounds_error=False, fill_value=None
        )
        interp_age = RegularGridInterpolator(
            (Om_grid, Ok_grid), grid_age, bounds_error=False, fill_value=None
        )
        interp_bao = None
        if len(z_bao) > 0:
            interp_bao = RegularGridInterpolator(
                (Om_grid, Ok_grid), grid_bao,
                bounds_error=False, fill_value=None
            )

        cls._shared = {
            'loaded': True,
            'z_sn': z_sn, 'mb_sn': mb_sn, 'inv_cov_sn': inv_cov_sn,
            'bao_data': bao_data, 'z_bao': z_bao,
            'interp_sn': interp_sn, 'interp_age': interp_age,
            'interp_bao': interp_bao,
        }
        print("  Shared data ready.")

    def log_prob_vectorized(self, theta_batch):
        s = self._shared
        z_sn, mb_sn, inv_cov_sn = s['z_sn'], s['mb_sn'], s['inv_cov_sn']
        z_bao, bao_data = s['z_bao'], s['bao_data']
        interp_sn, interp_age, interp_bao = (
            s['interp_sn'], s['interp_age'], s['interp_bao']
        )

        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]
        Om, Ok, H0, M_B, rd, dt_form = th.T

        pts = np.column_stack((Om, Ok))
        mask = ((0.1 < Om) & (Om < 0.6) & (-0.5 < Ok) & (Ok < 0.5) &
                (50 < H0) & (H0 < 90) & (-20 < M_B) & (M_B < -18) &
                (120 < rd) & (rd < 160) & (0.0 < dt_form) & (dt_form < 1.0))

        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask):
            return logL

        pts_v = pts[mask]
        Om_v, Ok_v, H0_v, MB_v, rd_v, dt_v = (
            Om[mask], Ok[mask], H0[mask], M_B[mask], rd[mask], dt_form[mask]
        )
        Ol_v = 1.0 - Om_v - Ok_v

        # SN likelihood
        d_sn_v = interp_sn(pts_v)
        const = c_light / H0_v[:, None]
        D_M_sn = d_sn_v * const
        mu_model = 5.0 * np.log10(D_M_sn * (1.0 + z_sn[None, :]) + 1e-10) + 25.0
        diff_sn = mb_sn[None, :] - (mu_model + MB_v[:, None])
        temp = diff_sn @ inv_cov_sn
        chi2_sn = np.sum(diff_sn * temp, axis=1)
        logL_v = -0.5 * chi2_sn

        # Age + dt_form
        age_int_v = interp_age(pts_v)
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * age_int_v
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig) ** 2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err) ** 2

        # BAO
        if len(z_bao) > 0 and interp_bao is not None:
            d_bao_v = interp_bao(pts_v)
            D_M_bao = d_bao_v * const
            chi2_bao = np.zeros(len(pts_v), dtype=np.float64)
            idx_start = 0
            for d in bao_data:
                N_pts = len(d['z'])
                z_arr, type_arr, y_obs, icov = (
                    d['z'], d['type'], d['y'], d['icov']
                )
                D_M_b = D_M_bao[:, idx_start:idx_start + N_pts]
                Ez = np.sqrt(
                    Om_v[:, None] * (1.0 + z_arr[None, :]) ** 3 +
                    Ok_v[:, None] * (1.0 + z_arr[None, :]) ** 2 +
                    Ol_v[:, None]
                )
                D_H_b = c_light / (H0_v[:, None] * Ez)
                D_V_b = (z_arr[None, :] * D_M_b ** 2 * D_H_b) ** (1.0 / 3.0)
                mod = np.zeros_like(D_M_b, dtype=np.float64)
                for k in range(N_pts):
                    t = type_arr[k]
                    if t == 'DM_over_rs':
                        mod[:, k] = D_M_b[:, k] / rd_v
                    elif t == 'DH_over_rs':
                        mod[:, k] = D_H_b[:, k] / rd_v
                    else:
                        mod[:, k] = D_V_b[:, k] / rd_v
                diff_b = y_obs[None, :] - mod
                temp_b = diff_b @ icov
                chi2_bao += np.sum(diff_b * temp_b, axis=1)
                idx_start += N_pts
            logL_v += -0.5 * chi2_bao

        logL[mask] = logL_v
        return logL


# =====================================================================
# Run one scenario
# =====================================================================
# (此处省略文件头部，直接看 run_one 函数的修改)
def run_one(age_obs, age_err, label, n_steps=10000, n_burnin=2500,
            nwalkers=40, ndim=6, seed=None):
    if seed is not None:
        np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"Scenario: {label}  (t = {age_obs} ± {age_err} Gyr)")
    print('=' * 60)

    like = LikelihoodNonFlatVectorizedCPU(age_obs, age_err)

    p0_center = np.array([0.30, 0.05, 68.0, -19.40, 147.0, 0.15])
    p0 = [p0_center + 1e-3 * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, like.log_prob_vectorized, vectorize=True
    )
    t0 = time.time()
    sampler.run_mcmc(p0, n_steps, progress=True)
    print(f"  MCMC done in {time.time() - t0:.1f} s")

    # --- 保存链并输出诊断 ---
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = np.max(tau)
        n_eff = np.mean(n_steps * nwalkers / tau)
        print(f"  [Convergence {label}] tau_max: {tau_max:.1f}, Chain/tau: {n_steps/tau_max:.1f}, N_eff: {n_eff:.0f}")
        
        sc_short = label.split('(')[-1].split(')')[0].replace(" ", "_")
        np.savez(f"chain_curvature_{sc_short}.npz", chain=sampler.get_chain())
    except Exception as e:
        print(f"  [Convergence check skipped: {e}]")
    # ----------------------

    flat = sampler.get_chain(discard=n_burnin, flat=True)

    # ---- Full 6-parameter posterior ----
    names = ["Om", "Ok", "H0", "MB", "rd", "dt_form"]
    # ... (原有代码保持不变) ...
    print(f"\n  --- 6-parameter posterior (median + 68% CI) ---")
    posterior = {}
    for i, n in enumerate(names):
        p = np.percentile(flat[:, i], [16, 50, 84])
        posterior[n] = (p[1], p[2] - p[1], p[1] - p[0])
        print(f"    {n:8s} = {p[1]:.4f}  +{p[2]-p[1]:.4f} / -{p[1]-p[0]:.4f}")

    # ---- Convergence diagnostics ----
    try:
        tau = sampler.get_autocorr_time(tol=0)
        kept_per_walker = flat.shape[0] // nwalkers
        neff_list = [nwalkers * kept_per_walker / (2 * t) for t in tau]
        print(f"  τ_max = {tau.max():.1f},  chain/τ = {n_steps / tau.max():.1f}")
        print(f"  N_eff median = {int(np.median(neff_list))},  "
              f"N_eff min = {int(min(neff_list))}")
    except Exception as e:
        print(f"  [autocorr: {e}]")
    print(f"  Acceptance fraction = "
          f"{np.mean(sampler.acceptance_fraction):.3f}")

    return flat, posterior


# =====================================================================
# Main driver
# =====================================================================
def main():
    print("=" * 60)
    print("Non-flat ΛCDM robustness — three age scenarios overlaid")
    print("Pantheon+ + age + DESI Y1 BAO, free Ω_k")
    print("=" * 60)

    print("\n[Step 1] Loading shared data...")
    LikelihoodNonFlatVectorizedCPU.load_shared()

    # ----- Match the JLA scenario styling for visual consistency -----
   
    scenarios = [
        {'age': 14.2, 'err': 0.40,
         'label': r'$t_0 = 14.2 \pm 0.4$ Gyr  (HD 140283)',
         'short': 'HD140283', 'color': '#666666'},
        {'age': 13.5, 'err': 0.27,
         'label': r'$t_0 = 13.5 \pm 0.27$ Gyr  (Globular Clusters)',
         'short': 'GC',       'color': '#0066cc'},
        {'age': 12.6, 'err': 0.27,
         'label': r'$t_0 = 12.6 \pm 0.27$ Gyr  (SH0ES Implied)',
         'short': 'SH0ES',    'color': '#cc0000'},
    ]

    print("\n[Step 2] Running MCMC for three scenarios...")
    results = []
    for sc in scenarios:
        flat, posterior = run_one(sc['age'], sc['err'], sc['label'])
        results.append({'config': sc, 'flat': flat, 'posterior': posterior})

    # ---------- Tension summary ----------
    PLANCK_H0 = (67.4, 0.5)
    SH0ES_H0 = (73.04, 1.04)
    SH0ES_MB = (-19.253, 0.027)
    BBN_RD = (146.8, 0.8)

    def tens(v1, e1, v2, e2):
        return abs(v1 - v2) / np.sqrt(e1**2 + e2**2)

    print(f"\n{'=' * 72}\nTENSION ANALYSIS\n{'=' * 72}")
    for r in results:
        sc, p = r['config'], r['posterior']
        h0v, h0e = p['H0'][0], max(p['H0'][1], p['H0'][2])
        mbv, mbe = p['MB'][0], max(p['MB'][1], p['MB'][2])
        rdv, rde = p['rd'][0], max(p['rd'][1], p['rd'][2])
        print(f"  [{sc['short']}]")
        print(f"    H₀  = {h0v:.2f} ± {h0e:.2f}   "
              f"(vs Planck {tens(h0v, h0e, *PLANCK_H0):.2f}σ, "
              f"vs SH0ES {tens(h0v, h0e, *SH0ES_H0):.2f}σ)")
        print(f"    M_B = {mbv:.3f} ± {mbe:.3f}   "
              f"(vs SH0ES M_B {tens(mbv, mbe, *SH0ES_MB):.2f}σ)")
        print(f"    r_d = {rdv:.2f} ± {rde:.2f}   "
              f"(vs BBN {tens(rdv, rde, *BBN_RD):.2f}σ)")

    # ---------- Machine-readable summary ----------
    with open('EDFig2_Curvature_AllScenarios_summary.txt', 'w') as f:
        f.write("# Non-flat ΛCDM, three age scenarios, full 6-parameter posterior\n")
        f.write("# columns: median, +68%, -68%\n\n")
        for r in results:
            sc = r['config']
            f.write(f"## {sc['short']} (age = {sc['age']} ± {sc['err']} Gyr)\n")
            for n, (med, hi, lo) in r['posterior'].items():
                f.write(f"  {n:10s}  {med:14.6f}  {hi:10.6f}  {lo:10.6f}\n")
            f.write("\n")
    print("\nWrote: EDFig2_Curvature_AllScenarios_summary.txt")

    # ====================================================================
    # Per-scenario stats via cosmo_tools.calculate_stats / print_results
    # ====================================================================
    print("\n[Step 3] Per-scenario stats (5 physical params, no dt_form)...")
    labels_phys = [r"\Omega_m", r"\Omega_k", r"H_0", r"M_B", r"r_d"]

    all_samples = []          # list of (N_chain, 5) arrays
    all_legend_labels = []
    all_colors = []

    for r in results:
        sc = r['config']
        marg = r['flat'][:, :5]   # drop dt_form
        all_samples.append(marg)
        all_legend_labels.append(sc['label'])
        all_colors.append(sc['color'])

        print(f"\n--- {sc['short']} ---")
        stats = cosmo_tools.calculate_stats(marg, labels_phys)
        cosmo_tools.print_results(stats)

    # ====================================================================
    # Overlay plot via cosmo_tools.plot_getdist_comparison
    # (same calling convention as JLA three-scenario figure)
    # ====================================================================
    print("\n[Step 4] Building overlay corner plot...")
    cosmo_tools.plot_getdist_comparison(
        all_samples,
        labels_phys,
        legend_labels=all_legend_labels,
        colors=all_colors,
        filename="EDFig2_Curvature_AllScenarios.pdf"
    )

    print(f"\n{'=' * 60}\nAll done.\n{'=' * 60}")


if __name__ == "__main__":
    main()