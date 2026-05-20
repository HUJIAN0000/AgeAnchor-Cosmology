# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:31:37 2026

@author: Administrator

Figure 3: CORE RESULT — Pantheon+ + cosmic age + DESI Y1 BAO.
优化：使用全局变量消除 Windows 多进程下的内存序列化延迟，达到最高并行效率。
"""
# -*- coding: utf-8 -*-
"""
Figure 3: Supporting result — Pantheon+ + cosmic age + BAO, with r_d free.

OPTIMIZED VERSION:
  - Numba JIT-compiled 1D Om grid precomputation (flat LCDM: w0=-1, wa=0)
  - SciPy RegularGridInterpolator -> O(1) lookup of SN distances + age
  - emcee with vectorize=True: batched likelihood across all walkers
  - Global cache shared across the three age scenarios (1x precompute)
  - BAO evaluated per-walker over the valid vectorized mask

Output: Figure3A_Triangle_BAO.pdf  (Om, H0, MB triangle)
        Figure3B_rd_distribution.pdf  (r_d posterior vs BBN/Planck)
"""
import time
import numpy as np
import emcee
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from getdist import plots, MCSamples

from cosmoc import (
    c_light, load_pantheon_plus, load_bao, bao_model,
    SCENARIOS, PLANCK_RD, BBN_RD
)

AGE_CONVERSION_FACTOR = 977.8  # Gyr * (km/s/Mpc)


# ====================================================================
# 1) Numba 1D grid precomputation over Om   (flat LCDM: w0 = -1, wa = 0)
# ====================================================================

@jit(nopython=True, fastmath=True)
def compute_grids_1d_lcdm_cpu(z_sn, Om_grid):
    """
    Precompute *dimensionless* cosmological integrals on a 1D Om grid.

        grid_sn[i, s] = ∫_0^{z_sn[s]} dz' / sqrt(Om*(1+z')^3 + (1-Om))
        grid_age[i]   = ∫_0^1 da / sqrt(Om/a + (1-Om)*a^2)

    Multiply by (c/H0) [-> Mpc] and (977.8/H0) [-> Gyr] at use time.
    """
    z_max = 0.0
    if len(z_sn) > 0:
        z_max = np.max(z_sn)

    n_z_steps = 1000
    dz = z_max / n_z_steps
    z_array = np.linspace(0.0, z_max, n_z_steps + 1)

    N_Om = len(Om_grid)
    N_sn = len(z_sn)
    grid_sn = np.zeros((N_Om, N_sn), dtype=np.float64)
    grid_age = np.zeros(N_Om, dtype=np.float64)

    for i in range(N_Om):
        Om = Om_grid[i]
        Ol = 1.0 - Om

        # --- 1. Age integral (midpoint rule over a in (0, 1]) ---
        age_int = 0.0
        n_steps_age = 500
        da = 1.0 / n_steps_age
        for step in range(n_steps_age):
            a = (step + 0.5) * da
            val = Om / a + Ol * a * a
            if val > 0.0:
                age_int += 1.0 / np.sqrt(val)
        grid_age[i] = age_int * da

        # --- 2. Cumulative distance integral (trapezoidal) ---
        cum_I = np.zeros(n_z_steps + 1, dtype=np.float64)
        prev_inv = 1.0  # E(z=0) = sqrt(Om + Ol) = 1 -> 1/E = 1
        for step in range(1, n_z_steps + 1):
            z = z_array[step]
            val = Om * (1.0 + z) ** 3 + Ol
            curr_inv = 1.0 / np.sqrt(val) if val > 0.0 else 0.0
            cum_I[step] = cum_I[step - 1] + 0.5 * (prev_inv + curr_inv) * dz
            prev_inv = curr_inv

        # --- 3. Linear interpolation at SN redshifts ---
        for s in range(N_sn):
            z = z_sn[s]
            idx = int(z / dz)
            if idx >= n_z_steps:
                idx = n_z_steps - 1
            w = (z - z_array[idx]) / dz
            grid_sn[i, s] = cum_I[idx] * (1.0 - w) + cum_I[idx + 1] * w

    return grid_sn, grid_age


class GlobalCosmoDataCache:
    """One-time data load + Numba precompute; reused for every scenario."""

    def __init__(self):
        print("\n=== Initialize data + 1D Om grid  (CPU FP64) ===")
        self.z_sn, self.mb_sn, self.inv_cov_sn = load_pantheon_plus()
        self.bao_data = load_bao()
        
        self.z_sn = self.z_sn.astype(np.float64)
        self.mb_sn = self.mb_sn.astype(np.float64)
        self.inv_cov_sn = self.inv_cov_sn.astype(np.float64)

        # 100 nodes over a window that brackets the 0.1 < Om < 0.6 prior.
        self.Om_grid = np.linspace(0.05, 0.65, 100, dtype=np.float64)

        print("Numba 1D precompute (Om: 100 nodes, z-steps: 1000)...")
        t0 = time.time()
        grid_sn, grid_age = compute_grids_1d_lcdm_cpu(self.z_sn, self.Om_grid)
        print(f"✅ Grid built in {time.time() - t0:.2f}s")

        self.interp_sn = RegularGridInterpolator(
            (self.Om_grid,), grid_sn,
            bounds_error=False, fill_value=None,
        )
        self.interp_age = RegularGridInterpolator(
            (self.Om_grid,), grid_age,
            bounds_error=False, fill_value=None,
        )


# ====================================================================
# 2) Vectorized likelihood   (emcee vectorize=True)
# ====================================================================

class LikelihoodBAOVectorized:
    def __init__(self, age_obs, age_err, cache: GlobalCosmoDataCache):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.dt_prior_mu = 0.15
        self.dt_prior_sig = 0.05
        self.cache = cache

    def log_prob_vectorized(self, theta_batch):
        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]

        Om = th[:, 0]; H0 = th[:, 1]; M_B = th[:, 2]; rd = th[:, 3]; dt_form = th[:, 4]

        mask = (
            (0.1 < Om) & (Om < 0.6) &
            (50.0 < H0) & (H0 < 90.0) &
            (-20.0 < M_B) & (M_B < -18.0) &
            (120.0 < rd) & (rd < 160.0) &
            (0.0 < dt_form) & (dt_form < 1.0)
        )

        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask):
            return logL

        Om_v = Om[mask]; H0_v = H0[mask]; MB_v = M_B[mask]; rd_v = rd[mask]; dt_v = dt_form[mask]
        pts_v = Om_v[:, None]  # shape (Nv, 1) for the 1-D interpolator

        # ---- 1. SN Ia chi^2 (vectorized over walkers) ----
        d_sn_v = self.cache.interp_sn(pts_v)                        # (Nv, N_sn)
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])                 # Mpc
        mu_model = 5.0 * np.log10(
            D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10
        ) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])
        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)

        # ---- 2. Age prior + dt prior (vectorized) ----
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v).ravel()
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig) ** 2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err) ** 2

        # ---- 3. BAO chi^2 (loop over valid walkers) ----
        # Since BAO points are few and bao_model is a black box, looping over valid walkers is fast enough.
        for i in range(len(Om_v)):
            o = Om_v[i]
            h = H0_v[i]
            r = rd_v[i]
            ll_bao = 0.0
            for d in self.cache.bao_data:
                mod = bao_model(d['z'], d['type'], o, h, r)
                diff_b = d['y'] - mod
                ll_bao += -0.5 * (diff_b.T @ d['icov'] @ diff_b)
            logL_v[i] += ll_bao

        logL[mask] = logL_v
        return logL


# ====================================================================
# 3) MCMC driver
# ====================================================================

def run_mcmc(like, ndim=5, nwalkers=32, max_iter=6000):
    p0_center = np.array([0.31, 67.4, -19.4, 147.0, 0.15])
    scatter   = np.array([1e-2, 0.3,  1e-2,  1.0,   1e-3])
    p0 = [p0_center + scatter * np.random.randn(ndim) for _ in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, like.log_prob_vectorized, vectorize=True,
    )

    old_tau = np.inf
    for _ in sampler.sample(p0, iterations=max_iter, progress=True):
        if sampler.iteration % 100:
            continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception:
            continue
        if (np.all(tau * 50 < sampler.iteration)
                and np.all(np.abs(old_tau - tau) / tau < 0.01)):
            break
        old_tau = tau
    
    # --- Convergence diagnostic ---
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = np.max(tau)
        chain_len = sampler.iteration
        n_eff = np.mean(chain_len * nwalkers / tau)
        print(f"\n[Convergence BAO] tau_max: {tau_max:.1f}, "
              f"Chain/tau: {chain_len / tau_max:.1f}, N_eff: {n_eff:.0f}")
    except Exception as e:
        print(f"\n[Convergence BAO skipped]: {e}")
    
    discard = int(sampler.iteration * 0.3)
    return sampler.get_chain(discard=discard, flat=True), sampler


# ====================================================================
# 4) Main
# ====================================================================

def main():
    print("=" * 70)
    print("Figure 3: Pantheon+ + cosmic age + BAO, with r_d free")
    print("=" * 70)

    # ONE-TIME precompute, shared across all scenarios.
    global_cache = GlobalCosmoDataCache()

    all_mc = []
    rd_medians = []
    names = ["Om", "H0", "MB", "rd", "dt"]
    labels = [r"\Omega_m", r"H_0", r"M_B", r"r_d", r"\Delta t"]

    for sc in SCENARIOS:
        print(f"\n=== {sc['label']} (t = {sc['mu']} +/- {sc['err']} Gyr) ===")
        t0 = time.time()
        like = LikelihoodBAOVectorized(sc['mu'], sc['err'], global_cache)
        flat, sampler = run_mcmc(like)
        print(f"  MCMC done in {time.time() - t0:.1f}s  (iters: {sampler.iteration})")

        print("--- Posterior ---")
        for i, n in enumerate(names):
            p = np.percentile(flat[:, i], [16, 50, 84])
            print(f"  {n}: {p[1]:.4f}  +{p[2]-p[1]:.4f} / -{p[1]-p[0]:.4f}")
        rd_medians.append(np.median(flat[:, 3]))
        
        mc = MCSamples(
            samples=flat, names=names, labels=labels,
            label=sc['label'],
            settings={'smooth_scale_1D': 0.7, 'smooth_scale_2D': 0.7}
        )
        all_mc.append(mc)

        # Save per-scenario chain
        np.savez_compressed(
            f"chain_with_BAO_{sc['short']}.npz",
            chain=flat,
            names=np.array(names),
            labels=np.array(labels),
            age_obs=sc['mu'], age_err=sc['err'],
            label=sc['label'], color=sc['color'],
        )
        print(f"  ✅ Saved: chain_with_BAO_{sc['short']}.npz")

    # Panel A: Om, H0, MB triangle
    print("\nGenerating Figure 3A (triangle)...")
    g = plots.get_subplot_plotter(width_inch=6)
    g.triangle_plot(
        all_mc, params=["Om", "H0", "MB"], filled=True,
        contour_colors=[s['color'] for s in SCENARIOS],
        legend_loc='upper right'
    )
    plt.savefig("Figure3A_Triangle_BAO.pdf", bbox_inches='tight')

    # Panel B: r_d distribution
    print("Generating Figure 3B (r_d)...")
    plt.figure(figsize=(7.5, 4.8))
    for i, mc in enumerate(all_mc):
        d = mc.get1DDensity('rd')
        plt.plot(d.x, d.P, color=SCENARIOS[i]['color'], lw=2,
                 label=SCENARIOS[i]['label'])
        plt.fill_between(d.x, d.P, color=SCENARIOS[i]['color'], alpha=0.15)
        plt.axvline(rd_medians[i], color=SCENARIOS[i]['color'],
                    ls='--', lw=1.2)
        plt.text(rd_medians[i] - 1.0, 0.45,
                 f"$r_d \\approx {rd_medians[i]:.1f}$ Mpc",
                 color=SCENARIOS[i]['color'], rotation=90,
                 ha='right', va='center', fontweight='bold', fontsize=9,
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    plt.axvspan(PLANCK_RD[0]-PLANCK_RD[1], PLANCK_RD[0]+PLANCK_RD[1],
                color='gold', alpha=0.3, label='Planck 2018')
    plt.axvspan(BBN_RD[0]-BBN_RD[1], BBN_RD[0]+BBN_RD[1],
                color='green', alpha=0.15, label='BBN Prediction')
    plt.xlabel(r"Sound Horizon $r_d$ [Mpc]", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.xlim(125, 162)
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    plt.savefig("Figure3B_rd_distribution.pdf", bbox_inches='tight')
    print("✅ Saved Figure3A_Triangle_BAO.pdf and Figure3B_rd_distribution.pdf")

if __name__ == "__main__":
    main()