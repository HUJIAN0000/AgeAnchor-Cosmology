# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:41:53 2026

@author: Administrator

Figure 4: H_0 as a continuous function of the cosmic age prior t_0.

OPTIMIZED VERSION:
  - Numba JIT-compiled 1D Om grid precomputation
  - SciPy RegularGridInterpolator -> O(1) lookup
  - emcee with vectorize=True: batched likelihood across all walkers
  - Global cache shared across the 15+3 MCMC runs

Output: Figure4_Sensitivity.pdf
"""
import time
import numpy as np
import emcee
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from cosmoc import (
    c_light, load_pantheon_plus,
    PLANCK_H0, SH0ES_H0
)

AGE_CONVERSION_FACTOR = 977.8  # Gyr * (km/s/Mpc)

# ====================================================================
# 1) Numba 1D grid precomputation over Om   (flat LCDM: w0 = -1, wa = 0)
# ====================================================================

@jit(nopython=True, fastmath=True)
def compute_grids_1d_lcdm_cpu(z_sn, Om_grid):
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

        # --- 1. Age integral ---
        age_int = 0.0
        n_steps_age = 500
        da = 1.0 / n_steps_age
        for step in range(n_steps_age):
            a = (step + 0.5) * da
            val = Om / a + Ol * a * a
            if val > 0.0:
                age_int += 1.0 / np.sqrt(val)
        grid_age[i] = age_int * da

        # --- 2. Cumulative distance integral ---
        cum_I = np.zeros(n_z_steps + 1, dtype=np.float64)
        prev_inv = 1.0
        for step in range(1, n_z_steps + 1):
            z = z_array[step]
            val = Om * (1.0 + z) ** 3 + Ol
            curr_inv = 1.0 / np.sqrt(val) if val > 0.0 else 0.0
            cum_I[step] = cum_I[step - 1] + 0.5 * (prev_inv + curr_inv) * dz
            prev_inv = curr_inv

        # --- 3. Linear interpolation ---
        for s in range(N_sn):
            z = z_sn[s]
            idx = int(z / dz)
            if idx >= n_z_steps:
                idx = n_z_steps - 1
            w = (z - z_array[idx]) / dz
            grid_sn[i, s] = cum_I[idx] * (1.0 - w) + cum_I[idx + 1] * w

    return grid_sn, grid_age


class GlobalCosmoDataCache:
    """One-time data load + Numba precompute; reused for all grid points."""
    def __init__(self):
        print("\n=== Initialize data + 1D Om grid ===")
        self.z_sn, self.mb_sn, self.inv_cov_sn = load_pantheon_plus()
        self.z_sn = self.z_sn.astype(np.float64)
        self.mb_sn = self.mb_sn.astype(np.float64)
        self.inv_cov_sn = self.inv_cov_sn.astype(np.float64)

        self.Om_grid = np.linspace(0.05, 0.65, 100, dtype=np.float64)
        
        t0 = time.time()
        grid_sn, grid_age = compute_grids_1d_lcdm_cpu(self.z_sn, self.Om_grid)
        print(f"✅ Grid built in {time.time() - t0:.2f}s")

        self.interp_sn = RegularGridInterpolator(
            (self.Om_grid,), grid_sn, bounds_error=False, fill_value=None
        )
        self.interp_age = RegularGridInterpolator(
            (self.Om_grid,), grid_age, bounds_error=False, fill_value=None
        )


# ====================================================================
# 2) Vectorized likelihood
# ====================================================================

class LikelihoodNoBAOVectorized:
    def __init__(self, age_obs, age_err, cache: GlobalCosmoDataCache):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.dt_prior_mu = 0.15
        self.dt_prior_sig = 0.05
        self.cache = cache

    def log_prob_vectorized(self, theta_batch):
        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]
        Om = th[:, 0]; H0 = th[:, 1]; M_B = th[:, 2]; dt_form = th[:, 3]

        mask = (
            (0.1 < Om) & (Om < 0.6) &
            (50.0 < H0) & (H0 < 90.0) &
            (-20.0 < M_B) & (M_B < -18.0) &
            (0.0 < dt_form) & (dt_form < 1.0)
        )

        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask):
            return logL

        Om_v = Om[mask]; H0_v = H0[mask]; MB_v = M_B[mask]; dt_v = dt_form[mask]
        pts_v = Om_v[:, None]

        # SN Ia
        d_sn_v = self.cache.interp_sn(pts_v)
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])
        mu_model = 5.0 * np.log10(D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])
        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)

        # Age + dt priors
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v).ravel()
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig) ** 2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err) ** 2

        logL[mask] = logL_v
        return logL

# ====================================================================
# 3) MCMC Driver & Main
# ====================================================================

def quick_mcmc(like, ndim=4, nwalkers=32, nsteps=2000):
    p0 = [np.array([0.31, 67.4, -19.4, 0.15]) + 1e-3*np.random.randn(ndim)
          for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, like.log_prob_vectorized, vectorize=True
    )
    sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=int(nsteps*0.3), flat=True)
    return flat

def main():
    print("=" * 70)
    print("Figure 4: Sensitivity Curve (Vectorized)")
    print("=" * 70)
    
    # Init cache once
    global_cache = GlobalCosmoDataCache()

    t0_grid = np.linspace(12.0, 15.0, 15)
    h0_med = np.zeros_like(t0_grid)
    h0_lo = np.zeros_like(t0_grid)
    h0_hi = np.zeros_like(t0_grid)

    print("\nScanning cosmic age values (15 grid points)...")
    for i, t0 in enumerate(tqdm(t0_grid)):
        like = LikelihoodNoBAOVectorized(t0, 0.27, global_cache)
        flat = quick_mcmc(like)
        p = np.percentile(flat[:, 1], [16, 50, 84])
        h0_med[i], h0_lo[i], h0_hi[i] = p[1], p[1]-p[0], p[2]-p[1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhspan(PLANCK_H0[0]-PLANCK_H0[1], PLANCK_H0[0]+PLANCK_H0[1],
               color='#0066cc', alpha=0.2, label='Planck 2018')
    ax.axhspan(SH0ES_H0[0]-SH0ES_H0[1], SH0ES_H0[0]+SH0ES_H0[1],
               color='#cc0000', alpha=0.2, label='SH0ES 2022')

    ax.plot(t0_grid, h0_med, 'k-', lw=2, label='Age-$H_0$ Relation')
    ax.fill_between(t0_grid, h0_med - h0_lo, h0_med + h0_hi,
                    color='sandybrown', alpha=0.4, label=r'$1\sigma$ Uncertainty')

    # Mark the three key scenarios
    print("\nRunning specific scenario markers...")
    markers = [
        (12.6, 0.27, 'SH0ES Implied\n(Young Age)', '#cc0000', 's'),
        (13.5, 0.27, 'Globular Clusters\n(This Work)', '#0066cc', '*'),
        (14.2, 0.40, 'HD 140283\n(Old Age)', 'gray', 'o'),
    ]
    for t, err, lab, c, m in markers:
        like = LikelihoodNoBAOVectorized(t, err, global_cache)
        flat = quick_mcmc(like, nsteps=2500)
        p = np.percentile(flat[:, 1], [16, 50, 84])
        ax.errorbar(t, p[1], yerr=[[p[1]-p[0]], [p[2]-p[1]]],
                    fmt=m, color=c, markersize=12, capsize=4,
                    label=lab.split('\n')[0])
        ax.annotate(lab, xy=(t, p[1]), xytext=(t, p[1]+1.5),
                    color=c, fontsize=10, ha='center', fontweight='bold')

    ax.set_xlabel(r'Cosmic Age $t_0$ [Gyr]', fontsize=13)
    ax.set_ylabel(r'Hubble Constant $H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=13)
    ax.set_xlim(12.0, 15.0)
    ax.set_ylim(60, 78)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Figure4_Sensitivity.pdf', bbox_inches='tight')
    plt.savefig('Figure4_Sensitivity.png', dpi=200, bbox_inches='tight')
    print("\n✅ Saved Figure4_Sensitivity.pdf / .png")

if __name__ == "__main__":
    main()