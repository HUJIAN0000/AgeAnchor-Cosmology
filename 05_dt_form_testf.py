# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:45:40 2026

@author: Administrator

Extended Data Figure 1: Sensitivity of inferred H_0 to the formation-time
delay Delta_t_form (instead of marginalizing).

OPTIMIZED VERSION:
  - Numba JIT-compiled 1D Om grid precomputation
  - SciPy RegularGridInterpolator -> O(1) lookup
  - emcee with vectorize=True: batched likelihood across all walkers
  - Global cache shared across the 13 MCMC runs

Output: EDFig1_dt_form.pdf
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

class LikelihoodFixedDtVectorized:
    """SN + age likelihood with Delta_t_form FIXED (not marginalized)."""
    def __init__(self, age_obs, age_err, dt_fixed, cache: GlobalCosmoDataCache):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.dt_fixed = float(dt_fixed)
        self.cache = cache

    def log_prob_vectorized(self, theta_batch):
        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]
        
        # NOTE: Only 3 parameters here, dt_form is fixed per class instance!
        Om = th[:, 0]; H0 = th[:, 1]; M_B = th[:, 2]

        mask = (
            (0.1 < Om) & (Om < 0.6) &
            (50.0 < H0) & (H0 < 90.0) &
            (-20.0 < M_B) & (M_B < -18.0)
        )

        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask):
            return logL

        Om_v = Om[mask]; H0_v = H0[mask]; MB_v = M_B[mask]
        pts_v = Om_v[:, None]

        # SN Ia
        d_sn_v = self.cache.interp_sn(pts_v)
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])
        mu_model = 5.0 * np.log10(D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])
        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)

        # Age prior (using the FIXED dt_form)
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v).ravel()
        logL_v += -0.5 * ((age_th - (self.age_obs + self.dt_fixed)) / self.age_err) ** 2

        logL[mask] = logL_v
        return logL

# ====================================================================
# 3) MCMC Driver & Main
# ====================================================================

def quick_mcmc(like, ndim=3, nwalkers=32, nsteps=2000):
    p0 = [np.array([0.31, 67.4, -19.4]) + 1e-3*np.random.randn(ndim)
          for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, like.log_prob_vectorized, vectorize=True
    )
    sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=int(nsteps*0.3), flat=True)
    return flat

def main():
    print("=" * 70)
    print("ED Figure 1: dt_form Sensitivity Curve (Vectorized)")
    print("=" * 70)
    
    # Init cache once
    global_cache = GlobalCosmoDataCache()

    dt_grid = np.linspace(0.0, 0.6, 13)
    h0_med = np.zeros_like(dt_grid)
    h0_lo = np.zeros_like(dt_grid)
    h0_hi = np.zeros_like(dt_grid)

    print("\nScanning Delta_t_form values (13 grid points)...")
    for i, dt in enumerate(tqdm(dt_grid)):
        like = LikelihoodFixedDtVectorized(13.5, 0.27, dt, global_cache)
        flat = quick_mcmc(like)
        p = np.percentile(flat[:, 1], [16, 50, 84])
        h0_med[i], h0_lo[i], h0_hi[i] = p[1], p[1]-p[0], p[2]-p[1]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.axhspan(PLANCK_H0[0]-PLANCK_H0[1], PLANCK_H0[0]+PLANCK_H0[1],
               color='#0066cc', alpha=0.25, label='Planck 2018')
    ax.axhspan(SH0ES_H0[0]-SH0ES_H0[1], SH0ES_H0[0]+SH0ES_H0[1],
               color='#cc0000', alpha=0.25, label='SH0ES 2022')

    ax.axvspan(0.1, 0.2, color='green', alpha=0.20,
               label='Standard Formation\nDelay (0.1-0.2 Gyr)')

    ax.plot(dt_grid, h0_med, 'k-', lw=2, label=r'Derived $H_0$')
    ax.fill_between(dt_grid, h0_med - h0_lo, h0_med + h0_hi,
                    color='gray', alpha=0.3, label=r'$1\sigma$ Uncertainty')

    ax.text(0.02, 67.5, r'Baseline ($\Delta t = 0$)', fontsize=9, color='black')

    ax.set_xlabel(r'Formation Time Delay $\Delta t_{\rm form}$ [Gyr]', fontsize=13)
    ax.set_ylabel(r'Derived $H_0$ ($t_{\rm tot}=13.5+\Delta t_{\rm form}$) [km s$^{-1}$ Mpc$^{-1}$]',
                  fontsize=12)
    ax.set_xlim(0.0, 0.6)
    ax.set_ylim(62, 74)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('EDFig1_dt_form.pdf', bbox_inches='tight')
    plt.savefig('EDFig1_dt_form.png', dpi=200, bbox_inches='tight')
    print("\n✅ Saved EDFig1_dt_form.pdf / .png")

if __name__ == "__main__":
    main()