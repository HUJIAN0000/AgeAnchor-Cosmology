# -*- coding: utf-8 -*-
"""
Extended Data: Sensitivity of the age-anchored H_0 to the assumed stellar-age
systematic uncertainty sigma_age.

【纯 CPU FP64 矢量化加速版】
- 自动保存 MCMC 统计数据 (.npz) 并融合了 10_pretty_replot.py 中的高级美化画图。
"""

import os
import time
import numpy as np
import emcee
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from scipy.interpolate import RegularGridInterpolator

from cosmoc import (
    load_pantheon_plus,
    PLANCK_H0, SH0ES_H0, SH0ES_MB # 引入绘图用参考带
)

c_light = 299792.458  # km/s
AGE_CONVERSION_FACTOR = 977.8

@jit(nopython=True, fastmath=True)
def compute_grids_1d_lcdm_cpu(z_sn, Om_grid):
    z_max = 0.0
    if len(z_sn) > 0: z_max = np.max(z_sn)
    n_z_steps = 2000
    dz = z_max / n_z_steps
    z_array = np.linspace(0, z_max, n_z_steps + 1)
    N_Om = len(Om_grid)
    N_sn = len(z_sn)

    grid_sn = np.zeros((N_Om, N_sn), dtype=np.float64)
    grid_age = np.zeros(N_Om, dtype=np.float64)
    
    for i in range(N_Om):
        Om = Om_grid[i]
        Ol = 1.0 - Om
        
        age_int = 0.0
        n_steps_age = 1000
        da = 1.0 / n_steps_age
        for step in range(n_steps_age):
            a = (step + 0.5) * da
            val = Om/a + Ol * a**2
            if val > 0: age_int += 1.0 / np.sqrt(val)
        grid_age[i] = age_int * da
        
        cum_I = np.zeros(n_z_steps + 1, dtype=np.float64)
        prev_val = Om + Ol
        prev_inv = 1.0 / np.sqrt(prev_val) if prev_val > 0 else 0.0
            
        for step in range(1, n_z_steps + 1):
            z = z_array[step]
            val = Om * (1.0+z)**3 + Ol
            curr_inv = 1.0 / np.sqrt(val) if val > 0 else 0.0
            cum_I[step] = cum_I[step-1] + 0.5 * (prev_inv + curr_inv) * dz
            prev_inv = curr_inv
            
        for s in range(N_sn):
            z = z_sn[s]
            idx = int(z / dz)
            if idx >= n_z_steps: idx = n_z_steps - 1
            w = (z - z_array[idx]) / dz
            grid_sn[i, s] = cum_I[idx] * (1.0 - w) + cum_I[idx+1] * w
                    
    return grid_sn, grid_age

class GlobalCosmoDataCache1D:
    def __init__(self):
        print("\n=== 初始化数据与 1D (Om) 积分网格 (纯 CPU FP64) ===")
        self.z_sn, self.mb_sn, self.inv_cov_sn = load_pantheon_plus()
        self.z_sn = self.z_sn.astype(np.float64)
        self.mb_sn = self.mb_sn.astype(np.float64)
        self.inv_cov_sn = self.inv_cov_sn.astype(np.float64)
        
        self.Om_grid = np.linspace(0.05, 0.65, 500, dtype=np.float64)
        t0 = time.time()
        grid_sn, grid_age = compute_grids_1d_lcdm_cpu(self.z_sn, self.Om_grid)
        print(f"✅ 1D 网格预计算完成，耗时: {time.time() - t0:.4f} 秒")
        
        self.interp_sn = RegularGridInterpolator((self.Om_grid,), grid_sn, bounds_error=False, fill_value=None)
        self.interp_age = RegularGridInterpolator((self.Om_grid,), grid_age, bounds_error=False, fill_value=None)


class LikelihoodAgeOnlyVectorizedCPU:
    def __init__(self, age_obs, age_err, cache: GlobalCosmoDataCache1D):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.cache = cache
        self.dt_prior_mu = 0.15
        self.dt_prior_sig = 0.05

    def log_prob_vectorized(self, theta_batch):
        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]
        Om = th[:, 0]; H0 = th[:, 1]; M_B = th[:, 2]; dt_form = th[:, 3]

        mask = ( (0.1 < Om) & (Om < 0.6) & (50.0 < H0) & (H0 < 90.0) &
                 (-20.0 < M_B) & (M_B < -18.0) & (0.0 < dt_form) & (dt_form < 1.0) )
        
        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask): return logL
            
        Om_v, H0_v, MB_v, dt_v = Om[mask], H0[mask], M_B[mask], dt_form[mask]
        pts_v = Om_v.reshape(-1, 1)
        
        d_sn_v = self.cache.interp_sn(pts_v) 
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])
        mu_model = 5.0 * np.log10(D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])
        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)
        
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v)
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig)**2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err)**2
        
        logL[mask] = logL_v
        return logL


def run_mcmc_vectorized(like, ndim=4, nwalkers=32, nsteps=4000, burn_frac=0.3):
    p0 = [np.array([0.31, 67.4, -19.4, 0.15]) + 1e-3 * np.random.randn(ndim) for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_prob_vectorized, vectorize=True)
    sampler.run_mcmc(p0, nsteps, progress=False)
    flat = sampler.get_chain(discard=int(nsteps * burn_frac), flat=True)
    return flat


def main():
    cache = GlobalCosmoDataCache1D()

    sigma_grid = [0.27, 0.40, 0.50, 0.70]
    tGC = 13.5

    H0_med = np.zeros(len(sigma_grid))
    H0_lo  = np.zeros(len(sigma_grid))
    H0_hi  = np.zeros(len(sigma_grid))
    MB_med = np.zeros(len(sigma_grid))
    MB_lo  = np.zeros(len(sigma_grid))
    MB_hi  = np.zeros(len(sigma_grid))

    print(f"Scanning sigma_age at fixed tGC = {tGC} Gyr ...\n")
    for i, sigma in enumerate(tqdm(sigma_grid)):
        t_start = time.time()
        like = LikelihoodAgeOnlyVectorizedCPU(tGC, sigma, cache)
        flat = run_mcmc_vectorized(like, nsteps=4000)
        
        h0p = np.percentile(flat[:, 1], [16, 50, 84])
        mbp = np.percentile(flat[:, 2], [16, 50, 84])
        H0_med[i], H0_lo[i], H0_hi[i] = h0p[1], h0p[1]-h0p[0], h0p[2]-h0p[1]
        MB_med[i], MB_lo[i], MB_hi[i] = mbp[1], mbp[1]-mbp[0], mbp[2]-mbp[1]
        
        print(f"  sigma_t = {sigma:.2f} Gyr (耗时 {time.time()-t_start:.1f}s) -> "
              f"H0 = {h0p[1]:.2f} +{h0p[2]-h0p[1]:.2f}/-{h0p[1]-h0p[0]:.2f}")

    # =============== [保存 MCMC 提取的统计数据] ===============
    np.savez_compressed(
        "chain_sigma_scan.npz",
        sigma_grid=np.array(sigma_grid),
        H0_med=H0_med, H0_lo=H0_lo, H0_hi=H0_hi,
        MB_med=MB_med, MB_lo=MB_lo, MB_hi=MB_hi,
    )
    print("  ✅ 已保存扫描数据至: chain_sigma_scan.npz")

    # =====================================================
    # 融合 10_pretty_replot.py 的美化画图
    # =====================================================
    PLANCK_COLOR = '#0066cc'
    SH0ES_COLOR  = '#cc0000'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- H0 Panel ---
    ax1.axhspan(PLANCK_H0[0] - PLANCK_H0[1], PLANCK_H0[0] + PLANCK_H0[1],
                color=PLANCK_COLOR, alpha=0.18, label='Planck 2018', zorder=0)
    ax1.axhspan(SH0ES_H0[0] - SH0ES_H0[1], SH0ES_H0[0] + SH0ES_H0[1],
                color=SH0ES_COLOR, alpha=0.18, label='SH0ES 2022', zorder=0)
    
    ax1.errorbar(sigma_grid, H0_med, yerr=[H0_lo, H0_hi],
                 fmt='ko', markersize=8, capsize=4, lw=1.8, label='Age-anchored $H_0$')
    
    ax1.set_xlabel(r'Assumed stellar-age systematic $\sigma_t$ [Gyr]', fontsize=14)
    ax1.set_ylabel(r'$H_0 \,\, [\mathrm{km\,s^{-1}\,Mpc^{-1}}]$', fontsize=14)
    # 缩紧 Y 轴以获得更好视觉效果
    ax1.set_ylim(62, 75)
    ax1.set_xlim(0.20, 0.75)
    ax1.grid(alpha=0.3, ls='--')
    ax1.legend(fontsize=11, loc='upper right', frameon=True)
    ax1.set_title('Inferred $H_0$ widens but stays clear of SH0ES band', fontsize=13)

    # --- MB Panel ---
    ax2.axhspan(SH0ES_MB[0] - SH0ES_MB[1], SH0ES_MB[0] + SH0ES_MB[1],
                color=SH0ES_COLOR, alpha=0.18, label='SH0ES Cepheid $M_B$', zorder=0)
    
    ax2.errorbar(sigma_grid, MB_med, yerr=[MB_lo, MB_hi],
                 fmt='ko', markersize=8, capsize=4, lw=1.8, label='Age-anchored $M_B$')
    
    ax2.set_xlabel(r'Assumed stellar-age systematic $\sigma_t$ [Gyr]', fontsize=14)
    ax2.set_ylabel(r'$M_B \,\, [\mathrm{mag}]$', fontsize=14)
    ax2.invert_yaxis()
    ax2.set_xlim(0.20, 0.75)
    # 缩紧 Y 轴
    ax2.set_ylim(-19.10, -19.65)
    ax2.grid(alpha=0.3, ls='--')
    ax2.legend(fontsize=11, loc='lower right', frameon=True)
    ax2.set_title(r'$M_B$ tension with SH0ES is preserved across $\sigma_t$', fontsize=13)

    plt.tight_layout()
    plt.savefig('EDFig4_age_sigma_scan.pdf', bbox_inches='tight')
    plt.savefig('EDFig4_age_sigma_scan.png', dpi=200, bbox_inches='tight')
    print("✅ 已保存完美排版的 EDFig4_age_sigma_scan.pdf / .png")

    with open('EDFig4_age_sigma_scan_summary.txt', 'w') as f:
        f.write("Sensitivity of age-anchored H0 / MB to assumed sigma_age\n")
        for i, s in enumerate(sigma_grid):
            f.write(f"sigma={s:.2f} | H0={H0_med[i]:.2f} | MB={MB_med[i]:.3f}\n")

if __name__ == "__main__":
    main()