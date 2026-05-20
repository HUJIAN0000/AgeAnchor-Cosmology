# -*- coding: utf-8 -*-
"""
Extended Data: joint inference using TWO independent stellar-age anchors
(Valcin et al. 2021 globular clusters + Lundkvist et al. 2025 HD 140283),
versus each one alone.

Pure CPU FP64 vectorised acceleration pipeline.

Revision notes
--------------
- Removed the redundant 'chain_joint.npz' save inside run_mcmc_vectorized
  (it was overwritten for every configuration). Each configuration now
  saves only to its own 'chain_joint_<safe_name>.npz' file in main().
- Convergence diagnostic line is now tagged with the configuration label
  ("[Convergence GC only]", etc.), so the printout in the log is
  unambiguous when looking at Table 3 of the manuscript.
"""

import os
import time
import numpy as np
import emcee
import pandas as pd
import matplotlib.pyplot as plt
from getdist import plots, MCSamples
from numba import jit
from scipy.interpolate import RegularGridInterpolator

from cosmoc import (
    load_pantheon_plus,
    PLANCK_H0, SH0ES_H0, SH0ES_MB,   # 绘图用参考带
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
        print(f"✅ 1D 网格预计算完成，耗时: {time.time() - t0:.4f} 秒\n")
        self.interp_sn = RegularGridInterpolator((self.Om_grid,), grid_sn, bounds_error=False, fill_value=None)
        self.interp_age = RegularGridInterpolator((self.Om_grid,), grid_age, bounds_error=False, fill_value=None)


class LikelihoodAgesVectorizedCPU:
    def __init__(self, age_specs, cache: GlobalCosmoDataCache1D):
        self.age_specs = age_specs
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

        for (lab, mu, sig) in self.age_specs:
            logL_v += -0.5 * ((age_th - (mu + dt_v)) / sig) ** 2

        logL[mask] = logL_v
        return logL


def run_mcmc_vectorized(like, ndim=4, nwalkers=32, nsteps=5000, burn_frac=0.3,
                        label="(unnamed)"):
    """Run a fixed-length MCMC and print a per-configuration convergence summary.

    NOTE: This function no longer writes any .npz file itself. The
    caller (main) is responsible for saving the flat chain under a
    configuration-specific filename, which avoids the previous bug where
    a generic 'chain_joint.npz' was overwritten for every config.
    """
    p0 = [np.array([0.31, 67.4, -19.4, 0.15]) + 1e-3 * np.random.randn(ndim)
          for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_prob_vectorized, vectorize=True)
    sampler.run_mcmc(p0, nsteps, progress=True)

    # --- Convergence diagnostic (per-configuration) ---
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.max(tau))
        n_eff = float(np.mean(nsteps * nwalkers / tau))
        print(f"\n[Convergence {label}] "
              f"tau_max: {tau_max:.1f}, "
              f"Chain/tau: {nsteps / tau_max:.1f}, "
              f"N_eff: {n_eff:.0f}, "
              f"acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    except Exception as e:
        print(f"\n[Convergence {label} skipped]: {e}")

    flat = sampler.get_chain(discard=int(nsteps * burn_frac), flat=True)
    return flat


def percentiles_to_str(name, samples_col):
    p = np.percentile(samples_col, [16, 50, 84])
    return p[1], p[2] - p[1], p[1] - p[0]


def main():
    cache = GlobalCosmoDataCache1D()

    configs = [
        ("GC only",             [("GC", 13.5, 0.27)],                          '#0066cc'),
        ("HD140283 only",       [("HD140283", 14.2, 0.40)],                    '#888888'),
        ("Joint GC + HD140283", [("GC", 13.5, 0.27), ("HD140283", 14.2, 0.40)], '#cc6600'),
    ]

    all_mc = []
    summary = []
    names  = ["Om", "H0", "MB", "dt"]
    labels = [r"\Omega_m", r"H_0", r"M_B", r"\Delta t"]

    print("开始遍历 3 种组合配置并进行 MCMC 采样...")
    for (cname, specs, color) in configs:
        print(f"\n[Config] {cname}")
        t_start = time.time()
        like = LikelihoodAgesVectorizedCPU(specs, cache)
        flat = run_mcmc_vectorized(like, nsteps=5000, label=cname)

        print(f"  --- posterior (耗时: {time.time() - t_start:.2f} 秒) ---")
        row = {"config": cname}
        for i, n in enumerate(names):
            med, hi, lo = percentiles_to_str(n, flat[:, i])
            print(f"    {n:<5} = {med:.4f}  +{hi:.4f} / -{lo:.4f}")
            row[n] = (med, hi, lo)
        summary.append(row)

        mc = MCSamples(samples=flat, names=names, labels=labels,
                       label=cname, settings={'smooth_scale_1D': 0.7, 'smooth_scale_2D': 0.7})
        all_mc.append(mc)

        # =============== [Save chain — per configuration] ===============
        safe_name = cname.replace(' ', '_').replace('+', 'plus')
        np.savez_compressed(
            f"chain_joint_{safe_name}.npz",
            chain=flat,
            names=np.array(names),
            labels=np.array(labels),
            label=cname, color=color,
        )
        print(f"  ✅ 已保存链至: chain_joint_{safe_name}.npz")

    # =====================================================
    # Pretty plot + reference bands
    # =====================================================
    print("\nGenerating pretty EDFig5...")
    g = plots.get_subplot_plotter(width_inch=8)
    g.settings.axes_fontsize   = 11
    g.settings.lab_fontsize    = 13
    g.settings.legend_fontsize = 11
    g.settings.figure_legend_frame = False

    g.triangle_plot(all_mc, params=["Om", "H0", "MB"], filled=True,
                    contour_colors=[c for (_, _, c) in configs],
                    legend_loc='upper right')

    PLANCK_COLOR = '#0066cc'
    SH0ES_COLOR  = '#cc0000'
    try:
        # H0 占对角线索引 1 (Om 是 0)
        axH0 = g.subplots[1, 1]
        if axH0 is not None:
            axH0.axvspan(PLANCK_H0[0] - PLANCK_H0[1], PLANCK_H0[0] + PLANCK_H0[1],
                         color=PLANCK_COLOR, alpha=0.18, zorder=0)
            axH0.axvspan(SH0ES_H0[0] - SH0ES_H0[1], SH0ES_H0[0] + SH0ES_H0[1],
                         color=SH0ES_COLOR, alpha=0.18, zorder=0)

        # M_B 占对角线索引 2
        axMB = g.subplots[2, 2]
        if axMB is not None:
            axMB.axvspan(SH0ES_MB[0] - SH0ES_MB[1], SH0ES_MB[0] + SH0ES_MB[1],
                         color=SH0ES_COLOR, alpha=0.18, zorder=0)
    except Exception as e:
        print("⚠️ 无法添加参考阴影带:", e)

    plt.savefig("EDFig5_joint_ages.pdf", bbox_inches='tight')
    plt.savefig("EDFig5_joint_ages.png", dpi=200, bbox_inches='tight')
    print("✅ 已保存完美排版的 EDFig5_joint_ages.pdf / .png")

    # ----- Text summary -----
    out = ["=" * 78, "EDFig5 — Joint age-prior comparison summary", "=" * 78]
    out.append(f"{'Configuration':<28} | {'H0':>14} | {'MB':>16} | {'Om':>14}")
    out.append("-" * 78)
    for row in summary:
        h0 = f"{row['H0'][0]:.2f} +/-{max(row['H0'][1], row['H0'][2]):.2f}"
        mb = f"{row['MB'][0]:.3f} +/-{max(row['MB'][1], row['MB'][2]):.3f}"
        om = f"{row['Om'][0]:.3f} +/-{max(row['Om'][1], row['Om'][2]):.3f}"
        out.append(f"{row['config']:<28} | {h0:>14} | {mb:>16} | {om:>14}")
    summary_text = "\n".join(out)
    print("\n" + summary_text)
    with open("EDFig5_joint_ages_summary.txt", "w") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
