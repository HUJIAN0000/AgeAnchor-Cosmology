# -*- coding: utf-8 -*-
"""
Extended Data Figure 3: w0waCDM robustness test.

Pure CPU FP64 vectorised acceleration pipeline (LCDMCPUFP64 style):
- emcee with vectorize=True (no Python-level walker loop).
- Numba precompute of a 3D (Om, w0, wa) cosmological-integral grid (FP64).
- Grid is built once globally and reused across all three age scenarios.
- MCMC chains are saved (.npz). Pretty corner plot at the end.

Revision notes
--------------
- Added convergence diagnostics printout for each scenario (tau_max,
  Chain/tau, N_eff) so that the values feeding the Methods Table 3 are
  reproducible from a single run.
- Highlighted the hard-coded flat-LCDM+BAO reference triplet
  (H0_lcdm, rd_lcdm) used to compute |Delta H0| in the summary block:
  ** these must be kept in sync with the latest output of
     03_with_BAO2f.py (Globular Cluster scenario). **
"""

import os
import time
import numpy as np
import pandas as pd
import emcee
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import RegularGridInterpolator

from getdist import plots, MCSamples

# 导入数据与基础常量
from cosmoc import (
    c_light,
    load_pantheon_plus, load_bao,
    SCENARIOS, PLANCK_RD, BBN_RD,
    PLANCK_H0, SH0ES_H0   # 绘图用的参考带
)

AGE_CONVERSION_FACTOR = 977.8

# ====================================================================
# !!! IMPORTANT: keep these in sync with the latest run of
#     03_with_BAO2f.py for the Globular Cluster scenario.
#     They are the LCDM+BAO baseline against which the CPL shift
#     |Delta H0| is reported in the summary text and in the paper.
# ====================================================================
H0_LCDM_GC_BASELINE = 68.91      # km/s/Mpc  (median, GC, flat LCDM + BAO)
H0_LCDM_GC_BASELINE_ERR = 1.45   # 1-sigma
RD_LCDM_GC_BASELINE = 146.57     # Mpc
RD_LCDM_GC_BASELINE_ERR = 2.96   # 1-sigma


# ====================================================================
# 1) Numba 3D 网格预计算引擎 (Om, w0, wa) -> 纯 FP64
# ====================================================================
@jit(nopython=True, fastmath=True)
def compute_grids_3d_w0wa_cpu(z_sn, z_bao, Om_grid, w0_grid, wa_grid):
    z_max = 0.0
    if len(z_sn) > 0: z_max = max(z_max, np.max(z_sn))
    if len(z_bao) > 0: z_max = max(z_max, np.max(z_bao))

    n_z_steps = 1000
    dz = z_max / n_z_steps
    z_array = np.linspace(0, z_max, n_z_steps + 1)

    N_Om = len(Om_grid)
    N_w0 = len(w0_grid)
    N_wa = len(wa_grid)
    N_sn = len(z_sn)
    N_bao = len(z_bao)

    grid_sn = np.zeros((N_Om, N_w0, N_wa, N_sn), dtype=np.float64)
    grid_bao = np.zeros((N_Om, N_w0, N_wa, N_bao), dtype=np.float64)
    grid_age = np.zeros((N_Om, N_w0, N_wa), dtype=np.float64)

    for i in range(N_Om):
        for j in range(N_w0):
            for k in range(N_wa):
                Om = Om_grid[i]
                w0 = w0_grid[j]
                wa = wa_grid[k]
                Ol = 1.0 - Om

                # --- 1. 计算年龄积分 ---
                age_int = 0.0
                n_steps_age = 500
                da = 1.0 / n_steps_age
                for step in range(n_steps_age):
                    a = (step + 0.5) * da
                    f_DE_a = a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
                    val = Om/a + Ol * a**2 * f_DE_a
                    if val > 0:
                        age_int += 1.0 / np.sqrt(val)
                grid_age[i, j, k] = age_int * da

                # --- 2. 计算距离累积积分 ---
                cum_I = np.zeros(n_z_steps + 1, dtype=np.float64)
                prev_val = Om + Ol
                prev_inv = 1.0 / np.sqrt(prev_val) if prev_val > 0 else 0.0

                for step in range(1, n_z_steps + 1):
                    z = z_array[step]
                    f_DE_z = (1.0+z)**(3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*z/(1.0+z))
                    val = Om * (1.0+z)**3 + Ol * f_DE_z
                    curr_inv = 1.0 / np.sqrt(val) if val > 0 else 0.0
                    cum_I[step] = cum_I[step-1] + 0.5 * (prev_inv + curr_inv) * dz
                    prev_inv = curr_inv

                # --- 3. 提取 SNe 距离 ---
                for s in range(N_sn):
                    z = z_sn[s]
                    idx = int(z / dz)
                    if idx >= n_z_steps: idx = n_z_steps - 1
                    w = (z - z_array[idx]) / dz
                    grid_sn[i, j, k, s] = cum_I[idx] * (1.0 - w) + cum_I[idx+1] * w

                # --- 4. 提取 BAO 距离 ---
                for b in range(N_bao):
                    z = z_bao[b]
                    idx = int(z / dz)
                    if idx >= n_z_steps: idx = n_z_steps - 1
                    w = (z - z_array[idx]) / dz
                    grid_bao[i, j, k, b] = cum_I[idx] * (1.0 - w) + cum_I[idx+1] * w

    return grid_sn, grid_bao, grid_age


class GlobalCosmoDataCache:
    def __init__(self):
        print("\n=== 初始化全局数据与 3D 积分网格 (纯 CPU FP64) ===")
        self.z_sn, self.mb_sn, self.inv_cov_sn = load_pantheon_plus()
        self.bao_data = load_bao()

        self.z_sn = self.z_sn.astype(np.float64)
        self.mb_sn = self.mb_sn.astype(np.float64)
        self.inv_cov_sn = self.inv_cov_sn.astype(np.float64)

        bao_z_list = []
        for d in self.bao_data:
            d['z'] = d['z'].astype(np.float64)
            d['y'] = d['y'].astype(np.float64)
            d['icov'] = d['icov'].astype(np.float64)
            bao_z_list.extend(d['z'])
        self.z_bao = np.array(bao_z_list, dtype=np.float64)

        self.Om_grid = np.linspace(0.08, 0.62, 30, dtype=np.float64)
        self.w0_grid = np.linspace(-2.2, 0.2, 30, dtype=np.float64)
        self.wa_grid = np.linspace(-3.2, 2.2, 30, dtype=np.float64)

        print("正在进行 3D Numba 预计算 (网格: 30x30x30)...")
        t0 = time.time()
        grid_sn, grid_bao, grid_age = compute_grids_3d_w0wa_cpu(
            self.z_sn, self.z_bao, self.Om_grid, self.w0_grid, self.wa_grid
        )
        print(f"✅ 3D 网格预计算完成，耗时: {time.time() - t0:.2f} 秒")

        grid_coords = (self.Om_grid, self.w0_grid, self.wa_grid)
        self.interp_sn = RegularGridInterpolator(grid_coords, grid_sn, bounds_error=False, fill_value=None)
        self.interp_age = RegularGridInterpolator(grid_coords, grid_age, bounds_error=False, fill_value=None)
        if len(self.z_bao) > 0:
            self.interp_bao = RegularGridInterpolator(grid_coords, grid_bao, bounds_error=False, fill_value=None)


class LikelihoodW0WaVectorizedCPU:
    def __init__(self, age_obs, age_err, cache: GlobalCosmoDataCache):
        self.age_obs = float(age_obs)
        self.age_err = float(age_err)
        self.dt_prior_mu = 0.15
        self.dt_prior_sig = 0.05
        self.cache = cache

    def log_prob_vectorized(self, theta_batch):
        th = np.asarray(theta_batch, dtype=np.float64)
        B = th.shape[0]

        Om = th[:, 0]; H0 = th[:, 1]; M_B = th[:, 2]; rd = th[:, 3]
        dt_form = th[:, 4]; w0 = th[:, 5]; wa = th[:, 6]

        pts = np.column_stack((Om, w0, wa))

        mask = ( (0.25 < Om) & (Om < 0.6) & (50.0 < H0) & (H0 < 90.0) &
                 (-20.0 < M_B) & (M_B < -18.0) & (120.0 < rd) & (rd < 160.0) &
                 (0.0 < dt_form) & (dt_form < 1.0) &
                 (-2.0 < w0) & (w0 < 0.0) & (-3.0 < wa) & (wa < 2.0) )

        logL = np.full(B, -np.inf, dtype=np.float64)
        if not np.any(mask): return logL

        pts_v = pts[mask]
        Om_v, H0_v, MB_v, rd_v, dt_v, w0_v, wa_v = (
            Om[mask], H0[mask], M_B[mask], rd[mask], dt_form[mask], w0[mask], wa[mask]
        )
        Ol_v = 1.0 - Om_v

        d_sn_v = self.cache.interp_sn(pts_v)
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])
        mu_model = 5.0 * np.log10(D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])

        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)

        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v)
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig)**2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err)**2

        if len(self.cache.z_bao) > 0:
            d_bao_v = self.cache.interp_bao(pts_v)
            D_M_bao = d_bao_v * (c_light / H0_v[:, None])
            chi2_bao = np.zeros(len(pts_v), dtype=np.float64)
            idx_start = 0

            for d in self.cache.bao_data:
                N_pts = len(d['z'])
                z_arr = d['z']; y_obs = d['y']; icov = d['icov']
                D_M_b = D_M_bao[:, idx_start : idx_start + N_pts]
                f_DE = (1.0 + z_arr[None, :])**(3.0*(1.0+w0_v[:, None]+wa_v[:, None])) * np.exp(-3.0*wa_v[:, None]*z_arr[None, :]/(1.0+z_arr[None, :]))
                Ez = np.sqrt(Om_v[:, None] * (1.0 + z_arr[None, :])**3 + Ol_v[:, None] * f_DE)
                D_H_b = c_light / (H0_v[:, None] * Ez)
                D_V_b = (z_arr[None, :] * D_M_b**2 * D_H_b)**(1.0/3.0)

                mod = np.zeros_like(D_M_b, dtype=np.float64)
                for k in range(N_pts):
                    t = d['type'][k]
                    if t == 'DM_over_rs': mod[:, k] = D_M_b[:, k] / rd_v
                    elif t == 'DH_over_rs': mod[:, k] = D_H_b[:, k] / rd_v
                    else: mod[:, k] = D_V_b[:, k] / rd_v

                diff_b = y_obs[None, :] - mod
                chi2_bao += np.sum(diff_b * (diff_b @ icov), axis=1)
                idx_start += N_pts

            logL_v += -0.5 * chi2_bao

        logL[mask] = logL_v
        return logL


def run_mcmc(like, ndim=7, nwalkers=48, max_iter=10000, sc_short="(unnamed)"):
    """Adaptive MCMC with convergence diagnostics printout at the end."""
    p0_center = np.array([0.31, 67.4, -19.4, 147.0, 0.15, -1.0, 0.0])
    p0 = [p0_center + np.array([1e-2, 0.3, 1e-2, 0.5, 1e-3, 5e-2, 5e-2]) * np.random.randn(ndim)
          for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, like.log_prob_vectorized, vectorize=True)

    old_tau = np.inf
    for _ in sampler.sample(p0, iterations=max_iter, progress=True):
        if sampler.iteration % 200: continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception:
            continue
        if (np.all(tau * 50 < sampler.iteration)
                and np.all(np.abs(old_tau - tau) / tau < 0.01)):
            break
        old_tau = tau

    # --- Convergence diagnostic (feeds Table 3 of the manuscript) ---
    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_max = float(np.max(tau))
        chain_len = int(sampler.iteration)
        n_eff = float(np.mean(chain_len * nwalkers / tau))
        print(f"\n[Convergence w0waCDM {sc_short}] "
              f"tau_max: {tau_max:.1f}, "
              f"Chain/tau: {chain_len / tau_max:.1f}, "
              f"N_eff: {n_eff:.0f}, "
              f"acceptance: {np.mean(sampler.acceptance_fraction):.3f}")
    except Exception as e:
        print(f"\n[Convergence w0waCDM {sc_short} skipped]: {e}")

    discard = int(sampler.iteration * 0.3)
    return sampler.get_chain(discard=discard, flat=True), sampler


def main():
    print("=" * 70)
    print("w0waCDM robustness test — three age scenarios")
    print("Pantheon+ + age + DESI Y1 BAO  ;  flat,  free (w0, wa)")
    print("=" * 70)

    global_cache = GlobalCosmoDataCache()

    names  = ["Om", "H0", "MB", "rd", "dt", "w0", "wa"]
    labels = [r"\Omega_m", r"H_0", r"M_B", r"r_d", r"\Delta t", r"w_0", r"w_a"]

    all_mc = []
    posteriors = {}

    for sc in SCENARIOS:
        print(f"\n[Scenario] {sc['label']} (t = {sc['mu']} +/- {sc['err']} Gyr)")
        t0 = time.time()
        like = LikelihoodW0WaVectorizedCPU(sc['mu'], sc['err'], global_cache)
        flat, sampler = run_mcmc(like, sc_short=sc['short'])
        print(f"  MCMC done in {time.time() - t0:.1f}s  (iterations: {sampler.iteration})")

        res = {}
        for i, n in enumerate(names):
            p = np.percentile(flat[:, i], [16, 50, 84])
            res[n] = (p[1], p[2] - p[1], p[1] - p[0])
        posteriors[sc['short']] = res

        mc = MCSamples(samples=flat, names=names, labels=labels,
                       label=sc['label'], settings={'smooth_scale_1D': 0.7, 'smooth_scale_2D': 0.7})
        all_mc.append(mc)

        # =============== [保存 MCMC 链] ===============
        safe_short = sc['short']
        np.savez_compressed(
            f"chain_w0wa_{safe_short}.npz",
            chain=flat,
            names=np.array(names),
            labels=np.array(labels),
            age_obs=sc['mu'], age_err=sc['err'],
            label=sc['label'], color=sc['color']
        )
        print(f"  ✅ 已保存 MCMC 链至: chain_w0wa_{safe_short}.npz")

    # ================= 汇总数据对比 =================
    # NOTE: The LCDM baseline numbers below are read from the
    # global constants at the top of this file. Update those
    # constants whenever 03_with_BAO2f.py is re-run.
    gc = posteriors['GC']
    H0_lcdm     = H0_LCDM_GC_BASELINE
    H0_err_lcdm = H0_LCDM_GC_BASELINE_ERR
    rd_lcdm     = RD_LCDM_GC_BASELINE
    rd_err_lcdm = RD_LCDM_GC_BASELINE_ERR

    H0_w0wa = gc['H0'][0]
    H0_err_w0wa = max(gc['H0'][1], gc['H0'][2])
    rd_w0wa = gc['rd'][0]
    rd_err_w0wa = max(gc['rd'][1], gc['rd'][2])

    dH0 = abs(H0_w0wa - H0_lcdm)
    dH0_sig = dH0 / np.sqrt(H0_err_w0wa**2 + H0_err_lcdm**2)
    drd_BBN_sig = abs(rd_w0wa - BBN_RD[0]) / np.sqrt(rd_err_w0wa**2 + BBN_RD[1]**2)

    summary_lines = [
        "=" * 70, "EDFig3 — w0waCDM robustness summary (baseline GC scenario)", "=" * 70,
        f"H0 (w0waCDM, GC)   = {H0_w0wa:.2f} +- {H0_err_w0wa:.2f}",
        f"H0 (LambdaCDM, GC) = {H0_lcdm:.2f} +- {H0_err_lcdm:.2f}  [reference]",
        f"  |delta H0|       = {dH0:.2f}  ({dH0_sig:.2f} sigma)",
        f"rd (w0waCDM, GC)   = {rd_w0wa:.2f} +- {rd_err_w0wa:.2f}",
        f"  rd vs BBN        = {drd_BBN_sig:.2f} sigma",
        f"w0 (GC)            = {gc['w0'][0]:.3f} (+{gc['w0'][1]:.3f}/-{gc['w0'][2]:.3f})",
        f"wa (GC)            = {gc['wa'][0]:.3f} (+{gc['wa'][1]:.3f}/-{gc['wa'][2]:.3f})",
        f"Om (GC)            = {gc['Om'][0]:.4f} (+{gc['Om'][1]:.4f}/-{gc['Om'][2]:.4f})",
        f"MB (GC)            = {gc['MB'][0]:.4f} (+{gc['MB'][1]:.4f}/-{gc['MB'][2]:.4f})",
        "", "--- EDFig3 caption placeholder values ---",
        f"  age-anchored H0 = {H0_w0wa:.2f} +/- {H0_err_w0wa:.2f}",
        f"  H0 vs LambdaCDM baseline ({H0_lcdm:.2f} +- {H0_err_lcdm:.2f}) at {dH0_sig:.1f} sigma",
        f"  rd = {rd_w0wa:.1f} +/- {rd_err_w0wa:.1f} Mpc",
        f"  rd vs BBN at {drd_BBN_sig:.2f} sigma",
        f"  (w0, wa) ~= ({gc['w0'][0]:.2f}, {gc['wa'][0]:.2f})",
        "", "--- Section 'Compatibility with JWST/DESI' placeholder ---",
        f"  |Delta H0| between LambdaCDM and best-fit w0waCDM = {dH0:.2f} km/s/Mpc",
        "=" * 70
    ]
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open("EDFig3_w0wa_summary.txt", "w") as f:
        f.write(summary_text + "\n")

    # =====================================================
    # 融合 10_pretty_replot.py 的美化画图 (附加参考带)
    # =====================================================
    print("\n[Plotting] Building pretty corner plot for EDFig3...")
    plot_params = ["Om", "w0", "wa", "H0", "MB", "rd"]
    g = plots.get_subplot_plotter(width_inch=10)
    g.settings.axes_fontsize   = 11
    g.settings.lab_fontsize    = 13
    g.settings.legend_fontsize = 11
    g.settings.figure_legend_frame = False

    g.triangle_plot(all_mc, params=plot_params, filled=True,
                    contour_colors=[s['color'] for s in SCENARIOS],
                    legend_loc='upper right')

    # [高级功能]: 添加 Planck 和 SH0ES 的参考阴影带到 H0 维度
    PLANCK_COLOR = '#0066cc'
    SH0ES_COLOR  = '#cc0000'
    try:
        axH0 = g.subplots[3, 3]  # H0 是 params 里的索引 3
        if axH0 is not None:
            axH0.axvspan(PLANCK_H0[0] - PLANCK_H0[1], PLANCK_H0[0] + PLANCK_H0[1],
                         color=PLANCK_COLOR, alpha=0.18, zorder=0)
            axH0.axvspan(SH0ES_H0[0] - SH0ES_H0[1], SH0ES_H0[0] + SH0ES_H0[1],
                         color=SH0ES_COLOR, alpha=0.18, zorder=0)
    except Exception as e:
        print("⚠️ 无法添加参考阴影带:", e)

    g.export("EDFig3_w0wa.pdf")
    plt.savefig("EDFig3_w0wa.png", dpi=200, bbox_inches='tight')
    print("✅ 已保存完美排版的 EDFig3_w0wa.pdf / .png")


if __name__ == "__main__":
    main()
