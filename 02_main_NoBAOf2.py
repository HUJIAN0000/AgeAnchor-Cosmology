# -*- coding: utf-8 -*-
"""
Figure 2: CORE RESULT — Pantheon+ + cosmic age (NO BAO, NO CMB, NO Cepheid).

Demonstrates that the age prior alone, combined with SN Ia luminosity
distances, breaks the M_B-H_0 degeneracy and yields independent constraints
on (Omega_m, H_0, M_B).

OPTIMIZED VERSION (follows the 07_w0wa_testf2.py pattern):
  - Numba JIT-compiled 1D Om grid precomputation (flat LCDM: w0=-1, wa=0)
  - SciPy RegularGridInterpolator -> O(1) lookup of SN distances + age
  - emcee with vectorize=True: batched likelihood across all walkers
  - Global cache shared across the three age scenarios (1x precompute)

Because LCDM only has one cosmological d.o.f. (Om), the grid is 1D instead
of 3D, so precomputation is ~1 second and the MCMC stage is essentially
pure NumPy matrix algebra.

Output: Figure2_Main_NoBAO.pdf
"""
import time
import numpy as np
import emcee
import matplotlib.pyplot as plt
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from getdist import plots, MCSamples

from cosmoc import (
    c_light, load_pantheon_plus,
    SCENARIOS, PLANCK_H0, SH0ES_H0, SH0ES_MB,
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
        pts_v = Om_v[:, None]  # shape (Nv, 1) for the 1-D interpolator

        # ---- SN Ia chi^2 (vectorized over walkers) ----
        d_sn_v = self.cache.interp_sn(pts_v)                        # (Nv, N_sn)
        D_M_sn = d_sn_v * (c_light / H0_v[:, None])                 # Mpc
        mu_model = 5.0 * np.log10(
            D_M_sn * (1.0 + self.cache.z_sn[None, :]) + 1e-10
        ) + 25.0
        diff_sn = self.cache.mb_sn[None, :] - (mu_model + MB_v[:, None])
        # quadratic form: sum over data of (diff @ icov @ diff)
        temp = diff_sn @ self.cache.inv_cov_sn
        logL_v = -0.5 * np.sum(diff_sn * temp, axis=1)

        # ---- Age prior + dt prior ----
        age_th = (AGE_CONVERSION_FACTOR / H0_v) * self.cache.interp_age(pts_v).ravel()
        logL_v += -0.5 * ((dt_v - self.dt_prior_mu) / self.dt_prior_sig) ** 2
        logL_v += -0.5 * ((age_th - (self.age_obs + dt_v)) / self.age_err) ** 2

        logL[mask] = logL_v
        return logL


# ====================================================================
# 3) MCMC driver
# ====================================================================

def run_mcmc(like, ndim=4, nwalkers=32, max_iter=6000):
    p0_center = np.array([0.31, 67.4, -19.4, 0.15])
    scatter   = np.array([1e-2, 0.3,  1e-2,  1e-3])
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
        print(f"\n[Convergence NoBAO] tau_max: {tau_max:.1f}, "
              f"Chain/tau: {chain_len / tau_max:.1f}, N_eff: {n_eff:.0f}")
    except Exception as e:
        print(f"\n[Convergence NoBAO skipped]: {e}")

    discard = int(sampler.iteration * 0.3)
    return sampler.get_chain(discard=discard, flat=True), sampler


def compute_tension(val1, err1, val2, err2):
    """Absolute tension in sigma."""
    return abs(val1 - val2) / np.sqrt(err1 ** 2 + err2 ** 2)


# ====================================================================
# 4) Main
# ====================================================================

def main():
    print("=" * 70)
    print("Figure 2 (NoBAO) — Pantheon+ + cosmic age, flat LCDM")
    print("=" * 70)

    # ONE-TIME precompute, shared across all scenarios.
    global_cache = GlobalCosmoDataCache()

    all_mc = []
    summary = {}
    names = ["Om", "H0", "MB", "dt"]
    labels = [r"\Omega_m", r"H_0", r"M_B", r"\Delta t"]

    for sc in SCENARIOS:
        print(f"\n=== {sc['label']} (t = {sc['mu']} +/- {sc['err']} Gyr) ===")
        t0 = time.time()
        like = LikelihoodNoBAOVectorized(sc['mu'], sc['err'], global_cache)
        flat, sampler = run_mcmc(like)
        print(f"  MCMC done in {time.time() - t0:.1f}s  (iters: {sampler.iteration})")

        print("--- Posterior ---")
        res = {}
        for i, n in enumerate(names):
            p = np.percentile(flat[:, i], [16, 50, 84])
            print(f"  {n}: {p[1]:.4f}  +{p[2]-p[1]:.4f} / -{p[1]-p[0]:.4f}")
            res[n] = (p[1], p[2] - p[1], p[1] - p[0])
        summary[sc['label']] = res

        mc = MCSamples(
            samples=flat, names=names, labels=labels,
            label=sc['label'],
            settings={'smooth_scale_1D': 0.7, 'smooth_scale_2D': 0.7},
        )
        all_mc.append(mc)

        # Save per-scenario chain (07_w0wa style)
        np.savez_compressed(
            f"chain_NoBAO_{sc['short']}.npz",
            chain=flat,
            names=np.array(names),
            labels=np.array(labels),
            age_obs=sc['mu'], age_err=sc['err'],
            label=sc['label'], color=sc['color'],
        )
        print(f"  ✅ Saved: chain_NoBAO_{sc['short']}.npz")

    # ===== Tension analysis (Globular Cluster scenario) =====
    gc = summary['Globular Clusters']
    print("\n" + "=" * 60)
    print("TENSION ANALYSIS — Globular Cluster scenario (core result)")
    print("=" * 60)
    h0_err = max(gc['H0'][1], gc['H0'][2])
    mb_err = max(gc['MB'][1], gc['MB'][2])

    t_planck_h0 = compute_tension(gc['H0'][0], h0_err, *PLANCK_H0)
    t_sh0es_h0  = compute_tension(gc['H0'][0], h0_err, *SH0ES_H0)
    t_sh0es_mb  = compute_tension(gc['MB'][0], mb_err, *SH0ES_MB)
    print(f"  H_0 (this work) = {gc['H0'][0]:.2f} +/- {h0_err:.2f}")
    print(f"    vs Planck:  {t_planck_h0:.2f} sigma")
    print(f"    vs SH0ES:   {t_sh0es_h0:.2f} sigma")
    print(f"  M_B (this work) = {gc['MB'][0]:.3f} +/- {mb_err:.3f}")
    print(f"    vs SH0ES M_B = {SH0ES_MB[0]:.3f}: {t_sh0es_mb:.2f} sigma")

    # ===== Triangle plot =====
    print("\nGenerating triangle plot...")
    g = plots.get_subplot_plotter(width_inch=7)
    g.triangle_plot(
        all_mc, params=["Om", "H0", "MB"], filled=True,
        contour_colors=[s['color'] for s in SCENARIOS],
        legend_loc='upper right',
    )
    plt.savefig("Figure2_Main_NoBAO.pdf", bbox_inches='tight')
    plt.savefig("Figure2_Main_NoBAO.png", dpi=200, bbox_inches='tight')
    print("✅ Saved Figure2_Main_NoBAO.pdf / .png")


if __name__ == "__main__":
    main()
