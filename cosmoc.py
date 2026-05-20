# -*- coding: utf-8 -*-
"""
@author: Jian Hu
Email: dg1626002@smail.nju.edu.cn
Shared physics / data-loading utilities.

All analysis scripts import from this module to avoid code duplication
and to guarantee identical SN likelihood treatment across all figures.

KEY DESIGN: Pantheon+ loading applies the standard Hubble-flow cuts
  - exclude IS_CALIBRATOR == 1 (77 SH0ES Cepheid hosts)
  - exclude zHD < 0.01 (peculiar velocity contamination)
  - use m_b_corr (Tripp-corrected magnitude), NOT raw mB
"""

import os
import numpy as np
import pandas as pd
from numba import jit

c_light = 299792.458  # km/s

# ================= Cosmology integrals =================
@jit(nopython=True)
def Ez_inverse(z, Om, Ok, Ol):
    return 1.0 / np.sqrt(Om * (1+z)**3 + Ok * (1+z)**2 + Ol)

@jit(nopython=True)
def compute_distance_vector(z_array, Om, Ok, H0):
    """Transverse comoving distance D_M [Mpc] for an array of redshifts."""
    Ol = 1.0 - Om - Ok
    dh = c_light / H0
    n = len(z_array)
    dm_list = np.empty(n)
    n_steps = 100
    for i in range(n):
        z = z_array[i]
        dz = z / n_steps
        total = 0.5 * (Ez_inverse(0.0, Om, Ok, Ol) + Ez_inverse(z, Om, Ok, Ol))
        for j in range(1, n_steps):
            total += Ez_inverse(j*dz, Om, Ok, Ol)
        dc = dh * total * dz
        if np.abs(Ok) < 1e-5:
            dm_list[i] = dc
        elif Ok > 0:
            dm_list[i] = (dh / np.sqrt(Ok)) * np.sinh(np.sqrt(Ok) * dc / dh)
        else:
            dm_list[i] = (dh / np.sqrt(-Ok)) * np.sin(np.sqrt(-Ok) * dc / dh)
    return dm_list

@jit(nopython=True)
def compute_age_gyr(H0, Om, Ok):
    """Cosmic age t0 in Gyr."""
    Ol = 1.0 - Om - Ok
    n_steps = 200
    da = 1.0 / n_steps
    total = 0.0
    for i in range(n_steps):
        a = (i + 0.5) * da
        if a > 0:
            total += 1.0 / np.sqrt(Om/a + Ok + Ol*a*a)
    return (977.8 / H0) * total * da

# ================= Data loaders =================
def load_pantheon_plus(data_path="Pantheon+SH0ES.dat",
                       cov_path="Pantheon+SH0ES_STAT+SYS.cov",
                       verbose=True):
    """
    Load Pantheon+ with standard Hubble-flow cuts.

    Returns
    -------
    z_sn : (N,) array of zHD
    mb_sn : (N,) array of m_b_corr (Tripp-corrected magnitudes)
    inv_cov : (N, N) inverse covariance matrix
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)
    if not os.path.exists(cov_path):
        raise FileNotFoundError(cov_path)

    pan = pd.read_csv(data_path, sep=r'\s+')
    N_full = len(pan)

    # Standard Pantheon+ Hubble-flow selection
    mask = (pan['IS_CALIBRATOR'] == 0) & (pan['zHD'] > 0.01)
    indices = np.where(mask)[0]

    pan_use = pan.iloc[indices].reset_index(drop=True)
    z_sn = pan_use['zHD'].values
    mb_sn = pan_use['m_b_corr'].values  # Tripp-corrected magnitudes

    cov_raw = np.loadtxt(cov_path, skiprows=1)
    N_check = int(round(np.sqrt(len(cov_raw))))
    cov_full = cov_raw.reshape(N_check, N_check)
    cov_sub = cov_full[np.ix_(indices, indices)]
    inv_cov = np.linalg.inv(cov_sub)

    if verbose:
        print(f"  Pantheon+ loaded: {len(z_sn)} SNe "
              f"(removed {N_full - len(z_sn)}: "
              f"{(pan['IS_CALIBRATOR']==1).sum()} calibrators + "
              f"{((pan['zHD']<0.01)&(pan['IS_CALIBRATOR']==0)).sum()} low-z)")

    return z_sn, mb_sn, inv_cov

def load_bao(desi_data="desi_gaussian_bao_ALL_GCcomb_mean.txt",
             desi_cov="desi_gaussian_bao_ALL_GCcomb_cov.txt",
             verbose=True):
    """
    Load DESI Y1 BAO. Returns a list of dicts (extensible to SDSS DR12 too).
    """
    bao_list = []
    if os.path.exists(desi_data) and os.path.exists(desi_cov):
        df = pd.read_csv(desi_data, sep=r'\s+', comment='#',
                         names=['z', 'val', 'type'])
        cov = np.loadtxt(desi_cov)
        bao_list.append({
            'z': df['z'].values,
            'y': df['val'].values,
            'type': df['type'].values,
            'icov': np.linalg.inv(cov),
            'label': 'DESI Y1'
        })
        if verbose:
            print(f"  DESI BAO loaded: {len(df)} points")
    else:
        if verbose:
            print("  WARNING: DESI BAO files not found, skipping")
    return bao_list

def bao_model(z_arr, type_arr, Om, H0, rd):
    """
    Compute BAO observables for given parameters.
    type can be 'DM_over_rs', 'DH_over_rs', or 'DV_over_rs'.
    """
    dm = compute_distance_vector(z_arr, Om, 0.0, H0)
    Ez = np.sqrt(Om * (1 + z_arr)**3 + (1 - Om))
    dh = c_light / (H0 * Ez)
    dv = (z_arr * dh * dm**2)**(1.0/3.0)
    out = np.where(type_arr == 'DM_over_rs', dm/rd,
          np.where(type_arr == 'DH_over_rs', dh/rd,
                                              dv/rd))
    return out

# ================= Scenarios =================
SCENARIOS = [
    {'mu': 14.2, 'err': 0.40, 'label': 'HD 140283',         'color': 'gray',    'short': 'HD140283'},
    {'mu': 13.5, 'err': 0.27, 'label': 'Globular Clusters', 'color': '#0066cc', 'short': 'GC'},
    {'mu': 12.6, 'err': 0.27, 'label': 'SH0ES Implied',     'color': '#cc0000', 'short': 'SH0ES'},
]

# Reference values
PLANCK_H0 = (67.4, 0.5)
SH0ES_H0 = (73.04, 1.04)
SH0ES_MB = (-19.253, 0.027)
PLANCK_RD = (147.0, 0.3)
BBN_RD = (146.8, 0.8)