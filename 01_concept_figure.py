# -*- coding: utf-8 -*-
"""
Figure 1: Conceptual illustration of how the cosmic age prior breaks
the H0-Omega_m degeneracy in flat LambdaCDM.

This figure is purely analytic - it does NOT use SN data. It shows
contours of constant cosmic age t0(H0, Om) overlaid with Planck/SH0ES bands.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.integrate import quad

def age_gyr(H0, Om):
    """Cosmic age in Gyr for flat LambdaCDM."""
    integ, _ = quad(lambda z: 1/((1+z)*np.sqrt(Om*(1+z)**3 + (1-Om))),
                     0, np.inf)
    return (977.8 / H0) * integ

def run_plot():
    Om_grid = np.linspace(0.2, 0.5, 120)
    H0_grid = np.linspace(55, 80, 120)
    X, Y = np.meshgrid(Om_grid, H0_grid)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = age_gyr(Y[i, j], X[i, j])

    fig, ax = plt.subplots(figsize=(8, 6))

    # Reference bands
    ax.axhspan(66.9, 67.9, color='#0066cc', alpha=0.2, zorder=1)
    ax.axhspan(72.0, 74.0, color='#cc0000', alpha=0.2, zorder=1)

    # Age contours
    levels = [12.6, 13.5, 14.2]
    colors = ['#cc0000', '#0066cc', 'gray']
    styles = ['--', '-', ':']
    labels = [
        r'SH0ES Implied ($12.6$ Gyr)',
        r'Globular Clusters ($13.5$ Gyr)',
        r'HD 140283 ($14.2$ Gyr)'
    ]

    CS = ax.contour(X, Y, Z, levels=levels, colors=colors,
                    linestyles=styles, linewidths=2.5)

    fmt = {l: rf'$\mathbf{{{l}\ Gyr}}$' for l in levels}
    ax.clabel(CS, inline=True, fontsize=11, fmt=fmt,
              manual=[(0.35, 73), (0.35, 67), (0.35, 63)])

    ax.text(0.42, 73.5, r"$\leftarrow$ Requires Young Age",
            fontsize=9, color='#cc0000', va='center')
    ax.text(0.42, 67.4, r"$\leftarrow$ Consistent w/ GC Age",
            fontsize=9, color='#0066cc', va='center')

    ax.set_xlabel(r'Matter Density $\Omega_m$', fontsize=14)
    ax.set_ylabel(r'Hubble Constant $H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=14)
    ax.set_xlim(0.25, 0.45)
    ax.set_ylim(60, 78)

    proxies = [Line2D([0], [0], color=c, ls=s, lw=2.5)
               for c, s in zip(colors, styles)]
    p_patch = mpatches.Patch(color='#0066cc', alpha=0.2, label='Planck 2018')
    s_patch = mpatches.Patch(color='#cc0000', alpha=0.2, label='SH0ES 2022')

    ax.legend(proxies + [p_patch, s_patch],
              labels + ['Planck 2018', 'SH0ES 2022'],
              loc='lower left', fontsize=10)

    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('Figure1_Concept.pdf', bbox_inches='tight')
    plt.savefig('Figure1_Concept.png', dpi=200, bbox_inches='tight')
    print("Saved Figure1_Concept.pdf / .png")

if __name__ == "__main__":
    run_plot()