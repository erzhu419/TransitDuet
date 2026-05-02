#!/usr/bin/env python3
"""
make_mechanism_figures.py
=========================
Generate the three mechanism-analysis figures referenced by the paper:
  fig:theta_evolution   — belief-weighted MORL weights over training
  fig:lambda_convergence — lower-level Lagrangian multiplier
  fig:delta_utilization  — per-episode upper action mean ± std

Reads 3-seed A_full diagnostics from logs_remote/ and writes PDFs
to paper/figures/.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs_remote'
OUT = ROOT.parent / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]
SEED_COLORS = {42: '#1f77b4', 123: '#ff7f0e', 456: '#2ca02c'}

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
})


def load_dfs(name='A_full'):
    out = {}
    for s in SEEDS:
        p = LOGS / f'{name}_seed{s}' / 'diagnostics.csv'
        df = pd.read_csv(p)
        df = df[df['ep'] < 9000].reset_index(drop=True)
        out[s] = df
    return out


def smooth(series, window=10):
    return series.rolling(window=window, min_periods=1).mean()


def fig_theta(dfs):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    # Mean over seeds
    eps = dfs[SEEDS[0]]['ep'].values
    for comp, ls, label in [('theta_wait', '-', r'$\theta_{\mathrm{wait}}$'),
                            ('theta_fleet', '--', r'$\theta_{\mathrm{fleet}}$'),
                            ('theta_cv', ':', r'$\theta_{\mathrm{cv}}$')]:
        stack = np.stack([smooth(dfs[s][comp], 10).values for s in SEEDS])
        m = stack.mean(axis=0)
        sd = stack.std(axis=0)
        line, = ax.plot(eps, m, ls, label=label, linewidth=1.5)
        ax.fill_between(eps, m - sd, m + sd, alpha=0.15, color=line.get_color())
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\theta$ weight')
    ax.set_xlim(0, eps[-1])
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', frameon=False, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / 'theta_evolution.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


def fig_lambda(dfs):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    eps = dfs[SEEDS[0]]['ep'].values
    stack = np.stack([smooth(dfs[s]['lower_lambda'], 10).values for s in SEEDS])
    m = stack.mean(axis=0)
    sd = stack.std(axis=0)
    ax.plot(eps, m, color='#d62728', linewidth=1.5, label=r'$\lambda$ (mean $\pm$ std, 3 seeds)')
    ax.fill_between(eps, m - sd, m + sd, alpha=0.2, color='#d62728')
    ax.axhline(0.57, linestyle='--', color='gray', linewidth=0.8,
               label=r'converged $\lambda = 0.57$')
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'Lagrangian $\lambda$')
    ax.set_xlim(0, eps[-1])
    ax.legend(loc='lower right', frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / 'lambda_convergence.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


def fig_delta(dfs):
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for s, df in dfs.items():
        eps = df['ep'].values
        mu = smooth(df['upper_delta_mean'], 5).values
        sd = smooth(df['upper_delta_std'], 5).values
        c = SEED_COLORS[s]
        ax.plot(eps, mu, color=c, linewidth=1.0, label=f'seed {s}')
        ax.fill_between(eps, mu - sd, mu + sd, alpha=0.15, color=c)
    ax.axhline(0, linestyle=':', color='black', linewidth=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel(r'$\delta_t$ (s)')
    ax.set_xlim(0, eps[-1])
    ax.set_ylim(-120, 120)
    ax.legend(loc='lower right', frameon=False, ncol=3)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / 'delta_utilization.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


def main():
    dfs = load_dfs('H_hiro')
    fig_theta(dfs)
    fig_lambda(dfs)
    fig_delta(dfs)


if __name__ == '__main__':
    main()
