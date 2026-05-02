#!/usr/bin/env python3
"""
make_result_figures.py
======================
Generate paper result figures from local aggregated data:
  fig:training_curves    — wait time over episodes, A_full vs baselines
  fig:ablation_bars      — composite cost per ablation with error bars
  fig:pareto_frontier    — wait vs N_fleet (Pareto curve, one policy)
  fig:generalization     — wait vs route_sigma (OOD evaluation)

Writes PDFs to paper/figures/.
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'logs_remote'
EVAL = ROOT / 'logs' / 'eval_generalization'
OUT = ROOT.parent / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456]

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

METHOD_COLOR = {
    'A_full': '#1f77b4',
    'fixed': '#2ca02c',
    'ga': '#ff7f0e',
    'cmaes': '#d62728',
}


def smooth(s, w=10):
    return s.rolling(window=w, min_periods=1).mean()


# ═══════════════════════════════════════════════════════════════
# fig:training_curves
# All methods' actual training trajectories. TransitDuet (H_hiro)
# trains for 300 episodes; the search/static baselines train for
# 120 episodes (their convergence rate does not require longer).
# A vertical dashed line at episode 120 marks where the baselines
# stop training; the TransitDuet curve continues past it.
# ═══════════════════════════════════════════════════════════════
def fig_training_curves():
    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    # TransitDuet (H_hiro: HIRO-style goal-conditioned coupling)
    dfs = []
    for s in SEEDS:
        df = pd.read_csv(LOGS / f'H_hiro_seed{s}' / 'diagnostics.csv')
        df = df[df['ep'] < 9000].reset_index(drop=True)
        dfs.append(smooth(df['avg_wait_min'], 15).values[:300])
    stack = np.stack(dfs)
    eps = np.arange(stack.shape[1])
    m = stack.mean(axis=0); sd = stack.std(axis=0)
    ax.plot(eps, m, color=METHOD_COLOR['A_full'], linewidth=1.8,
            label='TransitDuet (HIRO, Ours)')
    ax.fill_between(eps, m - sd, m + sd, color=METHOD_COLOR['A_full'], alpha=0.2)

    x_max = stack.shape[1]

    # Baselines: actual training curves from history.json (~120 eps each)
    for m_name, display in [('fixed', 'Fixed ($H=360$ s)'),
                            ('ga', 'GA'),
                            ('cmaes', 'CMA-ES')]:
        series = []
        for s in SEEDS:
            h = json.load(open(LOGS / f'upper_{m_name}_seed{s}' / 'history.json'))
            series.append(pd.Series(h['avg_wait']).rolling(15, min_periods=1).mean().values)
        min_len = min(len(x) for x in series)
        stack_b = np.stack([x[:min_len] for x in series])
        eps_b = np.arange(min_len)
        mm = stack_b.mean(axis=0); ss = stack_b.std(axis=0)
        ax.plot(eps_b, mm, color=METHOD_COLOR[m_name],
                linewidth=1.3, linestyle='--', label=display)
        ax.fill_between(eps_b, mm - ss, mm + ss,
                        color=METHOD_COLOR[m_name], alpha=0.12)

    ax.set_xlabel('Training episode')
    ax.set_ylabel('Average wait (min)')
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, max(25, ax.get_ylim()[1]))
    ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / 'training_curves.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


# ═══════════════════════════════════════════════════════════════
# fig:ablation_bars
# ═══════════════════════════════════════════════════════════════
def fig_ablation_bars():
    # HIRO-mode ablations, sorted by ΔComposite (best to worst); reads from
    # the per-ckpt eval CSVs (best-ckpt-per-seed aggregation).
    ablations = [
        ('H_hiro',                'Full'),
        ('H_hiro_no_morl',        r'$-$MORL'),
        ('H_hiro_no_tpc',         r'$-$TPC'),
        ('H_hiro_no_demand_noise',r'$-$DemNoise'),
        ('H_hiro_no_csbapr',      r'$-$CS-BAPR'),
        ('H_hiro_no_holdfb',      r'$-$HoldFB'),
        ('H_hiro_no_hindsight',   r'$-$Hindsight'),
        ('H_hiro_fixed_fleet',    r'$-$Elastic'),
    ]
    EVAL_ROOT = LOGS.parent / 'logs_remote' / 'eval_per_ckpt'

    means = []
    stds = []
    labels = []
    for key, disp in ablations:
        csv_path = EVAL_ROOT / key / f'{key}_per_ckpt.csv'
        if not csv_path.exists():
            continue
        rows = list(pd.read_csv(csv_path).itertuples())
        seeds = sorted(set(int(r.seed) for r in rows))
        bests = []
        for s in seeds:
            seed_rows = [r for r in rows if int(r.seed) == s]
            seed_rows.sort(key=lambda r: r.composite_mean)
            bests.append(seed_rows[0].composite_mean)
        means.append(np.mean(bests))
        stds.append(np.std(bests))
        labels.append(disp)

    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    x = np.arange(len(labels))
    colors = ['#1f77b4'] + ['#888888'] * (len(labels) - 1)
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(means[0], color='#1f77b4', linestyle=':', linewidth=0.8, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Composite cost')
    ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.15)
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    path = OUT / 'ablation_bars.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


# ═══════════════════════════════════════════════════════════════
# fig:pareto_frontier
# ═══════════════════════════════════════════════════════════════
def fig_pareto_frontier():
    # Load per-seed pareto_frontier.json and stack
    all_pts = {}  # N_fleet -> list of (wait, cv, overshoot)
    for s in SEEDS:
        p = json.load(open(LOGS / f'A_full_seed{s}' / 'pareto_frontier.json'))
        for row in p:
            N = int(row['N_fleet'])
            all_pts.setdefault(N, []).append((row['wait_mean'], row['cv_mean'],
                                              row.get('overshoot_mean', 0)))

    Ns = sorted(all_pts.keys())
    wait_m = np.array([np.mean([p[0] for p in all_pts[N]]) for N in Ns])
    wait_s = np.array([np.std([p[0] for p in all_pts[N]]) for N in Ns])
    over_m = np.array([np.mean([p[2] for p in all_pts[N]]) for N in Ns])

    fig, ax1 = plt.subplots(figsize=(4.8, 2.8))
    ax1.errorbar(Ns, wait_m, yerr=wait_s, marker='o', color='#1f77b4',
                 linewidth=1.5, capsize=3, label='Wait (min)')
    ax1.set_xlabel(r'Fleet budget $N_{\mathrm{fleet}}$')
    ax1.set_ylabel('Wait (min)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(Ns, over_m, alpha=0.2, color='#d62728', width=0.7, label='Overshoot')
    ax2.set_ylabel('Overshoot', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.spines['top'].set_visible(False)

    # Highlight knee at N=15
    knee_i = Ns.index(15) if 15 in Ns else None
    if knee_i is not None:
        ax1.annotate(f'knee\n({wait_m[knee_i]:.1f} min)',
                     xy=(15, wait_m[knee_i]),
                     xytext=(14, wait_m[knee_i] + 4),
                     fontsize=8,
                     arrowprops=dict(arrowstyle='->', lw=0.7, color='black'))
    fig.tight_layout()
    path = OUT / 'pareto_frontier.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


# ═══════════════════════════════════════════════════════════════
# fig:generalization
# ═══════════════════════════════════════════════════════════════
def fig_generalization():
    cross_dir = EVAL / 'cross_sigma'
    by_sigma = {}
    for f in cross_dir.glob('sigma_*_seed*.json'):
        r = json.loads(f.read_text())
        by_sigma.setdefault(r['sigma'], []).append(r)

    sigmas = sorted(by_sigma.keys())
    wait_m = [np.mean([r['wait_mean'] for r in by_sigma[s]]) for s in sigmas]
    wait_s = [np.std([r['wait_mean'] for r in by_sigma[s]]) for s in sigmas]
    cv_m = [np.mean([r['cv_mean'] for r in by_sigma[s]]) for s in sigmas]
    cv_s = [np.std([r['cv_mean'] for r in by_sigma[s]]) for s in sigmas]

    fig, ax1 = plt.subplots(figsize=(4.8, 2.8))
    ax1.errorbar(sigmas, wait_m, yerr=wait_s, marker='o', color='#1f77b4',
                 linewidth=1.5, capsize=3, label='Wait (min)')
    ax1.axvline(1.5, linestyle='--', color='gray', linewidth=0.8)
    ax1.annotate('train $\\sigma$', xy=(1.5, 4.5), fontsize=8, color='gray')
    ax1.set_xlabel(r'Route stochasticity $\sigma$')
    ax1.set_ylabel('Wait (min)', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.errorbar(sigmas, cv_m, yerr=cv_s, marker='s', color='#d62728',
                 linewidth=1.2, capsize=3, linestyle='--', label='CV')
    ax2.set_ylabel('Headway CV', color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.spines['top'].set_visible(False)

    fig.tight_layout()
    path = OUT / 'generalization.pdf'
    fig.savefig(path)
    print(f"  wrote {path}")


def main():
    fig_training_curves()
    fig_ablation_bars()
    fig_pareto_frontier()
    fig_generalization()


if __name__ == '__main__':
    main()
