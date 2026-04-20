#!/usr/bin/env python3
"""
aggregate.py
============
Aggregate results from all experiment runs into summary tables.

Outputs:
  results/main_table.csv   — Main comparison (methods × metrics, mean±std over seeds)
  results/ablation.csv     — Ablation table (A-G)
  results/pareto.csv       — Pareto frontier across seeds
  results/summary.txt      — Human-readable summary
"""

import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / 'logs'
RESULTS_DIR = ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def load_diag(path):
    """Load diagnostics.csv from a run directory."""
    p = Path(path) / 'diagnostics.csv'
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def last30_stats(df, metric):
    """Return mean, std over last 30 episodes."""
    if df is None or len(df) == 0:
        return np.nan, np.nan
    last = df.iloc[-30:]
    if metric not in last.columns:
        return np.nan, np.nan
    return float(last[metric].mean()), float(last[metric].std())


def aggregate_ablations():
    """Aggregate A-G ablation runs."""
    ablations = ['A_full', 'B_no_holding_feedback', 'C_no_csbapr',
                 'D_no_hindsight', 'E_no_morl', 'F_fixed_fleet',
                 'G_no_demand_noise']
    metrics = ['avg_wait_min', 'headway_cv', 'peak_fleet',
               'upper_delta_std', 'hold_fb_mean']

    rows = []
    for ab in ablations:
        run_dirs = sorted(LOGS_DIR.glob(f'{ab}_seed*'))
        seed_dfs = [load_diag(d) for d in run_dirs]
        seed_dfs = [d for d in seed_dfs if d is not None]
        if not seed_dfs:
            continue
        row = {'ablation': ab, 'n_seeds': len(seed_dfs)}
        for m in metrics:
            vals = [last30_stats(d, m)[0] for d in seed_dfs]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                row[f'{m}_mean'] = np.mean(vals)
                row[f'{m}_std'] = np.std(vals)
            else:
                row[f'{m}_mean'] = np.nan
                row[f'{m}_std'] = np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / 'ablation.csv'
    df.to_csv(out_path, index=False)
    print(f"  → {out_path}  ({len(df)} rows)")
    return df


def aggregate_baselines():
    """Aggregate baseline upper-level comparison."""
    methods = ['cmaes', 'ga', 'fixed', 'bo', 'contextual_cmaes', 'cmaes_rl']
    rows = []
    for m in methods:
        run_dirs = sorted(LOGS_DIR.glob(f'upper_{m}_seed*'))
        for d in run_dirs:
            hist_path = d / 'history.json'
            if not hist_path.exists():
                continue
            try:
                with open(hist_path) as f:
                    hist = json.load(f)
            except Exception:
                continue
            # Use last 30 eps of relevant metrics
            n = len(hist.get('avg_wait', []))
            if n < 10:
                continue
            tail = slice(max(0, n-30), n)
            row = {
                'method': m,
                'run': d.name,
                'wait_mean': np.mean(hist['avg_wait'][tail]) if 'avg_wait' in hist else np.nan,
                'peak_fleet_mean': np.mean(hist['peak_fleet'][tail]) if 'peak_fleet' in hist else np.nan,
                'headway_cv_mean': np.mean(hist.get('headway_cv', [np.nan])[tail]) if 'headway_cv' in hist else np.nan,
                'sys_reward_mean': np.mean(hist['sys_reward'][tail]) if 'sys_reward' in hist else np.nan,
                'n_eps': n,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    # Aggregate over seeds
    if len(df) > 0:
        agg = df.groupby('method').agg(
            {'wait_mean': ['mean', 'std'],
             'peak_fleet_mean': ['mean', 'std'],
             'headway_cv_mean': ['mean', 'std']}).round(3)
        out_path = RESULTS_DIR / 'baselines.csv'
        agg.to_csv(out_path)
        print(f"  → {out_path}  ({len(df)} runs)")
    return df


def aggregate_pareto():
    """Aggregate Pareto frontier from A_full runs."""
    rows = []
    for d in sorted(LOGS_DIR.glob('A_full_seed*')):
        p_path = d / 'pareto_frontier.json'
        if not p_path.exists():
            continue
        try:
            with open(p_path) as f:
                pareto = json.load(f)
        except Exception:
            continue
        seed = re.search(r'seed(\d+)', d.name)
        seed_id = seed.group(1) if seed else '?'
        for p in pareto:
            p['seed'] = seed_id
            rows.append(p)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # Aggregate over seeds per N_fleet
    agg = df.groupby('N_fleet').agg({
        'wait_mean': ['mean', 'std'],
        'cv_mean': ['mean', 'std'],
        'overshoot_mean': ['mean', 'std'],
    }).round(3)
    agg.columns = ['_'.join(c) for c in agg.columns]
    out_path = RESULTS_DIR / 'pareto.csv'
    agg.to_csv(out_path)
    print(f"  → {out_path}  ({len(df)} points, {df.seed.nunique()} seeds)")
    return agg


def write_summary(abl_df, base_df, pareto_df):
    """Human-readable summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("  TransitDuet Experiment Summary")
    lines.append("=" * 80)

    if len(abl_df) > 0:
        lines.append("\n## Ablation Study (wait_min, mean±std over seeds)")
        for _, r in abl_df.iterrows():
            w_m = r.get('avg_wait_min_mean', np.nan)
            w_s = r.get('avg_wait_min_std', np.nan)
            cv_m = r.get('headway_cv_mean', np.nan)
            n = int(r.get('n_seeds', 0))
            lines.append(f"  {r['ablation']:28s}  wait={w_m:4.1f}±{w_s:.1f}  cv={cv_m:.2f}  (n={n})")

    if len(pareto_df) > 0:
        lines.append("\n## Pareto Frontier (wait_mean ± std)")
        for N, row in pareto_df.iterrows():
            lines.append(f"  N_fleet={int(N):2d}  wait={row['wait_mean_mean']:4.1f}±{row['wait_mean_std']:.1f}  "
                         f"cv={row['cv_mean_mean']:.2f}  overshoot={row['overshoot_mean_mean']:.1f}")

    if len(base_df) > 0:
        lines.append("\n## Baseline Comparison")
        for method in base_df['method'].unique():
            sub = base_df[base_df.method == method]
            if len(sub) > 0:
                lines.append(f"  {method:20s}  wait={sub.wait_mean.mean():4.1f}±{sub.wait_mean.std():.1f}  "
                             f"cv={sub.headway_cv_mean.mean():.2f}  n={len(sub)}")

    txt = '\n'.join(lines)
    (RESULTS_DIR / 'summary.txt').write_text(txt)
    print(txt)


def main():
    print("Aggregating ablations...")
    abl = aggregate_ablations()
    print("Aggregating baselines...")
    base = aggregate_baselines()
    print("Aggregating Pareto frontier...")
    pareto = aggregate_pareto()
    print("\nWriting summary...")
    write_summary(abl, base, pareto)
    print(f"\nResults in: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
