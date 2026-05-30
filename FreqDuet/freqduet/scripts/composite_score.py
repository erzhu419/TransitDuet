#!/usr/bin/env python3
"""
composite_score.py
==================
Compute the composite cost used by TransitDuet (the training
objective), aggregated over the last K training episodes × 3 seeds.

  composite_per_episode = wait_min/10 + fleet_overshoot^2 / N_fleet + headway_cv

The reported per-seed composite is the *per-episode mean*:

  seed_composite = mean_t( wait_t/10 + overshoot_t^2 / N_fleet_t + cv_t )

NOT mean(wait)/10 + (mean(overshoot))^2 / mean(N_fleet) + mean(cv);
the two differ whenever overshoot varies across episodes (Jensen's
inequality), so this script computes per-episode first and aggregates
afterwards. Per-seed values are then mean/std-aggregated across seeds.

Usage:
    python scripts/composite_score.py               # read ./logs_remote
    python scripts/composite_score.py --logs logs/  # custom dir
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

EVAL_EP_MARKER = 9000


def composite_row(df_tail: pd.DataFrame) -> dict:
    """
    Compute per-episode composite, then average across episodes.
    The per-episode formula must match runner_v3's compute_system_reward:
      composite_t = wait_t/10 + overshoot_t^2 / N_fleet_t + cv_t.
    """
    wait_per_ep = df_tail['avg_wait_min'].astype(float)
    cv_per_ep = df_tail['headway_cv'].astype(float)
    if 'fleet_overshoot' in df_tail.columns:
        overshoot_per_ep = df_tail['fleet_overshoot'].astype(float)
    else:
        overshoot_per_ep = pd.Series(0.0, index=df_tail.index)
    if 'N_fleet' in df_tail.columns:
        n_fleet_per_ep = df_tail['N_fleet'].astype(float).clip(lower=1.0)
    else:
        n_fleet_per_ep = pd.Series(12.0, index=df_tail.index)

    fleet_pen_per_ep = (overshoot_per_ep ** 2) / n_fleet_per_ep
    composite_per_ep = wait_per_ep / 10.0 + fleet_pen_per_ep + cv_per_ep
    return dict(
        wait=wait_per_ep.mean(),
        overshoot=overshoot_per_ep.mean(),
        cv=cv_per_ep.mean(),
        n_fleet=n_fleet_per_ep.mean(),
        fleet_pen=fleet_pen_per_ep.mean(),
        composite=composite_per_ep.mean(),
    )


def aggregate(logs_dir: Path, last_k: int = 30):
    ablations = ['H_hiro', 'H_hiro_no_holdfb', 'H_hiro_no_csbapr',
                 'H_hiro_no_hindsight', 'H_hiro_no_morl',
                 'H_hiro_fixed_fleet', 'H_hiro_no_demand_noise',
                 'H_hiro_no_tpc']
    rows = []
    for ab in ablations:
        per_seed = []
        for d in sorted(logs_dir.glob(f'{ab}_seed*')):
            csv = d / 'diagnostics.csv'
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            if 'ep' in df.columns:
                df = df[df['ep'] < EVAL_EP_MARKER]
            if len(df) < last_k:
                continue
            tail = df.iloc[-last_k:]
            per_seed.append(composite_row(tail))
        if not per_seed:
            continue
        agg = {'ablation': ab, 'n_seeds': len(per_seed)}
        for k in ['wait', 'overshoot', 'cv', 'fleet_pen', 'composite']:
            vals = [r[k] for r in per_seed]
            agg[f'{k}_mean'] = np.mean(vals)
            agg[f'{k}_std'] = np.std(vals)
        rows.append(agg)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame):
    print("=" * 86)
    print(f"{'Config':26s}  {'wait(m)':>12s}  {'overshoot':>12s}  {'cv':>12s}  "
          f"{'composite':>12s}")
    print("-" * 86)
    for _, r in df.iterrows():
        print(f"{r['ablation']:26s}  "
              f"{r['wait_mean']:5.2f}±{r['wait_std']:4.2f}  "
              f"{r['overshoot_mean']:5.2f}±{r['overshoot_std']:4.2f}  "
              f"{r['cv_mean']:5.3f}±{r['cv_std']:5.3f}  "
              f"{r['composite_mean']:5.3f}±{r['composite_std']:5.3f}")
    print("=" * 86)
    base = df[df.ablation == 'H_hiro']
    if len(base):
        c0 = base['composite_mean'].iloc[0]
        print(f"\nRelative to H_hiro ({c0:.3f}):")
        for _, r in df.iterrows():
            delta = (r['composite_mean'] - c0) / c0 * 100
            sign = '+' if delta >= 0 else ''
            mark = '  (worse)' if delta > 2 else ('  (better)' if delta < -2 else '  (tie)')
            print(f"  {r['ablation']:26s}  {sign}{delta:5.1f}%{mark}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', default='logs_remote',
                    help='directory containing {ablation}_seed{N}/ run dirs')
    ap.add_argument('--last-k', type=int, default=30,
                    help='use last K training episodes per seed (default 30)')
    ap.add_argument('--out', default='results_remote/composite.csv')
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    df = aggregate(logs_dir, last_k=args.last_k)
    if df.empty:
        print(f"No data under {logs_dir}")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out, index=False)
    print(f"Wrote {out}\n")
    print_table(df)


if __name__ == '__main__':
    main()
