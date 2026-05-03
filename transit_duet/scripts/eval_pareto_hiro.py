#!/usr/bin/env python3
"""
eval_pareto_hiro.py
===================
Generate the Pareto frontier (wait vs N_fleet) for the validation-best
H_hiro checkpoint of each seed. Section V-D of the paper claims this
frontier is produced by ``one trained TransitDuet policy''; this script
makes that claim mechanically reproducible by:

  1. Reading the per-ckpt CSV (logs/eval_per_ckpt/<exp>/<exp>_per_ckpt.csv)
     to identify the validation-best checkpoint per seed.
  2. Restoring that checkpoint with runner_v3 (HIRO coupling).
  3. For each N_fleet in [8, 16], running n_eval episodes at the held-out
     evaluation distribution.
  4. Writing logs/<exp>_seed<N>/pareto_frontier.json which
     scripts/make_result_figures.py:fig_pareto_frontier reads.

This replaces the older flow of relying on training-time --eval_pareto
runs that wrote to A_full_seed*; we now (a) anchor on the validation-best
ckpt rather than the training-end ckpt, and (b) read from H_hiro_seed*
rather than the archived A_full_seed*.

Usage:
    python scripts/eval_pareto_hiro.py
    python scripts/eval_pareto_hiro.py --exp H_hiro --seeds 42,123,456 \
        --n_eval 5 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from runner_v3 import TransitDuetV2Runner, load_config


class _NullDiag:
    def append(self, row): pass
    def save_json(self): pass


def best_ckpt_episode(exp: str, seed: int) -> int | None:
    csv = SCRIPT_DIR / 'logs' / 'eval_per_ckpt' / exp / f'{exp}_per_ckpt.csv'
    if not csv.exists():
        return None
    rows = []
    with open(csv) as f:
        for r in csv_mod.DictReader(f):
            if int(r['seed']) == seed:
                rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: float(r['composite_mean']))
    return int(rows[0]['ep'])


def eval_at_fleet(runner, n_fleet: int, n_eps: int, seed: int) -> dict:
    waits, cvs, overs = [], [], []
    for _ in range(n_eps):
        runner.env._n_fleet_target = n_fleet
        row = runner.run_episode(ep=9999, training=False, N_fleet_override=n_fleet)
        waits.append(float(row['avg_wait_min']))
        cvs.append(float(row['headway_cv']))
        overs.append(float(row['fleet_overshoot']))
    return {
        'N_fleet': n_fleet,
        'wait_mean': float(np.mean(waits)),
        'wait_std': float(np.std(waits)),
        'cv_mean': float(np.mean(cvs)),
        'cv_std': float(np.std(cvs)),
        'overshoot_mean': float(np.mean(overs)),
        'overshoot_std': float(np.std(overs)),
        'n_eps': n_eps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='H_hiro')
    ap.add_argument('--config', default=None,
                    help='default: configs_ablation/<exp>.yaml')
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--n_fleet_min', type=int, default=8)
    ap.add_argument('--n_fleet_max', type=int, default=16)
    ap.add_argument('--n_eval', type=int, default=5)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    config_path = (SCRIPT_DIR / args.config) if args.config else \
        (SCRIPT_DIR / 'configs_ablation' / f'{args.exp}.yaml')

    for seed in seeds:
        exp_dir = SCRIPT_DIR / 'logs' / f'{args.exp}_seed{seed}'
        if not exp_dir.exists():
            print(f"[skip] {exp_dir} not found")
            continue
        ep = best_ckpt_episode(args.exp, seed)
        if ep is None:
            print(f"[skip] no per_ckpt CSV for {args.exp}_seed{seed}; "
                  f"run scripts/per_ckpt_eval.py first")
            continue

        cfg = load_config(str(config_path))
        runner = TransitDuetV2Runner(cfg, device=args.device)
        runner.log_dir = str(exp_dir)
        lower_p = exp_dir / 'checkpoints' / f'lower_ep{ep}.pt'
        upper_p = exp_dir / 'checkpoints' / f'upper_ep{ep}.pt'
        if not (lower_p.exists() and upper_p.exists()):
            print(f"[skip] ckpt files for ep{ep} missing under {exp_dir}")
            continue
        runner.lower_trainer.load(str(lower_p))
        runner.upper_trainer.load(str(upper_p))
        runner.diag = _NullDiag()
        print(f"=== {args.exp}_seed{seed} | best ckpt ep{ep} ===")

        rows = []
        for N in range(args.n_fleet_min, args.n_fleet_max + 1):
            t0 = time.time()
            r = eval_at_fleet(runner, n_fleet=N, n_eps=args.n_eval, seed=seed)
            r['best_ep'] = ep
            r['seed'] = seed
            rows.append(r)
            print(f"  N={N:2d}  wait={r['wait_mean']:5.2f}±{r['wait_std']:4.2f}  "
                  f"cv={r['cv_mean']:.3f}  over={r['overshoot_mean']:5.2f}  "
                  f"({time.time()-t0:.0f}s)")

        out = exp_dir / 'pareto_frontier.json'
        out.write_text(json.dumps(rows, indent=1))
        print(f"  wrote {out}\n")


if __name__ == '__main__':
    main()
