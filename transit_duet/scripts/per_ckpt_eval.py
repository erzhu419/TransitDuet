#!/usr/bin/env python3
"""
per_ckpt_eval.py
================
For each (seed, checkpoint episode) of a TransitDuet experiment, run N evaluation
episodes with deterministic policy on the same fair env (demand_noise=0.15,
elastic fleet [8,16]). Pick the best checkpoint per seed by mean composite cost.

Usage:
    python scripts/per_ckpt_eval.py --exp H_tpc --eps 49,99,149,199,249,299 \
        --n_eval 20 --seeds 42,123,456
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from runner_v2 import TransitDuetV2Runner, load_config


class _NullDiag:
    def append(self, row): pass
    def save_json(self): pass


def load_ckpt(exp_dir: Path, ep: int, config_path: Path, device: str):
    cfg = load_config(str(config_path))
    runner = TransitDuetV2Runner(cfg, device=device)
    runner.log_dir = str(exp_dir)
    ckpt_dir = exp_dir / 'checkpoints'
    lower_p = ckpt_dir / f'lower_ep{ep}.pt'
    upper_p = ckpt_dir / f'upper_ep{ep}.pt'
    if not (lower_p.exists() and upper_p.exists()):
        return None
    runner.lower_trainer.load(str(lower_p))
    runner.upper_trainer.load(str(upper_p))
    runner.diag = _NullDiag()
    return runner


def composite(wait, cv, overshoot, n_fleet):
    return wait / 10.0 + (overshoot ** 2) / max(n_fleet, 1) + cv


def eval_runner(runner, n_eps: int, base_seed: int):
    waits, cvs, overs, ns = [], [], [], []
    rng = np.random.RandomState(base_seed)
    for i in range(n_eps):
        N = int(rng.randint(runner.fleet_min, runner.fleet_max + 1))
        runner.env._n_fleet_target = N
        row = runner.run_episode(ep=9999, training=False, N_fleet_override=N)
        waits.append(float(row['avg_wait_min']))
        cvs.append(float(row['headway_cv']))
        overs.append(float(row['fleet_overshoot']))
        ns.append(N)
    waits, cvs, overs, ns = map(np.array, (waits, cvs, overs, ns))
    composites = waits / 10.0 + (overs ** 2) / np.maximum(ns, 1) + cvs
    return {
        'wait_mean': float(waits.mean()), 'wait_std': float(waits.std()),
        'cv_mean': float(cvs.mean()), 'cv_std': float(cvs.std()),
        'overshoot_mean': float(overs.mean()), 'overshoot_std': float(overs.std()),
        'composite_mean': float(composites.mean()),
        'composite_std': float(composites.std()),
        'n_eps': n_eps,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='H_tpc',
                    help='experiment name (matches logs/<exp>_seed*/)')
    ap.add_argument('--config', default='configs_ablation/H_tpc.yaml')
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--eps', default='49,99,149,199,249,299',
                    help='comma-separated checkpoint episodes to evaluate')
    ap.add_argument('--n_eval', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--eval_seed', type=int, default=12345,
                    help='base seed for shared eval rollouts')
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    eps = [int(e) for e in args.eps.split(',')]
    out_root = SCRIPT_DIR / 'logs' / 'eval_per_ckpt' / args.exp
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for seed in seeds:
        exp_dir = SCRIPT_DIR / 'logs' / f'{args.exp}_seed{seed}'
        if not exp_dir.exists():
            print(f"  [skip] {exp_dir} not found")
            continue
        for ep in eps:
            t0 = time.time()
            runner = load_ckpt(exp_dir, ep,
                               SCRIPT_DIR / args.config, args.device)
            if runner is None:
                print(f"  [skip] {args.exp}_seed{seed} ep={ep} missing")
                continue
            stats = eval_runner(runner, args.n_eval, args.eval_seed)
            row = {'exp': args.exp, 'seed': seed, 'ep': ep, **stats,
                   'wall_s': round(time.time() - t0, 1)}
            summary.append(row)
            out_path = out_root / f'seed{seed}_ep{ep}.json'
            out_path.write_text(json.dumps(row, indent=1))
            print(f"  seed{seed} ep{ep:3d}  "
                  f"wait={stats['wait_mean']:5.2f}±{stats['wait_std']:4.2f}  "
                  f"cv={stats['cv_mean']:.3f}  "
                  f"over={stats['overshoot_mean']:5.2f}  "
                  f"composite={stats['composite_mean']:5.3f}  "
                  f"({row['wall_s']}s)")
            del runner
            torch.cuda.empty_cache()

    if not summary:
        return

    # Pick best ckpt per seed by composite_mean
    print("\n" + "=" * 78)
    print("Best checkpoint per seed (lowest composite mean):")
    print("=" * 78)
    best_per_seed = {}
    for seed in seeds:
        rows = [r for r in summary if r['seed'] == seed]
        if not rows:
            continue
        best = min(rows, key=lambda r: r['composite_mean'])
        best_per_seed[seed] = best
        print(f"  seed{seed} best @ ep{best['ep']:3d}  "
              f"wait={best['wait_mean']:5.2f}  cv={best['cv_mean']:.3f}  "
              f"over={best['overshoot_mean']:5.2f}  "
              f"composite={best['composite_mean']:5.3f}")

    # 3-seed aggregate of best ckpts
    if len(best_per_seed) == 3:
        ws = [best_per_seed[s]['wait_mean'] for s in seeds]
        cs = [best_per_seed[s]['composite_mean'] for s in seeds]
        cvs = [best_per_seed[s]['cv_mean'] for s in seeds]
        os_ = [best_per_seed[s]['overshoot_mean'] for s in seeds]
        print("\n3-seed aggregate of best checkpoints:")
        print(f"  wait      = {np.mean(ws):.2f} ± {np.std(ws):.2f}")
        print(f"  cv        = {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")
        print(f"  overshoot = {np.mean(os_):.2f} ± {np.std(os_):.2f}")
        print(f"  composite = {np.mean(cs):.3f} ± {np.std(cs):.3f}")
        print("\nReference (Fixed baseline, fair env):")
        print("  wait=7.03 ± 1.63   composite=1.62 ± 0.27")

    # Write a CSV summary
    import csv as csv_mod
    csv_path = out_root / f'{args.exp}_per_ckpt.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv_mod.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)
    print(f"\nWrote {csv_path}")


if __name__ == '__main__':
    main()
