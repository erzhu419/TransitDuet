#!/usr/bin/env python3
"""
generalization_eval.py
======================
Evaluate validation-selected H_hiro checkpoints across out-of-distribution
travel-time stochasticity, and contrast a no-demand-noise variant against
the noisy training distribution. Uses ``runner_v3`` (the runner that
honours coupling_mode {channels, haar, hiro}); under ``runner_v2`` HIRO
checkpoints would be silently re-evaluated under launch-time-shift semantics
rather than goal-shift, breaking the comparison.

Modes:
  (a) cross_sigma --- the H_hiro policy (trained at sigma_route=1.5) is
      evaluated at sigma_route in {0.5, 1.0, 1.5, 2.0, 3.0}, including the
      training value 1.5 itself, so that Section V-F's table has a self-
      consistent row at the training distribution.
  (b) demand_shift --- the H_hiro_no_demand_noise policy is evaluated on
      both the deterministic-demand env it was trained on and the noisy
      env that the main paper uses, to quantify the brittleness of training
      under deterministic demand.

For each (mode, seed) the script picks the validation-best checkpoint by
reading the per-ckpt CSV produced by per_ckpt_eval.py. If that CSV is
missing, it falls back to the latest checkpoint on disk and prints a warning.

Outputs one JSON per (config, seed) under logs/eval_generalization/.

Usage:
    python scripts/generalization_eval.py --mode cross_sigma
    python scripts/generalization_eval.py --mode demand_shift
    python scripts/generalization_eval.py --mode all      # both, default
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from runner_v3 import TransitDuetV2Runner, load_config


class _NullDiag:
    """Stub so run_episode can append rows during eval without writing CSV."""
    def append(self, row): pass
    def save_json(self): pass


def best_ckpt_episode(exp: str, seed: int) -> int | None:
    """
    Return the episode index of the validation-best checkpoint for
    (exp, seed) by reading logs/eval_per_ckpt/<exp>/<exp>_per_ckpt.csv.
    Returns None if the CSV is missing or empty.
    """
    csv = SCRIPT_DIR / 'logs' / 'eval_per_ckpt' / exp / f'{exp}_per_ckpt.csv'
    if not csv.exists():
        return None
    import csv as _csv
    rows = []
    with open(csv) as f:
        for r in _csv.DictReader(f):
            if int(r['seed']) == seed:
                rows.append(r)
    if not rows:
        return None
    rows.sort(key=lambda r: float(r['composite_mean']))
    return int(rows[0]['ep'])


def load_runner_at_ckpt(exp_dir: Path, ep: int | None,
                        config_path: Path, device: str):
    """
    Load `runner_v3` and restore checkpoint at episode `ep` if specified;
    otherwise call maybe_resume() (latest ckpt).
    """
    cfg = load_config(str(config_path))
    runner = TransitDuetV2Runner(cfg, device=device)
    runner.log_dir = str(exp_dir)
    if ep is not None:
        lower_p = exp_dir / 'checkpoints' / f'lower_ep{ep}.pt'
        upper_p = exp_dir / 'checkpoints' / f'upper_ep{ep}.pt'
        if not (lower_p.exists() and upper_p.exists()):
            raise RuntimeError(
                f"Selected ckpt ep{ep} not found under {exp_dir}/checkpoints; "
                f"check that per_ckpt_eval.py was run for this seed.")
        runner.lower_trainer.load(str(lower_p))
        runner.upper_trainer.load(str(upper_p))
        print(f"  [Load] Validation-best ckpt ep{ep} from {exp_dir}")
    else:
        loaded = runner.maybe_resume()
        if loaded == 0:
            raise RuntimeError(f"No checkpoint found under {exp_dir}/checkpoints")
        print(f"  [Load] (fallback) latest ckpt ep{loaded - 1} from {exp_dir}")
    runner.diag = _NullDiag()
    return runner


def eval_at(runner, *, sigma=None, demand_noise=None, N_fleet=None,
            n_eps=10, seed=0):
    """Run n_eps eval episodes; return per-episode metrics as lists."""
    if sigma is not None:
        runner.env.route_sigma = sigma
    if demand_noise is not None:
        runner.env.demand_noise = demand_noise

    waits, cvs, overshoots, peaks, Ns = [], [], [], [], []
    rng = np.random.RandomState(seed)
    for i in range(n_eps):
        N_eval = int(rng.randint(runner.fleet_min, runner.fleet_max + 1)) \
            if N_fleet is None else N_fleet
        runner.env._n_fleet_target = N_eval
        row = runner.run_episode(ep=9999, training=False, N_fleet_override=N_eval)
        waits.append(float(row['avg_wait_min']))
        cvs.append(float(row['headway_cv']))
        overshoots.append(float(row['fleet_overshoot']))
        peaks.append(float(row['peak_fleet']))
        Ns.append(N_eval)
    return dict(wait=waits, cv=cvs, overshoot=overshoots,
                peak_fleet=peaks, N_fleet=Ns)


def summarize(metrics: dict) -> dict:
    out = {}
    for k, vs in metrics.items():
        out[f'{k}_mean'] = float(np.mean(vs))
        out[f'{k}_std'] = float(np.std(vs))
    return out


def mode_cross_sigma(args):
    """H_hiro (trained at sigma=1.5) eval across sigma_route values."""
    sigmas = [float(s) for s in args.sigmas.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    exp = args.exp
    config_path = SCRIPT_DIR / 'configs_ablation' / f'{exp}.yaml'
    out_root = SCRIPT_DIR / 'logs' / 'eval_generalization' / 'cross_sigma'
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        exp_dir = SCRIPT_DIR / 'logs' / f'{exp}_seed{seed}'
        if not exp_dir.exists():
            print(f"  [skip] {exp_dir} not found")
            continue
        ep = best_ckpt_episode(exp, seed)
        if ep is None:
            print(f"  [warn] no per_ckpt CSV for {exp}_seed{seed}; "
                  f"falling back to latest ckpt")
        runner = load_runner_at_ckpt(exp_dir, ep, config_path, args.device)
        for sigma in sigmas:
            key = f'sigma_{sigma}_seed{seed}'
            out_path = out_root / f'{key}.json'
            if out_path.exists() and not args.overwrite:
                print(f"  [skip] {out_path.name} exists")
                continue
            t0 = time.time()
            m = eval_at(runner, sigma=sigma, demand_noise=0.15,
                        n_eps=args.n_eps, seed=seed)
            rec = {'exp': exp, 'seed': seed, 'sigma': sigma,
                   'demand_noise': 0.15, 'best_ep': ep,
                   'n_eps': args.n_eps, **summarize(m), 'raw': m}
            out_path.write_text(json.dumps(rec, indent=1))
            print(f"  {key}  wait={rec['wait_mean']:.2f}±{rec['wait_std']:.2f}  "
                  f"cv={rec['cv_mean']:.3f}  over={rec['overshoot_mean']:.2f}  "
                  f"({time.time() - t0:.0f}s)")


def mode_demand_shift(args):
    """H_hiro_no_demand_noise policy evaluated on both deterministic and
    noisy demand to quantify brittleness."""
    seeds = [int(s) for s in args.seeds.split(',')]
    exp = 'H_hiro_no_demand_noise'
    config_path = SCRIPT_DIR / 'configs_ablation' / f'{exp}.yaml'
    out_root = SCRIPT_DIR / 'logs' / 'eval_generalization' / 'demand_shift'
    out_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        exp_dir = SCRIPT_DIR / 'logs' / f'{exp}_seed{seed}'
        if not exp_dir.exists():
            print(f"  [skip] {exp_dir} not found")
            continue
        ep = best_ckpt_episode(exp, seed)
        if ep is None:
            print(f"  [warn] no per_ckpt CSV for {exp}_seed{seed}; "
                  f"falling back to latest ckpt")
        runner = load_runner_at_ckpt(exp_dir, ep, config_path, args.device)
        for label, dn in [('in_dist', 0.0), ('ood_noisy', 0.15)]:
            key = f'demand_{label}_seed{seed}'
            out_path = out_root / f'{key}.json'
            if out_path.exists() and not args.overwrite:
                print(f"  [skip] {out_path.name} exists")
                continue
            t0 = time.time()
            m = eval_at(runner, sigma=1.5, demand_noise=dn,
                        n_eps=args.n_eps, seed=seed)
            rec = {'exp': exp, 'seed': seed, 'label': label,
                   'demand_noise': dn, 'best_ep': ep,
                   'n_eps': args.n_eps, **summarize(m), 'raw': m}
            out_path.write_text(json.dumps(rec, indent=1))
            print(f"  {key}  wait={rec['wait_mean']:.2f}±{rec['wait_std']:.2f}  "
                  f"cv={rec['cv_mean']:.3f}  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cross_sigma', 'demand_shift', 'all'],
                    default='all')
    ap.add_argument('--exp', default='H_hiro',
                    help='experiment for cross_sigma (default H_hiro)')
    ap.add_argument('--sigmas', default='0.5,1.0,1.5,2.0,3.0',
                    help='including the training value 1.5 by default')
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--n_eps', type=int, default=10)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.mode in ('cross_sigma', 'all'):
        print("=== cross_sigma ===")
        mode_cross_sigma(args)
    if args.mode in ('demand_shift', 'all'):
        print("=== demand_shift ===")
        mode_demand_shift(args)

    print("\nDone. Results in logs/eval_generalization/")


if __name__ == '__main__':
    main()
