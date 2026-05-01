#!/usr/bin/env python3
"""
eval_baseline.py
================
Per-checkpoint evaluation of any non-RL upper baseline (fixed / GA / CMA-ES).
Uses the final per-seed upper_params from history.json (or H=360 for Fixed).
Same protocol as per_ckpt_eval.py for H_tpc.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
from env.sim import env_bus
from lower.resac_lagrangian import RESACLagrangianTrainer


def make_env_and_lower(device):
    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = 0.15
    state_dim = env.state_dim
    lower = RESACLagrangianTrainer(
        state_dim=state_dim, action_dim=1, hidden_dim=64,
        action_range=60.0, cost_limit=0.5,
        ensemble_size=10, beta=-2.0, lr=3e-4,
        lambda_lr=1e-3, gamma=0.99, soft_tau=0.005,
        auto_entropy=True, maximum_alpha=0.3, device=device)
    return env, lower


def make_upper_callback(headways):
    """Time-of-day-aware headway selector. headways = (Hpeak, Hoff, Htrans)."""
    Hp, Ho, Ht = headways

    def cb(s_upper, trip):
        trip._upper_queried = True
        hour = 6 + trip.launch_time // 3600
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return float(Hp)
        elif 9 < hour < 17:
            return float(Ho)
        else:
            return float(Ht)
    return cb


def run_episode(env, lower, headways, N_fleet, device):
    env._n_fleet_target = N_fleet
    env._upper_policy_callback = make_upper_callback(headways)
    env.reset()
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: None for k in range(env.max_agent_num)}
    while not env.done:
        for key in state_dict:
            if len(state_dict[key]) == 1:
                if action_dict[key] is None:
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    action_dict[key] = lower.policy_net.get_action(
                        torch.from_numpy(obs).float().to(device),
                        deterministic=True)
            elif len(state_dict[key]) == 2:
                state_dict[key] = state_dict[key][1:]
                obs = np.array(state_dict[key][0], dtype=np.float32)
                action_dict[key] = lower.policy_net.get_action(
                    torch.from_numpy(obs).float().to(device),
                    deterministic=True)
        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)
    z = env.measurement_vector
    return float(z[0]), float(z[1]), float(z[2])


def eval_ckpt(env, lower, headways, n_eps, fleet_min, fleet_max, eval_seed, device):
    rng = np.random.RandomState(eval_seed)
    waits, cvs, overs, ns = [], [], [], []
    for _ in range(n_eps):
        N = int(rng.randint(fleet_min, fleet_max + 1))
        wait, peak, cv = run_episode(env, lower, headways, N, device)
        waits.append(wait); cvs.append(cv)
        overs.append(max(0.0, peak - N))
        ns.append(N)
    waits, cvs, overs, ns = map(np.array, (waits, cvs, overs, ns))
    cps = waits / 10.0 + (overs ** 2) / np.maximum(ns, 1) + cvs
    return {
        'wait_mean': float(waits.mean()), 'wait_std': float(waits.std()),
        'cv_mean': float(cvs.mean()), 'cv_std': float(cvs.std()),
        'overshoot_mean': float(overs.mean()), 'overshoot_std': float(overs.std()),
        'composite_mean': float(cps.mean()), 'composite_std': float(cps.std()),
        'n_eps': n_eps,
    }


def get_headways(method, seed):
    if method == 'fixed':
        return (360.0, 360.0, 360.0)
    h = json.load(open(SCRIPT_DIR / 'logs' / f'upper_{method}_seed{seed}' / 'history.json'))
    last = h['upper_params'][-1]
    return tuple(float(x) for x in last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=['fixed', 'ga', 'cmaes'], required=True)
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--eps', default='49,99,119')
    ap.add_argument('--n_eval', type=int, default=20)
    ap.add_argument('--fleet_min', type=int, default=8)
    ap.add_argument('--fleet_max', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--eval_seed', type=int, default=12345)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    eps = [int(e) for e in args.eps.split(',')]
    out_root = SCRIPT_DIR / 'logs' / 'eval_per_ckpt' / args.method
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for seed in seeds:
        run_dir = SCRIPT_DIR / 'logs' / f'upper_{args.method}_seed{seed}'
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue
        env, lower = make_env_and_lower(args.device)
        headways = get_headways(args.method, seed)
        print(f"  {args.method} seed{seed} headways = {headways}")
        for ep in eps:
            ckpt_p = ckpt_dir / f'lower_ep{ep}.pt'
            if not ckpt_p.exists():
                avail = sorted([int(p.stem.split('ep')[-1])
                                for p in ckpt_dir.glob('lower_ep*.pt')])
                if not avail: continue
                ep = min(avail, key=lambda x: abs(x - ep))
                ckpt_p = ckpt_dir / f'lower_ep{ep}.pt'
            t0 = time.time()
            lower.load(str(ckpt_p))
            stats = eval_ckpt(env, lower, headways, args.n_eval,
                              args.fleet_min, args.fleet_max,
                              args.eval_seed, args.device)
            row = {'method': args.method, 'seed': seed, 'ep': ep,
                   'headways': list(headways), **stats,
                   'wall_s': round(time.time() - t0, 1)}
            summary.append(row)
            (out_root / f'seed{seed}_ep{ep}.json').write_text(
                json.dumps(row, indent=1))
            print(f"  {args.method} seed{seed} ep{ep:3d}  "
                  f"wait={stats['wait_mean']:5.2f}±{stats['wait_std']:4.2f}  "
                  f"composite={stats['composite_mean']:5.3f}  "
                  f"({row['wall_s']}s)")

    if not summary:
        return
    print()
    print("Best per seed:")
    best_per_seed = {}
    for seed in seeds:
        rows = [r for r in summary if r['seed'] == seed]
        if not rows: continue
        best = min(rows, key=lambda r: r['composite_mean'])
        best_per_seed[seed] = best
        print(f"  {args.method} seed{seed} best @ ep{best['ep']:3d}  "
              f"wait={best['wait_mean']:5.2f}  composite={best['composite_mean']:5.3f}")
    if len(best_per_seed) == 3:
        ws = [best_per_seed[s]['wait_mean'] for s in seeds]
        cs = [best_per_seed[s]['composite_mean'] for s in seeds]
        cvs = [best_per_seed[s]['cv_mean'] for s in seeds]
        os_ = [best_per_seed[s]['overshoot_mean'] for s in seeds]
        print(f"\n3-seed aggregate ({args.method}):")
        print(f"  wait      = {np.mean(ws):.2f} +/- {np.std(ws):.2f}")
        print(f"  cv        = {np.mean(cvs):.3f} +/- {np.std(cvs):.3f}")
        print(f"  overshoot = {np.mean(os_):.2f} +/- {np.std(os_):.2f}")
        print(f"  composite = {np.mean(cs):.3f} +/- {np.std(cs):.3f}")


if __name__ == '__main__':
    main()
