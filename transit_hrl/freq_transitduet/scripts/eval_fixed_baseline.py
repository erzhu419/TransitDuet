#!/usr/bin/env python3
"""
eval_fixed_baseline.py
======================
Per-checkpoint evaluation of the Fixed-timetable baseline using the same
protocol as per_ckpt_eval.py for H_tpc: load lower checkpoint, run N
deterministic eval episodes with H=360s constant headway and elastic fleet
sampled per episode, pick best ckpt by composite cost.

Usage:
    python scripts/eval_fixed_baseline.py --eps 49,99,115 --n_eval 60 \
        --seeds 42,123,456
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


def composite(wait, cv, overshoot, n_fleet):
    return wait / 10.0 + (overshoot ** 2) / max(n_fleet, 1) + cv


def make_env_and_lower(device, seed):
    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env.demand_noise = 0.15
    state_dim = env.state_dim

    lower = RESACLagrangianTrainer(
        state_dim=state_dim, action_dim=1, hidden_dim=64,
        action_range=60.0, cost_limit=0.5,
        ensemble_size=10, beta=-2.0, lr=3e-4,
        lambda_lr=1e-4, gamma=0.99, soft_tau=0.005,
        auto_entropy=True, maximum_alpha=0.1, device=device)
    return env, lower


def run_episode_fixed(env, lower, N_fleet, device):
    """One deterministic eval episode with H=360, no upper."""
    env._n_fleet_target = N_fleet

    def upper_cb(s_upper, trip):
        trip._upper_queried = True
        return 360.0  # Fixed headway
    env._upper_policy_callback = upper_cb

    env.reset()
    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: None for k in range(env.max_agent_num)}

    while not env.done:
        for key in state_dict:
            if len(state_dict[key]) == 1:
                if action_dict[key] is None:
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    a = lower.policy_net.get_action(
                        torch.from_numpy(obs).float().to(device),
                        deterministic=True)  # deterministic eval
                    action_dict[key] = a
            elif len(state_dict[key]) == 2:
                state_dict[key] = state_dict[key][1:]
                obs = np.array(state_dict[key][0], dtype=np.float32)
                action_dict[key] = lower.policy_net.get_action(
                    torch.from_numpy(obs).float().to(device),
                    deterministic=True)
        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    z = env.measurement_vector
    return float(z[0]), float(z[1]), float(z[2])  # wait, peak_fleet, cv


def eval_ckpt(env, lower, n_eps, fleet_min, fleet_max, eval_seed, device):
    rng = np.random.RandomState(eval_seed)
    waits, cvs, overs, ns = [], [], [], []
    for i in range(n_eps):
        N = int(rng.randint(fleet_min, fleet_max + 1))
        wait, peak, cv = run_episode_fixed(env, lower, N, device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='42,123,456')
    ap.add_argument('--eps', default='49,99,149,199,249,299',
                    help='paper protocol: 6 evenly-spaced ckpts across 300-ep training')
    ap.add_argument('--n_eval', type=int, default=20,
                    help='paper protocol: 20 fresh eval episodes per ckpt')
    ap.add_argument('--fleet_min', type=int, default=8)
    ap.add_argument('--fleet_max', type=int, default=16)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--eval_seed', type=int, default=12345)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    eps = [int(e) for e in args.eps.split(',')]
    out_root = SCRIPT_DIR / 'logs' / 'eval_per_ckpt' / 'fixed'
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for seed in seeds:
        run_dir = SCRIPT_DIR / 'logs' / f'upper_fixed_seed{seed}'
        ckpt_dir = run_dir / 'checkpoints'
        if not ckpt_dir.exists():
            print(f"  [skip] {run_dir} no checkpoints"); continue
        env, lower = make_env_and_lower(args.device, seed)
        for ep in eps:
            ckpt_p = ckpt_dir / f'lower_ep{ep}.pt'
            if not ckpt_p.exists():
                # try a near match
                avail = sorted([int(p.stem.split('ep')[-1])
                                for p in ckpt_dir.glob('lower_ep*.pt')])
                if not avail:
                    continue
                ep_use = min(avail, key=lambda x: abs(x - ep))
                ckpt_p = ckpt_dir / f'lower_ep{ep_use}.pt'
                ep = ep_use
            t0 = time.time()
            lower.load(str(ckpt_p))
            stats = eval_ckpt(env, lower, args.n_eval,
                              args.fleet_min, args.fleet_max,
                              args.eval_seed, args.device)
            row = {'method': 'fixed', 'seed': seed, 'ep': ep, **stats,
                   'wall_s': round(time.time() - t0, 1)}
            summary.append(row)
            (out_root / f'seed{seed}_ep{ep}.json').write_text(
                json.dumps(row, indent=1))
            print(f"  fixed seed{seed} ep{ep:3d}  "
                  f"wait={stats['wait_mean']:5.2f}±{stats['wait_std']:4.2f}  "
                  f"cv={stats['cv_mean']:.3f}  "
                  f"over={stats['overshoot_mean']:5.2f}  "
                  f"composite={stats['composite_mean']:5.3f}  "
                  f"({row['wall_s']}s)")

    if not summary:
        return

    print()
    print("=" * 78)
    print("Best ckpt per seed:")
    best_per_seed = {}
    for seed in seeds:
        rows = [r for r in summary if r['seed'] == seed]
        if not rows: continue
        best = min(rows, key=lambda r: r['composite_mean'])
        best_per_seed[seed] = best
        print(f"  fixed seed{seed} best @ ep{best['ep']:3d}  "
              f"wait={best['wait_mean']:5.2f}  composite={best['composite_mean']:5.3f}")

    if len(best_per_seed) == 3:
        ws = [best_per_seed[s]['wait_mean'] for s in seeds]
        cs = [best_per_seed[s]['composite_mean'] for s in seeds]
        cvs = [best_per_seed[s]['cv_mean'] for s in seeds]
        os_ = [best_per_seed[s]['overshoot_mean'] for s in seeds]
        print(f"\n3-seed aggregate (Fixed retrained, fixed Lagrangian):")
        print(f"  wait      = {np.mean(ws):.2f} ± {np.std(ws):.2f}")
        print(f"  cv        = {np.mean(cvs):.3f} ± {np.std(cvs):.3f}")
        print(f"  overshoot = {np.mean(os_):.2f} ± {np.std(os_):.2f}")
        print(f"  composite = {np.mean(cs):.3f} ± {np.std(cs):.3f}")


if __name__ == '__main__':
    main()
