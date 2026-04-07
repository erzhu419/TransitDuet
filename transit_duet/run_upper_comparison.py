"""
run_upper_comparison.py
=======================
Compare upper-level algorithms: RE-SAC vs BO vs CMA-ES vs GA vs Fixed.

All variants share the same lower-level RE-SAC (pretrained in Stage 1).
Upper level starts from ep 0 (no warmup needed — lower is already trained).

Usage:
    python -u run_upper_comparison.py --method bo --episodes 100 --seed 42
    python -u run_upper_comparison.py --method cmaes --episodes 100 --seed 42
    python -u run_upper_comparison.py --method ga --episodes 100 --seed 42
    python -u run_upper_comparison.py --method resac --episodes 100 --seed 42
    python -u run_upper_comparison.py --method fixed --episodes 30 --seed 42
"""

import sys, os, time, argparse, json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from env.sim import env_bus
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from upper.upper_bo import BOUpperPolicy
from upper.upper_cmaes import CMAESUpperPolicy
from upper.upper_ga import GAUpperPolicy
from upper.resac_upper import RESACUpperTrainer
from upper.upper_contextual_cmaes import ContextualCMAESUpperPolicy


def create_upper(method, device='cpu'):
    """Create upper-level optimizer by method name."""
    action_low = [180., 300., 240.]
    action_high = [600., 1200., 900.]

    if method == 'bo':
        return BOUpperPolicy(action_low=action_low, action_high=action_high,
                             n_initial=10, length_scale=0.5)
    elif method == 'cmaes':
        return CMAESUpperPolicy(action_low=action_low, action_high=action_high,
                                pop_size=10, sigma0=0.3)
    elif method == 'contextual_cmaes':
        return ContextualCMAESUpperPolicy(
            state_dim=5, action_dim=3,
            action_low=action_low, action_high=action_high,
            pop_size=12, sigma0=0.5)
    elif method == 'ga':
        return GAUpperPolicy(action_low=action_low, action_high=action_high,
                             pop_size=15, mutation_sigma=0.15)
    elif method == 'resac':
        return RESACUpperTrainer(
            state_dim=5, action_dim=3, hidden_dim=64,
            action_low=action_low, action_high=action_high,
            ensemble_size=10, beta=-2.0, lr=3e-4, gamma=0.95,
            replay_capacity=50000, device=device)
    elif method == 'fixed':
        return None  # use default 360s
    else:
        raise ValueError(f"Unknown method: {method}")


def run_episode_with_upper(env, lower_trainer, upper_params, replay_buffer,
                           device='cpu', training=True, contextual_policy=None):
    """
    Run one episode with upper headway control and lower RE-SAC.
    upper_params: np.array(3) for static methods, or None.
    contextual_policy: callable(state) → headway(3) for contextual methods.
    Returns (avg_reward, z_measurement, steps).
    """
    env.reset()

    if contextual_policy is not None:
        # State-dependent: call policy with each dispatch's state
        def upper_cb(s_upper, trip):
            trip._upper_queried = True
            hw = contextual_policy(s_upper)
            hour = 6 + trip.launch_time // 3600
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return float(hw[0])
            elif 9 < hour < 17:
                return float(hw[1])
            else:
                return float(hw[2])
        env._upper_policy_callback = upper_cb
    elif upper_params is not None:
        def upper_cb(s_upper, trip):
            trip._upper_queried = True
            hour = 6 + trip.launch_time // 3600
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                return float(upper_params[0])
            elif 9 < hour < 17:
                return float(upper_params[1])
            else:
                return float(upper_params[2])
        env._upper_policy_callback = upper_cb
    else:
        env._upper_policy_callback = None

    state_dict, reward_dict, _ = env.initialize_state()
    action_dict = {k: None for k in range(env.max_agent_num)}

    ep_reward = 0.0
    ep_cost = 0.0
    ep_steps = 0

    while not env.done:
        for key in state_dict:
            if len(state_dict[key]) == 1:
                if action_dict[key] is None:
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    a = lower_trainer.policy_net.get_action(
                        torch.from_numpy(obs).float().to(device),
                        deterministic=not training)
                    action_dict[key] = a
            elif len(state_dict[key]) == 2:
                if state_dict[key][0][1] != state_dict[key][1][1]:
                    state = np.array(state_dict[key][0], dtype=np.float32)
                    next_state = np.array(state_dict[key][1], dtype=np.float32)
                    reward = reward_dict[key]
                    cost = env.cost.get(key, 0.0)
                    trip_id = int(state_dict[key][0][0])
                    if training:
                        replay_buffer.push(state, action_dict[key], reward, cost,
                                           next_state, False, trip_id)
                    ep_reward += reward
                    ep_cost += cost
                    ep_steps += 1
                state_dict[key] = state_dict[key][1:]
                obs = np.array(state_dict[key][0], dtype=np.float32)
                action_dict[key] = lower_trainer.policy_net.get_action(
                    torch.from_numpy(obs).float().to(device),
                    deterministic=not training)

        state_dict, reward_dict, cost_dict, done = env.step(action_dict, render=False)

    z = env.measurement_vector
    avg_r = ep_reward / max(ep_steps, 1)
    return avg_r, z, ep_steps


def compute_system_reward(z, N_fleet=12):
    """
    System-level reward for upper optimization.
    z = [avg_wait_min, peak_fleet, headway_cv]
    Higher is better.
    """
    wait_penalty = -z[0] / 10.0          # avg wait in [-inf, 0]
    fleet_penalty = -max(0, z[1] - N_fleet) ** 2 / N_fleet  # over-fleet penalty
    cv_penalty = -z[2]                    # headway CV in [-inf, 0]
    return wait_penalty + fleet_penalty + cv_penalty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['bo', 'cmaes', 'contextual_cmaes', 'ga', 'resac', 'fixed'])
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--lower_warmup', type=int, default=20,
                        help='Episodes to pretrain lower before upper starts')
    args = parser.parse_args()

    device = 'cuda:0' if args.gpu and torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Environment
    env_path = str(SCRIPT_DIR / 'env')
    env = env_bus(env_path, route_sigma=1.5)
    env.enable_plot = False
    env._n_fleet_target = 12
    state_dim = env.state_dim

    # Lower policy (RE-SAC Lagrangian)
    lower = RESACLagrangianTrainer(
        state_dim=state_dim, action_dim=1, hidden_dim=64,
        action_range=60.0, cost_limit=0.5,
        ensemble_size=10, beta=-2.0, lr=3e-4,
        lambda_lr=1e-3, gamma=0.99, soft_tau=0.005,
        auto_entropy=True, maximum_alpha=0.3, device=device)
    replay_buffer = CostReplayBuffer(500_000)

    # Upper policy
    upper = create_upper(args.method, device)

    # Logging
    log_dir = SCRIPT_DIR / 'logs' / f'upper_{args.method}_seed{args.seed}'
    os.makedirs(log_dir, exist_ok=True)
    history = defaultdict(list)

    print(f"Upper Comparison: method={args.method}, episodes={args.episodes}, seed={args.seed}")
    print(f"  Lower warmup: {args.lower_warmup} eps, then upper activates")
    print("=" * 80)

    batch_size = 512
    total_eps = args.lower_warmup + args.episodes

    for ep in range(total_eps):
        t0 = time.time()

        if ep < args.lower_warmup:
            # Stage 1: lower warmup, no upper
            avg_r, z, steps = run_episode_with_upper(
                env, lower, None, replay_buffer, device, training=True)
            upper_params = np.array([360., 360., 360.])
            phase = "Warmup"
        else:
            # Stage 2: upper active
            contextual_policy = None

            if args.method == 'fixed':
                upper_params = np.array([360., 360., 360.])
            elif args.method == 'contextual_cmaes':
                # Contextual: suggest returns a policy, not params
                contextual_policy = upper.suggest()
                upper_params = contextual_policy(
                    np.array([0.5, 0.5, 0.5, 0.5, 0.5]))  # for logging
            elif args.method == 'resac':
                s_upper = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                upper_params = upper.policy_net.get_action(s_upper, deterministic=False)
            else:
                upper_params = upper.suggest()

            avg_r, z, steps = run_episode_with_upper(
                env, lower, upper_params, replay_buffer, device,
                training=True, contextual_policy=contextual_policy)

            # Report reward to upper optimizer
            sys_reward = compute_system_reward(z)
            if args.method == 'resac':
                s_upper = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
                upper.replay_buffer.push(s_upper, upper_params, sys_reward,
                                         s_upper, False)
                if len(upper.replay_buffer) >= 64:
                    for _ in range(5):
                        upper.update(64)
            elif upper is not None:
                upper.report(sys_reward)

            phase = args.method.upper()

        env_time = time.time() - t0

        # Train lower
        train_metrics = {}
        if len(replay_buffer) > batch_size:
            t1 = time.time()
            for _ in range(30):
                train_metrics = lower.update(replay_buffer, batch_size, reward_scale=1.0)
            train_time = time.time() - t1
        else:
            train_time = 0

        # Log
        sys_r = compute_system_reward(z)
        history['avg_reward'].append(avg_r)
        history['sys_reward'].append(sys_r)
        history['avg_wait'].append(float(z[0]))
        history['peak_fleet'].append(float(z[1]))
        history['headway_cv'].append(float(z[2]))
        history['upper_params'].append(upper_params.tolist())
        history['lambda'].append(lower.lambda_param)

        if ep % 5 == 0 or ep < 3:
            print(f"[Ep {ep:3d}] {phase:8s} | "
                  f"R={avg_r:.3f} sys={sys_r:.3f} | "
                  f"w={z[0]:.1f}m f={z[1]:.0f} cv={z[2]:.2f} | "
                  f"H=[{upper_params[0]:.0f},{upper_params[1]:.0f},{upper_params[2]:.0f}] | "
                  f"λ={lower.lambda_param:.2f} | "
                  f"{env_time:.0f}+{train_time:.0f}s")

    # Save
    results = {k: [float(x) if not isinstance(x, list) else x
                    for x in v] for k, v in history.items()}
    with open(log_dir / 'history.json', 'w') as f:
        json.dump(results, f)

    if upper is not None and hasattr(upper, 'get_best'):
        best = upper.get_best()
        print(f"\nBest headway: H_peak={best[0]:.0f}s, H_off={best[1]:.0f}s, H_trans={best[2]:.0f}s")

    print(f"Results saved to {log_dir}")


if __name__ == '__main__':
    main()
