"""
runner.py
=========
TransitDuet main training loop — full 3-stage bi-level training.

Stage 1 (warmup):  β=0, only lower πL trains, fixed target_headway=360s
Stage 2 (ramp):    β ramps 0→0.5, upper πU activates, TAP couples layers
Stage 3 (full):    β=0.5, full coupling

Usage:
    cd transit_duet/
    python -u runner.py [--config config.yaml] [--episodes 300] [--gpu]
"""

import os
import sys
import argparse
import json
import time
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from env.sim import env_bus
from upper.upper_policy import UpperPolicy
from upper.measurement_proj import MeasurementProjection
from lower.resac_lagrangian import RESACLagrangianTrainer
from lower.cost_replay_buffer import CostReplayBuffer
from coupling.tap import TAPManager
from coupling.beta_schedule import BetaSchedule


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


class TransitDuetRunner:
    """Main training loop for bi-level bus control."""

    def __init__(self, config, device='cpu'):
        self.cfg = config
        self.device = device

        # Environment
        env_path = os.path.join(str(SCRIPT_DIR), config['env']['path'])
        self.env = env_bus(env_path, route_sigma=config['env']['route_sigma'])
        self.env.enable_plot = False

        state_dim = self.env.state_dim

        # Upper policy (independent network)
        self.upper_policy = UpperPolicy(
            state_dim=config['upper']['state_dim'],
            K=config['upper']['action_dim'],
            hidden_dim=config['upper']['hidden_dim'],
            action_low=config['upper']['action_low'],
            action_high=config['upper']['action_high'],
            device=device,
        ).to(device)
        self.upper_optimizer = torch.optim.Adam(
            self.upper_policy.parameters(), lr=config['upper']['lr']
        )

        # Lower policy (RE-SAC Lagrangian, parameter-sharing single-agent)
        self.replay_buffer = CostReplayBuffer(config['training']['replay_buffer_size'])
        self.lower_trainer = RESACLagrangianTrainer(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=config['lower']['hidden_dim'],
            action_range=config['lower']['action_range'],
            cost_limit=config['lower']['cost_limit'],
            ensemble_size=config['lower'].get('ensemble_size', 10),
            beta=config['lower'].get('resac_beta', -2.0),
            beta_ood=config['lower'].get('beta_ood', 0.01),
            weight_reg=config['lower'].get('weight_reg', 0.01),
            lr=config['lower']['lr'],
            lambda_lr=config['lower']['lambda_lr'],
            gamma=config['lower']['gamma'],
            soft_tau=config['lower']['soft_tau'],
            auto_entropy=config['lower']['auto_entropy'],
            maximum_alpha=config['lower']['maximum_alpha'],
            device=device,
        )

        # ApproPO measurement projection
        self.measurement_proj = MeasurementProjection(
            N_fleet=config['upper']['N_fleet'],
            lr=config['coupling']['measurement_lr'],
        )

        # TAP coupling
        self.beta_schedule = BetaSchedule(
            warmup_eps=config['coupling']['beta_warmup_eps'],
            ramp_eps=config['coupling']['beta_ramp_eps'],
            beta_max=config['coupling']['beta_max'],
        )
        self.tap = TAPManager(self.beta_schedule)

        # Training params
        self.batch_size = config['lower']['batch_size']
        self.updates_per_episode = config['lower'].get('updates_per_episode', 30)

        # Previous episode's upper reward per trip (for TAP reverse)
        self._prev_upper_reward_per_trip = {}

        # Logging
        self.log_dir = os.path.join(str(SCRIPT_DIR), 'logs', 'full_training')
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = defaultdict(list)

    def _upper_callback(self, s_upper, trip):
        """Called by env.step() at each dispatch event."""
        action = self.upper_policy.get_action(s_upper, deterministic=False)
        trip_id = trip.launch_turn

        # Record for TAP
        self.tap.record_upper_transition(s_upper, action, trip_id)

        # Select headway based on time-of-day
        hour = 6 + trip.launch_time // 3600
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            target_hw = action[0]  # H_peak
        elif 9 < hour < 17:
            target_hw = action[1]  # H_off_peak
        else:
            target_hw = action[2]  # H_transition

        return float(target_hw)

    def run_episode(self, ep, training=True):
        """Run one complete episode."""
        t0 = time.time()
        self.tap.clear()
        self.env.reset()

        beta = self.beta_schedule.get_beta(ep)

        # Register upper callback only after warmup
        if ep > self.beta_schedule.warmup and training:
            self.env._upper_policy_callback = self._upper_callback
        else:
            self.env._upper_policy_callback = None

        # Initialize
        state_dict, reward_dict, _ = self.env.initialize_state()
        action_dict = {k: None for k in range(self.env.max_agent_num)}

        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0

        while not self.env.done:
            for key in state_dict:
                if len(state_dict[key]) == 1:
                    if action_dict[key] is None:
                        obs = np.array(state_dict[key][0], dtype=np.float32)
                        a = self.lower_trainer.policy_net.get_action(
                            torch.from_numpy(obs).float().to(self.device),
                            deterministic=not training)
                        action_dict[key] = a

                elif len(state_dict[key]) == 2:
                    if state_dict[key][0][1] != state_dict[key][1][1]:
                        state = np.array(state_dict[key][0], dtype=np.float32)
                        next_state = np.array(state_dict[key][1], dtype=np.float32)
                        reward = reward_dict[key]
                        cost = self.env.cost.get(key, 0.0)
                        trip_id = int(state_dict[key][0][0])

                        if training:
                            self.replay_buffer.push(
                                state, action_dict[key], reward, cost,
                                next_state, False, trip_id)
                            self.tap.record_lower_reward(reward, trip_id)

                        episode_reward += reward
                        episode_cost += cost
                        episode_steps += 1

                    state_dict[key] = state_dict[key][1:]
                    obs = np.array(state_dict[key][0], dtype=np.float32)
                    action_dict[key] = self.lower_trainer.policy_net.get_action(
                        torch.from_numpy(obs).float().to(self.device),
                        deterministic=not training)

            state_dict, reward_dict, cost_dict, done = self.env.step(
                action_dict, render=False)

        env_time = time.time() - t0

        # ──── Episode-end training ────
        t1 = time.time()
        train_metrics = {}

        # Lower policy: batch training at episode end
        if training and len(self.replay_buffer) > self.batch_size:
            # Build TAP reverse signal: inject upper reward into lower transitions
            tap_signal = None
            if beta > 0 and self._prev_upper_reward_per_trip:
                tap_signal = {}
                for tid, r_up in self._prev_upper_reward_per_trip.items():
                    tap_signal[tid] = beta * r_up

            for _ in range(self.updates_per_episode):
                train_metrics = self.lower_trainer.update(
                    self.replay_buffer, self.batch_size,
                    reward_scale=1.0, tap_signal=tap_signal)

        # Measurement projection update
        z = self.env.measurement_vector
        self.measurement_proj.update(z)
        r_upper = self.measurement_proj.compute_upper_reward(z)

        # TAP forward: compute augmented upper returns & update upper policy
        if ep > self.beta_schedule.warmup and training:
            n_dispatches = max(self.tap.num_upper_transitions, 1)
            upper_reward_per_trip = {}
            for trans in self.tap._upper_transitions:
                upper_reward_per_trip[trans['trip_id']] = r_upper / n_dispatches

            # Store for next episode's TAP reverse signal
            self._prev_upper_reward_per_trip = upper_reward_per_trip

            augmented = self.tap.compute_augmented_upper_returns(
                ep, upper_reward_per_trip)

            if augmented:
                self._update_upper_policy(augmented)

        train_time = time.time() - t1

        # ──── Logging ────
        avg_r = episode_reward / max(episode_steps, 1)
        q_mean = train_metrics.get('q_mean', 0)
        q_std = train_metrics.get('q_std', 0)

        self.history['ep_reward'].append(episode_reward)
        self.history['avg_reward'].append(avg_r)
        self.history['ep_cost'].append(episode_cost / max(episode_steps, 1))
        self.history['r_upper'].append(r_upper)
        self.history['avg_wait'].append(float(z[0]))
        self.history['peak_fleet'].append(float(z[1]))
        self.history['headway_cv'].append(float(z[2]))
        self.history['lambda'].append(self.lower_trainer.lambda_param)
        self.history['alpha'].append(self.lower_trainer.alpha)
        self.history['beta'].append(beta)
        self.history['q_mean'].append(q_mean)
        self.history['q_std'].append(q_std)
        self.history['theta_weights'].append(
            self.measurement_proj.get_reward_weights().tolist())

        return {
            'ep': ep, 'reward': episode_reward, 'avg_r': avg_r,
            'r_upper': r_upper, 'steps': episode_steps,
            'z': z.tolist(), 'lambda': self.lower_trainer.lambda_param,
            'beta': beta, 'stage': self.beta_schedule.stage_name(ep),
            'q_mean': q_mean, 'q_std': q_std,
            'env_time': env_time, 'train_time': train_time,
        }

    def _update_upper_policy(self, augmented_returns):
        """REINFORCE update for upper policy with TAP-augmented returns."""
        states, actions, returns = [], [], []
        for item in augmented_returns:
            if isinstance(item['state'], np.ndarray):
                states.append(item['state'])
                actions.append(item['action'])
                returns.append(item['augmented_return'])

        if not states:
            return

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        _, log_probs, _ = self.upper_policy.evaluate(states_t)
        loss = -(log_probs * returns_t).mean()

        self.upper_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.upper_policy.parameters(), 1.0)
        self.upper_optimizer.step()

    def train(self, total_episodes=None):
        """Full 3-stage training loop."""
        if total_episodes is None:
            total_episodes = self.cfg['training']['total_episodes']

        print(f"TransitDuet Full Training: {total_episodes} episodes")
        print(f"  Stage 1 warmup: ep 0-{self.beta_schedule.warmup}")
        print(f"  Stage 2 ramp:   ep {self.beta_schedule.warmup}-"
              f"{self.beta_schedule.warmup + self.beta_schedule.ramp}")
        print(f"  Stage 3 full:   ep {self.beta_schedule.warmup + self.beta_schedule.ramp}+")
        print(f"  Device: {self.device}, state_dim: {self.env.state_dim}")
        print("=" * 80)

        for ep in range(total_episodes):
            info = self.run_episode(ep, training=True)

            if ep % 5 == 0 or ep < 5:
                print(f"[Ep {ep:3d}] {info['stage']:15s} | "
                      f"R={info['avg_r']:6.3f} | "
                      f"R_up={info['r_upper']:7.2f} | "
                      f"wait={info['z'][0]:4.1f}m fleet={info['z'][1]:2.0f} "
                      f"cv={info['z'][2]:.3f} | "
                      f"Q={info['q_mean']:.1f}±{info['q_std']:.1f} | "
                      f"λ={info['lambda']:.3f} β={info['beta']:.2f} | "
                      f"{info['env_time']:.0f}+{info['train_time']:.0f}s")

            if (ep + 1) % self.cfg['training']['save_freq'] == 0:
                self._save_checkpoint(ep)

        self._save_checkpoint(total_episodes - 1)
        self._save_history()
        print("Training complete.")

    def _save_checkpoint(self, ep):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.lower_trainer.save(os.path.join(ckpt_dir, f'lower_ep{ep}.pt'))
        torch.save(self.upper_policy.state_dict(),
                   os.path.join(ckpt_dir, f'upper_ep{ep}.pt'))
        print(f"  [Checkpoint ep {ep}]")

    def _save_history(self):
        results = {}
        for key, values in self.history.items():
            try:
                results[key] = [float(x) if not isinstance(x, list) else x
                                for x in values]
            except (TypeError, ValueError):
                results[key] = [str(x) for x in values]
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            json.dump(results, f)


def main():
    parser = argparse.ArgumentParser(description='TransitDuet Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    config_path = os.path.join(str(SCRIPT_DIR), args.config)
    config = load_config(config_path)

    device = 'cpu'
    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'

    runner = TransitDuetRunner(config, device=device)
    episodes = args.episodes or config['training']['total_episodes']
    runner.train(total_episodes=episodes)


if __name__ == '__main__':
    main()
