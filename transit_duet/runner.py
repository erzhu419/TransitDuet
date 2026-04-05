"""
runner.py
=========
TransitDuet main training loop.

Orchestrates:
  - env_bus (single-agent, event-driven)
  - πU: UpperPolicy (dispatch-level timetable decisions)
  - πL: DSACLagrangianTrainer (station-level holding control)
  - TAP: Temporal Advantage Propagation (cross-layer coupling)
  - MeasurementProjection: ApproPO θ-OGD for fleet constraint

Usage:
    python runner.py [--config config.yaml] [--episodes 300]
"""

import os
import sys
import argparse
import json
import yaml
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

# Ensure transit_duet package is importable
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from env.sim import env_bus
from upper.upper_policy import UpperPolicy
from upper.measurement_proj import MeasurementProjection
from lower.dsac_lagrangian import DSACLagrangianTrainer
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

        state_dim = self.env.state_dim  # 8 + routes//2

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

        # Lower policy (DSAC-Lagrangian, parameter-sharing single-agent)
        self.replay_buffer = CostReplayBuffer(config['training']['replay_buffer_size'])
        self.lower_trainer = DSACLagrangianTrainer(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=config['lower']['hidden_dim'],
            action_range=config['lower']['action_range'],
            cost_limit=config['lower']['cost_limit'],
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

        # Logging
        self.log_dir = os.path.join(str(SCRIPT_DIR), 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = defaultdict(list)

    def _upper_callback(self, s_upper, trip):
        """
        Called by env.step() at each dispatch event.
        Returns target_headway for this trip.
        """
        action = self.upper_policy.get_action(s_upper, deterministic=False)
        trip_id = trip.launch_turn

        # Record for TAP
        self.tap.record_upper_transition(s_upper, action, trip_id)

        # Determine which headway param to use based on time-of-day
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
        done = False
        action_dict = {k: None for k in range(self.env.max_agent_num)}

        episode_reward = 0.0
        episode_cost = 0.0
        episode_steps = 0
        training_steps = 0

        while not done:
            # Single-agent control loop (same pattern as dsac_lag_bus.py)
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
                        trip_id = int(state_dict[key][0][0])  # bus_id as proxy

                        if training:
                            self.replay_buffer.push(
                                state, action_dict[key], reward, cost,
                                next_state, done, trip_id)
                            # TAP: record lower reward for this trip
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

            # Lower policy training
            if (training and len(self.replay_buffer) > self.cfg['lower']['batch_size']
                    and episode_steps % self.cfg['lower']['training_freq'] == 0):
                # Build TAP signal for lower training
                tap_signal = None
                if beta > 0 and ep > self.beta_schedule.warmup:
                    # Simple: use per-trip upper reward from last episode (bootstrap)
                    tap_signal = {}  # Will be populated in future episodes

                metrics = self.lower_trainer.update(
                    self.replay_buffer,
                    self.cfg['lower']['batch_size'],
                    reward_scale=self.cfg['lower']['reward_scale'],
                    tap_signal=tap_signal,
                )
                training_steps += 1

        # ---- Episode end ----

        # ApproPO measurement update
        z = self.env.measurement_vector
        self.measurement_proj.update(z)

        # Upper reward
        r_upper = self.measurement_proj.compute_upper_reward(z)

        # TAP: compute augmented upper returns
        if ep > self.beta_schedule.warmup:
            # Distribute upper reward equally across all dispatch events
            n_dispatches = max(self.tap.num_upper_transitions, 1)
            upper_reward_per_trip = {}
            for trans in self.tap._upper_transitions:
                upper_reward_per_trip[trans['trip_id']] = r_upper / n_dispatches

            augmented = self.tap.compute_augmented_upper_returns(ep, upper_reward_per_trip)

            # Simple REINFORCE update for upper policy
            if training and augmented:
                self._update_upper_policy(augmented)

        # Logging
        self.history['episode_reward'].append(episode_reward)
        self.history['episode_cost'].append(episode_cost)
        self.history['r_upper'].append(r_upper)
        self.history['measurement_wait'].append(float(z[0]))
        self.history['measurement_fleet'].append(float(z[1]))
        self.history['measurement_bunching'].append(float(z[2]))
        self.history['lambda'].append(self.lower_trainer.lambda_param)
        self.history['beta'].append(beta)
        self.history['theta_weights'].append(
            self.measurement_proj.get_reward_weights().tolist())

        return {
            'ep': ep,
            'reward': episode_reward,
            'cost': episode_cost,
            'r_upper': r_upper,
            'steps': episode_steps,
            'z': z.tolist(),
            'lambda': self.lower_trainer.lambda_param,
            'beta': beta,
            'stage': self.beta_schedule.stage_name(ep),
        }

    def _update_upper_policy(self, augmented_returns):
        """Simple REINFORCE-style update for upper policy."""
        if not augmented_returns:
            return

        states = []
        actions = []
        returns = []
        for item in augmented_returns:
            if isinstance(item['state'], np.ndarray):
                states.append(item['state'])
                actions.append(item['action'])
                returns.append(item['augmented_return'])

        if not states:
            return

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        if len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Get log probabilities
        _, log_probs, _ = self.upper_policy.evaluate(states_t)

        # REINFORCE loss
        loss = -(log_probs * returns_t).mean()

        self.upper_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.upper_policy.parameters(), 1.0)
        self.upper_optimizer.step()

    def train(self, total_episodes=None):
        """Full training loop."""
        if total_episodes is None:
            total_episodes = self.cfg['training']['total_episodes']

        print(f"TransitDuet Training: {total_episodes} episodes")
        print(f"  Warmup: {self.beta_schedule.warmup} eps")
        print(f"  Ramp: {self.beta_schedule.ramp} eps")
        print(f"  Device: {self.device}")
        print(f"  State dim: {self.env.state_dim}")
        print("=" * 60)

        for ep in range(total_episodes):
            info = self.run_episode(ep, training=True)

            # Print progress
            if ep % 5 == 0 or ep < 5:
                print(f"[Ep {ep:3d}] {info['stage']:15s} | "
                      f"R={info['reward']:8.1f} | "
                      f"C={info['cost']:6.3f} | "
                      f"R_up={info['r_upper']:7.2f} | "
                      f"z=[{info['z'][0]:.1f}, {info['z'][1]:.0f}, {info['z'][2]:.2f}] | "
                      f"λ={info['lambda']:.3f} β={info['beta']:.2f}")

            # Save checkpoint
            if (ep + 1) % self.cfg['training']['save_freq'] == 0:
                self._save_checkpoint(ep)

        # Final save
        self._save_checkpoint(total_episodes - 1)
        self._save_history()
        print("Training complete.")

    def _save_checkpoint(self, ep):
        ckpt_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        self.lower_trainer.save(os.path.join(ckpt_dir, f'lower_ep{ep}.pt'))
        torch.save(self.upper_policy.state_dict(),
                    os.path.join(ckpt_dir, f'upper_ep{ep}.pt'))
        print(f"  Checkpoint saved at ep {ep}")

    def _save_history(self):
        for key, values in self.history.items():
            np.save(os.path.join(self.log_dir, f'{key}.npy'),
                    np.array(values, dtype=object))


def main():
    parser = argparse.ArgumentParser(description='TransitDuet Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override total_episodes from config')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
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
