"""Causal finite-horizon regularity value learning for lower holding actions.

The critic is trained after an episode from outcomes that occur after a
decision.  Future outcomes are labels only: deployment observations and the
categorical policy remain strictly causal.
"""

from __future__ import annotations

from collections import deque
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMultiStepValueReplay:
    """Build exact-H targets without crossing trip or episode boundaries."""

    def __init__(self, capacity, horizon_steps, discount=1.0, seed=None):
        self.capacity = int(capacity)
        self.horizon_steps = int(horizon_steps)
        self.discount = float(discount)
        if self.capacity <= 0:
            raise ValueError("multi-step replay capacity must be positive")
        if self.horizon_steps < 2:
            raise ValueError("multi-step horizon must be at least two")
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("multi-step discount must lie in (0, 1]")
        self.buffer = deque(maxlen=self.capacity)
        self.pending = {}
        self.targets_emitted = 0
        self.terminal_tails_discarded = 0
        self.episode_tails_discarded = 0
        self._rng = random if seed is None else random.Random(int(seed))
        weights = self.discount ** np.arange(
            self.horizon_steps, dtype=np.float64)
        self._target_weights = weights / weights.sum()

    def append(
        self,
        state,
        action,
        outcome_cost,
        baseline_cost,
        stream_id,
        done=False,
    ):
        state_value = np.asarray(state, dtype=np.float32).copy()
        action_value = float(np.asarray(action).reshape(-1)[0])
        outcome = float(outcome_cost)
        baseline = float(baseline_cost)
        if not np.isfinite(state_value).all():
            raise ValueError("multi-step state must be finite")
        if not all(np.isfinite(value) for value in (
                action_value, outcome, baseline)):
            raise ValueError("multi-step transition values must be finite")
        if not (0.0 <= outcome <= 1.0 and 0.0 <= baseline <= 1.0):
            raise ValueError("arrival costs must lie in [0, 1]")

        key = int(stream_id)
        queue = self.pending.setdefault(key, deque())
        queue.append((
            state_value,
            action_value,
            outcome,
            baseline,
            key,
        ))
        emitted = False
        if len(queue) >= self.horizon_steps:
            window = list(queue)[:self.horizon_steps]
            future_cost = float(np.dot(
                self._target_weights,
                np.asarray([row[2] for row in window], dtype=np.float64),
            ))
            first = queue.popleft()
            target_change = future_cost - float(first[3])
            self.buffer.append((
                first[0],
                np.asarray([first[1]], dtype=np.float32),
                float(target_change),
                first[4],
            ))
            self.targets_emitted += 1
            emitted = True

        if bool(done):
            tail = self.pending.pop(key, deque())
            self.terminal_tails_discarded += len(tail)
        return emitted

    def end_episode(self):
        discarded = sum(len(queue) for queue in self.pending.values())
        self.episode_tails_discarded += discarded
        self.pending.clear()
        return int(discarded)

    def sample(self, batch_size):
        batch = self._rng.sample(self.buffer, int(batch_size))
        states, actions, targets, stream_ids = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(targets, dtype=np.float32).reshape(-1, 1),
            np.asarray(stream_ids, dtype=np.int64),
        )

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "horizon_steps": self.horizon_steps,
            "discount": self.discount,
            "buffer": list(self.buffer),
            "pending": {
                int(key): list(queue) for key, queue in self.pending.items()
            },
            "targets_emitted": int(self.targets_emitted),
            "terminal_tails_discarded": int(
                self.terminal_tails_discarded),
            "episode_tails_discarded": int(self.episode_tails_discarded),
            "rng_state": (
                None if self._rng is random else self._rng.getstate()),
        }

    def load_state_dict(self, state):
        expected = (
            self.capacity,
            self.horizon_steps,
            self.discount,
        )
        observed = (
            int(state["capacity"]),
            int(state["horizon_steps"]),
            float(state["discount"]),
        )
        if observed != expected:
            raise ValueError("multi-step replay contract mismatch")
        self.buffer = deque(state["buffer"], maxlen=self.capacity)
        self.pending = {
            int(key): deque(copy.deepcopy(rows))
            for key, rows in state.get("pending", {}).items()
        }
        self.targets_emitted = int(state.get("targets_emitted", 0))
        self.terminal_tails_discarded = int(
            state.get("terminal_tails_discarded", 0))
        self.episode_tails_discarded = int(
            state.get("episode_tails_discarded", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            if self._rng is random:
                raise ValueError(
                    "cannot restore isolated replay RNG into global RNG")
            self._rng.setstate(rng_state)


class DiscreteValueEnsemble(nn.Module):
    """Vectorized ensemble with one output for each executable action."""

    def __init__(
        self,
        state_dim,
        action_candidates,
        hidden_dim=64,
        ensemble_size=5,
        n_layers=2,
    ):
        super().__init__()
        candidates = torch.as_tensor(
            action_candidates, dtype=torch.float32).reshape(-1)
        if candidates.numel() < 2 or torch.unique(candidates).numel() != (
                candidates.numel()):
            raise ValueError("multi-step actions must be unique")
        self.register_buffer("action_candidates", candidates)
        self.ensemble_size = int(ensemble_size)
        self.n_layers = int(n_layers)
        if self.ensemble_size < 2:
            raise ValueError("multi-step ensemble requires at least two members")
        if self.n_layers < 1:
            raise ValueError("multi-step critic requires a hidden layer")

        dims = (
            [int(state_dim)]
            + [int(hidden_dim)] * self.n_layers
            + [int(candidates.numel())]
        )
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            scale = 1.0 / np.sqrt(max(in_dim, 1))
            self.weights.append(nn.Parameter(torch.randn(
                self.ensemble_size, in_dim, out_dim) * scale))
            self.biases.append(nn.Parameter(torch.zeros(
                self.ensemble_size, 1, out_dim)))

    def all_values(self, state):
        x = state.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for index, (weight, bias) in enumerate(
                zip(self.weights, self.biases)):
            x = torch.bmm(x, weight) + bias
            if index < self.n_layers:
                x = F.relu(x)
        return x

    def action_indices(self, action):
        distances = (
            action.reshape(-1, 1)
            - self.action_candidates.reshape(1, -1)
        ).abs()
        minimum, indices = distances.min(dim=-1)
        if torch.any(minimum > 1e-6):
            raise ValueError(
                "multi-step critic received action outside its library")
        return indices

    def forward(self, state, action):
        values = self.all_values(state)
        indices = self.action_indices(action)
        gather = indices.view(1, -1, 1).expand(
            self.ensemble_size, -1, 1)
        return values.gather(-1, gather).squeeze(-1)


class CausalMultiStepRegularityObjective:
    """Learn downstream arrival-cost change and constrain positive regret."""

    MODE = "discounted_future_arrival_cost_change_v1"

    def __init__(
        self,
        state_dim,
        action_candidates,
        config,
        replay_seed=None,
        device="cpu",
    ):
        cfg = dict(config or {})
        if not bool(cfg.get("enable", False)):
            raise ValueError("multi-step value objective must be enabled")
        mode = str(cfg.get("mode", self.MODE)).strip().lower()
        if mode != self.MODE:
            raise ValueError("unknown multi-step regularity value mode")
        self.device = device
        self.horizon_steps = int(cfg.get("horizon_steps", 4))
        self.discount = float(cfg.get("discount", 1.0))
        self.ucb_beta = float(cfg.get("ucb_beta", 0.0))
        self.min_replay_size = int(cfg.get("min_replay_size", 512))
        self.min_critic_updates = int(cfg.get("min_critic_updates", 30))
        self.replay_capacity = int(cfg.get("replay_capacity", 500_000))
        self.hidden_dim = int(cfg.get("hidden_dim", 64))
        self.ensemble_size = int(cfg.get("ensemble_size", 5))
        self.n_layers = int(cfg.get("n_layers", 2))
        self.learning_rate = float(cfg.get("lr", 3e-4))
        self.weight_decay = float(cfg.get("weight_decay", 1e-5))
        if self.horizon_steps < 2:
            raise ValueError("multi-step horizon must be at least two")
        if not (0.0 < self.discount <= 1.0):
            raise ValueError("multi-step discount must lie in (0, 1]")
        if not np.isfinite(self.ucb_beta) or self.ucb_beta < 0.0:
            raise ValueError("multi-step UCB beta must be non-negative")
        if self.min_replay_size <= 0 or self.min_critic_updates <= 0:
            raise ValueError("multi-step warm-up thresholds must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid multi-step critic optimizer")

        actions = np.asarray(action_candidates, dtype=np.float32).reshape(-1)
        if actions.size < 2 or np.unique(actions).size != actions.size:
            raise ValueError("multi-step action library must be unique")
        self.action_candidates = actions
        self.zero_action_index = int(np.argmin(np.abs(actions)))
        if not np.isclose(actions[self.zero_action_index], 0.0):
            raise ValueError("multi-step value regret requires zero holding")

        self.critic = DiscreteValueEnsemble(
            state_dim=state_dim,
            action_candidates=actions,
            hidden_dim=self.hidden_dim,
            ensemble_size=self.ensemble_size,
            n_layers=self.n_layers,
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.replay = CausalMultiStepValueReplay(
            capacity=self.replay_capacity,
            horizon_steps=self.horizon_steps,
            discount=self.discount,
            seed=replay_seed,
        )
        self.critic_updates = 0
        self.last_update_metrics = self._empty_metrics()
        self.last_policy_metrics = self._empty_policy_metrics()
        self._deployment_training_evidence = None
        self.contract = {
            "enabled": True,
            "mode": mode,
            "target": "future_arrival_cost_mean_minus_decision_arrival_cost",
            "boundary": "same_trip_exact_horizon_no_episode_crossing",
            "deployment_inputs": "decision_time_state_only",
            "horizon_steps": self.horizon_steps,
            "discount": self.discount,
            "ucb_beta": self.ucb_beta,
            "min_replay_size": self.min_replay_size,
            "min_critic_updates": self.min_critic_updates,
            "replay_capacity": self.replay_capacity,
            "hidden_dim": self.hidden_dim,
            "ensemble_size": self.ensemble_size,
            "n_layers": self.n_layers,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "action_candidates": [float(value) for value in actions],
        }

    @property
    def ready(self):
        if self._deployment_training_evidence is not None:
            return bool(self._deployment_training_evidence["ready"])
        return (
            self.critic_updates >= self.min_critic_updates
            and len(self.replay) >= self.min_replay_size
        )

    @staticmethod
    def _empty_metrics():
        return {
            "multistep_value_critic_loss": 0.0,
            "multistep_value_target_mean": 0.0,
            "multistep_value_target_std": 0.0,
            "multistep_value_prediction_mean": 0.0,
            "multistep_value_prediction_std": 0.0,
            "multistep_value_grad_norm": 0.0,
        }

    @staticmethod
    def _empty_policy_metrics():
        return {
            "multistep_value_action_span_mean": 0.0,
            "multistep_value_advantage_mean": 0.0,
            "multistep_value_advantage_std_mean": 0.0,
            "multistep_value_positive_regret_mean": 0.0,
            "multistep_value_positive_regret_max": 0.0,
        }

    def observe(
        self,
        state,
        action,
        outcome_cost,
        baseline_cost,
        stream_id,
        done=False,
    ):
        action_value = float(np.asarray(action).reshape(-1)[0])
        if not np.any(np.isclose(
                self.action_candidates, action_value, atol=1e-6, rtol=0.0)):
            raise ValueError("observed holding action is outside seven bins")
        return self.replay.append(
            state=state,
            action=action_value,
            outcome_cost=outcome_cost,
            baseline_cost=baseline_cost,
            stream_id=stream_id,
            done=done,
        )

    def end_episode(self):
        return self.replay.end_episode()

    def update(self, batch_size, weight_fn=None):
        batch_size = int(batch_size)
        if len(self.replay) < max(self.min_replay_size, batch_size):
            self.last_update_metrics = self._empty_metrics()
            return dict(self.last_update_metrics)
        state, action, target, stream_ids = self.replay.sample(batch_size)
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(
            action, dtype=torch.float32, device=self.device)
        target_t = torch.as_tensor(
            target, dtype=torch.float32, device=self.device).reshape(-1)
        if weight_fn is None:
            weights = torch.ones_like(target_t)
        else:
            values = np.asarray(weight_fn(stream_ids), dtype=np.float32)
            if values.shape != (batch_size,):
                raise ValueError("multi-step importance weights have wrong shape")
            weights = torch.as_tensor(values, device=self.device)

        predictions = self.critic(state_t, action_t)
        loss = (
            (predictions - target_t.unsqueeze(0)).pow(2)
            * weights.unsqueeze(0)
        ).mean()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 20.0)
        self.optimizer.step()
        self.critic_updates += 1
        with torch.no_grad():
            prediction_mean = predictions.mean(dim=0)
            metrics = {
                "multistep_value_critic_loss": float(loss.item()),
                "multistep_value_target_mean": float(target_t.mean().item()),
                "multistep_value_target_std": float(
                    target_t.std(unbiased=False).item()),
                "multistep_value_prediction_mean": float(
                    prediction_mean.mean().item()),
                "multistep_value_prediction_std": float(
                    prediction_mean.std(unbiased=False).item()),
                "multistep_value_grad_norm": float(grad_norm),
            }
        self.last_update_metrics = metrics
        return dict(metrics)

    def policy_cost(self, state, action_probs, valid, cost_cap):
        if not self.ready:
            action_costs = torch.zeros_like(action_probs)
            self.last_policy_metrics = self._empty_policy_metrics()
            return (
                torch.zeros(state.shape[0], device=state.device),
                torch.zeros_like(valid),
                action_costs,
            )
        with torch.no_grad():
            values = self.critic.all_values(state)
            zero = values[..., self.zero_action_index:self.zero_action_index + 1]
            advantages = values - zero
            mean_advantage = advantages.mean(dim=0)
            std_advantage = advantages.std(dim=0)
            robust_advantage = (
                mean_advantage + self.ucb_beta * std_advantage)
            action_costs = robust_advantage.clamp(
                min=0.0, max=float(cost_cap))
            span = values.mean(dim=0).max(dim=-1).values - (
                values.mean(dim=0).min(dim=-1).values)
            self.last_policy_metrics = {
                "multistep_value_action_span_mean": float(span.mean().item()),
                "multistep_value_advantage_mean": float(
                    mean_advantage.mean().item()),
                "multistep_value_advantage_std_mean": float(
                    std_advantage.mean().item()),
                "multistep_value_positive_regret_mean": float(
                    action_costs.mean().item()),
                "multistep_value_positive_regret_max": float(
                    action_costs.max().item()),
            }
        expected_cost = (action_probs * action_costs).sum(dim=-1)
        return expected_cost, valid, action_costs

    def telemetry(self):
        evidence = self._deployment_training_evidence
        replay_size = (
            evidence["replay_size"] if evidence is not None
            else len(self.replay))
        targets_emitted = (
            evidence["targets_emitted"] if evidence is not None
            else self.replay.targets_emitted)
        terminal_tails_discarded = (
            evidence["terminal_tails_discarded"]
            if evidence is not None
            else self.replay.terminal_tails_discarded)
        episode_tails_discarded = (
            evidence["episode_tails_discarded"]
            if evidence is not None
            else self.replay.episode_tails_discarded)
        return {
            "multistep_value_enabled": 1.0,
            "multistep_value_horizon_steps": float(self.horizon_steps),
            "multistep_value_discount": float(self.discount),
            "multistep_value_ucb_beta": float(self.ucb_beta),
            "multistep_value_replay_size": float(replay_size),
            "multistep_value_targets_emitted": float(targets_emitted),
            "multistep_value_terminal_tails_discarded": float(
                terminal_tails_discarded),
            "multistep_value_episode_tails_discarded": float(
                episode_tails_discarded),
            "multistep_value_critic_updates": float(self.critic_updates),
            "multistep_value_ready": float(self.ready),
            **self.last_update_metrics,
            **self.last_policy_metrics,
        }

    def training_state_dict(self):
        return {
            "format": "freqduet-causal-multistep-value-training-v1",
            "contract": copy.deepcopy(self.contract),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "critic_updates": int(self.critic_updates),
            "last_update_metrics": copy.deepcopy(self.last_update_metrics),
            "last_policy_metrics": copy.deepcopy(self.last_policy_metrics),
        }

    def load_training_state_dict(self, state):
        if state.get("format") != (
                "freqduet-causal-multistep-value-training-v1"):
            raise ValueError("invalid multi-step training checkpoint")
        if state.get("contract") != self.contract:
            raise ValueError("multi-step value contract mismatch")
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.critic_updates = int(state["critic_updates"])
        self._deployment_training_evidence = None
        self.last_update_metrics = copy.deepcopy(
            state.get("last_update_metrics", self._empty_metrics()))
        self.last_policy_metrics = copy.deepcopy(
            state.get("last_policy_metrics", self._empty_policy_metrics()))

    def deployment_state_dict(self):
        training_evidence = copy.deepcopy(
            self._deployment_training_evidence)
        if training_evidence is None:
            training_evidence = {
                "ready": bool(self.ready),
                "replay_size": int(len(self.replay)),
                "targets_emitted": int(self.replay.targets_emitted),
                "terminal_tails_discarded": int(
                    self.replay.terminal_tails_discarded),
                "episode_tails_discarded": int(
                    self.replay.episode_tails_discarded),
            }
        return {
            "format": "freqduet-causal-multistep-value-deployment-v2",
            "contract": copy.deepcopy(self.contract),
            "critic": self.critic.state_dict(),
            "critic_updates": int(self.critic_updates),
            "training_evidence": training_evidence,
        }

    def load_deployment_state_dict(self, state):
        if state.get("format") != (
                "freqduet-causal-multistep-value-deployment-v2"):
            raise ValueError("invalid multi-step deployment checkpoint")
        if state.get("contract") != self.contract:
            raise ValueError("multi-step value contract mismatch")
        evidence = state.get("training_evidence")
        required_evidence = {
            "ready", "replay_size", "targets_emitted",
            "terminal_tails_discarded", "episode_tails_discarded",
        }
        if not isinstance(evidence, dict) or set(evidence) != required_evidence:
            raise ValueError(
                "multi-step deployment checkpoint lacks training evidence")
        counters = {
            key: int(evidence[key]) for key in required_evidence - {"ready"}
        }
        if any(value < 0 for value in counters.values()):
            raise ValueError("multi-step deployment counters must be non-negative")
        critic_updates = int(state["critic_updates"])
        expected_ready = bool(
            critic_updates >= self.min_critic_updates
            and counters["replay_size"] >= self.min_replay_size)
        if bool(evidence["ready"]) != expected_ready:
            raise ValueError("multi-step deployment readiness is inconsistent")
        self.critic.load_state_dict(state["critic"])
        self.critic_updates = critic_updates
        self._deployment_training_evidence = {
            "ready": expected_ready,
            **counters,
        }
