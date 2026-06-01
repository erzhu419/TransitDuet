"""Native TransitDuet bridge for the shared Freq-HRL PPO core.

This module does not modify the copied TransitDuet runner.  It provides a
small adapter that instantiates the native runner, reads its real state/action
contract, and maps the domain-agnostic Gaussian PPO actions into native
timetable and holding actions.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig, TrajectoryBatch


TRANSIT_HRL_ROOT = Path(__file__).resolve().parents[3]
TRANSIT_DUET_ROOT = TRANSIT_HRL_ROOT / "freq_transitduet"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _array(value: Any, dtype: Any = np.float32) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _state_key(value: Any) -> tuple[float, ...]:
    arr = _array(value, dtype=np.float64)
    return tuple(np.round(arr, 6).tolist())


def _set_reproducible_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        return


@dataclass
class NativeTransitContract:
    upper_state_dim: int
    lower_state_dim: int
    upper_action_dim: int
    upper_model_action_dim: int
    lower_action_dim: int
    upper_action_low: list[float]
    upper_action_high: list[float]
    lower_action_range_s: float
    lower_action_bins: list[float]
    frequency_method: str
    timetable_planner: bool
    terminal_dispatch: bool
    promotion_replan: bool
    learned_promotion_gate: bool = False
    shared_core: str = "freq_hrl.rl.DualActorCriticPPO"

    def as_dict(self) -> dict[str, Any]:
        return {
            "upper_state_dim": int(self.upper_state_dim),
            "lower_state_dim": int(self.lower_state_dim),
            "upper_action_dim": int(self.upper_action_dim),
            "upper_model_action_dim": int(self.upper_model_action_dim),
            "lower_action_dim": int(self.lower_action_dim),
            "upper_action_low": list(self.upper_action_low),
            "upper_action_high": list(self.upper_action_high),
            "lower_action_range_s": float(self.lower_action_range_s),
            "lower_action_bins": list(self.lower_action_bins),
            "frequency_method": str(self.frequency_method),
            "timetable_planner": bool(self.timetable_planner),
            "terminal_dispatch": bool(self.terminal_dispatch),
            "promotion_replan": bool(self.promotion_replan),
            "learned_promotion_gate": bool(self.learned_promotion_gate),
            "shared_core": self.shared_core,
        }


class NativeTransitPPOBridge:
    """Map shared PPO actions into native TransitDuet action spaces."""

    def __init__(
        self,
        contract: NativeTransitContract,
        model: DualActorCriticPPO | None = None,
        *,
        hidden_dim: int = 0,
        init_log_std: float = -2.0,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        initialize_gate_prior: bool = True,
        native_policy_init_seed: int | None = None,
    ) -> None:
        self.contract = contract
        self.upper_action_low = _array(contract.upper_action_low, dtype=np.float64)
        self.upper_action_high = _array(contract.upper_action_high, dtype=np.float64)
        if self.upper_action_low.size != int(contract.upper_action_dim):
            raise ValueError("upper_action_low must match upper_action_dim")
        if self.upper_action_high.size != int(contract.upper_action_dim):
            raise ValueError("upper_action_high must match upper_action_dim")
        self.lower_action_bins = _array(contract.lower_action_bins, dtype=np.float64)
        if model is None and native_policy_init_seed is not None:
            _set_reproducible_seed(int(native_policy_init_seed))
        self.model = model or DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=int(contract.upper_state_dim),
            lower_state_dim=int(contract.lower_state_dim),
            upper_action_dim=int(contract.upper_model_action_dim),
            lower_action_dim=int(contract.lower_action_dim),
            hidden_dim=int(hidden_dim),
            init_log_std=float(init_log_std),
            learning_rate=float(learning_rate),
            device=str(device),
        ))
        if bool(contract.learned_promotion_gate) and native_policy_init_seed is not None:
            self.align_native_policy_from_seed(
                int(native_policy_init_seed),
                hidden_dim=int(hidden_dim),
                init_log_std=float(init_log_std),
                learning_rate=float(learning_rate),
                device=str(device),
            )
        if bool(contract.learned_promotion_gate) and initialize_gate_prior:
            self.initialize_promotion_gate_prior()

    @classmethod
    def from_runner(
        cls,
        runner: Any,
        *,
        hidden_dim: int = 0,
        init_log_std: float = -2.0,
        learning_rate: float = 3e-4,
        device: str = "cpu",
        learned_promotion_gate: bool = False,
        initialize_gate_prior: bool = True,
        native_policy_init_seed: int | None = None,
    ) -> "NativeTransitPPOBridge":
        cfg = getattr(runner, "cfg", {})
        lower_cfg = cfg.get("lower", {}) if isinstance(cfg, dict) else {}
        freq_cfg = cfg.get("frequency", {}) if isinstance(cfg, dict) else {}
        planner_cfg = cfg.get("upper", {}).get("timetable_planner", {}) if isinstance(cfg, dict) else {}
        lower_action_range = float(lower_cfg.get("action_range", 60.0))
        lower_bins = getattr(runner, "lower_action_bins", None)
        contract = NativeTransitContract(
            upper_state_dim=int(runner.upper_state_dim),
            lower_state_dim=int(runner.lower_state_dim),
            upper_action_dim=int(runner.upper_action_dim),
            upper_model_action_dim=int(runner.upper_action_dim) + (1 if bool(learned_promotion_gate) else 0),
            lower_action_dim=1,
            upper_action_low=_array(runner.upper_action_low).astype(float).tolist(),
            upper_action_high=_array(runner.upper_action_high).astype(float).tolist(),
            lower_action_range_s=lower_action_range,
            lower_action_bins=(
                _array(lower_bins).astype(float).tolist()
                if lower_bins is not None else []
            ),
            frequency_method=str(freq_cfg.get("method", "unknown")),
            timetable_planner=bool(getattr(runner, "timetable_planner", None) is not None),
            terminal_dispatch=bool(getattr(runner, "timetable_terminal_dispatch", False)),
            promotion_replan=bool(planner_cfg.get(
                "promotion_replan",
                getattr(runner, "timetable_promotion_replan", False),
            )),
            learned_promotion_gate=bool(learned_promotion_gate),
        )
        return cls(
            contract,
            hidden_dim=hidden_dim,
            init_log_std=init_log_std,
            learning_rate=learning_rate,
            device=device,
            initialize_gate_prior=initialize_gate_prior,
            native_policy_init_seed=native_policy_init_seed,
        )

    def align_native_policy_from_seed(
        self,
        seed: int,
        *,
        hidden_dim: int,
        init_log_std: float,
        learning_rate: float,
        device: str,
    ) -> None:
        """Keep the native action policy identical when appending a gate head."""
        if not bool(self.contract.learned_promotion_gate):
            return
        _set_reproducible_seed(int(seed))
        baseline = DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=int(self.contract.upper_state_dim),
            lower_state_dim=int(self.contract.lower_state_dim),
            upper_action_dim=int(self.contract.upper_action_dim),
            lower_action_dim=int(self.contract.lower_action_dim),
            hidden_dim=int(hidden_dim),
            init_log_std=float(init_log_std),
            learning_rate=float(learning_rate),
            device=str(device),
        ))
        try:
            import torch

            with torch.no_grad():
                src = baseline.upper_actor.net[-1]
                dst = self.model.upper_actor.net[-1]
                dst.weight[:int(self.contract.upper_action_dim)].copy_(src.weight)
                dst.bias[:int(self.contract.upper_action_dim)].copy_(src.bias)
                self.model.upper_actor.log_std[:int(self.contract.upper_action_dim)].copy_(
                    baseline.upper_actor.log_std
                )
            self.model.lower_actor.load_state_dict(baseline.lower_actor.state_dict())
            self.model.upper_value.load_state_dict(baseline.upper_value.state_dict())
            self.model.lower_value.load_state_dict(baseline.lower_value.state_dict())
        except Exception:
            return

    def initialize_promotion_gate_prior(self) -> None:
        """Seed the optional native gate head from causal promotion features.

        Native Transit upper states append promotion features as
        `[flag, strength, age]` when enabled.  The prior keeps the gate closed
        without a promotion signal and opens it for persistent/high-strength
        shocks; PPO can still update the row during native episode training.
        """
        if not bool(self.contract.learned_promotion_gate):
            return
        try:
            linear = self.model.upper_actor.net[-1]
            if not hasattr(linear, "weight") or not hasattr(linear, "bias"):
                return
            import torch

            gate_row = int(self.contract.upper_model_action_dim) - 1
            with torch.no_grad():
                linear.weight[gate_row].zero_()
                linear.bias[gate_row] = -2.0
                if int(self.contract.upper_state_dim) >= 3:
                    linear.weight[gate_row, -3] = 2.0
                    linear.weight[gate_row, -2] = 3.0
                    linear.weight[gate_row, -1] = 1.0
        except Exception:
            return

    def upper_latent_to_native(self, latent_action: Any) -> np.ndarray:
        latent = _array(latent_action, dtype=np.float64)
        if latent.size != int(self.contract.upper_model_action_dim):
            raise ValueError("upper latent action has the wrong dimension")
        latent = latent[:int(self.contract.upper_action_dim)]
        weight = 0.5 * (np.tanh(latent) + 1.0)
        return (
            self.upper_action_low
            + weight * (self.upper_action_high - self.upper_action_low)
        ).astype(np.float32)

    def upper_native_to_latent(self, native_action: Any, gate_latent: float = 0.0) -> np.ndarray:
        native = _array(native_action, dtype=np.float64)
        if native.size != int(self.contract.upper_action_dim):
            raise ValueError("upper native action has the wrong dimension")
        denom = np.maximum(self.upper_action_high - self.upper_action_low, 1e-9)
        weight = np.clip((native - self.upper_action_low) / denom, 1e-6, 1.0 - 1e-6)
        latent = np.arctanh(np.clip(2.0 * weight - 1.0, -1.0 + 1e-6, 1.0 - 1e-6))
        if int(self.contract.upper_model_action_dim) > int(self.contract.upper_action_dim):
            latent = np.concatenate([
                latent,
                np.asarray([float(gate_latent)], dtype=np.float64),
            ])
        return latent.astype(np.float32)

    def promotion_gate_value(self, latent_action: Any) -> float:
        if not bool(self.contract.learned_promotion_gate):
            return 0.0
        latent = _array(latent_action, dtype=np.float64)
        if latent.size != int(self.contract.upper_model_action_dim):
            raise ValueError("upper latent action has the wrong dimension")
        return float(_sigmoid(latent[-1:])[0])

    def lower_latent_to_native(self, latent_action: Any) -> np.ndarray:
        latent = _array(latent_action, dtype=np.float64)
        if latent.size < 1:
            raise ValueError("lower latent action must have at least one dimension")
        value = float(self.contract.lower_action_range_s) * float(_sigmoid(latent[:1])[0])
        if self.lower_action_bins.size:
            idx = int(np.argmin(np.abs(self.lower_action_bins - value)))
            value = float(self.lower_action_bins[idx])
        value = float(np.clip(value, 0.0, float(self.contract.lower_action_range_s)))
        return np.asarray([value], dtype=np.float32)

    def act_upper_native(self, upper_state: Any, sample: bool = False) -> dict[str, Any]:
        out = self.model.act_upper(_array(upper_state), sample=sample)
        latent = _array(out["action"], dtype=np.float64)
        native = self.upper_latent_to_native(latent)
        return {
            "native_action": native,
            "latent_action": latent.astype(np.float32),
            "promotion_gate_value": self.promotion_gate_value(latent),
            "logp": float(out["logp"]),
            "value": float(out["value"]),
        }

    def act_lower_native(self, lower_state: Any, sample: bool = False) -> dict[str, Any]:
        out = self.model.act_lower(_array(lower_state), sample=sample)
        latent = _array(out["action"], dtype=np.float64)
        native = self.lower_latent_to_native(latent)
        return {
            "native_action": native,
            "latent_action": latent.astype(np.float32),
            "logp": float(out["logp"]),
            "value": float(out["value"]),
        }

    def contract_dict(self) -> dict[str, Any]:
        return self.contract.as_dict()


class _SharedPPOPolicyProxy:
    def __init__(self, bridge: NativeTransitPPOBridge, level: str) -> None:
        self.bridge = bridge
        self.level = str(level)
        self.pending: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        self.preselected: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        self.last_upper: dict[str, Any] | None = None
        self.decisions = 0
        self.gate_evaluations = 0
        self.gate_replans = 0
        self.gate_values: list[float] = []

    def _remember(self, state: np.ndarray, info: dict[str, Any]) -> None:
        key = _state_key(state)
        self.pending.setdefault(key, []).append(info)
        if self.level == "upper":
            self.last_upper = info
        self.decisions += 1

    def pop(self, state: Any) -> dict[str, Any] | None:
        key = _state_key(state)
        values = self.pending.get(key)
        if not values:
            return None
        info = values.pop(0)
        if not values:
            self.pending.pop(key, None)
        return info

    def _act_info(self, state_arr: np.ndarray, sample: bool) -> dict[str, Any]:
        if self.level == "upper":
            out = self.bridge.act_upper_native(state_arr, sample=sample)
        else:
            out = self.bridge.act_lower_native(state_arr, sample=sample)
        return {
            "state": state_arr.astype(np.float32).copy(),
            "latent_action": _array(out["latent_action"]).astype(np.float32),
            "native_action": _array(out["native_action"]).astype(np.float32),
            "promotion_gate_value": float(out.get("promotion_gate_value", 0.0)),
            "logp": float(out["logp"]),
            "value": float(out["value"]),
        }

    def evaluate_promotion_gate(
        self,
        state: Any,
        *,
        threshold: float,
        sample: bool,
        preselect_action: bool = False,
        native_action_override: Any | None = None,
        native_action_blend: float = 0.0,
    ) -> bool:
        if self.level != "upper" or not bool(self.bridge.contract.learned_promotion_gate):
            return False
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state_arr = _array(state)
        info = self._act_info(state_arr, sample=sample)
        gate_value = float(info.get("promotion_gate_value", 0.0))
        self.gate_evaluations += 1
        self.gate_values.append(gate_value)
        promote = gate_value >= float(threshold)
        if promote:
            if bool(preselect_action):
                if native_action_override is not None:
                    native_override = _array(native_action_override, dtype=np.float64)
                    blend = float(np.clip(native_action_blend, 0.0, 1.0))
                    native = (
                        (1.0 - blend) * native_override
                        + blend * _array(info["native_action"], dtype=np.float64)
                    )
                    native = np.clip(native, self.bridge.upper_action_low, self.bridge.upper_action_high)
                    gate_latent = float(_array(info["latent_action"], dtype=np.float64)[-1])
                    info = dict(info)
                    info["native_action"] = native.astype(np.float32)
                    info["latent_action"] = self.bridge.upper_native_to_latent(native, gate_latent=gate_latent)
                key = _state_key(state_arr)
                self.preselected.setdefault(key, []).append(info)
            self.gate_replans += 1
        return bool(promote)

    def get_action(self, state: Any, deterministic: bool = False) -> np.ndarray:
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state_arr = _array(state)
        sample = not bool(deterministic)
        key = _state_key(state_arr)
        preselected = self.preselected.get(key)
        if preselected:
            info = preselected.pop(0)
            if not preselected:
                self.preselected.pop(key, None)
        else:
            info = self._act_info(state_arr, sample=sample)
        self._remember(state_arr, info)
        return info["native_action"].copy()

    def log_prob(self, state: Any, action: Any) -> float:
        return 0.0


class _NativeUpperReplayCollector:
    def __init__(self, upper_proxy: _SharedPPOPolicyProxy) -> None:
        self.upper_proxy = upper_proxy
        self.rows: list[dict[str, Any]] = []

    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        info = self.upper_proxy.pop(state)
        self.rows.append({
            "state": _array(state).astype(np.float32),
            "native_action": _array(action).astype(np.float32),
            "latent_action": (
                _array(info["latent_action"]).astype(np.float32)
                if info is not None else np.zeros(
                    int(self.upper_proxy.bridge.contract.upper_model_action_dim),
                    dtype=np.float32,
                )
            ),
            "reward": float(reward),
            "next_state": _array(next_state).astype(np.float32),
            "done": float(done),
        })

    def __len__(self) -> int:
        return len(self.rows)


class _NativeLowerReplayCollector:
    def __init__(
        self,
        lower_proxy: _SharedPPOPolicyProxy,
        upper_proxy: _SharedPPOPolicyProxy,
        contract: NativeTransitContract,
    ) -> None:
        self.lower_proxy = lower_proxy
        self.upper_proxy = upper_proxy
        self.contract = contract
        self.rows: list[dict[str, Any]] = []

    def push(
        self,
        state: Any,
        action: Any,
        reward: float,
        cost: float,
        next_state: Any,
        done: bool,
        trip_id: int = 0,
    ) -> None:
        lower_info = self.lower_proxy.pop(state)
        upper_info = self.upper_proxy.last_upper
        if lower_info is None:
            lower_info = {
                "state": _array(state).astype(np.float32),
                "latent_action": np.zeros(int(self.contract.lower_action_dim), dtype=np.float32),
                "logp": 0.0,
                "value": 0.0,
            }
        if upper_info is None:
            upper_info = {
                "state": np.zeros(int(self.contract.upper_state_dim), dtype=np.float32),
                "latent_action": np.zeros(int(self.contract.upper_model_action_dim), dtype=np.float32),
                "logp": 0.0,
                "value": 0.0,
            }
        self.rows.append({
            "upper_state": _array(upper_info["state"]).astype(np.float32),
            "lower_state": _array(state).astype(np.float32),
            "upper_action": _array(upper_info["latent_action"]).astype(np.float32),
            "lower_action": _array(lower_info["latent_action"]).astype(np.float32),
            "reward": float(reward),
            "done": float(done),
            "old_upper_logp": float(upper_info["logp"]),
            "old_lower_logp": float(lower_info["logp"]),
            "old_upper_value": float(upper_info["value"]),
            "old_lower_value": float(lower_info["value"]),
            "constraint": float(cost),
            "trip_id": int(trip_id),
        })

    def __len__(self) -> int:
        return len(self.rows)

    def to_batch(self) -> TrajectoryBatch | None:
        if not self.rows:
            return None
        return TrajectoryBatch(
            upper_state=np.asarray([row["upper_state"] for row in self.rows], dtype=np.float32),
            lower_state=np.asarray([row["lower_state"] for row in self.rows], dtype=np.float32),
            upper_action=np.asarray([row["upper_action"] for row in self.rows], dtype=np.float32),
            lower_action=np.asarray([row["lower_action"] for row in self.rows], dtype=np.float32),
            reward=np.asarray([row["reward"] for row in self.rows], dtype=np.float32),
            done=np.asarray([row["done"] for row in self.rows], dtype=np.float32),
            old_upper_logp=np.asarray([row["old_upper_logp"] for row in self.rows], dtype=np.float32),
            old_lower_logp=np.asarray([row["old_lower_logp"] for row in self.rows], dtype=np.float32),
            old_upper_value=np.asarray([row["old_upper_value"] for row in self.rows], dtype=np.float32),
            old_lower_value=np.asarray([row["old_lower_value"] for row in self.rows], dtype=np.float32),
            constraint=np.asarray([row["constraint"] for row in self.rows], dtype=np.float32),
        )


def _native_row_score(row: dict[str, Any]) -> float:
    return -float(row.get("avg_wait_min", 0.0)) - 2.0 * float(row.get("headway_cv", 0.0))


def install_shared_ppo_episode_loop(
    runner: Any,
    bridge: NativeTransitPPOBridge,
    *,
    learned_promotion_gate: bool = False,
    promotion_gate_threshold: float = 0.55,
    promotion_gate_sample: bool = False,
    promotion_gate_strength_min: float = 0.0,
    promotion_gate_age_min: float = 0.0,
    promotion_gate_min_elapsed_s: float = 0.0,
    promotion_gate_cooldown_s: float = 0.0,
    promotion_gate_preselect_action: bool = False,
    promotion_gate_plan_blend: float = 0.0,
) -> dict[str, Any]:
    upper_proxy = _SharedPPOPolicyProxy(bridge, "upper")
    lower_proxy = _SharedPPOPolicyProxy(bridge, "lower")
    lower_collector = _NativeLowerReplayCollector(lower_proxy, upper_proxy, bridge.contract)
    upper_collector = _NativeUpperReplayCollector(upper_proxy)
    runner.upper_trainer.policy_net = upper_proxy
    runner.upper_trainer.replay_buffer = upper_collector
    runner.lower_trainer.policy_net = lower_proxy
    runner.replay_buffer = lower_collector
    runner.upper_warmup = 0
    runner.updates_per_episode = 0
    runner.upper_updates = 0
    runner.tpc_enable = False
    runner.target_upper_trainer = None
    if bool(learned_promotion_gate):
        last_gate_replan_by_key: dict[Any, float] = {}

        def learned_gate_hook(**kwargs: Any) -> bool:
            freq_summary = kwargs.get("freq_summary", {}) or {}
            if not bool(freq_summary.get("freq_promotion_flag", 0.0)):
                return False
            if float(freq_summary.get("freq_promotion_strength", 0.0)) < float(promotion_gate_strength_min):
                return False
            if float(freq_summary.get("freq_promotion_age", 0.0)) < float(promotion_gate_age_min):
                return False
            elapsed = float(kwargs.get("elapsed", 0.0))
            if elapsed < float(promotion_gate_min_elapsed_s):
                return False
            interval_s = float(getattr(runner, "timetable_replan_interval_s", 0.0))
            horizon_s = float(getattr(getattr(runner, "timetable_planner", None), "horizon_s", interval_s))
            if interval_s > 0.0 and elapsed >= interval_s:
                return False
            if horizon_s > 0.0 and elapsed > horizon_s:
                return False
            cooldown_s = float(promotion_gate_cooldown_s)
            if cooldown_s > 0.0:
                key = kwargs.get("planner_key", "__all__")
                active_plan = kwargs.get("active_plan", {}) or {}
                origin = float(active_plan.get("origin", 0.0))
                now_s = origin + elapsed
                last_s = last_gate_replan_by_key.get(key)
                if last_s is not None and now_s - last_s < cooldown_s:
                    return False
            active_plan = kwargs.get("active_plan", {}) or {}
            native_override = None
            if bool(promotion_gate_preselect_action) and "action" in active_plan:
                native_override = active_plan.get("action")
            promote = upper_proxy.evaluate_promotion_gate(
                kwargs["s_upper"],
                threshold=float(promotion_gate_threshold),
                sample=bool(promotion_gate_sample),
                preselect_action=bool(promotion_gate_preselect_action),
                native_action_override=native_override,
                native_action_blend=float(promotion_gate_plan_blend),
            )
            if promote and cooldown_s > 0.0:
                last_gate_replan_by_key[kwargs.get("planner_key", "__all__")] = now_s
            return bool(promote)

        runner.freq_hrl_learned_promotion_gate = learned_gate_hook
    return {
        "upper_proxy": upper_proxy,
        "lower_proxy": lower_proxy,
        "lower_collector": lower_collector,
        "upper_collector": upper_collector,
    }


def load_native_runner(
    config_path: Path,
    *,
    seed: int,
    logs_dir: Path | None,
    device: str = "cpu",
    config_overrides: dict[str, Any] | None = None,
) -> Any:
    if str(TRANSIT_HRL_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSIT_HRL_ROOT))
    if str(TRANSIT_DUET_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSIT_DUET_ROOT))
    from freq_transitduet.runner_v3 import TransitDuetV2Runner, load_config

    _set_reproducible_seed(int(seed))
    cfg = load_config(str(config_path))
    cfg["seed"] = int(seed)
    if config_overrides:
        _merge_dict(cfg, dict(config_overrides))
    if logs_dir is not None:
        cfg.setdefault("logging", {})["logs_dir"] = str(logs_dir)
    return TransitDuetV2Runner(cfg, device=device)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _native_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = [
        "avg_wait_min",
        "headway_cv",
        "ep_reward",
        "ep_cost",
        "ep_steps",
        "upper_plan_decisions",
        "upper_plan_reuse_ratio",
        "freq_wait_lower_net_mean",
        "lower_lf_drift_ratio",
        "upper_hf_power_ratio",
        "freq_promotion_strength",
        "shared_ppo_gate_evaluations",
        "shared_ppo_gate_replans",
        "shared_ppo_gate_value_mean",
    ]
    summary = {"n": len(rows)}
    for key in keys:
        vals = [float(row.get(key, 0.0)) for row in rows]
        summary[f"{key}_mean"] = float(np.mean(vals))
    summary["score_mean"] = float(np.mean([_native_row_score(row) for row in rows]))
    return summary


def run_native_shared_ppo_episode_loop(
    output_dir: Path,
    config_path: Path,
    *,
    seed: int = 19,
    episodes: int = 1,
    device: str = "cpu",
    hidden_dim: int = 0,
    init_log_std: float = -2.0,
    learning_rate: float = 3e-4,
    keep_native_log_dir: bool = False,
    config_overrides: dict[str, Any] | None = None,
    learned_promotion_gate: bool = False,
    promotion_gate_threshold: float = 0.55,
    promotion_gate_sample: bool = False,
    promotion_gate_strength_min: float = 0.0,
    promotion_gate_age_min: float = 0.0,
    promotion_gate_min_elapsed_s: float = 0.0,
    promotion_gate_cooldown_s: float = 0.0,
    promotion_gate_preselect_action: bool = False,
    promotion_gate_plan_blend: float = 0.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    native_logs = output_dir / "native_logs"
    overrides = {
        "coupling": {
            "upper_warmup_eps": 0,
            "tpc": {"enable": False},
        },
        "lower": {"updates_per_episode": 0},
        "upper": {
            "updates_per_episode": 0,
            "timetable_planner": {"action_ema_alpha": 1.0},
        },
        "training": {
            "diag_freq": max(1, int(episodes) + 1),
            "trip_dump_freq": max(1, int(episodes) + 1),
        },
    }
    if config_overrides:
        _merge_dict(overrides, dict(config_overrides))
    runner = load_native_runner(
        config_path,
        seed=int(seed),
        logs_dir=native_logs,
        device=str(device),
        config_overrides=overrides,
    )
    if runner.diag is None:
        if str(TRANSIT_DUET_ROOT) not in sys.path:
            sys.path.insert(0, str(TRANSIT_DUET_ROOT))
        from freq_transitduet.runner_v3 import DiagnosticLog

        runner.diag = DiagnosticLog(runner.log_dir, resume=False)
    bridge = NativeTransitPPOBridge.from_runner(
        runner,
        hidden_dim=hidden_dim,
        init_log_std=init_log_std,
        learning_rate=learning_rate,
        device=device,
        learned_promotion_gate=bool(learned_promotion_gate),
        native_policy_init_seed=int(seed),
    )
    installed = install_shared_ppo_episode_loop(
        runner,
        bridge,
        learned_promotion_gate=bool(learned_promotion_gate),
        promotion_gate_threshold=float(promotion_gate_threshold),
        promotion_gate_sample=bool(promotion_gate_sample),
        promotion_gate_strength_min=float(promotion_gate_strength_min),
        promotion_gate_age_min=float(promotion_gate_age_min),
        promotion_gate_min_elapsed_s=float(promotion_gate_min_elapsed_s),
        promotion_gate_cooldown_s=float(promotion_gate_cooldown_s),
        promotion_gate_preselect_action=bool(promotion_gate_preselect_action),
        promotion_gate_plan_blend=float(promotion_gate_plan_blend),
    )
    rows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    for ep in range(max(1, int(episodes))):
        collector: _NativeLowerReplayCollector = installed["lower_collector"]
        collector.rows.clear()
        row = runner.run_episode(ep, training=True)
        batch = collector.to_batch()
        update_metrics = bridge.model.update(batch) if batch is not None else {}
        row = dict(row)
        row.update({
            "native_shared_ppo": True,
            "shared_ppo_lower_samples": 0 if batch is None else int(batch.reward.size),
            "shared_ppo_upper_decisions": int(installed["upper_proxy"].decisions),
            "shared_ppo_lower_decisions": int(installed["lower_proxy"].decisions),
            "shared_ppo_gate_evaluations": int(installed["upper_proxy"].gate_evaluations),
            "shared_ppo_gate_replans": int(installed["upper_proxy"].gate_replans),
            "shared_ppo_gate_value_mean": (
                float(np.mean(installed["upper_proxy"].gate_values))
                if installed["upper_proxy"].gate_values else 0.0
            ),
            "shared_ppo_loss": float(update_metrics.get("loss", 0.0)),
            "shared_ppo_policy_loss": float(update_metrics.get("policy_loss", 0.0)),
            "shared_ppo_value_loss": float(update_metrics.get("value_loss", 0.0)),
        })
        rows.append(row)
        updates.append({"episode": int(ep), **update_metrics})
    summary = _native_summary(rows)
    payload = {
        "policy": "shared_dual_actor_critic_ppo",
        "trainer": "native_transit_episode_loop_shared_ppo",
        "domain": "transit_native",
        "seed": int(seed),
        "episodes": int(max(1, int(episodes))),
        "contract": bridge.contract_dict(),
        "learned_promotion_gate": bool(learned_promotion_gate),
        "promotion_gate_threshold": float(promotion_gate_threshold),
        "promotion_gate_sample": bool(promotion_gate_sample),
        "promotion_gate_strength_min": float(promotion_gate_strength_min),
        "promotion_gate_age_min": float(promotion_gate_age_min),
        "promotion_gate_min_elapsed_s": float(promotion_gate_min_elapsed_s),
        "promotion_gate_cooldown_s": float(promotion_gate_cooldown_s),
        "promotion_gate_preselect_action": bool(promotion_gate_preselect_action),
        "promotion_gate_plan_blend": float(promotion_gate_plan_blend),
        "rows": rows,
        "updates": updates,
        "summary": summary,
        "status": (
            "supported_native_episode_loop"
            if rows and rows[-1].get("shared_ppo_lower_samples", 0) > 0
            else "failed_native_episode_loop"
        ),
    }
    write_native_loop_outputs(output_dir, payload)
    if native_logs.exists() and not keep_native_log_dir:
        shutil.rmtree(native_logs)
    return payload


def write_native_loop_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rows = list(payload.get("rows", []))
    if rows:
        with (output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    summary = payload.get("summary", {})
    lines = [
        "# Native Transit Shared-PPO Episode Loop",
        "",
        f"- status: {payload.get('status', 'missing')}",
        f"- episodes: {payload.get('episodes', 0)}",
        f"- shared core: `{payload.get('contract', {}).get('shared_core', 'NA')}`",
        f"- upper contract: {payload.get('contract', {}).get('upper_state_dim', 'NA')}x{payload.get('contract', {}).get('upper_action_dim', 'NA')}",
        f"- upper model action dim: {payload.get('contract', {}).get('upper_model_action_dim', 'NA')}",
        f"- lower contract: {payload.get('contract', {}).get('lower_state_dim', 'NA')}x{payload.get('contract', {}).get('lower_action_dim', 'NA')}",
        f"- learned promotion gate: {payload.get('learned_promotion_gate', False)} threshold={payload.get('promotion_gate_threshold', 0.0)}",
        f"- gate guard: strength>={payload.get('promotion_gate_strength_min', 0.0)} age>={payload.get('promotion_gate_age_min', 0.0)} min_elapsed_s={payload.get('promotion_gate_min_elapsed_s', 0.0)} cooldown_s={payload.get('promotion_gate_cooldown_s', 0.0)} preselect_action={payload.get('promotion_gate_preselect_action', False)} plan_blend={payload.get('promotion_gate_plan_blend', 0.0)}",
        f"- mean wait: {summary.get('avg_wait_min_mean', 0.0):.4f}",
        f"- mean headway CV: {summary.get('headway_cv_mean', 0.0):.4f}",
        f"- mean shared-PPO score: {summary.get('score_mean', 0.0):.4f}",
        f"- mean gate value: {summary.get('shared_ppo_gate_value_mean_mean', 0.0):.4f}",
        "",
        "| ep | wait | cv | reward | lower samples | upper decisions | gate replans | lower decisions | loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row.get('ep', 0))} "
            f"| {float(row.get('avg_wait_min', 0.0)):.4f} "
            f"| {float(row.get('headway_cv', 0.0)):.4f} "
            f"| {float(row.get('ep_reward', 0.0)):.4f} "
            f"| {int(row.get('shared_ppo_lower_samples', 0))} "
            f"| {int(row.get('shared_ppo_upper_decisions', 0))} "
            f"| {int(row.get('shared_ppo_gate_replans', 0))} "
            f"| {int(row.get('shared_ppo_lower_decisions', 0))} "
            f"| {float(row.get('shared_ppo_loss', 0.0)):.4f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_native_shared_ppo_audit(
    output_dir: Path,
    config_path: Path,
    *,
    seed: int = 7,
    device: str = "cpu",
    keep_native_log_dir: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    native_logs = output_dir / "native_logs"
    runner = load_native_runner(
        config_path,
        seed=int(seed),
        logs_dir=native_logs,
        device=str(device),
    )
    bridge = NativeTransitPPOBridge.from_runner(runner, device=device)
    upper_state = np.zeros(bridge.contract.upper_state_dim, dtype=np.float32)
    lower_state = np.zeros(bridge.contract.lower_state_dim, dtype=np.float32)
    upper = bridge.act_upper_native(upper_state, sample=False)
    lower = bridge.act_lower_native(lower_state, sample=False)
    contract = bridge.contract_dict()
    checks = {
        "native_runner_instantiated": True,
        "uses_shared_core": isinstance(bridge.model, DualActorCriticPPO),
        "upper_action_dim_matches_native": (
            int(upper["native_action"].size) == int(contract["upper_action_dim"])
        ),
        "upper_action_in_bounds": bool(np.all(
            upper["native_action"] >= bridge.upper_action_low - 1e-6
        ) and np.all(upper["native_action"] <= bridge.upper_action_high + 1e-6)),
        "lower_action_in_bounds": bool(
            0.0 <= float(lower["native_action"][0]) <= float(contract["lower_action_range_s"])
        ),
        "native_timetable_terminal_dispatch": bool(contract["terminal_dispatch"]),
        "native_promotion_replan": bool(contract["promotion_replan"]),
    }
    summary = {
        "config_path": str(config_path),
        "seed": int(seed),
        "contract": contract,
        "smoke_actions": {
            "upper_native_action": _array(upper["native_action"]).astype(float).tolist(),
            "lower_native_action": _array(lower["native_action"]).astype(float).tolist(),
            "upper_logp": float(upper["logp"]),
            "lower_logp": float(lower["logp"]),
        },
        "checks": checks,
        "status": "supported_interface" if all(checks.values()) else "failed_interface",
    }
    write_outputs(output_dir, summary)
    if native_logs.exists() and not keep_native_log_dir:
        shutil.rmtree(native_logs)
    return summary


def write_outputs(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    row = {
        "status": summary["status"],
        **summary["contract"],
        **summary["checks"],
    }
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    lines = [
        "# Native Transit Shared-PPO Interface Audit",
        "",
        f"- status: {summary['status']}",
        f"- config: `{summary['config_path']}`",
        f"- shared core: `{summary['contract']['shared_core']}`",
        f"- upper contract: state={summary['contract']['upper_state_dim']} action={summary['contract']['upper_action_dim']}",
        f"- lower contract: state={summary['contract']['lower_state_dim']} action={summary['contract']['lower_action_dim']}",
        f"- terminal dispatch: {summary['contract']['terminal_dispatch']}",
        f"- promotion replan: {summary['contract']['promotion_replan']}",
        "",
        "| check | value |",
        "|---|---:|",
    ]
    for key, value in summary["checks"].items():
        lines.append(f"| {key} | {value} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TRANSIT_DUET_ROOT / "configs_freqduet" / "T_freqhrl_native_full.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_native_shared_ppo_audit"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--keep-native-log-dir", action="store_true")
    parser.add_argument("--episode-loop", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--learned-promotion-gate", action="store_true")
    parser.add_argument("--promotion-gate-threshold", type=float, default=0.55)
    parser.add_argument("--promotion-gate-sample", action="store_true")
    parser.add_argument("--promotion-gate-strength-min", type=float, default=0.0)
    parser.add_argument("--promotion-gate-age-min", type=float, default=0.0)
    parser.add_argument("--promotion-gate-min-elapsed-s", type=float, default=0.0)
    parser.add_argument("--promotion-gate-cooldown-s", type=float, default=0.0)
    parser.add_argument("--promotion-gate-preselect-action", action="store_true")
    parser.add_argument("--promotion-gate-plan-blend", type=float, default=0.0)
    args = parser.parse_args()
    if args.episode_loop:
        summary = run_native_shared_ppo_episode_loop(
            output_dir=args.output_dir,
            config_path=args.config,
            seed=int(args.seed),
            episodes=int(args.episodes),
            device=str(args.device),
            keep_native_log_dir=bool(args.keep_native_log_dir),
            learned_promotion_gate=bool(args.learned_promotion_gate),
            promotion_gate_threshold=float(args.promotion_gate_threshold),
            promotion_gate_sample=bool(args.promotion_gate_sample),
            promotion_gate_strength_min=float(args.promotion_gate_strength_min),
            promotion_gate_age_min=float(args.promotion_gate_age_min),
            promotion_gate_min_elapsed_s=float(args.promotion_gate_min_elapsed_s),
            promotion_gate_cooldown_s=float(args.promotion_gate_cooldown_s),
            promotion_gate_preselect_action=bool(args.promotion_gate_preselect_action),
            promotion_gate_plan_blend=float(args.promotion_gate_plan_blend),
        )
        print(f"wrote {args.output_dir}")
        print(
            "native_shared_ppo_loop "
            f"status={summary['status']} "
            f"episodes={summary['episodes']} "
            f"wait={summary['summary'].get('avg_wait_min_mean', 0.0):.3f}"
        )
    else:
        summary = run_native_shared_ppo_audit(
            output_dir=args.output_dir,
            config_path=args.config,
            seed=int(args.seed),
            device=str(args.device),
            keep_native_log_dir=bool(args.keep_native_log_dir),
        )
        print(f"wrote {args.output_dir}")
        print(
            "native_shared_ppo "
            f"status={summary['status']} "
            f"upper_dim={summary['contract']['upper_state_dim']}x{summary['contract']['upper_action_dim']} "
            f"lower_dim={summary['contract']['lower_state_dim']}x{summary['contract']['lower_action_dim']}"
        )


if __name__ == "__main__":
    main()
