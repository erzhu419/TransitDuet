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


def _low_signal_from_freq_summary(freq_summary: dict[str, Any]) -> float:
    low_level = float(freq_summary.get("freq_low_demand", 0.0))
    low_forecast = float(freq_summary.get("freq_low_forecast", low_level))
    return max(
        abs(float(freq_summary.get("freq_low_slope", 0.0))),
        abs(low_forecast - low_level),
        abs(float(freq_summary.get("freq_middle", 0.0))),
        abs(float(freq_summary.get("freq_middle_energy", 0.0))),
    )


def _state_hold_wait_feedback(state: Any, *, holdfb_dim: int = 4) -> dict[str, float]:
    """Read appended frequency hold/wait feedback features when present.

    Runner v3 appends `[same_hold, same_wait, other_hold, other_wait]`.
    The helper is intentionally conservative: if the state is too short or no
    hold-feedback block is present, it returns zero rather than guessing.
    """
    arr = _array(state, dtype=np.float64)
    if arr.size < int(holdfb_dim) or int(holdfb_dim) < 4:
        return {
            "same_hold": 0.0,
            "same_wait": 0.0,
            "other_hold": 0.0,
            "other_wait": 0.0,
        }
    tail = arr[-int(holdfb_dim):]
    return {
        "same_hold": max(float(tail[0]), 0.0),
        "same_wait": max(float(tail[1]), 0.0),
        "other_hold": max(float(tail[2]), 0.0),
        "other_wait": max(float(tail[3]), 0.0),
    }


def _state_wait_pressure(
    state: Any,
    *,
    holdfb_dim: int = 4,
    hold_guard_weight: float = 0.0,
) -> float:
    feedback = _state_hold_wait_feedback(state, holdfb_dim=int(holdfb_dim))
    same_hold = feedback["same_hold"]
    same_wait = feedback["same_wait"]
    other_wait = feedback["other_wait"]
    return max(
        same_wait
        - 0.5 * other_wait
        - max(float(hold_guard_weight), 0.0) * same_hold,
        0.0,
    )


def _state_dispatch_gap_ratio(state: Any) -> float:
    """Return current same-direction dispatch gap / scheduled headway if present."""
    arr = _array(state, dtype=np.float64)
    if arr.size <= 8:
        return 1.0
    scheduled = max(float(arr[8]), 1e-6)
    return max(float(arr[3]), 0.0) / scheduled


def _state_throughput_proxy(state: Any, *, holdfb_dim: int = 4) -> dict[str, float]:
    """Conservative upper-state proxy for passenger throughput risk.

    The native upper state does not expose exact onboard load or OD flow.  This
    proxy uses only causal dispatch features available at upper-decision time:
    fleet utilization, same-direction gap, holding pressure, and the appended
    high-frequency wait feedback block when enabled.
    """
    arr = _array(state, dtype=np.float64)
    hold_wait = _state_hold_wait_feedback(state, holdfb_dim=int(holdfb_dim))
    fleet_util = float(np.clip(arr[2], 0.0, 2.0)) if arr.size > 2 else 0.0
    gap_ratio = _state_dispatch_gap_ratio(state)
    holding_ratio = float(np.clip(arr[4], 0.0, 2.0)) if arr.size > 4 else 0.0
    same_hold = float(np.clip(arr[5], 0.0, 3.0)) if arr.size > 5 else 0.0
    same_wait = float(hold_wait.get("same_wait", 0.0))
    other_wait = float(hold_wait.get("other_wait", 0.0))
    gap_deviation = abs(float(gap_ratio) - 1.0)
    score = (
        0.75 * same_wait
        + 0.25 * max(same_wait - other_wait, 0.0)
        + 0.15 * min(fleet_util, 1.5)
        - 0.50 * holding_ratio
        - 0.30 * same_hold
        - 0.25 * gap_deviation
    )
    return {
        "throughput_proxy_score": float(score),
        "throughput_proxy_fleet_util": float(fleet_util),
        "throughput_proxy_holding_ratio": float(holding_ratio),
        "throughput_proxy_gap_deviation": float(gap_deviation),
        "throughput_proxy_same_hold": float(same_hold),
        "throughput_proxy_same_wait": float(same_wait),
    }


def _lower_wait_prior_context(
    state: Any,
    *,
    local_high_offset: int,
    context_dim: int = 0,
) -> dict[str, float]:
    """Read optional lower context just before the lower-frequency feature block."""
    arr = _array(state, dtype=np.float64)
    offset = max(int(local_high_offset), 1)
    context_dim = max(int(context_dim), 0)
    if context_dim <= 0 or arr.size < offset + context_dim:
        return {
            "load": 0.0,
            "queue": 0.0,
            "schedule_slack": 0.0,
        }
    start = int(arr.size) - offset - context_dim
    stop = int(arr.size) - offset
    context = arr[start:stop]
    return {
        "load": float(np.clip(context[0], 0.0, 2.0)) if context.size > 0 else 0.0,
        "queue": float(np.clip(context[1], 0.0, 2.0)) if context.size > 1 else 0.0,
        "schedule_slack": float(np.clip(context[2], -2.0, 2.0)) if context.size > 2 else 0.0,
    }


def _lower_wait_prior_scale(
    state: Any,
    *,
    local_high_offset: int,
    context_dim: int = 0,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
    load_damping_weight: float = 0.0,
    schedule_slack_damping_weight: float = 0.0,
    queue_boost_weight: float = 0.0,
) -> dict[str, float]:
    context = _lower_wait_prior_context(
        state,
        local_high_offset=int(local_high_offset),
        context_dim=int(context_dim),
    )
    scale = 1.0
    load_weight = max(float(load_damping_weight), 0.0)
    slack_weight = max(float(schedule_slack_damping_weight), 0.0)
    queue_weight = max(float(queue_boost_weight), 0.0)
    if load_weight > 0.0:
        scale /= 1.0 + load_weight * max(float(context["load"]), 0.0)
    if slack_weight > 0.0:
        scale /= 1.0 + slack_weight * max(float(context["schedule_slack"]), 0.0)
    if queue_weight > 0.0:
        scale *= 1.0 + queue_weight * max(float(context["queue"]), 0.0)
    high = max(float(max_scale), float(min_scale), 1.0)
    scale = float(np.clip(
        scale,
        max(min(float(min_scale), 1.0), 0.0),
        high,
    ))
    context["scale"] = scale
    return context


def _lower_wait_boarding_rescue_s(
    context: dict[str, float],
    *,
    local_high: float,
    gain_s: float = 0.0,
    max_s: float = 0.0,
    queue_min: float = 0.0,
    load_max: float = 0.0,
) -> float:
    gain = max(float(gain_s), 0.0)
    if gain <= 0.0:
        return 0.0
    queue_pressure = max(float(context.get("queue", 0.0)) - max(float(queue_min), 0.0), 0.0)
    if queue_pressure <= 0.0:
        return 0.0
    load_cap = max(float(load_max), 0.0)
    load_room = 1.0
    if load_cap > 0.0:
        load_room = max(load_cap - max(float(context.get("load", 0.0)), 0.0), 0.0) / load_cap
    rescue = gain * max(float(local_high), 0.0) * queue_pressure * load_room
    cap = max(float(max_s), 0.0)
    if cap > 0.0:
        rescue = min(rescue, cap)
    return float(max(rescue, 0.0))


def _direction_action_slice(action_size: int, planner_key: Any) -> slice:
    if planner_key == "__all__" or not isinstance(planner_key, (bool, np.bool_)):
        return slice(0, int(action_size))
    if int(action_size) < 2 or int(action_size) % 2 != 0:
        return slice(0, int(action_size))
    half = int(action_size) // 2
    return slice(0, half) if bool(planner_key) else slice(half, int(action_size))


def _candidate_target_headway_mean_s(runner: Any, action: Any, trip: Any) -> float:
    planner = getattr(runner, "timetable_planner", None)
    if planner is None or trip is None:
        return 0.0
    try:
        base = float(getattr(
            trip,
            "_freqduet_base_target_headway",
            getattr(trip, "target_headway", 360.0),
        ))
        horizon_s = max(float(getattr(planner, "horizon_s", 0.0)), 0.0)
        if horizon_s > 0.0:
            offsets = np.linspace(0.0, horizon_s, num=5, dtype=np.float64)
        else:
            offsets = np.asarray([0.0], dtype=np.float64)
        if not hasattr(planner, "target_headway"):
            action_arr = _array(action, dtype=np.float64)
            return float(base + np.mean(action_arr)) if action_arr.size else float(base)
        targets = [
            float(planner.target_headway(base, action, bool(getattr(trip, "direction", True)), offset))
            for offset in offsets
        ]
        return float(np.mean(targets)) if targets else 0.0
    except Exception:
        return 0.0


def _project_action_to_target_headway_cap(
    runner: Any,
    action: Any,
    *,
    trip: Any,
    planner_key: Any,
    target_headway_max_s: float,
    margin_s: float = 0.25,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project upper coefficients onto a mean target-headway cap.

    The existing guard rejected promotion replans when the learned actor base
    produced a high target headway.  For wait-aware replanning this is too
    conservative: a high-pressure promotion can still be made executable by
    shifting the active direction's coefficients down to the cap.
    """
    projected = _array(action, dtype=np.float64)
    target_cap = max(float(target_headway_max_s), 0.0)
    before = _candidate_target_headway_mean_s(runner, projected, trip)
    if target_cap <= 0.0 or before <= target_cap:
        return projected.astype(np.float32), {
            "target_headway_projection_active": 0.0,
            "target_headway_projection_before_s": float(before),
            "target_headway_projection_after_s": float(before),
            "target_headway_projection_correction_abs_s": 0.0,
            "target_headway_projection_clipped": 0.0,
        }
    sl = _direction_action_slice(projected.size, planner_key)
    if projected[sl].size == 0:
        return projected.astype(np.float32), {
            "target_headway_projection_active": 0.0,
            "target_headway_projection_before_s": float(before),
            "target_headway_projection_after_s": float(before),
            "target_headway_projection_correction_abs_s": 0.0,
            "target_headway_projection_clipped": 0.0,
        }
    original = projected.copy()
    clipped = 0.0
    for _ in range(6):
        current = _candidate_target_headway_mean_s(runner, projected, trip)
        excess = current - target_cap + max(float(margin_s), 0.0)
        if excess <= 1e-6:
            break
        next_action = projected.copy()
        next_action[sl] -= excess
        clipped_action = np.clip(
            next_action,
            _array(getattr(runner, "upper_action_low", []), dtype=np.float64),
            _array(getattr(runner, "upper_action_high", []), dtype=np.float64),
        )
        clipped = max(clipped, float(np.any(np.abs(clipped_action - next_action) > 1e-9)))
        projected = clipped_action
        if np.allclose(projected, original, atol=1e-9):
            break
    after = _candidate_target_headway_mean_s(runner, projected, trip)
    correction = float(np.mean(np.abs(projected[sl] - original[sl])))
    return projected.astype(np.float32), {
        "target_headway_projection_active": 1.0,
        "target_headway_projection_before_s": float(before),
        "target_headway_projection_after_s": float(after),
        "target_headway_projection_correction_abs_s": correction,
        "target_headway_projection_clipped": clipped,
    }


def _project_action_to_throughput_floor(
    active_action: Any,
    candidate_action: Any,
    *,
    metadata: dict[str, float],
    min_score: float = 0.0,
    min_delta_fraction: float = 0.0,
    fleet_util_max: float = 0.0,
    same_hold_max: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Shrink a replan delta when causal service-throughput proxies are weak."""
    active = _array(active_action, dtype=np.float64)
    candidate = _array(candidate_action, dtype=np.float64)
    if candidate.size != active.size:
        candidate = np.resize(candidate, active.size).astype(np.float64)
    score = float(metadata.get("throughput_proxy_score", 0.0))
    fleet_util = float(metadata.get("throughput_proxy_fleet_util", 0.0))
    same_hold = float(metadata.get("throughput_proxy_same_hold", 0.0))
    keep = 1.0
    floor = max(float(min_score), 0.0)
    if floor > 0.0 and score < floor:
        keep = min(keep, max(score, 0.0) / max(floor, 1e-9))
    fleet_cap = max(float(fleet_util_max), 0.0)
    if fleet_cap > 0.0 and fleet_util > fleet_cap:
        keep = min(keep, fleet_cap / max(fleet_util, 1e-9))
    hold_cap = max(float(same_hold_max), 0.0)
    if hold_cap > 0.0 and same_hold > hold_cap:
        keep = min(keep, hold_cap / max(same_hold, 1e-9))
    keep = float(np.clip(
        keep,
        max(min(float(min_delta_fraction), 1.0), 0.0),
        1.0,
    ))
    projected = active + keep * (candidate - active)
    return projected.astype(np.float32), {
        "throughput_floor_projection_active": float(keep < 1.0 - 1e-9),
        "throughput_floor_delta_fraction": float(keep),
        "throughput_floor_score": float(score),
        "throughput_floor_fleet_util": float(fleet_util),
        "throughput_floor_same_hold": float(same_hold),
        "throughput_floor_delta_abs_before_s": float(np.mean(np.abs(candidate - active)))
        if active.size else 0.0,
        "throughput_floor_delta_abs_after_s": float(np.mean(np.abs(projected - active)))
        if active.size else 0.0,
    }


def wait_aware_replan_action(
    active_action: Any,
    *,
    bridge: "NativeTransitPPOBridge",
    planner_key: Any,
    freq_summary: dict[str, Any],
    state: Any,
    wait_gain_s: float,
    max_shift_s: float,
    holdfb_dim: int = 0,
    state_wait_weight: float = 1.0,
    frequency_weight: float = 1.0,
    min_pressure: float = 0.0,
    max_pressure: float = 0.0,
    hold_guard_weight: float = 0.0,
    same_hold_max: float = 0.0,
    same_wait_min: float = 0.0,
    same_wait_max: float = 0.0,
    gap_guard_min_ratio: float = 0.0,
    gap_guard_max_ratio: float = 0.0,
    gap_risk_cap_start: float = 0.0,
    gap_risk_cap_full: float = 0.0,
    adaptive_drift_penalty_gain: float = 0.0,
    adaptive_drift_penalty_min_scale: float = 0.25,
    shift_sign: float = -1.0,
    soft_pressure_cap: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build a promotion-triggered timetable action that reacts to wait pressure.

    A positive pressure shortens the active direction's Bernstein headway
    coefficients, which lowers target headways in the rolling timetable plan.
    """
    active = _array(active_action, dtype=np.float64)
    if active.size != int(bridge.contract.upper_action_dim):
        active = np.resize(active, int(bridge.contract.upper_action_dim)).astype(np.float64)
    low_signal = _low_signal_from_freq_summary(freq_summary)
    hf_energy = max(float(freq_summary.get("freq_high_energy", 0.0)), 0.0)
    promotion_strength = max(float(freq_summary.get("freq_promotion_strength", 0.0)), 0.0)
    hold_wait = _state_hold_wait_feedback(state, holdfb_dim=int(holdfb_dim))
    gap_ratio = _state_dispatch_gap_ratio(state)
    state_wait = _state_wait_pressure(
        state,
        holdfb_dim=int(holdfb_dim),
        hold_guard_weight=float(hold_guard_weight),
    )
    freq_pressure = min(
        1.0,
        max(low_signal, 0.0)
        + 0.25 * min(hf_energy, 1.0)
        + 0.25 * min(promotion_strength, 1.0),
    )
    pressure = (
        float(frequency_weight) * freq_pressure
        + float(state_wait_weight) * max(state_wait, 0.0)
    )
    pressure = max(0.0, min(1.0, pressure))
    hf_to_lf = hf_energy / max(low_signal, 1e-6)
    drift_gain = max(float(adaptive_drift_penalty_gain), 0.0)
    drift_scale = 1.0
    if drift_gain > 0.0:
        drift_scale = 1.0 / (1.0 + drift_gain * max(hf_to_lf - 1.0, 0.0))
        drift_scale = float(np.clip(
            drift_scale,
            max(min(float(adaptive_drift_penalty_min_scale), 1.0), 0.0),
            1.0,
        ))
    gap_guard_active = (
        float(gap_guard_min_ratio) > 0.0
        and gap_ratio < float(gap_guard_min_ratio)
    )
    gap_guard_active = gap_guard_active or (
        float(gap_guard_max_ratio) > 0.0
        and gap_ratio > float(gap_guard_max_ratio)
    )
    wait_guard_active = (
        float(same_hold_max) > 0.0
        and float(hold_wait["same_hold"]) > float(same_hold_max)
    )
    wait_guard_active = wait_guard_active or (
        float(same_wait_min) > 0.0
        and float(hold_wait["same_wait"]) < float(same_wait_min)
    )
    wait_guard_active = wait_guard_active or (
        float(same_wait_max) > 0.0
        and float(hold_wait["same_wait"]) > float(same_wait_max)
    )
    pressure_guard_active = (
        float(max_pressure) > 0.0
        and pressure > float(max_pressure)
        and not bool(soft_pressure_cap)
    )
    pressure_soft_cap_active = (
        float(max_pressure) > 0.0
        and pressure > float(max_pressure)
        and bool(soft_pressure_cap)
    )
    effective_pressure = min(pressure, float(max_pressure)) if pressure_soft_cap_active else pressure
    pressure_cap_scale = (
        effective_pressure / max(pressure, 1e-6)
        if pressure_soft_cap_active else 1.0
    )
    if (
        pressure < float(min_pressure)
        or pressure_guard_active
        or gap_guard_active
        or wait_guard_active
    ):
        return active.astype(np.float32), {
            "pressure": float(pressure),
            "state_wait_pressure": float(state_wait),
            "frequency_pressure": float(freq_pressure),
            "state_same_hold": float(hold_wait["same_hold"]),
            "state_same_wait": float(hold_wait["same_wait"]),
            "state_other_hold": float(hold_wait["other_hold"]),
            "state_other_wait": float(hold_wait["other_wait"]),
            "state_dispatch_gap_ratio": float(gap_ratio),
            "pressure_guard_active": float(pressure_guard_active),
            "pressure_soft_cap_active": float(pressure_soft_cap_active),
            "pressure_cap_scale": float(pressure_cap_scale),
            "gap_guard_active": float(gap_guard_active),
            "wait_guard_active": float(wait_guard_active),
            "shift_pressure": 0.0,
            "gap_risk_scale": 1.0,
            "adaptive_drift_scale": float(drift_scale),
            "adaptive_drift_hf_to_lf": float(hf_to_lf),
            "signed_shift_s": 0.0,
            "abs_shift_s": 0.0,
        }
    if float(min_pressure) > 0.0:
        shift_pressure = (
            (effective_pressure - float(min_pressure))
            / max(1.0 - float(min_pressure), 1e-6)
        )
    else:
        shift_pressure = effective_pressure
    shift_pressure = max(0.0, min(1.0, float(shift_pressure)))
    gap_risk_scale = 1.0
    start = max(float(gap_risk_cap_start), 0.0)
    full = max(float(gap_risk_cap_full), start)
    if full > start:
        gap_deviation = abs(float(gap_ratio) - 1.0)
        if gap_deviation > start:
            gap_risk_scale = 1.0 - min(
                1.0,
                (gap_deviation - start) / max(full - start, 1e-6),
            )
            shift_pressure *= gap_risk_scale
    shift_pressure *= drift_scale
    sign = -1.0 if float(shift_sign) < 0.0 else 1.0
    shift = sign * min(abs(float(max_shift_s)), abs(float(wait_gain_s)) * shift_pressure)
    action = active.copy()
    sl = _direction_action_slice(action.size, planner_key)
    n_coeff = int(action[sl].size)
    if n_coeff > 1:
        profile = np.linspace(1.25, 0.75, n_coeff, dtype=np.float64)
        profile = profile / max(float(profile.mean()), 1e-9)
    else:
        profile = np.ones(n_coeff, dtype=np.float64)
    action[sl] = action[sl] + shift * profile
    action = np.clip(action, bridge.upper_action_low, bridge.upper_action_high)
    realized = action - active
    return action.astype(np.float32), {
        "pressure": float(pressure),
        "state_wait_pressure": float(state_wait),
        "frequency_pressure": float(freq_pressure),
        "state_same_hold": float(hold_wait["same_hold"]),
        "state_same_wait": float(hold_wait["same_wait"]),
        "state_other_hold": float(hold_wait["other_hold"]),
        "state_other_wait": float(hold_wait["other_wait"]),
        "state_dispatch_gap_ratio": float(gap_ratio),
        "pressure_guard_active": 0.0,
        "pressure_soft_cap_active": float(pressure_soft_cap_active),
        "pressure_cap_scale": float(pressure_cap_scale),
        "gap_guard_active": 0.0,
        "wait_guard_active": 0.0,
        "shift_pressure": float(shift_pressure),
        "gap_risk_scale": float(gap_risk_scale),
        "adaptive_drift_scale": float(drift_scale),
        "adaptive_drift_hf_to_lf": float(hf_to_lf),
        "shift_sign": float(sign),
        "signed_shift_s": float(np.mean(realized[sl])) if realized[sl].size else 0.0,
        "abs_shift_s": float(np.mean(np.abs(realized[sl]))) if realized[sl].size else 0.0,
    }


def _promotion_reward_floor_score(
    metadata: dict[str, float],
    *,
    active_target_headway_s: float,
    candidate_target_headway_s: float,
    reward_wait_weight: float,
    reward_target_weight: float,
    reward_throughput_weight: float,
    reward_fleet_weight: float,
    reward_action_cost: float,
    reward_gap_cost: float,
    reward_hold_cost: float,
) -> float:
    wait_pressure = max(float(metadata.get("state_wait_pressure", 0.0)), 0.0)
    same_wait = max(float(metadata.get("state_same_wait", 0.0)), 0.0)
    shift_pressure = max(float(metadata.get("shift_pressure", 0.0)), 0.0)
    abs_shift = max(
        float(metadata.get("abs_shift_s", 0.0)),
        float(metadata.get("final_action_delta_abs_s", 0.0)),
        0.0,
    )
    target_improvement = max(
        float(active_target_headway_s) - float(candidate_target_headway_s),
        0.0,
    ) / 600.0
    gap_deviation = abs(float(metadata.get("state_dispatch_gap_ratio", 1.0)) - 1.0)
    same_hold = max(float(metadata.get("state_same_hold", 0.0)), 0.0)
    throughput = float(metadata.get("throughput_proxy_score", 0.0))
    fleet_util = min(max(float(metadata.get("throughput_proxy_fleet_util", 0.0)), 0.0), 1.5)
    wait_shift_s = max(float(metadata.get("abs_shift_s", 0.0)), 0.0)
    action_effect = min(
        1.0,
        max(
            shift_pressure,
            wait_shift_s / 2.0,
            target_improvement,
        ),
    )
    return float(
        float(reward_wait_weight) * (same_wait + wait_pressure) * shift_pressure
        + float(reward_target_weight) * target_improvement
        + action_effect * float(reward_throughput_weight) * throughput
        + action_effect * float(reward_fleet_weight) * fleet_util
        - float(reward_action_cost) * abs_shift
        - float(reward_gap_cost) * gap_deviation
        - float(reward_hold_cost) * same_hold
    )


def _promotion_wait_credit_from_metadata(
    metadata: dict[str, float] | None,
    *,
    weight: float,
    clip: float,
) -> float:
    if metadata is None or float(weight) <= 0.0:
        return 0.0
    score = max(
        float(metadata.get("reward_floor_score", 0.0)),
        float(metadata.get("value_guard_score", 0.0)),
        0.0,
    )
    shift_pressure = max(float(metadata.get("shift_pressure", 0.0)), 0.0)
    abs_shift = max(float(metadata.get("abs_shift_s", 0.0)), 0.0)
    if score <= 0.0 or shift_pressure <= 0.0 or abs_shift <= 1e-9:
        return 0.0
    credit = float(weight) * score
    if float(clip) > 0.0:
        credit = min(credit, float(clip))
    return max(float(credit), 0.0)


def _parse_float_list(value: Any, *, default: tuple[float, ...] = (1.0,)) -> list[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return [float(v) for v in default]
        return [float(part) for part in parts]
    if isinstance(value, (list, tuple, np.ndarray)):
        values = [float(v) for v in value]
        return values or [float(v) for v in default]
    return [float(value)]


def value_guarded_replan_action(
    active_action: Any,
    replan_base: Any,
    *,
    runner: Any,
    bridge: "NativeTransitPPOBridge",
    planner_key: Any,
    freq_summary: dict[str, Any],
    state: Any,
    trip: Any,
    candidate_scales: Any,
    wait_gain_s: float,
    max_shift_s: float,
    holdfb_dim: int = 0,
    state_wait_weight: float = 1.0,
    frequency_weight: float = 1.0,
    min_pressure: float = 0.0,
    max_pressure: float = 0.0,
    hold_guard_weight: float = 0.0,
    same_hold_max: float = 0.0,
    same_wait_min: float = 0.0,
    same_wait_max: float = 0.0,
    gap_guard_min_ratio: float = 0.0,
    gap_guard_max_ratio: float = 0.0,
    gap_risk_cap_start: float = 0.0,
    gap_risk_cap_full: float = 0.0,
    adaptive_drift_penalty_gain: float = 0.0,
    adaptive_drift_penalty_min_scale: float = 0.25,
    shift_sign: float = -1.0,
    project_target_headway: bool = False,
    target_headway_max_s: float = 0.0,
    target_headway_project_margin_s: float = 0.25,
    reward_wait_weight: float = 1.0,
    reward_target_weight: float = 1.0,
    reward_throughput_weight: float = 0.0,
    reward_fleet_weight: float = 0.0,
    reward_action_cost: float = 0.05,
    reward_gap_cost: float = 0.25,
    reward_hold_cost: float = 0.35,
    final_delta_abs_max_s: float = 0.0,
    soft_pressure_cap: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    """Select a promotion timetable action from causal reward-aware candidates."""
    active = _array(active_action, dtype=np.float64)
    base = _array(replan_base, dtype=np.float64)
    if active.size != int(bridge.contract.upper_action_dim):
        active = np.resize(active, int(bridge.contract.upper_action_dim)).astype(np.float64)
    if base.size != active.size:
        base = np.resize(base, active.size).astype(np.float64)
    scales = _parse_float_list(candidate_scales, default=(1.0,))
    throughput = _state_throughput_proxy(state, holdfb_dim=int(holdfb_dim))
    active_target = _candidate_target_headway_mean_s(runner, active, trip)
    best_action: np.ndarray | None = None
    best_metadata: dict[str, float] | None = None
    best_score = -float("inf")
    evaluated = 0
    for raw_scale in scales:
        scale = max(float(raw_scale), 0.0)
        if scale <= 0.0:
            candidate = active.astype(np.float32)
            metadata = {
                "pressure": 0.0,
                "state_wait_pressure": 0.0,
                "frequency_pressure": 0.0,
                "state_same_hold": 0.0,
                "state_same_wait": 0.0,
                "state_other_hold": 0.0,
                "state_other_wait": 0.0,
                "state_dispatch_gap_ratio": 1.0,
                "pressure_guard_active": 0.0,
                "pressure_soft_cap_active": 0.0,
                "pressure_cap_scale": 1.0,
                "gap_guard_active": 0.0,
                "wait_guard_active": 0.0,
                "shift_pressure": 0.0,
                "gap_risk_scale": 1.0,
                "adaptive_drift_scale": 1.0,
                "adaptive_drift_hf_to_lf": 0.0,
                "signed_shift_s": 0.0,
                "abs_shift_s": 0.0,
            }
        else:
            candidate, metadata = wait_aware_replan_action(
                base,
                bridge=bridge,
                planner_key=planner_key,
                freq_summary=freq_summary,
                state=state,
                wait_gain_s=float(wait_gain_s) * scale,
                max_shift_s=float(max_shift_s) * scale,
                holdfb_dim=int(holdfb_dim),
                state_wait_weight=float(state_wait_weight),
                frequency_weight=float(frequency_weight),
                min_pressure=float(min_pressure),
                max_pressure=float(max_pressure),
                hold_guard_weight=float(hold_guard_weight),
                same_hold_max=float(same_hold_max),
                same_wait_min=float(same_wait_min),
                same_wait_max=float(same_wait_max),
                gap_guard_min_ratio=float(gap_guard_min_ratio),
                gap_guard_max_ratio=float(gap_guard_max_ratio),
                gap_risk_cap_start=float(gap_risk_cap_start),
                gap_risk_cap_full=float(gap_risk_cap_full),
                adaptive_drift_penalty_gain=float(adaptive_drift_penalty_gain),
                adaptive_drift_penalty_min_scale=float(adaptive_drift_penalty_min_scale),
                shift_sign=float(shift_sign),
                soft_pressure_cap=bool(soft_pressure_cap),
            )
        metadata = dict(metadata)
        metadata.update(throughput)
        if bool(project_target_headway) and float(target_headway_max_s) > 0.0:
            candidate, projection = _project_action_to_target_headway_cap(
                runner,
                candidate,
                trip=trip,
                planner_key=planner_key,
                target_headway_max_s=float(target_headway_max_s),
                margin_s=float(target_headway_project_margin_s),
            )
            metadata.update(projection)
        candidate_target = _candidate_target_headway_mean_s(runner, candidate, trip)
        metadata["active_target_headway_mean_s"] = float(active_target)
        metadata["candidate_target_headway_mean_s"] = float(candidate_target)
        metadata["value_guard_candidate_scale"] = float(scale)
        metadata["value_guard_candidate_count"] = float(len(scales))
        metadata["final_action_delta_abs_s"] = float(np.mean(
            np.abs(np.asarray(candidate, dtype=np.float64) - active)
        ))
        metadata["base_action_delta_abs_s"] = float(np.mean(
            np.abs(base - active)
        ))
        metadata["final_delta_guard_active"] = 0.0
        final_delta = float(metadata["final_action_delta_abs_s"])
        if (
            scale > 0.0
            and float(final_delta_abs_max_s) > 0.0
            and final_delta > float(final_delta_abs_max_s)
        ):
            metadata["final_delta_guard_active"] = 1.0
            score = -float("inf")
        elif scale <= 0.0:
            score = 0.0
        else:
            metadata["final_delta_guard_active"] = 0.0
            score = _promotion_reward_floor_score(
                metadata,
                active_target_headway_s=float(active_target),
                candidate_target_headway_s=float(candidate_target),
                reward_wait_weight=float(reward_wait_weight),
                reward_target_weight=float(reward_target_weight),
                reward_throughput_weight=float(reward_throughput_weight),
                reward_fleet_weight=float(reward_fleet_weight),
                reward_action_cost=float(reward_action_cost),
                reward_gap_cost=float(reward_gap_cost),
                reward_hold_cost=float(reward_hold_cost),
            )
        metadata["value_guard_score"] = float(score)
        metadata["reward_floor_score"] = float(score)
        evaluated += 1
        if score > best_score:
            best_score = float(score)
            best_action = np.asarray(candidate, dtype=np.float32).reshape(-1)
            best_metadata = metadata
    if best_action is None or best_metadata is None:
        best_action = active.astype(np.float32)
        best_metadata = {
            "pressure": 0.0,
            "state_wait_pressure": 0.0,
            "frequency_pressure": 0.0,
            "state_same_hold": 0.0,
            "state_same_wait": 0.0,
            "state_other_hold": 0.0,
            "state_other_wait": 0.0,
            "state_dispatch_gap_ratio": 1.0,
            "pressure_guard_active": 0.0,
            "gap_guard_active": 0.0,
            "wait_guard_active": 0.0,
            "shift_pressure": 0.0,
            "gap_risk_scale": 1.0,
            "adaptive_drift_scale": 1.0,
            "adaptive_drift_hf_to_lf": 0.0,
            "signed_shift_s": 0.0,
            "abs_shift_s": 0.0,
            "value_guard_score": -float("inf"),
            "reward_floor_score": -float("inf"),
            "value_guard_candidate_scale": 0.0,
            "value_guard_candidate_count": float(len(scales)),
        }
    best_metadata["value_guard_active"] = 1.0
    best_metadata["value_guard_candidate_evaluations"] = float(evaluated)
    return best_action.astype(np.float32), best_metadata


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
    upper_hold_feedback_dim: int = 0
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
            "upper_hold_feedback_dim": int(self.upper_hold_feedback_dim),
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
        if (native_policy_init_seed is not None
                and (bool(contract.learned_promotion_gate)
                     or int(contract.upper_hold_feedback_dim) > 0)):
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
            upper_hold_feedback_dim=int(getattr(runner, "freq_holdfb_dim", 0)),
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
        """Keep native actions identical when adding gate/hold-feedback inputs."""
        _set_reproducible_seed(int(seed))
        reference_upper_state_dim = max(
            1,
            int(self.contract.upper_state_dim)
            - max(int(self.contract.upper_hold_feedback_dim), 0),
        )
        baseline = DualActorCriticPPO(DualPPOConfig(
            upper_state_dim=int(reference_upper_state_dim),
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

            def copy_network_prefix(src_net, dst_net, *, output_rows: int | None = None) -> None:
                src_linears = [
                    module for module in src_net
                    if isinstance(module, torch.nn.Linear)
                ]
                dst_linears = [
                    module for module in dst_net
                    if isinstance(module, torch.nn.Linear)
                ]
                if len(src_linears) != len(dst_linears):
                    return
                for idx, (src, dst) in enumerate(zip(src_linears, dst_linears)):
                    is_first = idx == 0
                    is_last = idx == len(src_linears) - 1
                    rows = int(src.weight.shape[0])
                    if is_last and output_rows is not None:
                        rows = min(rows, int(output_rows))
                    cols = int(src.weight.shape[1])
                    if is_first:
                        cols = min(cols, int(reference_upper_state_dim))
                    rows = min(rows, int(dst.weight.shape[0]))
                    cols = min(cols, int(dst.weight.shape[1]))
                    dst.weight.zero_()
                    if dst.bias is not None:
                        dst.bias.zero_()
                    dst.weight[:rows, :cols].copy_(src.weight[:rows, :cols])
                    if src.bias is not None and dst.bias is not None:
                        dst.bias[:rows].copy_(src.bias[:rows])

            with torch.no_grad():
                copy_network_prefix(
                    baseline.upper_actor.net,
                    self.model.upper_actor.net,
                    output_rows=int(self.contract.upper_action_dim),
                )
                copy_network_prefix(
                    baseline.upper_value.net,
                    self.model.upper_value.net,
                )
                self.model.upper_actor.log_std[:int(self.contract.upper_action_dim)].copy_(
                    baseline.upper_actor.log_std
                )
            self.model.lower_actor.load_state_dict(baseline.lower_actor.state_dict())
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
                feature_end = int(self.contract.upper_state_dim) - max(
                    int(self.contract.upper_hold_feedback_dim), 0)
                if feature_end >= 3:
                    linear.weight[gate_row, feature_end - 3] = 2.0
                    linear.weight[gate_row, feature_end - 2] = 3.0
                    linear.weight[gate_row, feature_end - 1] = 1.0
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
        if bool(self.contract.learned_promotion_gate):
            import torch

            state_arr = _array(upper_state)
            state_t = self.model._state_tensor(state_arr)
            actor = self.model.upper_actor
            with torch.no_grad():
                mean = actor.net(state_t)
                std = torch.exp(actor.log_std).clamp(1e-4, 3.0).view(1, -1)
                native_dim = int(self.contract.upper_action_dim)
                native_mean = mean[:, :native_dim]
                native_std = std[:, :native_dim]
                if sample:
                    native_latent = native_mean + torch.randn_like(native_mean) * native_std
                else:
                    native_latent = native_mean
                if mean.shape[-1] > native_dim:
                    gate_latent = mean[:, native_dim:native_dim + 1]
                    latent_t = torch.cat([native_latent, gate_latent], dim=-1)
                else:
                    latent_t = native_latent
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(latent_t).sum(dim=-1)
                value = self.model.upper_value(state_t)
                latent = latent_t.cpu().numpy().reshape(-1)
                out = {"logp": float(logp.item()), "value": float(value.item())}
        else:
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
    def __init__(
        self,
        bridge: NativeTransitPPOBridge,
        level: str,
        *,
        lower_hf_wait_action_gain_s: float = 0.0,
        lower_hf_wait_feature_offset: int = 11,
        lower_hf_wait_context_dim: int = 0,
        lower_hf_wait_min_scale: float = 0.0,
        lower_hf_wait_max_scale: float = 1.0,
        lower_hf_wait_load_damping_weight: float = 0.0,
        lower_hf_wait_schedule_slack_damping_weight: float = 0.0,
        lower_hf_wait_queue_boost_weight: float = 0.0,
        lower_hf_wait_boarding_rescue_gain_s: float = 0.0,
        lower_hf_wait_boarding_rescue_max_s: float = 0.0,
        lower_hf_wait_boarding_rescue_queue_min: float = 0.0,
        lower_hf_wait_boarding_rescue_load_max: float = 0.0,
    ) -> None:
        self.bridge = bridge
        self.level = str(level)
        self.lower_hf_wait_action_gain_s = max(float(lower_hf_wait_action_gain_s), 0.0)
        self.lower_hf_wait_feature_offset = max(int(lower_hf_wait_feature_offset), 1)
        self.lower_hf_wait_context_dim = max(int(lower_hf_wait_context_dim), 0)
        self.lower_hf_wait_min_scale = max(min(float(lower_hf_wait_min_scale), 1.0), 0.0)
        self.lower_hf_wait_max_scale = max(float(lower_hf_wait_max_scale), 1.0)
        self.lower_hf_wait_load_damping_weight = max(float(lower_hf_wait_load_damping_weight), 0.0)
        self.lower_hf_wait_schedule_slack_damping_weight = max(
            float(lower_hf_wait_schedule_slack_damping_weight), 0.0)
        self.lower_hf_wait_queue_boost_weight = max(float(lower_hf_wait_queue_boost_weight), 0.0)
        self.lower_hf_wait_boarding_rescue_gain_s = max(
            float(lower_hf_wait_boarding_rescue_gain_s), 0.0)
        self.lower_hf_wait_boarding_rescue_max_s = max(
            float(lower_hf_wait_boarding_rescue_max_s), 0.0)
        self.lower_hf_wait_boarding_rescue_queue_min = max(
            float(lower_hf_wait_boarding_rescue_queue_min), 0.0)
        self.lower_hf_wait_boarding_rescue_load_max = max(
            float(lower_hf_wait_boarding_rescue_load_max), 0.0)
        self.lower_hf_wait_prior_scales: list[float] = []
        self.lower_hf_wait_prior_loads: list[float] = []
        self.lower_hf_wait_prior_queues: list[float] = []
        self.lower_hf_wait_prior_schedule_slacks: list[float] = []
        self.lower_hf_wait_boarding_rescues: list[float] = []
        self.pending: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        self.preselected: dict[tuple[float, ...], list[dict[str, Any]]] = {}
        self.last_upper: dict[str, Any] | None = None
        self.decisions = 0
        self.gate_evaluations = 0
        self.gate_replans = 0
        self.gate_values: list[float] = []
        self.wait_replan_pressures: list[float] = []
        self.wait_replan_shift_pressures: list[float] = []
        self.wait_replan_gap_risk_scales: list[float] = []
        self.wait_replan_gap_ratios: list[float] = []
        self.wait_replan_same_holds: list[float] = []
        self.wait_replan_same_waits: list[float] = []
        self.wait_replan_adaptive_drift_scales: list[float] = []
        self.wait_replan_adaptive_drift_hf_to_lf: list[float] = []
        self.wait_replan_throughput_scores: list[float] = []
        self.wait_replan_throughput_floor_delta_fractions: list[float] = []
        self.wait_replan_reward_floor_scores: list[float] = []
        self.wait_replan_value_guard_scores: list[float] = []
        self.wait_replan_value_guard_scales: list[float] = []
        self.wait_replan_value_guard_candidate_counts: list[float] = []
        self.wait_replan_pressure_overrides: list[float] = []
        self.wait_replan_signed_shifts: list[float] = []
        self.wait_replan_abs_shifts: list[float] = []
        self.wait_replan_base_delta_abs: list[float] = []
        self.wait_replan_final_delta_abs: list[float] = []
        self.wait_replan_actor_base_used: list[float] = []

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
        preselect_metadata: dict[str, float] | None = None,
        act_info_override: dict[str, Any] | None = None,
        force_promote: bool = False,
    ) -> bool:
        if self.level != "upper" or not bool(self.bridge.contract.learned_promotion_gate):
            return False
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state_arr = _array(state)
        info = (
            dict(act_info_override)
            if act_info_override is not None
            else self._act_info(state_arr, sample=sample)
        )
        gate_value = float(info.get("promotion_gate_value", 0.0))
        self.gate_evaluations += 1
        self.gate_values.append(gate_value)
        promote = bool(force_promote) or gate_value >= float(threshold)
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
                if preselect_metadata is not None:
                    self.wait_replan_pressures.append(float(
                        preselect_metadata.get("pressure", 0.0)
                    ))
                    self.wait_replan_shift_pressures.append(float(
                        preselect_metadata.get("shift_pressure", 0.0)
                    ))
                    self.wait_replan_gap_risk_scales.append(float(
                        preselect_metadata.get("gap_risk_scale", 1.0)
                    ))
                    self.wait_replan_gap_ratios.append(float(
                        preselect_metadata.get("state_dispatch_gap_ratio", 0.0)
                    ))
                    self.wait_replan_same_holds.append(float(
                        preselect_metadata.get("state_same_hold", 0.0)
                    ))
                    self.wait_replan_same_waits.append(float(
                        preselect_metadata.get("state_same_wait", 0.0)
                    ))
                    self.wait_replan_adaptive_drift_scales.append(float(
                        preselect_metadata.get("adaptive_drift_scale", 1.0)
                    ))
                    self.wait_replan_adaptive_drift_hf_to_lf.append(float(
                        preselect_metadata.get("adaptive_drift_hf_to_lf", 0.0)
                    ))
                    self.wait_replan_throughput_scores.append(float(
                        preselect_metadata.get("throughput_proxy_score", 0.0)
                    ))
                    self.wait_replan_throughput_floor_delta_fractions.append(float(
                        preselect_metadata.get("throughput_floor_delta_fraction", 1.0)
                    ))
                    self.wait_replan_reward_floor_scores.append(float(
                        preselect_metadata.get("reward_floor_score", 0.0)
                    ))
                    self.wait_replan_value_guard_scores.append(float(
                        preselect_metadata.get("value_guard_score", 0.0)
                    ))
                    self.wait_replan_value_guard_scales.append(float(
                        preselect_metadata.get("value_guard_candidate_scale", 0.0)
                    ))
                    self.wait_replan_value_guard_candidate_counts.append(float(
                        preselect_metadata.get("value_guard_candidate_count", 0.0)
                    ))
                    self.wait_replan_pressure_overrides.append(float(
                        preselect_metadata.get("gate_wait_pressure_override_active", 0.0)
                    ))
                    self.wait_replan_signed_shifts.append(float(
                        preselect_metadata.get("signed_shift_s", 0.0)
                    ))
                    self.wait_replan_abs_shifts.append(float(
                        preselect_metadata.get("abs_shift_s", 0.0)
                    ))
                    self.wait_replan_base_delta_abs.append(float(
                        preselect_metadata.get("base_action_delta_abs_s", 0.0)
                    ))
                    self.wait_replan_final_delta_abs.append(float(
                        preselect_metadata.get("final_action_delta_abs_s", 0.0)
                    ))
                    self.wait_replan_actor_base_used.append(float(
                        preselect_metadata.get("actor_base_used", 0.0)
                    ))
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
        if self.level == "lower" and self.lower_hf_wait_action_gain_s > 0.0:
            offset = min(self.lower_hf_wait_feature_offset, int(state_arr.size))
            local_high = max(float(state_arr[-offset]), 0.0) if offset > 0 else 0.0
            if local_high > 0.0:
                context = _lower_wait_prior_scale(
                    state_arr,
                    local_high_offset=offset,
                    context_dim=self.lower_hf_wait_context_dim,
                    min_scale=self.lower_hf_wait_min_scale,
                    max_scale=self.lower_hf_wait_max_scale,
                    load_damping_weight=self.lower_hf_wait_load_damping_weight,
                    schedule_slack_damping_weight=(
                        self.lower_hf_wait_schedule_slack_damping_weight),
                    queue_boost_weight=self.lower_hf_wait_queue_boost_weight,
                )
                scale = float(context["scale"])
                self.lower_hf_wait_prior_scales.append(scale)
                self.lower_hf_wait_prior_loads.append(float(context["load"]))
                self.lower_hf_wait_prior_queues.append(float(context["queue"]))
                self.lower_hf_wait_prior_schedule_slacks.append(float(context["schedule_slack"]))
                rescue_s = _lower_wait_boarding_rescue_s(
                    context,
                    local_high=local_high,
                    gain_s=self.lower_hf_wait_boarding_rescue_gain_s,
                    max_s=self.lower_hf_wait_boarding_rescue_max_s,
                    queue_min=self.lower_hf_wait_boarding_rescue_queue_min,
                    load_max=self.lower_hf_wait_boarding_rescue_load_max,
                )
                self.lower_hf_wait_boarding_rescues.append(float(rescue_s))
                adjusted = _array(info["native_action"]).astype(np.float32)
                adjusted_value = (
                    float(adjusted[0])
                    - self.lower_hf_wait_action_gain_s * local_high * scale
                    + float(rescue_s)
                )
                adjusted[0] = float(np.clip(
                    adjusted_value,
                    0.0,
                    float(self.bridge.contract.lower_action_range_s),
                ))
                info = dict(info)
                info["native_action"] = adjusted
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
    promotion_gate_wait_pressure_override: bool = False,
    promotion_gate_wait_pressure_override_min: float = 0.0,
    promotion_gate_strength_min: float = 0.0,
    promotion_gate_age_min: float = 0.0,
    promotion_gate_min_elapsed_s: float = 0.0,
    promotion_gate_cooldown_s: float = 0.0,
    promotion_gate_preselect_action: bool = False,
    promotion_gate_plan_blend: float = 0.0,
    promotion_gate_low_signal_min: float = 0.0,
    promotion_gate_max_hf_to_lf_ratio: float = 0.0,
    promotion_gate_max_replans: int = 0,
    promotion_gate_max_total_replans: int = 0,
    promotion_replan_policy: str = "actor",
    promotion_replan_wait_gain_s: float = 0.0,
    promotion_replan_max_shift_s: float = 30.0,
    promotion_replan_state_wait_weight: float = 1.0,
    promotion_replan_frequency_weight: float = 1.0,
    promotion_replan_min_pressure: float = 0.0,
    promotion_replan_max_pressure: float = 0.0,
    promotion_replan_soft_pressure_cap: bool = False,
    promotion_replan_require_shift: bool = False,
    promotion_replan_hold_guard_weight: float = 0.0,
    promotion_replan_same_hold_max: float = 0.0,
    promotion_replan_same_wait_min: float = 0.0,
    promotion_replan_same_wait_max: float = 0.0,
    promotion_replan_gap_guard_min_ratio: float = 0.0,
    promotion_replan_gap_guard_max_ratio: float = 0.0,
    promotion_replan_gap_risk_cap_start: float = 0.0,
    promotion_replan_gap_risk_cap_full: float = 0.0,
    promotion_replan_adaptive_drift_penalty_gain: float = 0.0,
    promotion_replan_adaptive_drift_penalty_min_scale: float = 0.25,
    promotion_replan_adaptive_drift_accept_min_scale: float = 0.0,
    promotion_replan_gap_risk_accept_max_scale: float = 0.0,
    promotion_replan_reward_floor_min_score: float = 0.0,
    promotion_replan_reward_floor_wait_weight: float = 1.0,
    promotion_replan_reward_floor_target_weight: float = 1.0,
    promotion_replan_reward_floor_throughput_weight: float = 0.0,
    promotion_replan_reward_floor_fleet_weight: float = 0.0,
    promotion_replan_reward_floor_action_cost: float = 0.05,
    promotion_replan_reward_floor_gap_cost: float = 0.25,
    promotion_replan_reward_floor_hold_cost: float = 0.35,
    promotion_replan_value_guard_min_score: float = 0.0,
    promotion_replan_value_guard_candidate_scales: Any = "",
    promotion_replan_throughput_guard_min_score: float = 0.0,
    promotion_replan_throughput_floor_min_score: float = 0.0,
    promotion_replan_throughput_floor_min_delta_fraction: float = 0.0,
    promotion_replan_throughput_floor_fleet_util_max: float = 0.0,
    promotion_replan_throughput_floor_same_hold_max: float = 0.0,
    promotion_replan_active_target_headway_min_s: float = 0.0,
    promotion_replan_target_headway_min_s: float = 0.0,
    promotion_replan_target_headway_max_s: float = 0.0,
    promotion_replan_project_target_headway: bool = False,
    promotion_replan_target_headway_project_margin_s: float = 0.25,
    promotion_replan_base_delta_abs_max_s: float = 0.0,
    promotion_replan_final_delta_abs_min_s: float = 0.0,
    promotion_replan_final_delta_abs_max_s: float = 0.0,
    promotion_replan_shift_sign: float = -1.0,
    promotion_replan_base_action: str = "active",
    promotion_replan_actor_base_trust_s: float = 0.0,
    promotion_replan_terminal_early_cap_s: float = 0.0,
    promotion_replan_terminal_early_relax: bool = False,
    promotion_replan_confirm_min_strength: float = 0.0,
    promotion_replan_confirm_min_low_signal: float = 0.0,
    promotion_replan_wait_credit_weight: float = 0.0,
    promotion_replan_wait_credit_clip: float = 0.0,
    lower_hf_wait_action_gain_s: float = 0.0,
    lower_hf_wait_feature_offset: int = 11,
    lower_hf_wait_context_dim: int = 0,
    lower_hf_wait_min_scale: float = 0.0,
    lower_hf_wait_max_scale: float = 1.0,
    lower_hf_wait_load_damping_weight: float = 0.0,
    lower_hf_wait_schedule_slack_damping_weight: float = 0.0,
    lower_hf_wait_queue_boost_weight: float = 0.0,
    lower_hf_wait_boarding_rescue_gain_s: float = 0.0,
    lower_hf_wait_boarding_rescue_max_s: float = 0.0,
    lower_hf_wait_boarding_rescue_queue_min: float = 0.0,
    lower_hf_wait_boarding_rescue_load_max: float = 0.0,
    adaptive_lower_drift_penalty_gain: float = 0.0,
    adaptive_lower_drift_penalty_min_scale: float = 0.25,
) -> dict[str, Any]:
    upper_proxy = _SharedPPOPolicyProxy(bridge, "upper")
    lower_proxy = _SharedPPOPolicyProxy(
        bridge,
        "lower",
        lower_hf_wait_action_gain_s=float(lower_hf_wait_action_gain_s),
        lower_hf_wait_feature_offset=int(lower_hf_wait_feature_offset),
        lower_hf_wait_context_dim=int(lower_hf_wait_context_dim),
        lower_hf_wait_min_scale=float(lower_hf_wait_min_scale),
        lower_hf_wait_max_scale=float(lower_hf_wait_max_scale),
        lower_hf_wait_load_damping_weight=float(lower_hf_wait_load_damping_weight),
        lower_hf_wait_schedule_slack_damping_weight=float(
            lower_hf_wait_schedule_slack_damping_weight),
        lower_hf_wait_queue_boost_weight=float(lower_hf_wait_queue_boost_weight),
        lower_hf_wait_boarding_rescue_gain_s=float(
            lower_hf_wait_boarding_rescue_gain_s),
        lower_hf_wait_boarding_rescue_max_s=float(
            lower_hf_wait_boarding_rescue_max_s),
        lower_hf_wait_boarding_rescue_queue_min=float(
            lower_hf_wait_boarding_rescue_queue_min),
        lower_hf_wait_boarding_rescue_load_max=float(
            lower_hf_wait_boarding_rescue_load_max),
    )
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
    runner.freq_hrl_promotion_terminal_early_cap_s = max(
        float(promotion_replan_terminal_early_cap_s), 0.0)
    runner.freq_hrl_promotion_terminal_early_relax = bool(
        promotion_replan_terminal_early_relax)
    runner.freq_hrl_promotion_target_headway_guard_rejects = 0
    runner.freq_hrl_promotion_pressure_guard_rejects = 0
    runner.freq_hrl_promotion_soft_pressure_cap_count = 0
    runner.freq_hrl_promotion_soft_pressure_cap_scale_sum = 0.0
    runner.freq_hrl_promotion_base_delta_guard_rejects = 0
    runner.freq_hrl_promotion_final_delta_floor_rejects = 0
    runner.freq_hrl_promotion_final_delta_guard_rejects = 0
    runner.freq_hrl_promotion_reward_floor_guard_rejects = 0
    runner.freq_hrl_promotion_confirm_guard_rejects = 0
    runner.freq_hrl_promotion_value_guard_rejects = 0
    runner.freq_hrl_promotion_throughput_guard_rejects = 0
    runner.freq_hrl_promotion_throughput_floor_project_count = 0
    runner.freq_hrl_promotion_throughput_floor_delta_fraction_sum = 0.0
    runner.freq_hrl_promotion_adaptive_drift_guard_rejects = 0
    runner.freq_hrl_promotion_gap_risk_guard_rejects = 0
    runner.freq_hrl_promotion_active_target_headway_floor_rejects = 0
    runner.freq_hrl_promotion_target_headway_floor_rejects = 0
    runner.freq_hrl_promotion_target_headway_project_count = 0
    runner.freq_hrl_promotion_target_headway_project_correction_abs_sum_s = 0.0
    runner.freq_hrl_promotion_wait_credit_budget = 0.0
    runner.freq_hrl_promotion_wait_credit_granted = 0.0
    runner.freq_hrl_promotion_wait_credit_consumed = 0.0
    runner.freq_hrl_promotion_wait_credit_events = 0
    credit_weight = max(float(promotion_replan_wait_credit_weight), 0.0)
    credit_clip = max(float(promotion_replan_wait_credit_clip), 0.0)
    if credit_weight > 0.0 and hasattr(runner, "_record_frequency_wait_credit"):
        original_record_frequency_wait_credit = runner._record_frequency_wait_credit

        def credit_aligned_record_frequency_wait_credit(*args: Any, **kwargs: Any) -> float:
            penalty = float(original_record_frequency_wait_credit(*args, **kwargs))
            budget = max(float(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_budget",
                0.0,
            )), 0.0)
            if budget <= 0.0:
                return penalty
            per_event_cap = credit_clip if credit_clip > 0.0 else budget
            consume = min(budget, per_event_cap)
            runner.freq_hrl_promotion_wait_credit_budget = float(budget - consume)
            runner.freq_hrl_promotion_wait_credit_consumed = float(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_consumed",
                0.0,
            )) + consume
            runner.freq_hrl_promotion_wait_credit_events = int(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_events",
                0,
            )) + 1
            return float(penalty - consume)

        runner._record_frequency_wait_credit = credit_aligned_record_frequency_wait_credit
    drift_gain = max(float(adaptive_lower_drift_penalty_gain), 0.0)
    if drift_gain > 0.0 and hasattr(runner, "_lower_drift_penalty"):
        original_lower_drift_penalty = runner._lower_drift_penalty
        runner.freq_hrl_adaptive_lower_drift_penalty_scales = []
        runner.freq_hrl_adaptive_lower_drift_penalty_hf_to_lf = []

        def adaptive_lower_drift_penalty(direction: Any, action_s: Any) -> float:
            penalty = float(original_lower_drift_penalty(direction, action_s))
            freq_summary = {}
            try:
                freq_summary = runner.env.frequency_summary()
            except Exception:
                freq_summary = {}
            low_signal = max(_low_signal_from_freq_summary(freq_summary), 1e-6)
            hf_energy = max(float(freq_summary.get("freq_high_energy", 0.0)), 0.0)
            hf_to_lf = hf_energy / low_signal
            scale = 1.0 / (1.0 + drift_gain * max(hf_to_lf - 1.0, 0.0))
            scale = float(np.clip(
                scale,
                max(min(float(adaptive_lower_drift_penalty_min_scale), 1.0), 0.0),
                1.0,
            ))
            runner.freq_hrl_adaptive_lower_drift_penalty_scales.append(scale)
            runner.freq_hrl_adaptive_lower_drift_penalty_hf_to_lf.append(float(hf_to_lf))
            return float(penalty * scale)

        runner._lower_drift_penalty = adaptive_lower_drift_penalty
    if bool(learned_promotion_gate):
        last_gate_replan_by_key: dict[Any, float] = {}
        gate_replans_by_key: dict[Any, int] = {}
        gate_replans_total = 0

        def learned_gate_hook(**kwargs: Any) -> bool:
            nonlocal gate_replans_total
            freq_summary = kwargs.get("freq_summary", {}) or {}
            freq_promotion_active = bool(freq_summary.get("freq_promotion_flag", 0.0))
            wait_pressure_override = bool(promotion_gate_wait_pressure_override)
            if not freq_promotion_active and not wait_pressure_override:
                return False
            if (
                freq_promotion_active
                and float(freq_summary.get("freq_promotion_strength", 0.0))
                < float(promotion_gate_strength_min)
            ):
                return False
            if (
                freq_promotion_active
                and float(freq_summary.get("freq_promotion_age", 0.0))
                < float(promotion_gate_age_min)
            ):
                return False
            low_level = float(freq_summary.get("freq_low_demand", 0.0))
            low_forecast = float(freq_summary.get("freq_low_forecast", low_level))
            low_signal = max(
                abs(float(freq_summary.get("freq_low_slope", 0.0))),
                abs(low_forecast - low_level),
                abs(float(freq_summary.get("freq_middle", 0.0))),
                abs(float(freq_summary.get("freq_middle_energy", 0.0))),
            )
            if (
                float(promotion_gate_low_signal_min) > 0.0
                and low_signal < float(promotion_gate_low_signal_min)
            ):
                return False
            hf_energy = max(float(freq_summary.get("freq_high_energy", 0.0)), 0.0)
            if float(promotion_gate_max_hf_to_lf_ratio) > 0.0:
                hf_to_lf = hf_energy / max(low_signal, 1e-6)
                if hf_to_lf > float(promotion_gate_max_hf_to_lf_ratio):
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
            key = kwargs.get("planner_key", "__all__")
            max_replans = max(0, int(promotion_gate_max_replans))
            if max_replans > 0 and gate_replans_by_key.get(key, 0) >= max_replans:
                return False
            max_total_replans = max(0, int(promotion_gate_max_total_replans))
            if max_total_replans > 0 and gate_replans_total >= max_total_replans:
                return False
            if cooldown_s > 0.0:
                active_plan = kwargs.get("active_plan", {}) or {}
                origin = float(active_plan.get("origin", 0.0))
                now_s = origin + elapsed
                last_s = last_gate_replan_by_key.get(key)
                if last_s is not None and now_s - last_s < cooldown_s:
                    return False
            active_plan = kwargs.get("active_plan", {}) or {}
            native_override = None
            preselect_metadata = None
            actor_replan_info = None
            force_promote = False
            if bool(promotion_gate_preselect_action) and "action" in active_plan:
                active_action = np.asarray(
                    active_plan.get("action"), dtype=np.float32).reshape(-1)
                native_override = active_action.copy()
                policy_name = str(promotion_replan_policy).lower()
                base_mode = str(promotion_replan_base_action).lower()
                use_actor_base = (
                    base_mode in {"actor", "policy", "learned", "current_actor"}
                    or policy_name in {
                        "actor_wait_aware",
                        "learned_actor_wait_aware",
                        "learned_wait_aware",
                    }
                )
                if use_actor_base:
                    actor_replan_info = upper_proxy._act_info(
                        _array(kwargs["s_upper"]), sample=bool(promotion_gate_sample))
                    replan_base = np.asarray(
                        actor_replan_info["native_action"], dtype=np.float32).reshape(-1)
                    actor_delta = (
                        np.asarray(replan_base, dtype=np.float64)
                        - np.asarray(active_action, dtype=np.float64)
                    )
                    actor_delta_abs = float(np.mean(np.abs(actor_delta)))
                    trust_s = max(float(promotion_replan_actor_base_trust_s), 0.0)
                    if trust_s > 0.0 and actor_delta_abs > trust_s:
                        scale = trust_s / max(actor_delta_abs, 1e-9)
                        replan_base = (
                            np.asarray(active_action, dtype=np.float64)
                            + actor_delta * scale
                        ).astype(np.float32)
                else:
                    replan_base = active_action.copy()
                    if base_mode in {"neutral", "zero", "zero_delta"}:
                        replan_base = np.zeros_like(replan_base, dtype=np.float32)
                    elif base_mode in {"midpoint", "mid"}:
                        replan_base = (
                            0.5 * (
                                np.asarray(bridge.upper_action_low, dtype=np.float32)
                                + np.asarray(bridge.upper_action_high, dtype=np.float32)
                            )
                        ).reshape(-1)
                if policy_name in {
                    "wait_aware",
                    "wait_aware_replan",
                    "learned_wait_aware",
                    "actor_wait_aware",
                    "learned_actor_wait_aware",
                }:
                    if _parse_float_list(
                        promotion_replan_value_guard_candidate_scales,
                        default=(),
                    ):
                        native_override, preselect_metadata = value_guarded_replan_action(
                            active_action,
                            replan_base,
                            runner=runner,
                            bridge=bridge,
                            planner_key=key,
                            freq_summary=freq_summary,
                            state=kwargs["s_upper"],
                            trip=kwargs.get("trip", None),
                            candidate_scales=promotion_replan_value_guard_candidate_scales,
                            wait_gain_s=float(promotion_replan_wait_gain_s),
                            max_shift_s=float(promotion_replan_max_shift_s),
                            holdfb_dim=int(getattr(runner, "freq_holdfb_dim", 0)),
                            state_wait_weight=float(promotion_replan_state_wait_weight),
                            frequency_weight=float(promotion_replan_frequency_weight),
                            min_pressure=float(promotion_replan_min_pressure),
                            max_pressure=float(promotion_replan_max_pressure),
                            hold_guard_weight=float(promotion_replan_hold_guard_weight),
                            same_hold_max=float(promotion_replan_same_hold_max),
                            same_wait_min=float(promotion_replan_same_wait_min),
                            same_wait_max=float(promotion_replan_same_wait_max),
                            gap_guard_min_ratio=float(promotion_replan_gap_guard_min_ratio),
                            gap_guard_max_ratio=float(promotion_replan_gap_guard_max_ratio),
                            gap_risk_cap_start=float(promotion_replan_gap_risk_cap_start),
                            gap_risk_cap_full=float(promotion_replan_gap_risk_cap_full),
                            adaptive_drift_penalty_gain=float(
                                promotion_replan_adaptive_drift_penalty_gain),
                            adaptive_drift_penalty_min_scale=float(
                                promotion_replan_adaptive_drift_penalty_min_scale),
                            shift_sign=float(promotion_replan_shift_sign),
                            project_target_headway=bool(
                                promotion_replan_project_target_headway),
                            target_headway_max_s=float(
                                promotion_replan_target_headway_max_s),
                            target_headway_project_margin_s=float(
                                promotion_replan_target_headway_project_margin_s),
                            reward_wait_weight=float(
                                promotion_replan_reward_floor_wait_weight),
                            reward_target_weight=float(
                                promotion_replan_reward_floor_target_weight),
                            reward_throughput_weight=float(
                                promotion_replan_reward_floor_throughput_weight),
                            reward_fleet_weight=float(
                                promotion_replan_reward_floor_fleet_weight),
                            reward_action_cost=float(
                                promotion_replan_reward_floor_action_cost),
                            reward_gap_cost=float(
                                promotion_replan_reward_floor_gap_cost),
                            reward_hold_cost=float(
                                promotion_replan_reward_floor_hold_cost),
                            final_delta_abs_max_s=float(
                                promotion_replan_final_delta_abs_max_s),
                            soft_pressure_cap=bool(promotion_replan_soft_pressure_cap),
                        )
                    else:
                        native_override, preselect_metadata = wait_aware_replan_action(
                            replan_base,
                            bridge=bridge,
                            planner_key=key,
                            freq_summary=freq_summary,
                            state=kwargs["s_upper"],
                            wait_gain_s=float(promotion_replan_wait_gain_s),
                            max_shift_s=float(promotion_replan_max_shift_s),
                            holdfb_dim=int(getattr(runner, "freq_holdfb_dim", 0)),
                            state_wait_weight=float(promotion_replan_state_wait_weight),
                            frequency_weight=float(promotion_replan_frequency_weight),
                            min_pressure=float(promotion_replan_min_pressure),
                            max_pressure=float(promotion_replan_max_pressure),
                            hold_guard_weight=float(promotion_replan_hold_guard_weight),
                            same_hold_max=float(promotion_replan_same_hold_max),
                            same_wait_min=float(promotion_replan_same_wait_min),
                            same_wait_max=float(promotion_replan_same_wait_max),
                            gap_guard_min_ratio=float(promotion_replan_gap_guard_min_ratio),
                            gap_guard_max_ratio=float(promotion_replan_gap_guard_max_ratio),
                            gap_risk_cap_start=float(promotion_replan_gap_risk_cap_start),
                            gap_risk_cap_full=float(promotion_replan_gap_risk_cap_full),
                            adaptive_drift_penalty_gain=float(
                                promotion_replan_adaptive_drift_penalty_gain),
                            adaptive_drift_penalty_min_scale=float(
                                promotion_replan_adaptive_drift_penalty_min_scale),
                            shift_sign=float(promotion_replan_shift_sign),
                            soft_pressure_cap=bool(promotion_replan_soft_pressure_cap),
                        )
                    preselect_metadata = dict(preselect_metadata)
                    preselect_metadata.update(_state_throughput_proxy(
                        kwargs["s_upper"],
                        holdfb_dim=int(getattr(runner, "freq_holdfb_dim", 0)),
                    ))
                    preselect_metadata["actor_base_used"] = float(use_actor_base)
                    preselect_metadata["actor_base_trust_s"] = float(
                        max(float(promotion_replan_actor_base_trust_s), 0.0))
                    preselect_metadata["base_action_delta_abs_s"] = float(np.mean(
                        np.abs(np.asarray(replan_base, dtype=np.float64)
                               - np.asarray(active_action, dtype=np.float64))
                    ))
                    preselect_metadata["final_action_delta_abs_s"] = float(np.mean(
                        np.abs(np.asarray(native_override, dtype=np.float64)
                               - np.asarray(active_action, dtype=np.float64))
                    ))
                    if float(preselect_metadata.get("pressure_soft_cap_active", 0.0)) > 0.0:
                        runner.freq_hrl_promotion_soft_pressure_cap_count = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_soft_pressure_cap_count",
                                0,
                            )
                        ) + 1
                        runner.freq_hrl_promotion_soft_pressure_cap_scale_sum = float(
                            getattr(
                                runner,
                                "freq_hrl_promotion_soft_pressure_cap_scale_sum",
                                0.0,
                            )
                        ) + float(preselect_metadata.get("pressure_cap_scale", 1.0))
                    value_guard_min_score = float(
                        promotion_replan_value_guard_min_score)
                    if (
                        abs(value_guard_min_score) > 1e-12
                        and float(preselect_metadata.get("value_guard_score", 0.0))
                        < value_guard_min_score
                    ):
                        runner.freq_hrl_promotion_value_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_value_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    if float(preselect_metadata.get("pressure_guard_active", 0.0)) > 0.0:
                        runner.freq_hrl_promotion_pressure_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_pressure_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    if (
                        float(preselect_metadata.get("gap_guard_active", 0.0)) > 0.0
                        or float(preselect_metadata.get("wait_guard_active", 0.0)) > 0.0
                    ):
                        return False
                    drift_accept_min_scale = max(
                        float(promotion_replan_adaptive_drift_accept_min_scale), 0.0)
                    if (
                        drift_accept_min_scale > 0.0
                        and float(preselect_metadata.get("adaptive_drift_scale", 1.0))
                        < drift_accept_min_scale
                    ):
                        runner.freq_hrl_promotion_adaptive_drift_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_adaptive_drift_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    gap_risk_accept_max_scale = max(
                        float(promotion_replan_gap_risk_accept_max_scale), 0.0)
                    if (
                        gap_risk_accept_max_scale > 0.0
                        and float(preselect_metadata.get("gap_risk_scale", 1.0))
                        > gap_risk_accept_max_scale
                    ):
                        runner.freq_hrl_promotion_gap_risk_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_gap_risk_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    throughput_guard_min = float(
                        promotion_replan_throughput_guard_min_score)
                    if (
                        throughput_guard_min > 0.0
                        and float(preselect_metadata.get("throughput_proxy_score", 0.0))
                        < throughput_guard_min
                    ):
                        runner.freq_hrl_promotion_throughput_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_throughput_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    throughput_floor_min = float(
                        promotion_replan_throughput_floor_min_score)
                    if (
                        throughput_floor_min > 0.0
                        or float(promotion_replan_throughput_floor_fleet_util_max) > 0.0
                        or float(promotion_replan_throughput_floor_same_hold_max) > 0.0
                    ):
                        native_override, throughput_floor_metadata = (
                            _project_action_to_throughput_floor(
                                active_action,
                                native_override,
                                metadata=preselect_metadata,
                                min_score=throughput_floor_min,
                                min_delta_fraction=float(
                                    promotion_replan_throughput_floor_min_delta_fraction),
                                fleet_util_max=float(
                                    promotion_replan_throughput_floor_fleet_util_max),
                                same_hold_max=float(
                                    promotion_replan_throughput_floor_same_hold_max),
                            )
                        )
                        preselect_metadata.update(throughput_floor_metadata)
                        if float(throughput_floor_metadata.get(
                                "throughput_floor_projection_active", 0.0)) > 0.0:
                            runner.freq_hrl_promotion_throughput_floor_project_count = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_throughput_floor_project_count",
                                    0,
                                )
                            ) + 1
                            runner.freq_hrl_promotion_throughput_floor_delta_fraction_sum = float(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_throughput_floor_delta_fraction_sum",
                                    0.0,
                                )
                            ) + float(throughput_floor_metadata.get(
                                "throughput_floor_delta_fraction", 1.0))
                        preselect_metadata["final_action_delta_abs_s"] = float(np.mean(
                            np.abs(np.asarray(native_override, dtype=np.float64)
                                   - np.asarray(active_action, dtype=np.float64))
                        ))
                    base_delta_abs_max_s = max(
                        float(promotion_replan_base_delta_abs_max_s), 0.0)
                    if (
                        base_delta_abs_max_s > 0.0
                        and float(preselect_metadata.get("base_action_delta_abs_s", 0.0))
                        > base_delta_abs_max_s
                    ):
                        runner.freq_hrl_promotion_base_delta_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_base_delta_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    final_delta_abs_max_s = max(
                        float(promotion_replan_final_delta_abs_max_s), 0.0)
                    if (
                        final_delta_abs_max_s > 0.0
                        and float(preselect_metadata.get("final_action_delta_abs_s", 0.0))
                        > final_delta_abs_max_s
                    ):
                        runner.freq_hrl_promotion_final_delta_guard_rejects = int(
                            getattr(
                                runner,
                                "freq_hrl_promotion_final_delta_guard_rejects",
                                0,
                            )
                        ) + 1
                        return False
                    target_headway_max_s = max(
                        float(promotion_replan_target_headway_max_s), 0.0)
                    if target_headway_max_s > 0.0:
                        candidate_target = _candidate_target_headway_mean_s(
                            runner, native_override, kwargs.get("trip", None))
                        if bool(promotion_replan_project_target_headway):
                            native_override, projection_metadata = (
                                _project_action_to_target_headway_cap(
                                    runner,
                                    native_override,
                                    trip=kwargs.get("trip", None),
                                    planner_key=key,
                                    target_headway_max_s=target_headway_max_s,
                                    margin_s=float(
                                        promotion_replan_target_headway_project_margin_s),
                                )
                            )
                            preselect_metadata.update(projection_metadata)
                            if float(projection_metadata.get(
                                    "target_headway_projection_active", 0.0)) > 0.0:
                                runner.freq_hrl_promotion_target_headway_project_count = int(
                                    getattr(
                                        runner,
                                        "freq_hrl_promotion_target_headway_project_count",
                                        0,
                                    )
                                ) + 1
                                runner.freq_hrl_promotion_target_headway_project_correction_abs_sum_s = float(
                                    getattr(
                                        runner,
                                        "freq_hrl_promotion_target_headway_project_correction_abs_sum_s",
                                        0.0,
                                    )
                                ) + float(projection_metadata.get(
                                    "target_headway_projection_correction_abs_s", 0.0))
                            preselect_metadata["final_action_delta_abs_s"] = float(np.mean(
                                np.abs(np.asarray(native_override, dtype=np.float64)
                                       - np.asarray(active_action, dtype=np.float64))
                            ))
                            candidate_target = float(projection_metadata.get(
                                "target_headway_projection_after_s", candidate_target))
                        preselect_metadata["candidate_target_headway_mean_s"] = float(
                            candidate_target)
                        if candidate_target > target_headway_max_s:
                            runner.freq_hrl_promotion_target_headway_guard_rejects = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_target_headway_guard_rejects",
                                    0,
                                )
                            ) + 1
                            return False
                    active_target_headway_min_s = max(
                        float(promotion_replan_active_target_headway_min_s), 0.0)
                    target_headway_min_s = max(
                        float(promotion_replan_target_headway_min_s), 0.0)
                    reward_floor_min_score = float(
                        promotion_replan_reward_floor_min_score)
                    if (
                        active_target_headway_min_s > 0.0
                        or target_headway_min_s > 0.0
                        or reward_floor_min_score > 0.0
                    ):
                        candidate_target = float(preselect_metadata.get(
                            "candidate_target_headway_mean_s",
                            _candidate_target_headway_mean_s(
                                runner, native_override, kwargs.get("trip", None)),
                        ))
                        active_target = float(_candidate_target_headway_mean_s(
                            runner, active_action, kwargs.get("trip", None)))
                        preselect_metadata["active_target_headway_mean_s"] = active_target
                        preselect_metadata["candidate_target_headway_mean_s"] = candidate_target
                        if (
                            active_target_headway_min_s > 0.0
                            and active_target < active_target_headway_min_s
                        ):
                            runner.freq_hrl_promotion_active_target_headway_floor_rejects = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_active_target_headway_floor_rejects",
                                    0,
                                )
                            ) + 1
                            return False
                        if (
                            target_headway_min_s > 0.0
                            and candidate_target < target_headway_min_s
                        ):
                            runner.freq_hrl_promotion_target_headway_floor_rejects = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_target_headway_floor_rejects",
                                    0,
                                )
                            ) + 1
                            return False
                        reward_score = _promotion_reward_floor_score(
                            preselect_metadata,
                            active_target_headway_s=active_target,
                            candidate_target_headway_s=candidate_target,
                            reward_wait_weight=float(
                                promotion_replan_reward_floor_wait_weight),
                            reward_target_weight=float(
                                promotion_replan_reward_floor_target_weight),
                            reward_throughput_weight=float(
                                promotion_replan_reward_floor_throughput_weight),
                            reward_fleet_weight=float(
                                promotion_replan_reward_floor_fleet_weight),
                            reward_action_cost=float(
                                promotion_replan_reward_floor_action_cost),
                            reward_gap_cost=float(
                                promotion_replan_reward_floor_gap_cost),
                            reward_hold_cost=float(
                                promotion_replan_reward_floor_hold_cost),
                        )
                        preselect_metadata["reward_floor_score"] = float(reward_score)
                        if (
                            abs(reward_floor_min_score) > 1e-12
                            and reward_score < reward_floor_min_score
                        ):
                            runner.freq_hrl_promotion_reward_floor_guard_rejects = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_reward_floor_guard_rejects",
                                    0,
                                )
                            ) + 1
                            return False
                    if (bool(promotion_replan_require_shift)
                            and float(preselect_metadata.get("abs_shift_s", 0.0)) <= 1e-6):
                        return False
                    confirm_strength_min = max(
                        float(promotion_replan_confirm_min_strength), 0.0)
                    confirm_low_signal_min = max(
                        float(promotion_replan_confirm_min_low_signal), 0.0)
                    if confirm_strength_min > 0.0 or confirm_low_signal_min > 0.0:
                        confirm_strength = max(
                            float(freq_summary.get("freq_promotion_strength", 0.0)),
                            0.0,
                        )
                        confirmed_by_strength = (
                            confirm_strength_min <= 0.0
                            or confirm_strength >= confirm_strength_min
                        )
                        confirmed_by_low_signal = (
                            confirm_low_signal_min <= 0.0
                            or low_signal >= confirm_low_signal_min
                        )
                        if not (confirmed_by_strength and confirmed_by_low_signal):
                            runner.freq_hrl_promotion_confirm_guard_rejects = int(
                                getattr(
                                    runner,
                                    "freq_hrl_promotion_confirm_guard_rejects",
                                    0,
                                )
                            ) + 1
                            return False
                        preselect_metadata["confirm_strength"] = float(confirm_strength)
                        preselect_metadata["confirm_low_signal"] = float(low_signal)
                    override_min = max(float(promotion_gate_wait_pressure_override_min), 0.0)
                    force_promote = (
                        wait_pressure_override
                        and float(preselect_metadata.get("pressure", 0.0)) >= override_min
                        and float(preselect_metadata.get("shift_pressure", 0.0)) > 0.0
                    )
                    if force_promote:
                        preselect_metadata["gate_wait_pressure_override_active"] = 1.0
                elif use_actor_base:
                    native_override = replan_base.astype(np.float32)
                    preselect_metadata = {
                        "pressure": 0.0,
                        "state_wait_pressure": 0.0,
                        "frequency_pressure": 0.0,
                        "state_same_hold": 0.0,
                        "state_same_wait": 0.0,
                        "state_other_hold": 0.0,
                        "state_other_wait": 0.0,
                        "state_dispatch_gap_ratio": _state_dispatch_gap_ratio(kwargs["s_upper"]),
                        "gap_guard_active": 0.0,
                        "wait_guard_active": 0.0,
                        "shift_pressure": 0.0,
                        "signed_shift_s": 0.0,
                        "abs_shift_s": 0.0,
                        "actor_base_used": 1.0,
                        "actor_base_trust_s": float(
                            max(float(promotion_replan_actor_base_trust_s), 0.0)),
                        "base_action_delta_abs_s": float(np.mean(
                            np.abs(np.asarray(replan_base, dtype=np.float64)
                                   - np.asarray(active_action, dtype=np.float64))
                        )),
                        "final_action_delta_abs_s": float(np.mean(
                            np.abs(np.asarray(native_override, dtype=np.float64)
                                   - np.asarray(active_action, dtype=np.float64))
                        )),
                    }
                final_delta_abs_min_s = max(
                    float(promotion_replan_final_delta_abs_min_s), 0.0)
                if (
                    final_delta_abs_min_s > 0.0
                    and preselect_metadata is not None
                    and float(preselect_metadata.get("final_action_delta_abs_s", 0.0))
                    < final_delta_abs_min_s
                ):
                    runner.freq_hrl_promotion_final_delta_floor_rejects = int(
                        getattr(
                            runner,
                            "freq_hrl_promotion_final_delta_floor_rejects",
                            0,
                        )
                    ) + 1
                    return False
            promote = upper_proxy.evaluate_promotion_gate(
                kwargs["s_upper"],
                threshold=float(promotion_gate_threshold),
                sample=bool(promotion_gate_sample),
                preselect_action=bool(promotion_gate_preselect_action),
                native_action_override=native_override,
                native_action_blend=float(promotion_gate_plan_blend),
                preselect_metadata=preselect_metadata,
                act_info_override=actor_replan_info,
                force_promote=bool(force_promote),
            )
            if promote and native_override is not None:
                wait_credit = _promotion_wait_credit_from_metadata(
                    preselect_metadata,
                    weight=credit_weight,
                    clip=credit_clip,
                )
                if wait_credit > 0.0:
                    runner.freq_hrl_promotion_wait_credit_budget = float(getattr(
                        runner,
                        "freq_hrl_promotion_wait_credit_budget",
                        0.0,
                    )) + wait_credit
                    runner.freq_hrl_promotion_wait_credit_granted = float(getattr(
                        runner,
                        "freq_hrl_promotion_wait_credit_granted",
                        0.0,
                    )) + wait_credit
                    if preselect_metadata is not None:
                        preselect_metadata["promotion_wait_credit_granted"] = float(
                            wait_credit)
                runner.freq_hrl_promotion_action_override = np.asarray(
                    native_override, dtype=np.float32).reshape(-1).copy()
                runner.freq_hrl_promotion_action_metadata = dict(
                    preselect_metadata or {})
            else:
                if hasattr(runner, "freq_hrl_promotion_action_override"):
                    delattr(runner, "freq_hrl_promotion_action_override")
                if hasattr(runner, "freq_hrl_promotion_action_metadata"):
                    delattr(runner, "freq_hrl_promotion_action_metadata")
            if promote and cooldown_s > 0.0:
                last_gate_replan_by_key[key] = now_s
            if promote and max_replans > 0:
                gate_replans_by_key[key] = gate_replans_by_key.get(key, 0) + 1
            if promote and max_total_replans > 0:
                gate_replans_total += 1
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
        "upper_plan_target_mean",
        "upper_plan_target_std",
        "terminal_launch_shift_mean",
        "terminal_launch_shift_std",
        "freq_wait_lower_net_mean",
        "freq_wait_lower_improvement_credit_mean",
        "freq_wait_lower_improvement_credit_max",
        "freq_wait_upper_credit_mean",
        "freq_wait_upper_credit_std",
        "freq_wait_low_share_mean",
        "freq_wait_lower_high_share_mean",
        "freq_wait_boarded_pax",
        "lower_lf_drift_ratio",
        "upper_hf_power_ratio",
        "freq_promotion_strength",
        "shared_ppo_gate_evaluations",
        "shared_ppo_gate_replans",
        "shared_ppo_target_headway_guard_rejects",
        "shared_ppo_target_headway_project_count",
        "shared_ppo_target_headway_project_correction_mean_s",
        "shared_ppo_soft_pressure_cap_count",
        "shared_ppo_soft_pressure_cap_scale_mean",
        "shared_ppo_final_delta_floor_rejects",
        "shared_ppo_final_delta_guard_rejects",
        "shared_ppo_reward_floor_guard_rejects",
        "shared_ppo_confirm_guard_rejects",
        "shared_ppo_value_guard_rejects",
        "shared_ppo_throughput_guard_rejects",
        "shared_ppo_throughput_floor_project_count",
        "shared_ppo_throughput_floor_delta_fraction_mean",
        "shared_ppo_adaptive_drift_guard_rejects",
        "shared_ppo_gap_risk_guard_rejects",
        "shared_ppo_active_target_headway_floor_rejects",
        "shared_ppo_target_headway_floor_rejects",
        "shared_ppo_gate_value_mean",
        "shared_ppo_wait_replan_count",
        "shared_ppo_wait_replan_pressure_mean",
        "shared_ppo_wait_replan_shift_pressure_mean",
        "shared_ppo_wait_replan_gap_ratio_mean",
        "shared_ppo_wait_replan_adaptive_drift_scale_mean",
        "shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean",
        "shared_ppo_wait_replan_throughput_score_mean",
        "shared_ppo_wait_replan_throughput_floor_delta_fraction_mean",
        "shared_ppo_wait_replan_reward_floor_score_mean",
        "shared_ppo_wait_replan_value_guard_score_mean",
        "shared_ppo_wait_replan_value_guard_scale_mean",
        "shared_ppo_wait_replan_value_guard_candidate_count_mean",
        "shared_ppo_adaptive_lower_drift_penalty_scale_mean",
        "shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean",
        "shared_ppo_lower_hf_wait_prior_scale_mean",
        "shared_ppo_lower_hf_wait_prior_load_mean",
        "shared_ppo_lower_hf_wait_prior_queue_mean",
        "shared_ppo_lower_hf_wait_prior_schedule_slack_mean",
        "shared_ppo_lower_hf_wait_boarding_rescue_mean",
        "shared_ppo_wait_replan_pressure_override_count",
        "shared_ppo_wait_replan_pressure_override_mean",
        "shared_ppo_wait_replan_same_hold_mean",
        "shared_ppo_wait_replan_same_wait_mean",
        "shared_ppo_wait_replan_shift_mean_s",
        "shared_ppo_wait_replan_shift_abs_mean_s",
        "shared_ppo_wait_replan_actor_base_used_mean",
        "shared_ppo_wait_replan_base_delta_abs_mean_s",
        "shared_ppo_wait_replan_final_delta_abs_mean_s",
        "shared_ppo_promotion_wait_credit_granted",
        "shared_ppo_promotion_wait_credit_consumed",
        "shared_ppo_promotion_wait_credit_budget",
        "shared_ppo_promotion_wait_credit_events",
        "native_real_profile",
        "native_boarded_pax",
        "native_alighted_pax",
        "native_avg_board_wait_min",
        "native_avg_onboard_load",
        "native_peak_onboard_load",
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
    promotion_gate_wait_pressure_override: bool = False,
    promotion_gate_wait_pressure_override_min: float = 0.0,
    promotion_gate_strength_min: float = 0.0,
    promotion_gate_age_min: float = 0.0,
    promotion_gate_min_elapsed_s: float = 0.0,
    promotion_gate_cooldown_s: float = 0.0,
    promotion_gate_preselect_action: bool = False,
    promotion_gate_plan_blend: float = 0.0,
    promotion_gate_low_signal_min: float = 0.0,
    promotion_gate_max_hf_to_lf_ratio: float = 0.0,
    promotion_gate_max_replans: int = 0,
    promotion_gate_max_total_replans: int = 0,
    promotion_replan_policy: str = "actor",
    promotion_replan_wait_gain_s: float = 0.0,
    promotion_replan_max_shift_s: float = 30.0,
    promotion_replan_state_wait_weight: float = 1.0,
    promotion_replan_frequency_weight: float = 1.0,
    promotion_replan_min_pressure: float = 0.0,
    promotion_replan_max_pressure: float = 0.0,
    promotion_replan_soft_pressure_cap: bool = False,
    promotion_replan_require_shift: bool = False,
    promotion_replan_hold_guard_weight: float = 0.0,
    promotion_replan_same_hold_max: float = 0.0,
    promotion_replan_same_wait_min: float = 0.0,
    promotion_replan_same_wait_max: float = 0.0,
    promotion_replan_gap_guard_min_ratio: float = 0.0,
    promotion_replan_gap_guard_max_ratio: float = 0.0,
    promotion_replan_gap_risk_cap_start: float = 0.0,
    promotion_replan_gap_risk_cap_full: float = 0.0,
    promotion_replan_adaptive_drift_penalty_gain: float = 0.0,
    promotion_replan_adaptive_drift_penalty_min_scale: float = 0.25,
    promotion_replan_adaptive_drift_accept_min_scale: float = 0.0,
    promotion_replan_gap_risk_accept_max_scale: float = 0.0,
    promotion_replan_reward_floor_min_score: float = 0.0,
    promotion_replan_reward_floor_wait_weight: float = 1.0,
    promotion_replan_reward_floor_target_weight: float = 1.0,
    promotion_replan_reward_floor_throughput_weight: float = 0.0,
    promotion_replan_reward_floor_fleet_weight: float = 0.0,
    promotion_replan_reward_floor_action_cost: float = 0.05,
    promotion_replan_reward_floor_gap_cost: float = 0.25,
    promotion_replan_reward_floor_hold_cost: float = 0.35,
    promotion_replan_value_guard_min_score: float = 0.0,
    promotion_replan_value_guard_candidate_scales: Any = "",
    promotion_replan_throughput_guard_min_score: float = 0.0,
    promotion_replan_throughput_floor_min_score: float = 0.0,
    promotion_replan_throughput_floor_min_delta_fraction: float = 0.0,
    promotion_replan_throughput_floor_fleet_util_max: float = 0.0,
    promotion_replan_throughput_floor_same_hold_max: float = 0.0,
    promotion_replan_active_target_headway_min_s: float = 0.0,
    promotion_replan_target_headway_min_s: float = 0.0,
    promotion_replan_target_headway_max_s: float = 0.0,
    promotion_replan_project_target_headway: bool = False,
    promotion_replan_target_headway_project_margin_s: float = 0.25,
    promotion_replan_base_delta_abs_max_s: float = 0.0,
    promotion_replan_final_delta_abs_min_s: float = 0.0,
    promotion_replan_final_delta_abs_max_s: float = 0.0,
    promotion_replan_shift_sign: float = -1.0,
    promotion_replan_base_action: str = "active",
    promotion_replan_actor_base_trust_s: float = 0.0,
    promotion_replan_terminal_early_cap_s: float = 0.0,
    promotion_replan_terminal_early_relax: bool = False,
    promotion_replan_confirm_min_strength: float = 0.0,
    promotion_replan_confirm_min_low_signal: float = 0.0,
    promotion_replan_wait_credit_weight: float = 0.0,
    promotion_replan_wait_credit_clip: float = 0.0,
    lower_hf_wait_action_gain_s: float = 0.0,
    lower_hf_wait_feature_offset: int = 11,
    lower_hf_wait_context_dim: int = 0,
    lower_hf_wait_min_scale: float = 0.0,
    lower_hf_wait_max_scale: float = 1.0,
    lower_hf_wait_load_damping_weight: float = 0.0,
    lower_hf_wait_schedule_slack_damping_weight: float = 0.0,
    lower_hf_wait_queue_boost_weight: float = 0.0,
    lower_hf_wait_boarding_rescue_gain_s: float = 0.0,
    lower_hf_wait_boarding_rescue_max_s: float = 0.0,
    lower_hf_wait_boarding_rescue_queue_min: float = 0.0,
    lower_hf_wait_boarding_rescue_load_max: float = 0.0,
    adaptive_lower_drift_penalty_gain: float = 0.0,
    adaptive_lower_drift_penalty_min_scale: float = 0.25,
    offpolicy_replay_updates: int = 1,
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
    Path(getattr(runner, "log_dir", native_logs)).mkdir(parents=True, exist_ok=True)
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
        promotion_gate_wait_pressure_override=bool(promotion_gate_wait_pressure_override),
        promotion_gate_wait_pressure_override_min=float(promotion_gate_wait_pressure_override_min),
        promotion_gate_strength_min=float(promotion_gate_strength_min),
        promotion_gate_age_min=float(promotion_gate_age_min),
        promotion_gate_min_elapsed_s=float(promotion_gate_min_elapsed_s),
        promotion_gate_cooldown_s=float(promotion_gate_cooldown_s),
        promotion_gate_preselect_action=bool(promotion_gate_preselect_action),
        promotion_gate_plan_blend=float(promotion_gate_plan_blend),
        promotion_gate_low_signal_min=float(promotion_gate_low_signal_min),
        promotion_gate_max_hf_to_lf_ratio=float(promotion_gate_max_hf_to_lf_ratio),
        promotion_gate_max_replans=int(promotion_gate_max_replans),
        promotion_gate_max_total_replans=int(promotion_gate_max_total_replans),
        promotion_replan_policy=str(promotion_replan_policy),
        promotion_replan_wait_gain_s=float(promotion_replan_wait_gain_s),
        promotion_replan_max_shift_s=float(promotion_replan_max_shift_s),
        promotion_replan_state_wait_weight=float(promotion_replan_state_wait_weight),
        promotion_replan_frequency_weight=float(promotion_replan_frequency_weight),
        promotion_replan_min_pressure=float(promotion_replan_min_pressure),
        promotion_replan_max_pressure=float(promotion_replan_max_pressure),
        promotion_replan_soft_pressure_cap=bool(promotion_replan_soft_pressure_cap),
        promotion_replan_require_shift=bool(promotion_replan_require_shift),
        promotion_replan_hold_guard_weight=float(promotion_replan_hold_guard_weight),
        promotion_replan_same_hold_max=float(promotion_replan_same_hold_max),
        promotion_replan_same_wait_min=float(promotion_replan_same_wait_min),
        promotion_replan_same_wait_max=float(promotion_replan_same_wait_max),
        promotion_replan_gap_guard_min_ratio=float(promotion_replan_gap_guard_min_ratio),
        promotion_replan_gap_guard_max_ratio=float(promotion_replan_gap_guard_max_ratio),
        promotion_replan_gap_risk_cap_start=float(promotion_replan_gap_risk_cap_start),
        promotion_replan_gap_risk_cap_full=float(promotion_replan_gap_risk_cap_full),
        promotion_replan_adaptive_drift_penalty_gain=float(
            promotion_replan_adaptive_drift_penalty_gain),
        promotion_replan_adaptive_drift_penalty_min_scale=float(
            promotion_replan_adaptive_drift_penalty_min_scale),
        promotion_replan_adaptive_drift_accept_min_scale=float(
            promotion_replan_adaptive_drift_accept_min_scale),
        promotion_replan_gap_risk_accept_max_scale=float(
            promotion_replan_gap_risk_accept_max_scale),
        promotion_replan_reward_floor_min_score=float(
            promotion_replan_reward_floor_min_score),
        promotion_replan_reward_floor_wait_weight=float(
            promotion_replan_reward_floor_wait_weight),
        promotion_replan_reward_floor_target_weight=float(
            promotion_replan_reward_floor_target_weight),
        promotion_replan_reward_floor_throughput_weight=float(
            promotion_replan_reward_floor_throughput_weight),
        promotion_replan_reward_floor_fleet_weight=float(
            promotion_replan_reward_floor_fleet_weight),
        promotion_replan_reward_floor_action_cost=float(
            promotion_replan_reward_floor_action_cost),
        promotion_replan_reward_floor_gap_cost=float(
            promotion_replan_reward_floor_gap_cost),
        promotion_replan_reward_floor_hold_cost=float(
            promotion_replan_reward_floor_hold_cost),
        promotion_replan_value_guard_min_score=float(
            promotion_replan_value_guard_min_score),
        promotion_replan_value_guard_candidate_scales=(
            promotion_replan_value_guard_candidate_scales),
        promotion_replan_throughput_guard_min_score=float(
            promotion_replan_throughput_guard_min_score),
        promotion_replan_throughput_floor_min_score=float(
            promotion_replan_throughput_floor_min_score),
        promotion_replan_throughput_floor_min_delta_fraction=float(
            promotion_replan_throughput_floor_min_delta_fraction),
        promotion_replan_throughput_floor_fleet_util_max=float(
            promotion_replan_throughput_floor_fleet_util_max),
        promotion_replan_throughput_floor_same_hold_max=float(
            promotion_replan_throughput_floor_same_hold_max),
        promotion_replan_active_target_headway_min_s=float(
            promotion_replan_active_target_headway_min_s),
        promotion_replan_target_headway_min_s=float(promotion_replan_target_headway_min_s),
        promotion_replan_target_headway_max_s=float(promotion_replan_target_headway_max_s),
        promotion_replan_project_target_headway=bool(promotion_replan_project_target_headway),
        promotion_replan_target_headway_project_margin_s=float(
            promotion_replan_target_headway_project_margin_s),
        promotion_replan_base_delta_abs_max_s=float(promotion_replan_base_delta_abs_max_s),
        promotion_replan_final_delta_abs_min_s=float(promotion_replan_final_delta_abs_min_s),
        promotion_replan_final_delta_abs_max_s=float(promotion_replan_final_delta_abs_max_s),
        promotion_replan_shift_sign=float(promotion_replan_shift_sign),
        promotion_replan_base_action=str(promotion_replan_base_action),
        promotion_replan_actor_base_trust_s=float(promotion_replan_actor_base_trust_s),
        promotion_replan_terminal_early_cap_s=float(promotion_replan_terminal_early_cap_s),
        promotion_replan_terminal_early_relax=bool(promotion_replan_terminal_early_relax),
        promotion_replan_confirm_min_strength=float(
            promotion_replan_confirm_min_strength),
        promotion_replan_confirm_min_low_signal=float(
            promotion_replan_confirm_min_low_signal),
        promotion_replan_wait_credit_weight=float(promotion_replan_wait_credit_weight),
        promotion_replan_wait_credit_clip=float(promotion_replan_wait_credit_clip),
        lower_hf_wait_action_gain_s=float(lower_hf_wait_action_gain_s),
        lower_hf_wait_feature_offset=int(lower_hf_wait_feature_offset),
        lower_hf_wait_context_dim=int(lower_hf_wait_context_dim),
        lower_hf_wait_min_scale=float(lower_hf_wait_min_scale),
        lower_hf_wait_max_scale=float(lower_hf_wait_max_scale),
        lower_hf_wait_load_damping_weight=float(lower_hf_wait_load_damping_weight),
        lower_hf_wait_schedule_slack_damping_weight=float(
            lower_hf_wait_schedule_slack_damping_weight),
        lower_hf_wait_queue_boost_weight=float(lower_hf_wait_queue_boost_weight),
        lower_hf_wait_boarding_rescue_gain_s=float(
            lower_hf_wait_boarding_rescue_gain_s),
        lower_hf_wait_boarding_rescue_max_s=float(
            lower_hf_wait_boarding_rescue_max_s),
        lower_hf_wait_boarding_rescue_queue_min=float(
            lower_hf_wait_boarding_rescue_queue_min),
        lower_hf_wait_boarding_rescue_load_max=float(
            lower_hf_wait_boarding_rescue_load_max),
        adaptive_lower_drift_penalty_gain=float(adaptive_lower_drift_penalty_gain),
        adaptive_lower_drift_penalty_min_scale=float(adaptive_lower_drift_penalty_min_scale),
    )
    rows: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    replay_updates = max(1, int(offpolicy_replay_updates))
    for ep in range(max(1, int(episodes))):
        collector: _NativeLowerReplayCollector = installed["lower_collector"]
        collector.rows.clear()
        row = runner.run_episode(ep, training=True)
        batch = collector.to_batch()
        update_metrics: dict[str, Any] = {}
        if batch is not None:
            for replay_idx in range(replay_updates):
                update_metrics = bridge.model.update(batch)
                updates.append({
                    "episode": int(ep),
                    "replay_update": int(replay_idx),
                    **update_metrics,
                })
        row = dict(row)
        row.update({
            "native_shared_ppo": True,
            "shared_ppo_lower_samples": 0 if batch is None else int(batch.reward.size),
            "shared_ppo_replay_updates": int(replay_updates if batch is not None else 0),
            "shared_ppo_upper_decisions": int(installed["upper_proxy"].decisions),
            "shared_ppo_lower_decisions": int(installed["lower_proxy"].decisions),
            "shared_ppo_gate_evaluations": int(installed["upper_proxy"].gate_evaluations),
            "shared_ppo_gate_replans": int(installed["upper_proxy"].gate_replans),
            "shared_ppo_target_headway_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_target_headway_guard_rejects",
                0,
            )),
            "shared_ppo_target_headway_project_count": int(getattr(
                runner,
                "freq_hrl_promotion_target_headway_project_count",
                0,
            )),
            "shared_ppo_target_headway_project_correction_mean_s": (
                float(getattr(
                    runner,
                    "freq_hrl_promotion_target_headway_project_correction_abs_sum_s",
                    0.0,
                ))
                / max(int(getattr(
                    runner,
                    "freq_hrl_promotion_target_headway_project_count",
                    0,
                )), 1)
            ),
            "shared_ppo_pressure_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_pressure_guard_rejects",
                0,
            )),
            "shared_ppo_soft_pressure_cap_count": int(getattr(
                runner,
                "freq_hrl_promotion_soft_pressure_cap_count",
                0,
            )),
            "shared_ppo_soft_pressure_cap_scale_mean": (
                float(getattr(
                    runner,
                    "freq_hrl_promotion_soft_pressure_cap_scale_sum",
                    0.0,
                ))
                / int(getattr(
                    runner,
                    "freq_hrl_promotion_soft_pressure_cap_count",
                    0,
                ))
                if int(getattr(
                    runner,
                    "freq_hrl_promotion_soft_pressure_cap_count",
                    0,
                )) > 0 else 1.0
            ),
            "shared_ppo_base_delta_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_base_delta_guard_rejects",
                0,
            )),
            "shared_ppo_final_delta_floor_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_final_delta_floor_rejects",
                0,
            )),
            "shared_ppo_final_delta_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_final_delta_guard_rejects",
                0,
            )),
            "shared_ppo_reward_floor_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_reward_floor_guard_rejects",
                0,
            )),
            "shared_ppo_confirm_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_confirm_guard_rejects",
                0,
            )),
            "shared_ppo_value_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_value_guard_rejects",
                0,
            )),
            "shared_ppo_throughput_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_throughput_guard_rejects",
                0,
            )),
            "shared_ppo_throughput_floor_project_count": int(getattr(
                runner,
                "freq_hrl_promotion_throughput_floor_project_count",
                0,
            )),
            "shared_ppo_throughput_floor_delta_fraction_mean": (
                float(getattr(
                    runner,
                    "freq_hrl_promotion_throughput_floor_delta_fraction_sum",
                    0.0,
                ))
                / int(getattr(
                    runner,
                    "freq_hrl_promotion_throughput_floor_project_count",
                    0,
                ))
                if int(getattr(
                    runner,
                    "freq_hrl_promotion_throughput_floor_project_count",
                    0,
                )) > 0 else 1.0
            ),
            "shared_ppo_adaptive_drift_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_adaptive_drift_guard_rejects",
                0,
            )),
            "shared_ppo_gap_risk_guard_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_gap_risk_guard_rejects",
                0,
            )),
            "shared_ppo_active_target_headway_floor_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_active_target_headway_floor_rejects",
                0,
            )),
            "shared_ppo_target_headway_floor_rejects": int(getattr(
                runner,
                "freq_hrl_promotion_target_headway_floor_rejects",
                0,
            )),
            "shared_ppo_gate_value_mean": (
                float(np.mean(installed["upper_proxy"].gate_values))
                if installed["upper_proxy"].gate_values else 0.0
            ),
            "shared_ppo_wait_replan_count": int(len(installed["upper_proxy"].wait_replan_abs_shifts)),
            "shared_ppo_wait_replan_pressure_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_pressures))
                if installed["upper_proxy"].wait_replan_pressures else 0.0
            ),
            "shared_ppo_wait_replan_shift_pressure_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_shift_pressures))
                if installed["upper_proxy"].wait_replan_shift_pressures else 0.0
            ),
            "shared_ppo_wait_replan_gap_ratio_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_gap_ratios))
                if installed["upper_proxy"].wait_replan_gap_ratios else 0.0
            ),
            "shared_ppo_wait_replan_gap_risk_scale_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_gap_risk_scales))
                if installed["upper_proxy"].wait_replan_gap_risk_scales else 0.0
            ),
            "shared_ppo_wait_replan_adaptive_drift_scale_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_adaptive_drift_scales))
                if installed["upper_proxy"].wait_replan_adaptive_drift_scales else 1.0
            ),
            "shared_ppo_wait_replan_adaptive_drift_hf_to_lf_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_adaptive_drift_hf_to_lf))
                if installed["upper_proxy"].wait_replan_adaptive_drift_hf_to_lf else 0.0
            ),
            "shared_ppo_wait_replan_throughput_score_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_throughput_scores))
                if installed["upper_proxy"].wait_replan_throughput_scores else 0.0
            ),
            "shared_ppo_wait_replan_throughput_floor_delta_fraction_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_throughput_floor_delta_fractions))
                if installed["upper_proxy"].wait_replan_throughput_floor_delta_fractions else 1.0
            ),
            "shared_ppo_wait_replan_reward_floor_score_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_reward_floor_scores))
                if installed["upper_proxy"].wait_replan_reward_floor_scores else 0.0
            ),
            "shared_ppo_wait_replan_value_guard_score_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_value_guard_scores))
                if installed["upper_proxy"].wait_replan_value_guard_scores else 0.0
            ),
            "shared_ppo_wait_replan_value_guard_scale_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_value_guard_scales))
                if installed["upper_proxy"].wait_replan_value_guard_scales else 0.0
            ),
            "shared_ppo_wait_replan_value_guard_candidate_count_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_value_guard_candidate_counts))
                if installed["upper_proxy"].wait_replan_value_guard_candidate_counts else 0.0
            ),
            "shared_ppo_adaptive_lower_drift_penalty_scale_mean": (
                float(np.mean(getattr(
                    runner,
                    "freq_hrl_adaptive_lower_drift_penalty_scales",
                    [],
                )))
                if getattr(runner, "freq_hrl_adaptive_lower_drift_penalty_scales", [])
                else 1.0
            ),
            "shared_ppo_adaptive_lower_drift_penalty_hf_to_lf_mean": (
                float(np.mean(getattr(
                    runner,
                    "freq_hrl_adaptive_lower_drift_penalty_hf_to_lf",
                    [],
                )))
                if getattr(runner, "freq_hrl_adaptive_lower_drift_penalty_hf_to_lf", [])
                else 0.0
            ),
            "shared_ppo_lower_hf_wait_prior_scale_mean": (
                float(np.mean(installed["lower_proxy"].lower_hf_wait_prior_scales))
                if installed["lower_proxy"].lower_hf_wait_prior_scales else 1.0
            ),
            "shared_ppo_lower_hf_wait_prior_load_mean": (
                float(np.mean(installed["lower_proxy"].lower_hf_wait_prior_loads))
                if installed["lower_proxy"].lower_hf_wait_prior_loads else 0.0
            ),
            "shared_ppo_lower_hf_wait_prior_queue_mean": (
                float(np.mean(installed["lower_proxy"].lower_hf_wait_prior_queues))
                if installed["lower_proxy"].lower_hf_wait_prior_queues else 0.0
            ),
            "shared_ppo_lower_hf_wait_prior_schedule_slack_mean": (
                float(np.mean(installed["lower_proxy"].lower_hf_wait_prior_schedule_slacks))
                if installed["lower_proxy"].lower_hf_wait_prior_schedule_slacks else 0.0
            ),
            "shared_ppo_lower_hf_wait_boarding_rescue_mean": (
                float(np.mean(installed["lower_proxy"].lower_hf_wait_boarding_rescues))
                if installed["lower_proxy"].lower_hf_wait_boarding_rescues else 0.0
            ),
            "shared_ppo_wait_replan_pressure_override_count": int(np.sum(
                installed["upper_proxy"].wait_replan_pressure_overrides
            )),
            "shared_ppo_wait_replan_pressure_override_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_pressure_overrides))
                if installed["upper_proxy"].wait_replan_pressure_overrides else 0.0
            ),
            "shared_ppo_wait_replan_same_hold_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_same_holds))
                if installed["upper_proxy"].wait_replan_same_holds else 0.0
            ),
            "shared_ppo_wait_replan_same_wait_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_same_waits))
                if installed["upper_proxy"].wait_replan_same_waits else 0.0
            ),
            "shared_ppo_wait_replan_shift_mean_s": (
                float(np.mean(installed["upper_proxy"].wait_replan_signed_shifts))
                if installed["upper_proxy"].wait_replan_signed_shifts else 0.0
            ),
            "shared_ppo_wait_replan_shift_abs_mean_s": (
                float(np.mean(installed["upper_proxy"].wait_replan_abs_shifts))
                if installed["upper_proxy"].wait_replan_abs_shifts else 0.0
            ),
            "shared_ppo_wait_replan_actor_base_used_mean": (
                float(np.mean(installed["upper_proxy"].wait_replan_actor_base_used))
                if installed["upper_proxy"].wait_replan_actor_base_used else 0.0
            ),
            "shared_ppo_wait_replan_base_delta_abs_mean_s": (
                float(np.mean(installed["upper_proxy"].wait_replan_base_delta_abs))
                if installed["upper_proxy"].wait_replan_base_delta_abs else 0.0
            ),
            "shared_ppo_wait_replan_final_delta_abs_mean_s": (
                float(np.mean(installed["upper_proxy"].wait_replan_final_delta_abs))
                if installed["upper_proxy"].wait_replan_final_delta_abs else 0.0
            ),
            "shared_ppo_promotion_wait_credit_granted": float(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_granted",
                0.0,
            )),
            "shared_ppo_promotion_wait_credit_consumed": float(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_consumed",
                0.0,
            )),
            "shared_ppo_promotion_wait_credit_budget": float(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_budget",
                0.0,
            )),
            "shared_ppo_promotion_wait_credit_events": int(getattr(
                runner,
                "freq_hrl_promotion_wait_credit_events",
                0,
            )),
            "shared_ppo_loss": float(update_metrics.get("loss", 0.0)),
            "shared_ppo_policy_loss": float(update_metrics.get("policy_loss", 0.0)),
            "shared_ppo_value_loss": float(update_metrics.get("value_loss", 0.0)),
        })
        rows.append(row)
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
        "promotion_gate_low_signal_min": float(promotion_gate_low_signal_min),
        "promotion_gate_max_hf_to_lf_ratio": float(promotion_gate_max_hf_to_lf_ratio),
        "promotion_gate_max_replans": int(max(0, int(promotion_gate_max_replans))),
        "promotion_gate_max_total_replans": int(max(0, int(promotion_gate_max_total_replans))),
        "promotion_replan_policy": str(promotion_replan_policy),
        "promotion_replan_wait_gain_s": float(promotion_replan_wait_gain_s),
        "promotion_replan_max_shift_s": float(promotion_replan_max_shift_s),
        "promotion_replan_project_target_headway": bool(promotion_replan_project_target_headway),
        "promotion_replan_target_headway_project_margin_s": float(
            promotion_replan_target_headway_project_margin_s),
        "promotion_replan_state_wait_weight": float(promotion_replan_state_wait_weight),
        "promotion_replan_frequency_weight": float(promotion_replan_frequency_weight),
        "promotion_replan_min_pressure": float(promotion_replan_min_pressure),
        "promotion_replan_max_pressure": float(promotion_replan_max_pressure),
        "promotion_replan_soft_pressure_cap": bool(promotion_replan_soft_pressure_cap),
        "promotion_replan_require_shift": bool(promotion_replan_require_shift),
        "promotion_replan_hold_guard_weight": float(promotion_replan_hold_guard_weight),
        "promotion_replan_same_hold_max": float(promotion_replan_same_hold_max),
        "promotion_replan_same_wait_min": float(promotion_replan_same_wait_min),
        "promotion_replan_same_wait_max": float(promotion_replan_same_wait_max),
        "promotion_replan_gap_guard_min_ratio": float(promotion_replan_gap_guard_min_ratio),
        "promotion_replan_gap_guard_max_ratio": float(promotion_replan_gap_guard_max_ratio),
        "promotion_replan_gap_risk_cap_start": float(promotion_replan_gap_risk_cap_start),
        "promotion_replan_gap_risk_cap_full": float(promotion_replan_gap_risk_cap_full),
        "promotion_replan_adaptive_drift_penalty_gain": float(
            promotion_replan_adaptive_drift_penalty_gain),
        "promotion_replan_adaptive_drift_penalty_min_scale": float(
            promotion_replan_adaptive_drift_penalty_min_scale),
        "promotion_replan_reward_floor_min_score": float(
            promotion_replan_reward_floor_min_score),
        "promotion_replan_reward_floor_wait_weight": float(
            promotion_replan_reward_floor_wait_weight),
        "promotion_replan_reward_floor_target_weight": float(
            promotion_replan_reward_floor_target_weight),
        "promotion_replan_reward_floor_throughput_weight": float(
            promotion_replan_reward_floor_throughput_weight),
        "promotion_replan_reward_floor_fleet_weight": float(
            promotion_replan_reward_floor_fleet_weight),
        "promotion_replan_reward_floor_action_cost": float(
            promotion_replan_reward_floor_action_cost),
        "promotion_replan_reward_floor_gap_cost": float(
            promotion_replan_reward_floor_gap_cost),
        "promotion_replan_reward_floor_hold_cost": float(
            promotion_replan_reward_floor_hold_cost),
        "promotion_replan_value_guard_min_score": float(
            promotion_replan_value_guard_min_score),
        "promotion_replan_value_guard_candidate_scales": (
            str(promotion_replan_value_guard_candidate_scales)),
        "promotion_replan_throughput_guard_min_score": float(
            promotion_replan_throughput_guard_min_score),
        "promotion_replan_throughput_floor_min_score": float(
            promotion_replan_throughput_floor_min_score),
        "promotion_replan_throughput_floor_min_delta_fraction": float(
            promotion_replan_throughput_floor_min_delta_fraction),
        "promotion_replan_throughput_floor_fleet_util_max": float(
            promotion_replan_throughput_floor_fleet_util_max),
        "promotion_replan_throughput_floor_same_hold_max": float(
            promotion_replan_throughput_floor_same_hold_max),
        "promotion_replan_active_target_headway_min_s": float(
            promotion_replan_active_target_headway_min_s),
        "promotion_replan_target_headway_min_s": float(promotion_replan_target_headway_min_s),
        "promotion_replan_target_headway_max_s": float(promotion_replan_target_headway_max_s),
        "promotion_replan_base_delta_abs_max_s": float(promotion_replan_base_delta_abs_max_s),
        "promotion_replan_final_delta_abs_min_s": float(promotion_replan_final_delta_abs_min_s),
        "promotion_replan_final_delta_abs_max_s": float(promotion_replan_final_delta_abs_max_s),
        "promotion_replan_shift_sign": float(promotion_replan_shift_sign),
        "promotion_replan_base_action": str(promotion_replan_base_action),
        "promotion_replan_actor_base_trust_s": float(promotion_replan_actor_base_trust_s),
        "promotion_replan_terminal_early_cap_s": float(promotion_replan_terminal_early_cap_s),
        "promotion_replan_terminal_early_relax": bool(promotion_replan_terminal_early_relax),
        "promotion_replan_confirm_min_strength": float(
            promotion_replan_confirm_min_strength),
        "promotion_replan_confirm_min_low_signal": float(
            promotion_replan_confirm_min_low_signal),
        "promotion_replan_wait_credit_weight": float(promotion_replan_wait_credit_weight),
        "promotion_replan_wait_credit_clip": float(promotion_replan_wait_credit_clip),
        "lower_hf_wait_action_gain_s": float(lower_hf_wait_action_gain_s),
        "lower_hf_wait_feature_offset": int(lower_hf_wait_feature_offset),
        "lower_hf_wait_context_dim": int(lower_hf_wait_context_dim),
        "lower_hf_wait_min_scale": float(lower_hf_wait_min_scale),
        "lower_hf_wait_max_scale": float(lower_hf_wait_max_scale),
        "lower_hf_wait_load_damping_weight": float(lower_hf_wait_load_damping_weight),
        "lower_hf_wait_schedule_slack_damping_weight": float(
            lower_hf_wait_schedule_slack_damping_weight),
        "lower_hf_wait_queue_boost_weight": float(lower_hf_wait_queue_boost_weight),
        "lower_hf_wait_boarding_rescue_gain_s": float(
            lower_hf_wait_boarding_rescue_gain_s),
        "lower_hf_wait_boarding_rescue_max_s": float(
            lower_hf_wait_boarding_rescue_max_s),
        "lower_hf_wait_boarding_rescue_queue_min": float(
            lower_hf_wait_boarding_rescue_queue_min),
        "lower_hf_wait_boarding_rescue_load_max": float(
            lower_hf_wait_boarding_rescue_load_max),
        "adaptive_lower_drift_penalty_gain": float(adaptive_lower_drift_penalty_gain),
        "adaptive_lower_drift_penalty_min_scale": float(adaptive_lower_drift_penalty_min_scale),
        "offpolicy_replay_updates": int(replay_updates),
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
        f"- gate LF/HF guard: low_signal_min={payload.get('promotion_gate_low_signal_min', 0.0)} max_hf_to_lf={payload.get('promotion_gate_max_hf_to_lf_ratio', 0.0)} max_replans={payload.get('promotion_gate_max_replans', 0)} max_total_replans={payload.get('promotion_gate_max_total_replans', 0)}",
        f"- replan target-headway guard: max_s={payload.get('promotion_replan_target_headway_max_s', 0.0)}",
        f"- replan target-headway projection: enabled={payload.get('promotion_replan_project_target_headway', False)} margin_s={payload.get('promotion_replan_target_headway_project_margin_s', 0.0)}",
        f"- replan throughput/reward floor: throughput_min={payload.get('promotion_replan_throughput_guard_min_score', 0.0)} floor_min={payload.get('promotion_replan_throughput_floor_min_score', 0.0)} reward_floor={payload.get('promotion_replan_reward_floor_min_score', 0.0)} target_min_s={payload.get('promotion_replan_target_headway_min_s', 0.0)}",
        f"- adaptive drift penalty: gain={payload.get('promotion_replan_adaptive_drift_penalty_gain', 0.0)} min_scale={payload.get('promotion_replan_adaptive_drift_penalty_min_scale', 0.0)}",
        f"- replan final-delta guard: min_s={payload.get('promotion_replan_final_delta_abs_min_s', 0.0)} max_s={payload.get('promotion_replan_final_delta_abs_max_s', 0.0)}",
        f"- promotion replan policy: {payload.get('promotion_replan_policy', 'actor')} wait_gain_s={payload.get('promotion_replan_wait_gain_s', 0.0)} max_shift_s={payload.get('promotion_replan_max_shift_s', 0.0)}",
        f"- lower HF wait action prior: gain_s={payload.get('lower_hf_wait_action_gain_s', 0.0)} offset={payload.get('lower_hf_wait_feature_offset', 0)} context_dim={payload.get('lower_hf_wait_context_dim', 0)} min_scale={payload.get('lower_hf_wait_min_scale', 0.0)} max_scale={payload.get('lower_hf_wait_max_scale', 1.0)}",
        f"- lower HF boarding rescue: gain_s={payload.get('lower_hf_wait_boarding_rescue_gain_s', 0.0)} max_s={payload.get('lower_hf_wait_boarding_rescue_max_s', 0.0)} queue_min={payload.get('lower_hf_wait_boarding_rescue_queue_min', 0.0)} load_max={payload.get('lower_hf_wait_boarding_rescue_load_max', 0.0)}",
        f"- adaptive lower drift penalty: gain={payload.get('adaptive_lower_drift_penalty_gain', 0.0)} min_scale={payload.get('adaptive_lower_drift_penalty_min_scale', 0.0)}",
        f"- off-policy replay updates per native batch: {payload.get('offpolicy_replay_updates', 1)}",
        f"- mean wait: {summary.get('avg_wait_min_mean', 0.0):.4f}",
        f"- mean headway CV: {summary.get('headway_cv_mean', 0.0):.4f}",
        f"- mean shared-PPO score: {summary.get('score_mean', 0.0):.4f}",
        f"- mean gate value: {summary.get('shared_ppo_gate_value_mean_mean', 0.0):.4f}",
        f"- mean wait-aware replan pressure: {summary.get('shared_ppo_wait_replan_pressure_mean_mean', 0.0):.4f}",
        f"- mean adaptive drift scale: {summary.get('shared_ppo_wait_replan_adaptive_drift_scale_mean_mean', 1.0):.4f}",
        f"- mean throughput proxy score: {summary.get('shared_ppo_wait_replan_throughput_score_mean_mean', 0.0):.4f}",
        f"- mean throughput floor delta fraction: {summary.get('shared_ppo_wait_replan_throughput_floor_delta_fraction_mean_mean', 1.0):.4f}",
        f"- mean reward-floor score: {summary.get('shared_ppo_wait_replan_reward_floor_score_mean_mean', 0.0):.4f}",
        f"- mean value-guard score: {summary.get('shared_ppo_wait_replan_value_guard_score_mean_mean', 0.0):.4f}",
        f"- mean value-guard selected scale: {summary.get('shared_ppo_wait_replan_value_guard_scale_mean_mean', 0.0):.4f}",
        f"- mean adaptive lower drift scale: {summary.get('shared_ppo_adaptive_lower_drift_penalty_scale_mean_mean', 1.0):.4f}",
        f"- mean lower prior scale: {summary.get('shared_ppo_lower_hf_wait_prior_scale_mean_mean', 1.0):.4f}",
        f"- mean lower boarding rescue: {summary.get('shared_ppo_lower_hf_wait_boarding_rescue_mean_mean', 0.0):.4f}s",
        f"- mean wait-pressure override count: {summary.get('shared_ppo_wait_replan_pressure_override_count_mean', 0.0):.4f}",
        f"- mean wait-aware replan shift: {summary.get('shared_ppo_wait_replan_shift_mean_s_mean', 0.0):.4f}s",
        f"- mean learned replan base delta: {summary.get('shared_ppo_wait_replan_base_delta_abs_mean_s_mean', 0.0):.4f}s",
        f"- mean learned replan final delta: {summary.get('shared_ppo_wait_replan_final_delta_abs_mean_s_mean', 0.0):.4f}s",
        f"- native boarded pax: {summary.get('native_boarded_pax_mean', 0.0):.1f}",
        f"- native alighted pax: {summary.get('native_alighted_pax_mean', 0.0):.1f}",
        f"- native onboard load: avg={summary.get('native_avg_onboard_load_mean', 0.0):.4f}, peak={summary.get('native_peak_onboard_load_mean', 0.0):.4f}",
        "",
        "| ep | wait | cv | reward | boarded | alighted | load | lower samples | upper decisions | gate replans | lower decisions | loss |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row.get('ep', 0))} "
            f"| {float(row.get('avg_wait_min', 0.0)):.4f} "
            f"| {float(row.get('headway_cv', 0.0)):.4f} "
            f"| {float(row.get('ep_reward', 0.0)):.4f} "
            f"| {int(row.get('native_boarded_pax', 0))} "
            f"| {int(row.get('native_alighted_pax', 0))} "
            f"| {float(row.get('native_avg_onboard_load', 0.0)):.4f} "
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
    parser.add_argument("--promotion-gate-low-signal-min", type=float, default=0.0)
    parser.add_argument("--promotion-gate-max-hf-to-lf-ratio", type=float, default=0.0)
    parser.add_argument("--promotion-gate-max-replans", type=int, default=0)
    parser.add_argument("--promotion-gate-max-total-replans", type=int, default=0)
    parser.add_argument("--promotion-replan-policy", default="actor")
    parser.add_argument("--promotion-replan-wait-gain-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-max-shift-s", type=float, default=30.0)
    parser.add_argument("--promotion-replan-state-wait-weight", type=float, default=1.0)
    parser.add_argument("--promotion-replan-frequency-weight", type=float, default=1.0)
    parser.add_argument("--promotion-replan-min-pressure", type=float, default=0.0)
    parser.add_argument("--promotion-replan-max-pressure", type=float, default=0.0)
    parser.add_argument("--promotion-replan-soft-pressure-cap", action="store_true")
    parser.add_argument("--promotion-replan-require-shift", action="store_true")
    parser.add_argument("--promotion-replan-hold-guard-weight", type=float, default=0.0)
    parser.add_argument("--promotion-replan-same-hold-max", type=float, default=0.0)
    parser.add_argument("--promotion-replan-same-wait-min", type=float, default=0.0)
    parser.add_argument("--promotion-replan-same-wait-max", type=float, default=0.0)
    parser.add_argument("--promotion-replan-gap-guard-min-ratio", type=float, default=0.0)
    parser.add_argument("--promotion-replan-gap-guard-max-ratio", type=float, default=0.0)
    parser.add_argument("--promotion-replan-gap-risk-cap-start", type=float, default=0.0)
    parser.add_argument("--promotion-replan-gap-risk-cap-full", type=float, default=0.0)
    parser.add_argument("--promotion-replan-target-headway-max-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-value-guard-min-score", type=float, default=0.0)
    parser.add_argument("--promotion-replan-value-guard-candidate-scales", default="")
    parser.add_argument("--promotion-replan-reward-floor-throughput-weight", type=float, default=0.0)
    parser.add_argument("--promotion-replan-reward-floor-fleet-weight", type=float, default=0.0)
    parser.add_argument("--promotion-replan-throughput-floor-min-score", type=float, default=0.0)
    parser.add_argument("--promotion-replan-throughput-floor-min-delta-fraction", type=float, default=0.0)
    parser.add_argument("--promotion-replan-throughput-floor-fleet-util-max", type=float, default=0.0)
    parser.add_argument("--promotion-replan-throughput-floor-same-hold-max", type=float, default=0.0)
    parser.add_argument("--promotion-replan-active-target-headway-min-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-target-headway-min-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-project-target-headway", action="store_true")
    parser.add_argument("--promotion-replan-target-headway-project-margin-s", type=float, default=0.25)
    parser.add_argument("--promotion-replan-base-delta-abs-max-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-final-delta-abs-min-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-final-delta-abs-max-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-shift-sign", type=float, default=-1.0)
    parser.add_argument("--promotion-replan-base-action", default="active")
    parser.add_argument("--promotion-replan-actor-base-trust-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-terminal-early-cap-s", type=float, default=0.0)
    parser.add_argument("--promotion-replan-terminal-early-relax", action="store_true")
    parser.add_argument("--promotion-replan-confirm-min-strength", type=float, default=0.0)
    parser.add_argument("--promotion-replan-confirm-min-low-signal", type=float, default=0.0)
    parser.add_argument("--promotion-replan-wait-credit-weight", type=float, default=0.0)
    parser.add_argument("--promotion-replan-wait-credit-clip", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-action-gain-s", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-feature-offset", type=int, default=11)
    parser.add_argument("--lower-hf-wait-context-dim", type=int, default=0)
    parser.add_argument("--lower-hf-wait-min-scale", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-max-scale", type=float, default=1.0)
    parser.add_argument("--lower-hf-wait-load-damping-weight", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-schedule-slack-damping-weight", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-queue-boost-weight", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-boarding-rescue-gain-s", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-boarding-rescue-max-s", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-boarding-rescue-queue-min", type=float, default=0.0)
    parser.add_argument("--lower-hf-wait-boarding-rescue-load-max", type=float, default=0.0)
    parser.add_argument("--adaptive-lower-drift-penalty-gain", type=float, default=0.0)
    parser.add_argument("--adaptive-lower-drift-penalty-min-scale", type=float, default=0.25)
    parser.add_argument("--offpolicy-replay-updates", type=int, default=1)
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
            promotion_gate_low_signal_min=float(args.promotion_gate_low_signal_min),
            promotion_gate_max_hf_to_lf_ratio=float(args.promotion_gate_max_hf_to_lf_ratio),
            promotion_gate_max_replans=int(args.promotion_gate_max_replans),
            promotion_gate_max_total_replans=int(args.promotion_gate_max_total_replans),
            promotion_replan_policy=str(args.promotion_replan_policy),
            promotion_replan_wait_gain_s=float(args.promotion_replan_wait_gain_s),
            promotion_replan_max_shift_s=float(args.promotion_replan_max_shift_s),
            promotion_replan_state_wait_weight=float(args.promotion_replan_state_wait_weight),
            promotion_replan_frequency_weight=float(args.promotion_replan_frequency_weight),
            promotion_replan_min_pressure=float(args.promotion_replan_min_pressure),
            promotion_replan_max_pressure=float(args.promotion_replan_max_pressure),
            promotion_replan_soft_pressure_cap=bool(args.promotion_replan_soft_pressure_cap),
            promotion_replan_require_shift=bool(args.promotion_replan_require_shift),
            promotion_replan_hold_guard_weight=float(args.promotion_replan_hold_guard_weight),
            promotion_replan_same_hold_max=float(args.promotion_replan_same_hold_max),
            promotion_replan_same_wait_min=float(args.promotion_replan_same_wait_min),
            promotion_replan_same_wait_max=float(args.promotion_replan_same_wait_max),
            promotion_replan_gap_guard_min_ratio=float(args.promotion_replan_gap_guard_min_ratio),
            promotion_replan_gap_guard_max_ratio=float(args.promotion_replan_gap_guard_max_ratio),
            promotion_replan_gap_risk_cap_start=float(args.promotion_replan_gap_risk_cap_start),
            promotion_replan_gap_risk_cap_full=float(args.promotion_replan_gap_risk_cap_full),
            promotion_replan_target_headway_max_s=float(args.promotion_replan_target_headway_max_s),
            promotion_replan_value_guard_min_score=float(
                args.promotion_replan_value_guard_min_score),
            promotion_replan_value_guard_candidate_scales=str(
                args.promotion_replan_value_guard_candidate_scales),
            promotion_replan_reward_floor_throughput_weight=float(
                args.promotion_replan_reward_floor_throughput_weight),
            promotion_replan_reward_floor_fleet_weight=float(
                args.promotion_replan_reward_floor_fleet_weight),
            promotion_replan_throughput_floor_min_score=float(
                args.promotion_replan_throughput_floor_min_score),
            promotion_replan_throughput_floor_min_delta_fraction=float(
                args.promotion_replan_throughput_floor_min_delta_fraction),
            promotion_replan_throughput_floor_fleet_util_max=float(
                args.promotion_replan_throughput_floor_fleet_util_max),
            promotion_replan_throughput_floor_same_hold_max=float(
                args.promotion_replan_throughput_floor_same_hold_max),
            promotion_replan_active_target_headway_min_s=float(
                args.promotion_replan_active_target_headway_min_s),
            promotion_replan_target_headway_min_s=float(
                args.promotion_replan_target_headway_min_s),
            promotion_replan_project_target_headway=bool(args.promotion_replan_project_target_headway),
            promotion_replan_target_headway_project_margin_s=float(
                args.promotion_replan_target_headway_project_margin_s),
            promotion_replan_base_delta_abs_max_s=float(args.promotion_replan_base_delta_abs_max_s),
            promotion_replan_final_delta_abs_min_s=float(args.promotion_replan_final_delta_abs_min_s),
            promotion_replan_final_delta_abs_max_s=float(args.promotion_replan_final_delta_abs_max_s),
            promotion_replan_shift_sign=float(args.promotion_replan_shift_sign),
            promotion_replan_base_action=str(args.promotion_replan_base_action),
            promotion_replan_actor_base_trust_s=float(args.promotion_replan_actor_base_trust_s),
            promotion_replan_terminal_early_cap_s=float(args.promotion_replan_terminal_early_cap_s),
            promotion_replan_terminal_early_relax=bool(args.promotion_replan_terminal_early_relax),
            promotion_replan_confirm_min_strength=float(
                args.promotion_replan_confirm_min_strength),
            promotion_replan_confirm_min_low_signal=float(
                args.promotion_replan_confirm_min_low_signal),
            promotion_replan_wait_credit_weight=float(args.promotion_replan_wait_credit_weight),
            promotion_replan_wait_credit_clip=float(args.promotion_replan_wait_credit_clip),
            lower_hf_wait_action_gain_s=float(args.lower_hf_wait_action_gain_s),
            lower_hf_wait_feature_offset=int(args.lower_hf_wait_feature_offset),
            lower_hf_wait_context_dim=int(args.lower_hf_wait_context_dim),
            lower_hf_wait_min_scale=float(args.lower_hf_wait_min_scale),
            lower_hf_wait_max_scale=float(args.lower_hf_wait_max_scale),
            lower_hf_wait_load_damping_weight=float(args.lower_hf_wait_load_damping_weight),
            lower_hf_wait_schedule_slack_damping_weight=float(
                args.lower_hf_wait_schedule_slack_damping_weight),
            lower_hf_wait_queue_boost_weight=float(args.lower_hf_wait_queue_boost_weight),
            lower_hf_wait_boarding_rescue_gain_s=float(
                args.lower_hf_wait_boarding_rescue_gain_s),
            lower_hf_wait_boarding_rescue_max_s=float(
                args.lower_hf_wait_boarding_rescue_max_s),
            lower_hf_wait_boarding_rescue_queue_min=float(
                args.lower_hf_wait_boarding_rescue_queue_min),
            lower_hf_wait_boarding_rescue_load_max=float(
                args.lower_hf_wait_boarding_rescue_load_max),
            adaptive_lower_drift_penalty_gain=float(args.adaptive_lower_drift_penalty_gain),
            adaptive_lower_drift_penalty_min_scale=float(args.adaptive_lower_drift_penalty_min_scale),
            offpolicy_replay_updates=int(args.offpolicy_replay_updates),
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
