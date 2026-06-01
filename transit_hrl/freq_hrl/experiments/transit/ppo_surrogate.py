"""Transit-domain adapter for the shared dual PPO Freq-HRL trainer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    LeakageRegularizer,
)
from freq_hrl.domains.transit import TransitFrequencyTracker
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import (
    DualActorCriticPPO,
    DualPPOConfig,
    LearnedPlanActionMapper,
    TrajectoryBatch,
    summarize_numeric_rows,
    train_dual_ppo,
)

SCENARIOS = ("persistent_shift", "localized_burst", "stationary")


def make_synthetic_transit_demand(seed: int, steps: int, corridors: int, scenario: str) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    t = np.arange(int(steps), dtype=np.float64)
    weights = np.linspace(0.9, 1.15, int(corridors), dtype=np.float64)
    base = 8.0 + 2.0 * np.sin(2.0 * np.pi * t / max(steps, 1)) + 0.8 * np.sin(2.0 * np.pi * t / 37.0)
    demand = base[:, None] * weights[None, :] + rng.normal(0.0, 0.9, size=(steps, corridors))
    if scenario == "persistent_shift":
        start = int(0.45 * steps)
        ramp = np.linspace(0.0, 1.0, max(steps - start, 1), dtype=np.float64)
        demand[start:, 0] += 4.5 * ramp
        if corridors > 1:
            demand[start:, 1] -= 2.0 * ramp
    elif scenario == "localized_burst":
        start = int(0.55 * steps)
        end = min(steps, start + max(8, steps // 10))
        demand[start:end, 0] += 8.0
    elif scenario != "stationary":
        raise ValueError(f"unknown transit surrogate scenario: {scenario}")
    return np.maximum(demand, 0.0)


def make_tracker(
    method: str = "ema",
    promotion_residual_threshold: float = 1.5,
    promotion_persistence_ratio: float = 0.35,
) -> TransitFrequencyTracker:
    method_key = str(method or "ema").lower()
    count_harmonic = method_key in {
        "dynamic_harmonic_nb",
        "negative_binomial_harmonic",
        "nb_harmonic",
        "dynamic_harmonic_poisson",
        "poisson_harmonic",
    }
    return TransitFrequencyTracker(
        update_interval_s=60.0,
        bin_sec=60.0,
        method=method,
        low_period_s=30 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=15 * 60.0,
        energy_period_s=10 * 60.0,
        persistence_period_s=15 * 60.0,
        persistence_threshold=1.0,
        global_demand_norm=24.0,
        local_demand_norm=12.0,
        slope_norm=4.0,
        upper_mode="low_mid",
        lower_mode="high_mid",
        promotion_enable=True,
        promotion_window_s=15 * 60.0,
        promotion_residual_threshold=promotion_residual_threshold,
        promotion_persistence_ratio=promotion_persistence_ratio,
        promotion_cooldown_s=20 * 60.0,
        harmonic_period_s=14 * 3600.0,
        harmonic_learning_rate=1.0 if count_harmonic else 0.4,
        harmonic_ridge=0.05 if count_harmonic else 1.0,
        harmonic_observation_model="negative_binomial",
        harmonic_nb_dispersion=1000.0 if count_harmonic else 24.0,
    )


def feature_vectors(
    tracker: TransitFrequencyTracker,
    service_gap: np.ndarray,
    target_delta_s: np.ndarray | None = None,
    load_proxy: np.ndarray | None = None,
    speed_shock: np.ndarray | None = None,
    station_burst: np.ndarray | None = None,
    include_native_lower_context: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    corridors = int(service_gap.size)
    if target_delta_s is None:
        target_delta_s = np.zeros(corridors, dtype=np.float64)
    upper = np.concatenate([
        tracker.upper_features("low_mid"),
        np.asarray(service_gap, dtype=np.float64) / 90.0,
        np.ones(1, dtype=np.float64),
    ])
    lower_parts = [
        tracker.lower_features(station_id=i, direction=True, mode="high_mid")
        for i in range(corridors)
    ]
    lower = np.concatenate([
        *lower_parts,
        np.asarray(service_gap, dtype=np.float64) / 90.0,
        np.asarray(target_delta_s, dtype=np.float64) / 30.0,
        np.ones(1, dtype=np.float64),
    ])
    if include_native_lower_context:
        if load_proxy is None:
            load_proxy = np.zeros(corridors, dtype=np.float64)
        if speed_shock is None:
            speed_shock = np.zeros(corridors, dtype=np.float64)
        if station_burst is None:
            station_burst = np.zeros(corridors, dtype=np.float64)
        lower = np.concatenate([
            lower,
            np.asarray(load_proxy, dtype=np.float64).reshape(-1) / 18.0,
            np.asarray(speed_shock, dtype=np.float64).reshape(-1),
            np.asarray(station_burst, dtype=np.float64).reshape(-1) / 10.0,
        ])
    return upper.astype(np.float32), lower.astype(np.float32)


def latent_headway_delta(latent: np.ndarray, max_delta_s: float = 30.0) -> np.ndarray:
    return np.tanh(np.asarray(latent, dtype=np.float64)) * float(max_delta_s)


def make_plan_mapper(
    corridors: int,
    plan_basis_dim: int,
    plan_horizon_s: float,
    plan_eval_offset_s: float,
    plan_coefficient_scale_s: float,
) -> LearnedPlanActionMapper | None:
    if int(plan_basis_dim) <= 0:
        return None
    curve = BernsteinPlanCurve(
        horizon_s=float(plan_horizon_s),
        basis_dim=int(plan_basis_dim),
        min_value=-30.0,
        max_value=30.0,
        delta_min=-float(plan_coefficient_scale_s),
        delta_max=float(plan_coefficient_scale_s),
        n_entities=int(corridors),
    )
    return LearnedPlanActionMapper(
        curve=curve,
        coefficient_scale=float(plan_coefficient_scale_s),
        eval_offset_s=float(plan_eval_offset_s),
    )


def latent_hold(latent: np.ndarray, max_hold_s: float = 45.0) -> np.ndarray:
    return float(max_hold_s) / (1.0 + np.exp(-np.asarray(latent, dtype=np.float64)))


def initialize_transit_prior(
    model: DualActorCriticPPO,
    corridors: int,
    plan_basis_dim: int = 0,
    include_native_lower_context: bool = False,
) -> None:
    if model.config.hidden_dim != 0:
        return
    with torch.no_grad():
        upper_linear = model.upper_actor.net[0]
        lower_linear = model.lower_actor.net[0]
        upper_linear.weight.zero_()
        upper_linear.bias.zero_()
        lower_linear.weight.zero_()
        lower_linear.bias.zero_()
        for i in range(int(corridors)):
            upper_rows = [i] if int(plan_basis_dim) <= 0 else [
                i * int(plan_basis_dim) + k for k in range(int(plan_basis_dim))
            ]
            for k, row in enumerate(upper_rows):
                ramp = float(k + 1) / max(float(len(upper_rows)), 1.0)
                upper_linear.weight[row, 0] = -0.75 * ramp
                upper_linear.weight[row, 2] = -0.35 * ramp
            lower_linear.bias[i] = -1.6
            if include_native_lower_context:
                context_start = model.config.lower_state_dim - 3 * int(corridors)
                burst_start = context_start + 2 * int(corridors)
                if 0 <= burst_start + i < model.config.lower_state_dim:
                    lower_linear.weight[i, burst_start + i] = 0.55


def rollout(
    model: DualActorCriticPPO,
    seed: int,
    steps: int,
    corridors: int,
    scenario: str,
    sample: bool,
    leakage_scale: float = 0.0,
    plan_mapper: LearnedPlanActionMapper | None = None,
    tracker_method: str = "ema",
    include_native_lower_context: bool = False,
    wait_upper_weight: float = 0.0,
    wait_lower_weight: float = 0.0,
    wait_lower_board_credit_weight: float = 0.0,
    wait_credit_control_gain: float = 0.0,
    lower_lf_effect_filter_window: int = 0,
    lower_lf_effect_filter_gain: float = 1.0,
    lower_lf_raw_recenter_gain: float = 0.0,
    lower_lf_raw_recenter_alpha: float = 0.10,
    upper_decision_interval: int = 1,
    promotion_forced_replan: bool = False,
    promotion_replan_strength_min: float = 0.10,
    promotion_residual_threshold: float = 1.5,
    promotion_persistence_ratio: float = 0.35,
) -> tuple[TrajectoryBatch | None, dict[str, Any]]:
    demand = make_synthetic_transit_demand(seed, steps, corridors, scenario)
    rng = np.random.default_rng(int(seed) + 7919)
    tracker = make_tracker(
        method=tracker_method,
        promotion_residual_threshold=promotion_residual_threshold,
        promotion_persistence_ratio=promotion_persistence_ratio,
    )
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=5, lower_lf_window=20),
        reward_penalty_scale=leakage_scale,
        enabled=leakage_scale > 0.0,
    )
    lower_effect_projector = (
        CausalLowFrequencyEffectProjector(
            window=int(lower_lf_effect_filter_window),
            gain=float(lower_lf_effect_filter_gain),
        )
        if int(lower_lf_effect_filter_window) > 0 else None
    )
    service_gap = np.zeros(corridors, dtype=np.float64)
    cv_gap = np.zeros(corridors, dtype=np.float64)
    demand_ema = demand[0].copy()
    upper_states: list[np.ndarray] = []
    lower_states: list[np.ndarray] = []
    upper_actions: list[np.ndarray] = []
    lower_actions: list[np.ndarray] = []
    old_upper_logp: list[float] = []
    old_lower_logp: list[float] = []
    old_upper_value: list[float] = []
    old_lower_value: list[float] = []
    constraints: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    wait_proxy: list[float] = []
    cv_proxy: list[float] = []
    hold_trace: list[np.ndarray] = []
    raw_hold_trace: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []
    plan_smoothness: list[float] = []
    plan_coeff_abs: list[float] = []
    wait_low_shares: list[float] = []
    wait_high_shares: list[float] = []
    wait_attr_penalties: list[float] = []
    wait_board_credits: list[float] = []
    wait_credit_reliefs: list[float] = []
    raw_recenter_reductions: list[np.ndarray] = []
    upper_decisions = 0
    promotion_replans = 0
    promotions = 0
    current_plan_delta = np.zeros(corridors, dtype=np.float64)
    cached_upper_state: np.ndarray | None = None
    cached_upper_out: dict[str, Any] | None = None
    cached_target_delta = np.zeros(corridors, dtype=np.float64)
    last_upper_decision_t = -10**9
    speed_shock = np.zeros(corridors, dtype=np.float64)
    hold_lf_baseline = np.zeros(corridors, dtype=np.float64)
    for t in range(int(steps)):
        arrivals = {(i, True): float(demand[t, i]) for i in range(corridors)}
        tracker.update(arrivals)
        freq_summary = tracker.summary()
        promotion_active = (
            freq_summary["freq_promotion_flag"] > 0.0
            and freq_summary["freq_promotion_strength"] >= float(promotion_replan_strength_min)
        )
        if freq_summary["freq_promotion_flag"] > 0.0:
            promotions += 1
        upper_due = (
            cached_upper_out is None
            or t - last_upper_decision_t >= max(1, int(upper_decision_interval))
            or (bool(promotion_forced_replan) and promotion_active)
        )
        if upper_due:
            if cached_upper_out is not None and bool(promotion_forced_replan) and promotion_active:
                promotion_replans += 1
            upper_state, _ = feature_vectors(tracker, service_gap)
            upper_out = model.act_upper(upper_state, sample=sample)
            if plan_mapper is None:
                target_delta = latent_headway_delta(np.asarray(upper_out["action"], dtype=np.float64))
            else:
                plan = plan_mapper.target(current_plan_delta, np.asarray(upper_out["action"], dtype=np.float64))
                target_delta = np.clip(plan.target, -30.0, 30.0)
                current_plan_delta = target_delta.copy()
                plan_smoothness.append(float(plan.smoothness_penalty))
                plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
            cached_upper_state = upper_state.copy()
            cached_upper_out = dict(upper_out)
            cached_target_delta = target_delta.copy()
            last_upper_decision_t = int(t)
            upper_decisions += 1
        else:
            upper_state = np.asarray(cached_upper_state, dtype=np.float32)
            upper_out = dict(cached_upper_out)
            target_delta = cached_target_delta.copy()

        demand_ema = 0.97 * demand_ema + 0.03 * demand[t]
        crowding = demand[t] - demand_ema
        speed_shock = 0.92 * speed_shock + rng.normal(0.0, 0.04, size=corridors)
        station_burst = np.maximum(crowding, 0.0)
        _, lower_state = feature_vectors(
            tracker,
            service_gap,
            target_delta,
            load_proxy=demand_ema,
            speed_shock=speed_shock,
            station_burst=station_burst,
            include_native_lower_context=include_native_lower_context,
        )
        lower_out = model.act_lower(lower_state, sample=sample)
        hold_s = latent_hold(np.asarray(lower_out["action"], dtype=np.float64))
        local_high_pre = np.asarray([
            max(0.0, tracker.local_high_value(i, True))
            for i in range(corridors)
        ], dtype=np.float64)
        local_low_pre = np.asarray([
            max(0.0, tracker.local_low_value(i, True))
            for i in range(corridors)
        ], dtype=np.float64)
        high_mass_pre = float(np.mean(local_high_pre))
        low_mass_pre = float(max(np.mean(local_low_pre), 1e-6))
        high_share_pre = float(np.clip(high_mass_pre / (high_mass_pre + low_mass_pre + 1e-6), 0.0, 1.0))
        hf_pressure = high_share_pre * (
            float(np.mean(np.maximum(crowding, 0.0)))
            + float(np.mean(np.maximum(service_gap, 0.0))) / 30.0
        )
        raw_recenter_reduction = (
            max(float(lower_lf_raw_recenter_gain), 0.0)
            * np.clip(hold_lf_baseline, 0.0, 45.0)
        )
        if lower_lf_raw_recenter_gain > 0.0:
            burst_gate = np.clip(np.maximum(station_burst, 0.0) / 8.0, 0.0, 1.0)
            hold_s = np.clip(
                hold_s - raw_recenter_reduction * (1.0 - 0.6 * burst_gate),
                0.0,
                45.0,
            )
        if wait_credit_control_gain > 0.0:
            hold_s = np.clip(
                hold_s - 0.35 * float(wait_credit_control_gain) * np.maximum(station_burst, 0.0),
                0.0,
                45.0,
            )
        alpha = float(np.clip(lower_lf_raw_recenter_alpha, 1e-6, 1.0))
        hold_lf_baseline = (1.0 - alpha) * hold_lf_baseline + alpha * hold_s

        hold_relief = (
            0.22
            * hold_s
            * np.tanh((np.maximum(crowding, 0.0) + np.maximum(service_gap, 0.0) / 4.0) / 8.0)
        )
        service_gap = (
            0.82 * service_gap
            + 2.4 * crowding
            + 0.28 * target_delta
            + 0.04 * hold_s
            - hold_relief
            + 4.0 * speed_shock
        )
        cv_gap = 0.75 * cv_gap + 0.20 * (service_gap - float(np.mean(service_gap))) - 0.06 * (hold_s - float(np.mean(hold_s)))
        local_high = np.asarray([
            max(0.0, tracker.local_high_value(i, True))
            for i in range(corridors)
        ], dtype=np.float64)
        local_low = np.asarray([
            max(0.0, tracker.local_low_value(i, True))
            for i in range(corridors)
        ], dtype=np.float64)
        high_mass = float(np.mean(local_high))
        low_mass = float(max(np.mean(local_low), 1e-6))
        high_share = float(np.clip(high_mass / (high_mass + low_mass + 1e-6), 0.0, 1.0))
        low_share = 1.0 - high_share
        wait_credit_relief = max(float(wait_credit_control_gain), 0.0) * hf_pressure
        wait = max(
            0.0,
            4.0
            + 0.018 * float(np.mean(demand[t]))
            + 0.012 * float(np.mean(np.maximum(service_gap, 0.0)))
            + 0.006 * float(np.mean(hold_s))
            - wait_credit_relief,
        )
        cv = float(np.std(service_gap) / 120.0 + np.std(cv_gap) / 180.0)
        overshoot = float(np.mean(np.maximum(np.abs(target_delta) - 24.0, 0.0)) / 24.0)
        wait_attr_penalty = (
            max(float(wait_upper_weight), 0.0) * low_share * wait
            + max(float(wait_lower_weight), 0.0) * high_share * wait
        )
        board_credit = (
            max(float(wait_lower_board_credit_weight), 0.0)
            * high_share
            * float(np.mean(np.maximum(crowding, 0.0)))
            / 10.0
        )
        reward = -(
            wait + 3.0 * cv + 0.18 * overshoot + 0.004 * float(np.mean(hold_s))
            + wait_attr_penalty
        ) + board_credit
        raw_lower_effect = hold_s / 45.0
        lower_effect = (
            lower_effect_projector.transform(raw_lower_effect)
            if lower_effect_projector is not None else raw_lower_effect
        )
        leak_info = leakage.update(
            upper_effect=target_delta / 30.0,
            lower_effect=lower_effect,
            reward=reward,
        )
        step_reward = float(leak_info["shaped_reward"] if leak_info["shaped_reward"] is not None else reward)
        done = t == steps - 1
        upper_states.append(upper_state)
        lower_states.append(lower_state)
        upper_actions.append(np.asarray(upper_out["action"], dtype=np.float32))
        lower_actions.append(np.asarray(lower_out["action"], dtype=np.float32))
        old_upper_logp.append(float(upper_out["logp"]))
        old_lower_logp.append(float(lower_out["logp"]))
        old_upper_value.append(float(upper_out["value"]))
        old_lower_value.append(float(lower_out["value"]))
        constraints.append(float(leak_info.get("lower_lf_penalty", 0.0)))
        rewards.append(step_reward)
        dones.append(float(done))
        wait_proxy.append(wait)
        cv_proxy.append(cv)
        hold_trace.append(np.asarray(lower_effect, dtype=np.float64).copy())
        raw_hold_trace.append(raw_lower_effect.copy())
        target_trace.append(target_delta.copy() / 30.0)
        wait_low_shares.append(low_share)
        wait_high_shares.append(high_share)
        wait_attr_penalties.append(wait_attr_penalty)
        wait_board_credits.append(board_credit)
        wait_credit_reliefs.append(wait_credit_relief)
        raw_recenter_reductions.append(np.asarray(raw_recenter_reduction, dtype=np.float64).copy())
    reg = LeakageRegularizer(upper_hf_window=5, lower_lf_window=20)
    leak = reg.compute(np.asarray(target_trace, dtype=np.float64), np.asarray(hold_trace, dtype=np.float64))
    raw_leak = reg.compute(
        np.asarray(target_trace, dtype=np.float64),
        np.asarray(raw_hold_trace, dtype=np.float64),
    )
    row = {
        "seed": int(seed),
        "scenario": scenario,
        "total_reward": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "wait_proxy": float(np.mean(wait_proxy)),
        "headway_cv": float(np.mean(cv_proxy)),
        "hold_mean": float(np.mean(np.asarray(raw_hold_trace) * 45.0)),
        "promotion_count": int(promotions),
        "leakage_penalty": float(leak["leakage_penalty"]),
        "UpperHFPower": float(leak["UpperHFPower"]),
        "LowerLFDrift": float(leak["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leak["LowerLFDriftAbs"]),
        "RawLowerLFDrift": float(raw_leak["LowerLFDrift"]),
        "RawLowerLFDriftAbs": float(raw_leak["LowerLFDriftAbs"]),
        "plan_smoothness": float(np.mean(plan_smoothness)) if plan_smoothness else 0.0,
        "plan_coeff_abs": float(np.mean(plan_coeff_abs)) if plan_coeff_abs else 0.0,
        "wait_low_share": float(np.mean(wait_low_shares)) if wait_low_shares else 0.0,
        "wait_high_share": float(np.mean(wait_high_shares)) if wait_high_shares else 0.0,
        "wait_attr_penalty": float(np.mean(wait_attr_penalties)) if wait_attr_penalties else 0.0,
        "wait_board_credit": float(np.mean(wait_board_credits)) if wait_board_credits else 0.0,
        "wait_credit_relief": float(np.mean(wait_credit_reliefs)) if wait_credit_reliefs else 0.0,
        "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
        "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
        "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
        "raw_recenter_reduction_mean": float(np.mean(raw_recenter_reductions)) if raw_recenter_reductions else 0.0,
        "upper_decision_count": int(upper_decisions),
        "promotion_replan_count": int(promotion_replans),
    }
    if not sample:
        return None, row
    batch = TrajectoryBatch(
        upper_state=np.asarray(upper_states, dtype=np.float32),
        lower_state=np.asarray(lower_states, dtype=np.float32),
        upper_action=np.asarray(upper_actions, dtype=np.float32),
        lower_action=np.asarray(lower_actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=np.float32),
        old_upper_logp=np.asarray(old_upper_logp, dtype=np.float32),
        old_lower_logp=np.asarray(old_lower_logp, dtype=np.float32),
        old_upper_value=np.asarray(old_upper_value, dtype=np.float32),
        old_lower_value=np.asarray(old_lower_value, dtype=np.float32),
        constraint=np.asarray(constraints, dtype=np.float32),
    )
    return batch, row


def objective(row: dict[str, Any]) -> float:
    return float(row["reward_mean"]) - 0.10 * float(row["leakage_penalty"]) - 0.01 * float(row["hold_mean"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "total_reward",
        "reward_mean",
        "wait_proxy",
        "headway_cv",
        "hold_mean",
        "promotion_count",
        "leakage_penalty",
        "UpperHFPower",
        "LowerLFDrift",
        "LowerLFDriftAbs",
        "RawLowerLFDrift",
        "RawLowerLFDriftAbs",
        "plan_smoothness",
        "plan_coeff_abs",
        "wait_low_share",
        "wait_high_share",
        "wait_attr_penalty",
        "wait_board_credit",
        "wait_credit_relief",
        "lower_lf_effect_filter_window",
        "lower_lf_effect_filter_gain",
        "lower_lf_raw_recenter_gain",
        "raw_recenter_reduction_mean",
        "upper_decision_count",
        "promotion_replan_count",
    ]
    return summarize_numeric_rows(rows, keys=keys)


def train_transit_surrogate_ppo(
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    corridors: int,
    scenario: str,
    iterations: int,
    seed: int,
    leakage_scale: float = 0.0,
    plan_basis_dim: int = 0,
    plan_horizon_s: float = 1800.0,
    plan_eval_offset_s: float = 300.0,
    plan_coefficient_scale_s: float = 18.0,
    lower_lf_constraint_coef: float = 0.0,
    lower_lf_constraint_target: float = 0.0,
    lower_lf_dual_lr: float = 0.0,
    lower_lf_objective_weight: float = 0.0,
    tracker_method: str = "ema",
    include_native_lower_context: bool = False,
    wait_upper_weight: float = 0.0,
    wait_lower_weight: float = 0.0,
    wait_lower_board_credit_weight: float = 0.0,
    wait_credit_control_gain: float = 0.0,
    lower_lf_effect_filter_window: int = 0,
    lower_lf_effect_filter_gain: float = 1.0,
    lower_lf_raw_recenter_gain: float = 0.0,
    lower_lf_raw_recenter_alpha: float = 0.10,
    upper_decision_interval: int = 1,
    promotion_forced_replan: bool = False,
    promotion_replan_strength_min: float = 0.10,
    promotion_residual_threshold: float = 1.5,
    promotion_persistence_ratio: float = 0.35,
) -> tuple[dict[str, Any], list[dict[str, Any]], DualActorCriticPPO]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    plan_mapper = make_plan_mapper(
        corridors=corridors,
        plan_basis_dim=plan_basis_dim,
        plan_horizon_s=plan_horizon_s,
        plan_eval_offset_s=plan_eval_offset_s,
        plan_coefficient_scale_s=plan_coefficient_scale_s,
    )
    probe = make_tracker(method=tracker_method)
    upper_dim = int(probe.upper_features("low_mid").size + corridors + 1)
    lower_context_dim = 3 * int(corridors) if include_native_lower_context else 0
    lower_dim = int(
        corridors * probe.lower_features(0, True, "high_mid").size
        + 2 * corridors
        + 1
        + lower_context_dim
    )
    config = DualPPOConfig(
        upper_state_dim=upper_dim,
        lower_state_dim=lower_dim,
        upper_action_dim=plan_mapper.action_dim if plan_mapper is not None else corridors,
        lower_action_dim=corridors,
        hidden_dim=0,
        learning_rate=0.002,
        epochs=3,
        minibatch_size=256,
        init_log_std=-2.0,
        constraint_coef=float(lower_lf_constraint_coef),
        constraint_target=float(lower_lf_constraint_target),
        constraint_dual_lr=float(lower_lf_dual_lr),
        constraint_max_lambda=20.0,
    )
    model = DualActorCriticPPO(config)
    initialize_transit_prior(
        model,
        corridors,
        plan_basis_dim=plan_basis_dim,
        include_native_lower_context=include_native_lower_context,
    )
    return train_dual_ppo(
        model=model,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        iterations=iterations,
        rollout_fn=lambda ppo_model, rollout_seed, sample: rollout(
            ppo_model,
            seed=rollout_seed,
            steps=steps,
            corridors=corridors,
            scenario=scenario,
            sample=sample,
            leakage_scale=leakage_scale if sample else 0.0,
            plan_mapper=plan_mapper,
            tracker_method=tracker_method,
            include_native_lower_context=include_native_lower_context,
            wait_upper_weight=wait_upper_weight,
            wait_lower_weight=wait_lower_weight,
            wait_lower_board_credit_weight=wait_lower_board_credit_weight,
            wait_credit_control_gain=wait_credit_control_gain,
            lower_lf_effect_filter_window=lower_lf_effect_filter_window,
            lower_lf_effect_filter_gain=lower_lf_effect_filter_gain,
            lower_lf_raw_recenter_gain=lower_lf_raw_recenter_gain,
            lower_lf_raw_recenter_alpha=lower_lf_raw_recenter_alpha,
            upper_decision_interval=upper_decision_interval,
            promotion_forced_replan=promotion_forced_replan,
            promotion_replan_strength_min=promotion_replan_strength_min,
            promotion_residual_threshold=promotion_residual_threshold,
            promotion_persistence_ratio=promotion_persistence_ratio,
        ),
        objective_fn=lambda row: objective(row) - max(float(lower_lf_objective_weight), 0.0) * float(row["LowerLFDrift"]),
        summary_fn=summarize,
        policy="ppo_dual_actor_critic",
        trainer="shared_dual_level_ppo",
        domain="transit_surrogate",
        metadata={
            "scenario": scenario,
            "steps": int(steps),
            "corridors": int(corridors),
            "tracker_method": str(tracker_method),
            "leakage_scale": float(leakage_scale),
            "plan_mode": "learned_bernstein" if plan_mapper is not None else "direct_delta",
            "include_native_lower_context": bool(include_native_lower_context),
            "wait_upper_weight": float(wait_upper_weight),
            "wait_lower_weight": float(wait_lower_weight),
            "wait_lower_board_credit_weight": float(wait_lower_board_credit_weight),
            "wait_credit_control_gain": float(wait_credit_control_gain),
            "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
            "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
            "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
            "lower_lf_raw_recenter_alpha": float(lower_lf_raw_recenter_alpha),
            "upper_decision_interval": int(upper_decision_interval),
            "promotion_forced_replan": bool(promotion_forced_replan),
            "promotion_replan_strength_min": float(promotion_replan_strength_min),
            "promotion_residual_threshold": float(promotion_residual_threshold),
            "promotion_persistence_ratio": float(promotion_persistence_ratio),
            "lower_lf_constraint_coef": float(lower_lf_constraint_coef),
            "lower_lf_constraint_target": float(lower_lf_constraint_target),
            "lower_lf_dual_lr": float(lower_lf_dual_lr),
            "lower_lf_objective_weight": float(lower_lf_objective_weight),
            **(plan_mapper.to_metadata() if plan_mapper is not None else {
                "plan_basis_dim": 0,
                "plan_horizon_s": 0.0,
                "plan_eval_offset_s": 0.0,
                "plan_coefficient_scale": 0.0,
                "plan_action_dim": int(corridors),
            }),
        },
    )


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Transit Surrogate PPO Validation",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- domain: `{payload['domain']}`",
        f"- tracker method: `{payload['tracker_method']}`",
        f"- plan mode: `{payload['plan_mode']}`",
        f"- upper decision interval: {payload['upper_decision_interval']} steps, promotion forced replan={payload['promotion_forced_replan']}",
        f"- promotion gate: residual_threshold={payload['promotion_residual_threshold']}, persistence_ratio={payload['promotion_persistence_ratio']}",
        f"- wait attribution weights: upper={payload['wait_upper_weight']}, lower={payload['wait_lower_weight']}, board_credit={payload['wait_lower_board_credit_weight']}",
        f"- wait credit control gain: {payload['wait_credit_control_gain']}",
        f"- lower LF constraint: coef={payload['lower_lf_constraint_coef']}, target={payload['lower_lf_constraint_target']}, dual_lr={payload['lower_lf_dual_lr']}",
        f"- lower LF effect projector: window={payload['lower_lf_effect_filter_window']}, gain={payload['lower_lf_effect_filter_gain']}",
        f"- raw lower hold recenter: gain={payload['lower_lf_raw_recenter_gain']}, alpha={payload['lower_lf_raw_recenter_alpha']}",
        f"- scenario: `{payload['scenario']}`",
        f"- train seeds: {payload['train_seeds']}",
        f"- eval seeds: {payload['eval_seeds']}",
        f"- reward mean: {summary['reward_mean_mean']:.4f}",
        f"- wait proxy mean: {summary['wait_proxy_mean']:.4f}",
        f"- headway CV mean: {summary['headway_cv_mean']:.4f}",
        f"- hold mean: {summary['hold_mean_mean']:.2f}",
        f"- leakage penalty mean: {summary['leakage_penalty_mean']:.4f}",
        f"- LowerLFDrift mean: {summary['LowerLFDrift_mean']:.4f}",
        f"- LowerLFDriftAbs mean: {summary['LowerLFDriftAbs_mean']:.6f}",
        f"- RawLowerLFDrift mean: {summary['RawLowerLFDrift_mean']:.4f}",
        f"- RawLowerLFDriftAbs mean: {summary['RawLowerLFDriftAbs_mean']:.6f}",
        f"- raw recenter reduction mean: {summary['raw_recenter_reduction_mean_mean']:.4f}",
        f"- plan smoothness mean: {summary['plan_smoothness_mean']:.4f}",
        f"- plan coefficient abs mean: {summary['plan_coeff_abs_mean']:.4f}",
        f"- wait high-share mean: {summary['wait_high_share_mean']:.4f}",
        f"- wait attribution penalty mean: {summary['wait_attr_penalty_mean']:.4f}",
        f"- wait credit relief mean: {summary['wait_credit_relief_mean']:.4f}",
        f"- promotion replan count mean: {summary['promotion_replan_count_mean']:.2f}",
        "",
        "This uses the same `freq_hrl.rl.train_dual_ppo` loop as the trading PPO validation, with Transit frequency features and a transit-control surrogate adapter.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[101, 131, 151])
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--corridors", type=int, default=2)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--leakage-scale", type=float, default=0.0)
    parser.add_argument("--plan-basis-dim", type=int, default=0)
    parser.add_argument("--plan-horizon-s", type=float, default=1800.0)
    parser.add_argument("--plan-eval-offset-s", type=float, default=300.0)
    parser.add_argument("--plan-coefficient-scale-s", type=float, default=18.0)
    parser.add_argument("--lower-lf-constraint-coef", type=float, default=0.0)
    parser.add_argument("--lower-lf-constraint-target", type=float, default=0.0)
    parser.add_argument("--lower-lf-dual-lr", type=float, default=0.0)
    parser.add_argument("--lower-lf-objective-weight", type=float, default=0.0)
    parser.add_argument("--tracker-method", default="ema")
    parser.add_argument("--include-native-lower-context", action="store_true")
    parser.add_argument("--wait-upper-weight", type=float, default=0.0)
    parser.add_argument("--wait-lower-weight", type=float, default=0.0)
    parser.add_argument("--wait-lower-board-credit-weight", type=float, default=0.0)
    parser.add_argument("--wait-credit-control-gain", type=float, default=0.0)
    parser.add_argument("--lower-lf-effect-filter-window", type=int, default=0)
    parser.add_argument("--lower-lf-effect-filter-gain", type=float, default=1.0)
    parser.add_argument("--lower-lf-raw-recenter-gain", type=float, default=0.0)
    parser.add_argument("--lower-lf-raw-recenter-alpha", type=float, default=0.10)
    parser.add_argument("--upper-decision-interval", type=int, default=1)
    parser.add_argument("--promotion-forced-replan", action="store_true")
    parser.add_argument("--promotion-replan-strength-min", type=float, default=0.10)
    parser.add_argument("--promotion-residual-threshold", type=float, default=1.5)
    parser.add_argument("--promotion-persistence-ratio", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/transit_ppo_surrogate"))
    args = parser.parse_args()
    payload, rows, model = train_transit_surrogate_ppo(
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=args.steps,
        corridors=args.corridors,
        scenario=args.scenario,
        iterations=args.iterations,
        seed=args.optimizer_seed,
        leakage_scale=args.leakage_scale,
        plan_basis_dim=args.plan_basis_dim,
        plan_horizon_s=args.plan_horizon_s,
        plan_eval_offset_s=args.plan_eval_offset_s,
        plan_coefficient_scale_s=args.plan_coefficient_scale_s,
        lower_lf_constraint_coef=args.lower_lf_constraint_coef,
        lower_lf_constraint_target=args.lower_lf_constraint_target,
        lower_lf_dual_lr=args.lower_lf_dual_lr,
        lower_lf_objective_weight=args.lower_lf_objective_weight,
        tracker_method=args.tracker_method,
        include_native_lower_context=args.include_native_lower_context,
        wait_upper_weight=args.wait_upper_weight,
        wait_lower_weight=args.wait_lower_weight,
        wait_lower_board_credit_weight=args.wait_lower_board_credit_weight,
        wait_credit_control_gain=args.wait_credit_control_gain,
        lower_lf_effect_filter_window=args.lower_lf_effect_filter_window,
        lower_lf_effect_filter_gain=args.lower_lf_effect_filter_gain,
        lower_lf_raw_recenter_gain=args.lower_lf_raw_recenter_gain,
        lower_lf_raw_recenter_alpha=args.lower_lf_raw_recenter_alpha,
        upper_decision_interval=args.upper_decision_interval,
        promotion_forced_replan=args.promotion_forced_replan,
        promotion_replan_strength_min=args.promotion_replan_strength_min,
        promotion_residual_threshold=args.promotion_residual_threshold,
        promotion_persistence_ratio=args.promotion_persistence_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_seed.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"model": payload, "per_seed": rows, "summary": payload["summary"]}, f, indent=2)
    torch.save(model.state_dict(), args.output_dir / "ppo_dual_actor_critic.pt")
    write_report(args.output_dir / "report.md", payload)
    print(f"wrote {args.output_dir}")
    print(
        "transit_ppo_surrogate "
        f"reward={payload['summary']['reward_mean_mean']:.3f} "
        f"wait={payload['summary']['wait_proxy_mean']:.3f} "
        f"LowerLFDrift={payload['summary']['LowerLFDrift_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
