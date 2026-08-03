"""PPO-style dual actor-critic validation for Freq-HRL trading."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    FrequencyDiagnostics,
    FrequencyRouter,
    LeakageRegularizer,
)
from freq_hrl.experiments.reproducibility import (
    derive_seed,
    training_rollout_seed,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
)
from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingCreditAssigner,
    TradingFrequencyTracker,
)
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import (
    FrequencySeparatedActorCriticPPO,
    HierarchicalRolloutBuilder,
    HierarchicalTrajectoryBatch,
    JointActorCriticPPO,
    JointPPOConfig,
    JointTrajectoryBatch,
    LearnedPlanActionMapper,
    LearnedPlanCurveState,
    SMDPPPOConfig,
    TemporalDecisionScheduler,
    causal_gru_actor_parameter_count,
    causal_gru_value_parameter_count,
    summarize_numeric_rows,
    train_frequency_separated_ppo,
    train_joint_ppo,
)

from .metrics import (
    DEFAULT_TRAINING_REWARD_SCALE,
    SELECTION_OBJECTIVE_VERSION,
    periods_per_year_from_bar_seconds,
    summarize_pnl_series,
    validation_utility,
)
from .performance_validation import SCENARIOS, make_synthetic_market


FLAT_PPO_MODES = ("flat_ppo", "flat_gru_ppo")
GENERIC_HRL_MODES = ("generic_hrl_ppo", "generic_hrl_gru_ppo")
POLICY_MODES = ("freq_hrl",) + FLAT_PPO_MODES + GENERIC_HRL_MODES
LEARNED_BASELINE_IMPLEMENTATION_VERSION = (
    "learned_baselines_v5_causal_gru_controls_2026_08_03"
)
FULL_METHOD_IMPLEMENTATION_VERSION = "freq_hrl_full_v3_credit_plan_leakage_2026_08_03"
EXECUTION_TIMELINE_CONTRACTS = (
    "legacy_pre_trade_v2",
    "causal_post_trade_v3",
)
METHOD_CONTRACTS = (
    "routing_core_v2",
    "curve_credit_control_v3",
    "full_freq_hrl_v3",
)

RAW_HISTORY_WINDOW = 120


def resolve_method_contract(method_contract: str) -> dict[str, bool]:
    contract = str(method_contract)
    if contract not in METHOD_CONTRACTS:
        raise ValueError(
            f"unknown method_contract: {contract}; expected one of {METHOD_CONTRACTS}"
        )
    return {
        "execute_plan_curve": contract != "routing_core_v2",
        "use_additive_frequency_credit": contract != "routing_core_v2",
        "constrain_raw_lower_effect": contract == "full_freq_hrl_v3",
    }


def gross_cap(target: np.ndarray, max_gross: float = 1.0) -> np.ndarray:
    out = np.asarray(target, dtype=np.float64).reshape(-1)
    gross = float(np.sum(np.abs(out)))
    if gross > max_gross and gross > 1e-12:
        out = out * (max_gross / gross)
    return out


def resize(value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


def make_tracker(assets: int) -> TradingFrequencyTracker:
    return TradingFrequencyTracker(
        bar_sec=60.0,
        method="ema",
        low_period_s=120 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=30 * 60.0,
        energy_period_s=10 * 60.0,
        persistence_period_s=30 * 60.0,
        persistence_threshold=0.0010,
        feature_norm=np.ones(assets) * 0.0015,
        promotion_enable=True,
        promotion_window_s=30 * 60.0,
        promotion_residual_threshold=0.00035,
        promotion_persistence_ratio=0.50,
        promotion_cooldown_s=10 * 60.0,
        promotion_regime_threshold=3e-05,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.05,
    )


def frequency_separated_feature_vectors(
    freq: dict[str, Any],
    position: np.ndarray,
    target: np.ndarray | None = None,
    *,
    leakage_feedback: float = 0.0,
    progress: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build v2 policy vectors through the machine-checked router contract."""
    dim = int(position.size)
    scale = 0.0014
    promotion = dict(freq.get("promotion", {}) or {})
    router = FrequencyRouter()
    upper = router.upper_view(
        freq,
        z_upper={"position": np.asarray(position, dtype=np.float64).copy()},
        promotion=promotion,
        leakage_feedback=float(leakage_feedback),
    )
    current_target = (
        np.zeros(dim, dtype=np.float64)
        if target is None else np.asarray(target, dtype=np.float64).reshape(-1)
    )
    if current_target.size != dim:
        current_target = np.resize(current_target, dim)
    lower = router.lower_view(
        freq,
        z_lower={"position": np.asarray(position, dtype=np.float64).copy()},
        current_plan=current_target,
        target_error=current_target - np.asarray(position, dtype=np.float64),
    )

    forecast = resize(upper["x_low_forecast"], dim) / scale
    energy = np.sqrt(np.maximum(resize(upper["x_high_energy"], dim), 0.0)) / scale
    promote = 1.0 if bool(promotion.get("promote", False)) else 0.0
    upper_state = np.concatenate([
        resize(upper["x_low"], dim) / scale,
        forecast,
        resize(upper["x_low_uncertainty"], dim) / scale,
        np.tanh(energy),
        np.tanh(resize(upper["x_high_persistence"], dim)),
        np.asarray(position, dtype=np.float64),
        np.asarray([
            promote,
            float(promotion.get("promotion_strength", 0.0)),
            float(promotion.get("shock_age", 0.0)) / 30.0,
            float(leakage_feedback),
            float(np.clip(progress, 0.0, 1.0)),
        ], dtype=np.float64),
    ])

    lower_energy = np.sqrt(np.maximum(resize(freq.get("x_high_energy", 0.0), dim), 0.0)) / scale
    lower_state = np.concatenate([
        current_target,
        np.asarray(position, dtype=np.float64),
        current_target - np.asarray(position, dtype=np.float64),
        resize(lower["x_high"], dim) / scale,
        resize(lower["x_mid"], dim) / scale,
        resize(freq.get("x_high_delta", 0.0), dim) / scale,
        np.tanh(lower_energy),
        np.tanh(resize(lower["shock_age"], dim) / 30.0),
        np.asarray([float(np.clip(progress, 0.0, 1.0))], dtype=np.float64),
    ])
    return upper_state.astype(np.float32), lower_state.astype(np.float32)


def causal_raw_history_window(
    raw_history: np.ndarray,
    *,
    assets: int,
    window: int = RAW_HISTORY_WINDOW,
) -> np.ndarray:
    """Return every causal sample in a fixed-size oldest-to-newest window."""

    history = np.asarray(raw_history, dtype=np.float64)
    if history.ndim == 1:
        history = history.reshape(1, -1)
    if history.ndim != 2 or history.shape[1] != int(assets):
        raise ValueError(
            f"raw_history must have shape (time, {assets}), got {history.shape}"
        )
    if history.shape[0] == 0:
        raise ValueError("raw_history must contain at least one causal observation")
    size = int(window)
    if size < 1:
        raise ValueError("window must be positive")
    observed = history[-size:]
    if observed.shape[0] < size:
        padding = np.repeat(
            observed[:1], size - observed.shape[0], axis=0
        )
        observed = np.concatenate([padding, observed], axis=0)
    return observed.reshape(-1)


def raw_hierarchical_feature_vectors(
    raw_history: np.ndarray,
    position: np.ndarray,
    target: np.ndarray | None = None,
    *,
    progress: float = 0.0,
    history_window: int = RAW_HISTORY_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Capacity-matched raw-history features for non-frequency baselines."""

    position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
    dim = int(position_arr.size)
    raw_window = causal_raw_history_window(
        raw_history, assets=dim, window=int(history_window)
    ) / 0.0014
    current_target = (
        np.zeros(dim, dtype=np.float64)
        if target is None else resize(target, dim)
    )
    gap = current_target - position_arr
    history_coverage = min(
        float(np.asarray(raw_history).reshape(-1, dim).shape[0])
        / float(max(int(history_window), 1)),
        1.0,
    )
    upper_state = np.concatenate([
        raw_window,
        np.asarray([history_coverage], dtype=np.float64),
        position_arr,
        current_target,
        np.asarray([
            0.0,
            0.0,
            0.0,
            float(np.clip(progress, 0.0, 1.0)),
        ], dtype=np.float64),
    ])
    lower_state = np.concatenate([
        raw_window,
        np.asarray([history_coverage], dtype=np.float64),
        current_target,
        position_arr,
        gap,
        np.asarray([float(np.clip(progress, 0.0, 1.0))], dtype=np.float64),
    ])
    return upper_state.astype(np.float32), lower_state.astype(np.float32)


def smdp_policy_feature_vectors(
    *,
    policy_mode: str,
    freq: dict[str, Any],
    raw_history: np.ndarray,
    position: np.ndarray,
    target: np.ndarray | None,
    leakage_feedback: float,
    progress: float = 0.0,
    history_window: int = RAW_HISTORY_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    if str(policy_mode) == "freq_hrl":
        return frequency_separated_feature_vectors(
            freq,
            position,
            target=target,
            leakage_feedback=leakage_feedback,
            progress=progress,
        )
    return raw_hierarchical_feature_vectors(
        raw_history,
        position,
        target=target,
        progress=progress,
        history_window=int(history_window),
    )


def flat_joint_feature_vector(
    raw_history: np.ndarray,
    position: np.ndarray,
    previous_target: np.ndarray | None,
    *,
    progress: float,
    history_window: int = RAW_HISTORY_WINDOW,
) -> np.ndarray:
    """Observation for standard flat PPO with the same causal history span."""

    _, lower = raw_hierarchical_feature_vectors(
        raw_history,
        position,
        target=previous_target,
        progress=progress,
        history_window=int(history_window),
    )
    return lower


def raw_upper_state_dim(
    assets: int, history_window: int = RAW_HISTORY_WINDOW
) -> int:
    return (int(history_window) + 2) * int(assets) + 5


def raw_lower_state_dim(
    assets: int, history_window: int = RAW_HISTORY_WINDOW
) -> int:
    return (int(history_window) + 3) * int(assets) + 2


def capacity_match_status(ratio: float, tolerance: float = 0.05) -> str:
    return (
        "matched_within_5pct"
        if abs(float(ratio) - 1.0) <= float(tolerance)
        else "closest_available_outside_5pct"
    )


def _actor_parameter_count(state_dim: int, action_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    if hidden <= 0:
        return int(state_dim * action_dim + 2 * action_dim)
    return int(
        hidden * hidden
        + hidden * (int(state_dim) + int(action_dim) + 2)
        + 2 * int(action_dim)
    )


def _value_parameter_count(state_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    if hidden <= 0:
        return int(state_dim + 1)
    return int(hidden * hidden + hidden * (int(state_dim) + 3) + 1)


def smdp_parameter_count(config: SMDPPPOConfig) -> int:
    """Analytic active-parameter count for the two-level PPO core."""

    if str(config.state_encoder) == "causal_gru":
        def actor(state_dim: int, action_dim: int) -> int:
            return causal_gru_actor_parameter_count(
                state_dim=state_dim,
                action_dim=action_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
            )

        def value(state_dim: int) -> int:
            return causal_gru_value_parameter_count(
                state_dim=state_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
            )

        return int(
            actor(config.upper_state_dim, config.upper_action_dim)
            + actor(config.lower_state_dim, config.lower_action_dim)
            + value(config.upper_state_dim)
            + 2 * value(config.lower_state_dim)
        )
    if str(config.state_encoder) != "mlp":
        raise ValueError(f"unknown state_encoder: {config.state_encoder}")
    return int(
        _actor_parameter_count(
            config.upper_state_dim, config.upper_action_dim, config.hidden_dim
        )
        + _actor_parameter_count(
            config.lower_state_dim, config.lower_action_dim, config.hidden_dim
        )
        + _value_parameter_count(config.upper_state_dim, config.hidden_dim)
        + 2 * _value_parameter_count(config.lower_state_dim, config.hidden_dim)
    )


def joint_parameter_count(config: JointPPOConfig) -> int:
    """Analytic active-parameter count for canonical flat PPO."""

    if str(config.state_encoder) == "causal_gru":
        return int(
            causal_gru_actor_parameter_count(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
            )
            + causal_gru_value_parameter_count(
                state_dim=config.state_dim,
                history_window=config.raw_history_window,
                raw_feature_dim=config.raw_feature_dim,
                hidden_dim=config.hidden_dim,
            )
        )
    if str(config.state_encoder) != "mlp":
        raise ValueError(f"unknown state_encoder: {config.state_encoder}")
    return int(
        _actor_parameter_count(config.state_dim, config.action_dim, config.hidden_dim)
        + _value_parameter_count(config.state_dim, config.hidden_dim)
    )


def capacity_matched_joint_hidden_dim(
    *,
    target_parameter_count: int,
    state_dim: int,
    action_dim: int,
    requested_hidden_dim: int,
    state_encoder: str = "mlp",
    raw_history_window: int = 0,
    raw_feature_dim: int = 0,
) -> tuple[int, int, float]:
    """Choose the closest active flat-PPO capacity without dummy parameters."""

    requested = int(requested_hidden_dim)
    if requested <= 0:
        config = JointPPOConfig(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            hidden_dim=0,
            state_encoder=str(state_encoder),
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(raw_feature_dim),
        )
        count = joint_parameter_count(config)
        return 0, count, float(count / max(int(target_parameter_count), 1))
    upper = max(32, 4 * requested)
    candidates = []
    for hidden in range(1, upper + 1):
        config = JointPPOConfig(
            state_dim=int(state_dim),
            action_dim=int(action_dim),
            hidden_dim=hidden,
            state_encoder=str(state_encoder),
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(raw_feature_dim),
        )
        count = joint_parameter_count(config)
        candidates.append((abs(count - int(target_parameter_count)), hidden, count))
    _, hidden, count = min(candidates)
    return hidden, count, float(count / max(int(target_parameter_count), 1))


def capacity_matched_smdp_hidden_dim(
    *,
    target_parameter_count: int,
    upper_state_dim: int,
    lower_state_dim: int,
    upper_action_dim: int,
    lower_action_dim: int,
    requested_hidden_dim: int,
    state_encoder: str = "mlp",
    raw_history_window: int = 0,
    raw_feature_dim: int = 0,
) -> tuple[int, int, float]:
    """Match a full-window generic HRL to the active Freq-HRL capacity."""

    requested = int(requested_hidden_dim)
    if requested <= 0:
        config = SMDPPPOConfig(
            upper_state_dim=int(upper_state_dim),
            lower_state_dim=int(lower_state_dim),
            upper_action_dim=int(upper_action_dim),
            lower_action_dim=int(lower_action_dim),
            hidden_dim=0,
            state_encoder=str(state_encoder),
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(raw_feature_dim),
        )
        count = smdp_parameter_count(config)
        return 0, count, float(count / max(int(target_parameter_count), 1))
    upper = max(32, 4 * max(requested, 1))
    candidates = []
    for hidden in range(1, upper + 1):
        config = SMDPPPOConfig(
            upper_state_dim=int(upper_state_dim),
            lower_state_dim=int(lower_state_dim),
            upper_action_dim=int(upper_action_dim),
            lower_action_dim=int(lower_action_dim),
            hidden_dim=hidden,
            state_encoder=str(state_encoder),
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(raw_feature_dim),
        )
        count = smdp_parameter_count(config)
        candidates.append((abs(count - int(target_parameter_count)), hidden, count))
    _, hidden, count = min(candidates)
    return hidden, count, float(count / max(int(target_parameter_count), 1))


def initialize_smdp_frequency_prior(
    model: FrequencySeparatedActorCriticPPO,
    assets: int,
    plan_basis_dim: int = 0,
) -> None:
    """Initialize v2 actors without introducing raw HF into the upper policy."""
    if model.config.hidden_dim != 0:
        return
    with torch.no_grad():
        upper_linear = model.upper_actor.net[0]
        lower_linear = model.lower_actor.net[0]
        upper_linear.weight.zero_()
        upper_linear.bias.zero_()
        lower_linear.weight.zero_()
        lower_linear.bias.zero_()
        for i in range(assets):
            upper_rows = [i] if int(plan_basis_dim) <= 0 else [
                i * int(plan_basis_dim) + k for k in range(int(plan_basis_dim))
            ]
            for k, row in enumerate(upper_rows):
                ramp = float(k + 1) / max(float(len(upper_rows)), 1.0)
                upper_linear.weight[row, i] = 0.65 * ramp
                upper_linear.weight[row, assets + i] = 0.25 * ramp
                upper_linear.weight[row, 3 * assets + i] = -0.05 * ramp
                upper_linear.weight[row, 5 * assets + i] = -0.10 * ramp
            lower_linear.weight[i, 2 * assets + i] = 0.30
            lower_linear.weight[i, 3 * assets + i] = 0.08
            lower_linear.weight[i, 5 * assets + i] = 0.04
            lower_linear.bias[i] = 0.20


def latent_target(latent: np.ndarray) -> np.ndarray:
    return gross_cap(np.tanh(np.asarray(latent, dtype=np.float64)))


def make_plan_mapper(
    assets: int,
    plan_basis_dim: int,
    plan_horizon_s: float,
    plan_eval_offset_s: float,
    plan_coefficient_scale: float,
    *,
    anchor_first_coefficient: bool = False,
) -> LearnedPlanActionMapper | None:
    if int(plan_basis_dim) <= 0:
        return None
    curve = BernsteinPlanCurve(
        horizon_s=float(plan_horizon_s),
        basis_dim=int(plan_basis_dim),
        min_value=-1.0,
        max_value=1.0,
        delta_min=-float(plan_coefficient_scale),
        delta_max=float(plan_coefficient_scale),
        n_entities=int(assets),
    )
    return LearnedPlanActionMapper(
        curve=curve,
        coefficient_scale=float(plan_coefficient_scale),
        eval_offset_s=float(plan_eval_offset_s),
        anchor_first_coefficient=bool(anchor_first_coefficient),
    )


def latent_speed(latent: np.ndarray) -> np.ndarray:
    return np.clip(0.05 + 0.95 / (1.0 + np.exp(-np.asarray(latent, dtype=np.float64))), 0.05, 1.0)


def bounded_speed(action: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return np.clip(0.05 + 0.95 * (bounded + 1.0) / 2.0, 0.05, 1.0)


def flat_latent_speed(latent: np.ndarray) -> np.ndarray:
    return bounded_speed(np.tanh(np.asarray(latent, dtype=np.float64)))


def joint_flat_rollout(
    model: JointActorCriticPPO,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    sample: bool,
    leakage_scale: float = 0.0,
    lower_lf_effect_filter_window: int = 0,
    lower_lf_effect_filter_gain: float = 1.0,
    lower_lf_raw_recenter_gain: float = 0.0,
    lower_lf_raw_recenter_scale: float = 0.10,
    policy_mode: str = "flat_ppo",
    reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
    mark_to_market_timing: str = "pre_trade",
    volume_impact_bps: float = 0.0,
    execution_timeline_contract: str = "legacy_pre_trade_v2",
    method_contract: str = "routing_core_v2",
) -> tuple[JointTrajectoryBatch | None, dict[str, float]]:
    policy_mode = str(policy_mode)
    if policy_mode not in FLAT_PPO_MODES:
        raise ValueError(f"joint flat rollout does not support {policy_mode}")
    history_window = int(
        getattr(model.config, "raw_history_window", 0) or RAW_HISTORY_WINDOW
    )
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
    env = PortfolioExecutionEnv(
        data["returns"],
        volumes=data["volume"],
        config=PortfolioExecutionConfig(
            transaction_cost_bps=50.0,
            slippage_bps=10.0,
            volume_impact_bps=float(volume_impact_bps),
            max_leverage=1.0,
            inventory_drift_penalty=0.002,
            drawdown_penalty=0.0,
            mark_to_market_timing=str(mark_to_market_timing),
        ),
    )
    tracker = make_tracker(assets)
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
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
    diagnostics = FrequencyDiagnostics(mi_bins=8)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    old_logp: list[float] = []
    old_value: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    targets: list[np.ndarray] = []
    lower_effects: list[np.ndarray] = []
    raw_lower_effects: list[np.ndarray] = []
    raw_recenter_boosts: list[np.ndarray] = []
    task_credits: list[float] = []
    promotions = 0
    raw_history: list[np.ndarray] = []
    previous_target = np.zeros(assets, dtype=np.float64)
    env.reset()
    for t in range(steps):
        raw_history.append(np.asarray(data["predictor"][t], dtype=np.float64).copy())
        freq = tracker.update_bar(data["predictor"][t], t=float(t * 60.0))
        if bool(dict(freq.get("promotion", {}) or {}).get("promote", False)):
            promotions += 1
        state = flat_joint_feature_vector(
            np.asarray(raw_history, dtype=np.float64),
            env.position.copy(),
            previous_target,
            progress=t / max(int(steps) - 1, 1),
            history_window=history_window,
        )
        policy_out = model.act(state, sample=sample)
        latent = np.asarray(policy_out["action"], dtype=np.float64)
        if latent.size != 2 * int(assets):
            raise RuntimeError(
                f"flat PPO action must have {2 * int(assets)} coordinates, "
                f"got {latent.size}"
            )
        target = latent_target(latent[:assets])
        speed = flat_latent_speed(latent[assets:])
        pre_gap = np.asarray(target, dtype=np.float64) - env.position.copy()
        raw_recenter_boost = max(float(lower_lf_raw_recenter_gain), 0.0) * np.tanh(
            np.abs(pre_gap) / max(float(lower_lf_raw_recenter_scale), 1e-9)
        )
        if lower_lf_raw_recenter_gain > 0.0:
            speed = np.clip(speed + raw_recenter_boost, 0.05, 1.0)
        env.set_target(target)
        _, reward, done, info = env.lower_step({
            "execution_speed": speed,
            "residual_order": np.zeros(assets, dtype=np.float64),
        })
        raw_lower_effect = np.asarray(info["position"], dtype=np.float64) - np.asarray(info["target"], dtype=np.float64)
        lower_effect = (
            lower_effect_projector.transform(raw_lower_effect)
            if lower_effect_projector is not None else raw_lower_effect
        )
        diagnostics.log_step(
            t=float(t * 60.0),
            states={
                "regime_shift": t == int(data["regime_shift_t"][0]),
                "shock": bool(np.any(data["shock_mask"][t])),
                "lower_responded": float(info["turnover"]) > 0.02,
            },
            actions={
                "upper": target,
                "lower": np.asarray(info["trade"], dtype=np.float64),
            },
            freq_features=dict(freq),
            effects={
                "upper": target,
                "lower": lower_effect,
            },
        )
        leak_info = leakage.update(upper_effect=target, lower_effect=lower_effect, reward=float(reward))
        step_reward = float(leak_info["shaped_reward"] if leak_info["shaped_reward"] is not None else reward)
        states.append(state)
        actions.append(latent.astype(np.float32))
        old_logp.append(float(policy_out["logp"]))
        old_value.append(float(policy_out["value"]))
        rewards.append(float(reward_scale) * step_reward)
        task_credits.append(step_reward)
        dones.append(float(done))
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        lower_effects.append(lower_effect.copy())
        raw_lower_effects.append(raw_lower_effect.copy())
        raw_recenter_boosts.append(np.asarray(raw_recenter_boost, dtype=np.float64).copy())
        previous_target = target.copy()
        if done:
            break
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    reg = LeakageRegularizer(upper_hf_window=6, lower_lf_window=24)
    leak = reg.compute(np.asarray(targets, dtype=np.float64), np.asarray(lower_effects, dtype=np.float64)) if targets else {
        "leakage_penalty": 0.0,
        "UpperHFPower": 0.0,
        "LowerLFDrift": 0.0,
        "LowerLFDriftAbs": 0.0,
    }
    raw_leak = reg.compute(np.asarray(targets, dtype=np.float64), np.asarray(raw_lower_effects, dtype=np.float64)) if targets else {
        "LowerLFDrift": 0.0,
        "LowerLFDriftAbs": 0.0,
    }
    diag = diagnostics.summarize_episode()
    financial = summarize_pnl_series(
        pnl,
        eq,
        periods_per_year=periods_per_year_from_bar_seconds(60.0),
    )
    row = {
        "baseline": policy_mode,
        "policy_mode": policy_mode,
        "seed": int(seed),
        "scenario": scenario,
        **financial,
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotions),
        "promotion_replan_count": 0,
        "scheduled_replan_count": 0,
        "upper_decision_count": int(len(targets)),
        "lower_decision_count": int(len(targets)),
        "upper_mean_duration": 1.0,
        "upper_to_lower_ratio": 1.0,
        "leakage_penalty": float(leak["leakage_penalty"]),
        "UpperHFPower": float(leak["UpperHFPower"]),
        "LowerLFDrift": float(leak["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leak["LowerLFDriftAbs"]),
        "RawLowerLFDrift": float(raw_leak["LowerLFDrift"]),
        "RawLowerLFDriftAbs": float(raw_leak["LowerLFDriftAbs"]),
        "FocusScore": float(diag["FocusScore"]),
        "upper_low_mi": float(diag.get("upper_low_mi", 0.0)),
        "upper_high_mi": float(diag.get("upper_high_mi", 0.0)),
        "lower_high_mi": float(diag.get("lower_high_mi", 0.0)),
        "lower_low_mi": float(diag.get("lower_low_mi", 0.0)),
        "upper_credit_mean": float(np.mean(task_credits)) if task_credits else 0.0,
        "lower_credit_mean": float(np.mean(task_credits)) if task_credits else 0.0,
        "plan_smoothness": 0.0,
        "plan_coeff_abs": 0.0,
        "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
        "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
        "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
        "raw_recenter_boost_mean": float(np.mean(raw_recenter_boosts)) if raw_recenter_boosts else 0.0,
        "protocol_valid": 1.0,
        "mark_to_market_timing": str(mark_to_market_timing),
        "volume_impact_bps": float(volume_impact_bps),
        "execution_timeline_contract": str(execution_timeline_contract),
        "method_contract": str(method_contract),
        "routing_contract": (
            "causal_raw_full_episode_gru"
            if policy_mode == "flat_gru_ppo"
            else "causal_raw_contiguous_window"
        ),
        "temporal_contract": "primitive_joint_action",
    }
    if not sample:
        return None, row
    batch = JointTrajectoryBatch(
        state=np.asarray(states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=np.float32),
        old_logp=np.asarray(old_logp, dtype=np.float32),
        old_value=np.asarray(old_value, dtype=np.float32),
    )
    return batch, row


def smdp_rollout(
    model: FrequencySeparatedActorCriticPPO,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    sample: bool,
    leakage_scale: float = 0.0,
    plan_mapper: LearnedPlanActionMapper | None = None,
    lower_lf_effect_filter_window: int = 0,
    lower_lf_effect_filter_gain: float = 1.0,
    lower_lf_raw_recenter_gain: float = 0.0,
    lower_lf_raw_recenter_scale: float = 0.10,
    upper_period: int = 30,
    min_upper_duration: int = 5,
    policy_mode: str = "freq_hrl",
    reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
    mark_to_market_timing: str = "pre_trade",
    volume_impact_bps: float = 0.0,
    execute_plan_curve: bool = False,
    use_additive_frequency_credit: bool = False,
    constrain_raw_lower_effect: bool = False,
    plan_smoothness_weight: float = 0.0,
    execution_timeline_contract: str = "legacy_pre_trade_v2",
    method_contract: str = "routing_core_v2",
) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, float]]:
    """Roll out generic-HRL or Freq-HRL on asynchronous SMDP streams."""
    policy_mode = str(policy_mode)
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"unknown policy_mode: {policy_mode}")
    if policy_mode in FLAT_PPO_MODES:
        raise ValueError(
            f"{policy_mode} must use joint_flat_rollout; the SMDP flat path is forbidden"
        )
    if execute_plan_curve and plan_mapper is None:
        raise ValueError("execute_plan_curve requires a learned plan mapper")
    if use_additive_frequency_credit and str(mark_to_market_timing) != "post_trade":
        raise ValueError(
            "additive frequency credit requires post_trade mark-to-market timing"
        )
    history_window = int(
        getattr(model.config, "raw_history_window", 0) or RAW_HISTORY_WINDOW
    )
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
    env_config = PortfolioExecutionConfig(
        transaction_cost_bps=50.0,
        slippage_bps=10.0,
        volume_impact_bps=float(volume_impact_bps),
        max_leverage=1.0,
        inventory_drift_penalty=0.002,
        drawdown_penalty=0.0,
        mark_to_market_timing=str(mark_to_market_timing),
    )
    env = PortfolioExecutionEnv(data["returns"], volumes=data["volume"], config=env_config)
    tracker = make_tracker(assets)
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
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
    scheduler = TemporalDecisionScheduler(
        upper_period=int(upper_period),
        min_upper_duration=int(min_upper_duration),
    )
    builder = HierarchicalRolloutBuilder(gamma=float(model.config.gamma))
    diagnostics = FrequencyDiagnostics(mi_bins=8)
    plan_state = (
        LearnedPlanCurveState(mapper=plan_mapper, gross_cap=1.0)
        if execute_plan_curve and plan_mapper is not None else None
    )
    credit_assigner = TradingCreditAssigner()

    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    targets: list[np.ndarray] = []
    lower_effects: list[np.ndarray] = []
    raw_lower_effects: list[np.ndarray] = []
    raw_recenter_boosts: list[np.ndarray] = []
    plan_smoothness: list[float] = []
    plan_coeff_abs: list[float] = []
    upper_credits: list[float] = []
    lower_credits: list[float] = []
    upper_task_credits: list[float] = []
    lower_task_credits: list[float] = []
    plan_returns: list[float] = []
    execution_deviation_returns: list[float] = []
    credit_reconstruction_errors: list[float] = []
    upper_leakage_costs: list[float] = []
    lower_leakage_costs: list[float] = []
    decision_reasons: list[str] = []
    promotion_signals = 0
    latest_leakage_feedback = 0.0
    current_target: np.ndarray | None = None
    raw_history: list[np.ndarray] = []
    pending_plan_smoothness_cost = 0.0

    env.reset()
    for t in range(steps):
        raw_history.append(np.asarray(data["predictor"][t], dtype=np.float64).copy())
        freq = tracker.update_bar(data["predictor"][t], t=float(t * 60.0))
        promotion = dict(freq.get("promotion", {}) or {})
        promote = bool(promotion.get("promote", False))
        if promote:
            promotion_signals += 1
        reason = scheduler.decision_reason(
            t,
            promotion=bool(promote and policy_mode == "freq_hrl"),
        )
        if reason is not None:
            upper_state, _ = smdp_policy_feature_vectors(
                policy_mode=policy_mode,
                freq=dict(freq),
                raw_history=np.asarray(raw_history, dtype=np.float64),
                position=env.position.copy(),
                target=current_target,
                leakage_feedback=latest_leakage_feedback,
                progress=t / max(int(steps) - 1, 1),
                history_window=history_window,
            )
            upper_out = model.act_upper(upper_state, sample=sample)
            if plan_mapper is None:
                current_target = latent_target(np.asarray(upper_out["action"], dtype=np.float64))
            elif plan_state is not None:
                plan = plan_state.activate(
                    now_s=float(t * 60.0),
                    current_value=(
                        env.position.copy()
                        if current_target is None else current_target
                    ),
                    latent_action=np.asarray(
                        upper_out["action"], dtype=np.float64
                    ),
                )
                current_target = gross_cap(plan.target)
                plan_smoothness.append(float(plan.smoothness_penalty))
                plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
                pending_plan_smoothness_cost = max(
                    float(plan_smoothness_weight), 0.0
                ) * float(plan.smoothness_penalty)
            else:
                plan = plan_mapper.target(
                    env.position.copy(), np.asarray(upper_out["action"], dtype=np.float64)
                )
                current_target = gross_cap(plan.target)
                plan_smoothness.append(float(plan.smoothness_penalty))
                plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
            builder.begin_upper(
                state=upper_state,
                action=np.asarray(upper_out["action"], dtype=np.float32),
                logp=float(upper_out["logp"]),
                value=float(upper_out["value"]),
            )
            scheduler.mark_decision(t)
            decision_reasons.append(reason)

        if plan_state is not None and plan_state.active:
            current_target = gross_cap(plan_state.value_at(float(t * 60.0)))
        if current_target is None:
            raise RuntimeError("upper scheduler did not create an initial target")
        _, lower_state = smdp_policy_feature_vectors(
            policy_mode=policy_mode,
            freq=dict(freq),
            raw_history=np.asarray(raw_history, dtype=np.float64),
            position=env.position.copy(),
            target=current_target,
            leakage_feedback=latest_leakage_feedback,
            progress=t / max(int(steps) - 1, 1),
            history_window=history_window,
        )
        lower_out = model.act_lower(lower_state, sample=sample)
        speed = latent_speed(np.asarray(lower_out["action"], dtype=np.float64))
        pre_gap = np.asarray(current_target, dtype=np.float64) - env.position.copy()
        raw_recenter_boost = max(float(lower_lf_raw_recenter_gain), 0.0) * np.tanh(
            np.abs(pre_gap) / max(float(lower_lf_raw_recenter_scale), 1e-9)
        )
        if lower_lf_raw_recenter_gain > 0.0:
            speed = np.clip(speed + raw_recenter_boost, 0.05, 1.0)
        env.set_target(current_target)
        _, reward, done, info = env.lower_step({
            "execution_speed": speed,
            "residual_order": np.zeros(assets, dtype=np.float64),
        })
        raw_lower_effect = np.asarray(info["position"], dtype=np.float64) - np.asarray(
            info["target"], dtype=np.float64
        )
        lower_effect = (
            lower_effect_projector.transform(raw_lower_effect)
            if lower_effect_projector is not None else raw_lower_effect
        )
        diagnostics.log_step(
            t=float(t * 60.0),
            states={
                "regime_shift": t == int(data["regime_shift_t"][0]),
                "shock": bool(np.any(data["shock_mask"][t])),
                "lower_responded": float(info["turnover"]) > 0.02,
            },
            actions={"upper": current_target, "lower": np.asarray(info["trade"], dtype=np.float64)},
            freq_features=dict(freq),
            effects={"upper": current_target, "lower": lower_effect},
        )
        leak_info = leakage.update(
            upper_effect=current_target,
            lower_effect=(raw_lower_effect if constrain_raw_lower_effect else lower_effect),
            reward=float(reward),
        )
        shaped_reward = float(
            leak_info["shaped_reward"] if leak_info["shaped_reward"] is not None else reward
        )
        leakage_reward_penalty = max(float(reward) - shaped_reward, 0.0)
        if use_additive_frequency_credit:
            upper_leakage_cost = max(float(leakage_scale), 0.0) * float(
                leak_info.get("upper_hf_penalty", 0.0)
            )
            lower_leakage_cost = max(float(leakage_scale), 0.0) * float(
                leak_info.get("lower_lf_penalty", 0.0)
            )
            credit = credit_assigner.assign(
                info,
                active_plan=current_target,
                upper_leakage_cost=upper_leakage_cost,
                lower_leakage_cost=lower_leakage_cost,
                plan_smoothness_cost=pending_plan_smoothness_cost,
            )
            upper_credit = float(credit.upper_training_credit)
            lower_credit = float(credit.lower_training_credit)
            upper_task_credit = float(credit.upper_task_credit)
            lower_task_credit = float(credit.lower_task_credit)
            plan_return = float(credit.plan_return)
            execution_deviation_return = float(
                credit.execution_deviation_return
            )
            reconstruction_error = float(credit.task_reconstruction_error)
            pending_plan_smoothness_cost = 0.0
        else:
            upper_leakage_cost = 0.0
            lower_leakage_cost = leakage_reward_penalty
            upper_credit = float(info["portfolio_return"]) - float(
                info.get("drawdown_cost", 0.0)
            )
            lower_credit = (
                -float(info["transaction_cost"])
                - float(info.get("inventory_drift_cost", 0.0))
                - leakage_reward_penalty
            )
            upper_task_credit = upper_credit
            lower_task_credit = lower_credit + leakage_reward_penalty
            plan_return = float(info["portfolio_return"])
            execution_deviation_return = 0.0
            reconstruction_error = float("nan")
        latest_leakage_feedback = float(leak_info.get("lower_lf_penalty", 0.0))
        builder.add_lower(
            state=lower_state,
            action=np.asarray(lower_out["action"], dtype=np.float32),
            logp=float(lower_out["logp"]),
            value=float(lower_out["value"]),
            reward=float(reward_scale) * lower_credit,
            upper_reward=float(reward_scale) * upper_credit,
            cost=latest_leakage_feedback,
            upper_cost=0.0,
            done=bool(done),
        )

        upper_credits.append(upper_credit)
        lower_credits.append(lower_credit)
        upper_task_credits.append(upper_task_credit)
        lower_task_credits.append(lower_task_credit)
        plan_returns.append(plan_return)
        execution_deviation_returns.append(execution_deviation_return)
        credit_reconstruction_errors.append(reconstruction_error)
        upper_leakage_costs.append(upper_leakage_cost)
        lower_leakage_costs.append(lower_leakage_cost)
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        lower_effects.append(lower_effect.copy())
        raw_lower_effects.append(raw_lower_effect.copy())
        raw_recenter_boosts.append(np.asarray(raw_recenter_boost, dtype=np.float64).copy())
        if done:
            break

    builder.finish(terminal=True)
    trajectory = builder.build()
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    reg = LeakageRegularizer(upper_hf_window=6, lower_lf_window=24)
    leak = reg.compute(np.asarray(targets), np.asarray(lower_effects))
    raw_leak = reg.compute(np.asarray(targets), np.asarray(raw_lower_effects))
    reported_leak = raw_leak if constrain_raw_lower_effect else leak
    finite_reconstruction_errors = np.asarray([
        value for value in credit_reconstruction_errors if np.isfinite(value)
    ], dtype=np.float64)
    target_steps = (
        np.diff(np.asarray(targets, dtype=np.float64), axis=0)
        if len(targets) > 1 else np.zeros((0, assets), dtype=np.float64)
    )
    diag = diagnostics.summarize_episode()
    financial = summarize_pnl_series(
        pnl,
        eq,
        periods_per_year=periods_per_year_from_bar_seconds(60.0),
    )
    row = {
        "baseline": policy_mode,
        "policy_mode": policy_mode,
        "seed": int(seed),
        "scenario": scenario,
        **financial,
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotion_signals),
        "promotion_replan_count": int(sum(reason == "promotion" for reason in decision_reasons)),
        "scheduled_replan_count": int(sum(reason == "scheduled" for reason in decision_reasons)),
        "upper_decision_count": int(trajectory.upper.size),
        "lower_decision_count": int(trajectory.lower.size),
        "upper_mean_duration": float(np.mean(trajectory.upper.duration)),
        "upper_to_lower_ratio": float(trajectory.upper.size / max(trajectory.lower.size, 1)),
        "leakage_penalty": float(reported_leak["leakage_penalty"]),
        "UpperHFPower": float(reported_leak["UpperHFPower"]),
        "LowerLFDrift": float(reported_leak["LowerLFDrift"]),
        "LowerLFDriftAbs": float(reported_leak["LowerLFDriftAbs"]),
        "ProjectedLowerLFDrift": float(leak["LowerLFDrift"]),
        "ProjectedLowerLFDriftAbs": float(leak["LowerLFDriftAbs"]),
        "RawLowerLFDrift": float(raw_leak["LowerLFDrift"]),
        "RawLowerLFDriftAbs": float(raw_leak["LowerLFDriftAbs"]),
        "FocusScore": float(diag["FocusScore"]),
        "upper_low_mi": float(diag.get("upper_low_mi", 0.0)),
        "upper_high_mi": float(diag.get("upper_high_mi", 0.0)),
        "lower_high_mi": float(diag.get("lower_high_mi", 0.0)),
        "lower_low_mi": float(diag.get("lower_low_mi", 0.0)),
        "upper_credit_mean": float(np.mean(upper_credits)) if upper_credits else 0.0,
        "lower_credit_mean": float(np.mean(lower_credits)) if lower_credits else 0.0,
        "upper_task_credit_mean": float(np.mean(upper_task_credits)) if upper_task_credits else 0.0,
        "lower_task_credit_mean": float(np.mean(lower_task_credits)) if lower_task_credits else 0.0,
        "plan_return_mean": float(np.mean(plan_returns)) if plan_returns else 0.0,
        "execution_deviation_return_mean": float(
            np.mean(execution_deviation_returns)
        ) if execution_deviation_returns else 0.0,
        "upper_leakage_cost_mean": float(np.mean(upper_leakage_costs)) if upper_leakage_costs else 0.0,
        "lower_leakage_cost_mean": float(np.mean(lower_leakage_costs)) if lower_leakage_costs else 0.0,
        "task_credit_reconstruction_max_abs_error": float(
            np.max(np.abs(finite_reconstruction_errors))
        ) if finite_reconstruction_errors.size else float("nan"),
        "plan_smoothness": float(np.mean(plan_smoothness)) if plan_smoothness else 0.0,
        "plan_coeff_abs": float(np.mean(plan_coeff_abs)) if plan_coeff_abs else 0.0,
        "plan_target_step_change_mean": float(
            np.mean(np.abs(target_steps))
        ) if target_steps.size else 0.0,
        "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
        "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
        "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
        "raw_recenter_boost_mean": float(np.mean(raw_recenter_boosts)) if raw_recenter_boosts else 0.0,
        "protocol_valid": 1.0,
        "full_method_implementation_version": (
            FULL_METHOD_IMPLEMENTATION_VERSION
            if method_contract == "full_freq_hrl_v3" else "not_applicable"
        ),
        "execution_timeline_contract": str(execution_timeline_contract),
        "method_contract": str(method_contract),
        "mark_to_market_timing": str(mark_to_market_timing),
        "volume_impact_bps": float(volume_impact_bps),
        "executed_plan_curve": float(bool(execute_plan_curve)),
        "additive_frequency_credit": float(bool(use_additive_frequency_credit)),
        "raw_lower_effect_constraint": float(bool(constrain_raw_lower_effect)),
        "plan_smoothness_weight": float(plan_smoothness_weight),
        "routing_contract": (
            "frequency_responsibility"
            if policy_mode == "freq_hrl" else (
                "causal_raw_full_episode_gru"
                if policy_mode == "generic_hrl_gru_ppo"
                else "causal_raw_contiguous_window"
            )
        ),
        "temporal_contract": (
            "asynchronous_hierarchy"
        ),
    }
    return (trajectory if sample else None), row


def objective(row: dict[str, float]) -> float:
    return validation_utility(row)


def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = [
        "total_return",
        "sharpe",
        "episode_information_ratio",
        "max_drawdown",
        "turnover",
        "promotion_count",
        "promotion_replan_count",
        "scheduled_replan_count",
        "upper_decision_count",
        "lower_decision_count",
        "upper_mean_duration",
        "upper_to_lower_ratio",
        "leakage_penalty",
        "UpperHFPower",
        "LowerLFDrift",
        "LowerLFDriftAbs",
        "ProjectedLowerLFDrift",
        "ProjectedLowerLFDriftAbs",
        "RawLowerLFDrift",
        "RawLowerLFDriftAbs",
        "FocusScore",
        "upper_low_mi",
        "upper_high_mi",
        "lower_high_mi",
        "lower_low_mi",
        "upper_credit_mean",
        "lower_credit_mean",
        "upper_task_credit_mean",
        "lower_task_credit_mean",
        "plan_return_mean",
        "execution_deviation_return_mean",
        "upper_leakage_cost_mean",
        "lower_leakage_cost_mean",
        "task_credit_reconstruction_max_abs_error",
        "plan_smoothness",
        "plan_coeff_abs",
        "plan_target_step_change_mean",
        "lower_lf_effect_filter_window",
        "lower_lf_effect_filter_gain",
        "lower_lf_raw_recenter_gain",
        "raw_recenter_boost_mean",
        "executed_plan_curve",
        "additive_frequency_credit",
        "raw_lower_effect_constraint",
        "volume_impact_bps",
        "plan_smoothness_weight",
        "protocol_valid",
    ]
    return summarize_numeric_rows(rows, keys=keys)


def train_ppo_actor_critic(
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    iterations: int,
    seed: int,
    validation_seeds: list[int] | None = None,
    hidden_dim: int = 64,
    learning_rate: float = 3e-4,
    ppo_epochs: int = 4,
    minibatch_size: int = 512,
    init_log_std: float = -1.0,
    resample_training_paths: bool = True,
    leakage_scale: float = 0.0,
    plan_basis_dim: int = 0,
    plan_horizon_s: float = 1800.0,
    plan_eval_offset_s: float = 300.0,
    plan_coefficient_scale: float = 0.75,
    lower_lf_constraint_coef: float = 0.0,
    lower_lf_constraint_target: float = 0.0,
    lower_lf_dual_lr: float = 0.0,
    lower_lf_objective_weight: float = 0.0,
    lower_lf_effect_filter_window: int = 0,
    lower_lf_effect_filter_gain: float = 1.0,
    lower_lf_raw_recenter_gain: float = 0.0,
    lower_lf_raw_recenter_scale: float = 0.10,
    policy_mode: str = "freq_hrl",
    upper_period: int = 30,
    min_upper_duration: int = 5,
    use_handcrafted_frequency_prior: bool = False,
    evaluation_role: str = "heldout_test",
    reward_scale: float = DEFAULT_TRAINING_REWARD_SCALE,
    execution_timeline_contract: str = "legacy_pre_trade_v2",
    method_contract: str = "routing_core_v2",
    volume_impact_bps: float = 0.0,
    plan_smoothness_weight: float = 0.0,
) -> tuple[
    dict[str, Any],
    list[dict[str, float]],
    FrequencySeparatedActorCriticPPO | JointActorCriticPPO,
]:
    policy_mode = str(policy_mode or "freq_hrl")
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"unknown policy_mode: {policy_mode}")
    evaluation_role = str(evaluation_role)
    if evaluation_role not in {"heldout_test", "tuning_validation"}:
        raise ValueError(
            "evaluation_role must be 'heldout_test' or 'tuning_validation'"
        )
    if not np.isfinite(float(reward_scale)) or float(reward_scale) <= 0.0:
        raise ValueError("reward_scale must be positive and finite")
    execution_timeline_contract = str(execution_timeline_contract)
    if execution_timeline_contract not in EXECUTION_TIMELINE_CONTRACTS:
        raise ValueError(
            "unknown execution_timeline_contract: "
            f"{execution_timeline_contract}; expected one of "
            f"{EXECUTION_TIMELINE_CONTRACTS}"
        )
    method_contract = str(method_contract)
    method_flags = resolve_method_contract(method_contract)
    mark_to_market_timing = (
        "post_trade"
        if execution_timeline_contract == "causal_post_trade_v3"
        else "pre_trade"
    )
    if not np.isfinite(float(volume_impact_bps)) or float(volume_impact_bps) < 0.0:
        raise ValueError("volume_impact_bps must be finite and non-negative")
    for name, value in (("plan_smoothness_weight", plan_smoothness_weight),):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if method_contract != "routing_core_v2":
        if policy_mode in FLAT_PPO_MODES:
            raise ValueError(
                f"{method_contract} is hierarchical and cannot be used by {policy_mode}"
            )
        if execution_timeline_contract != "causal_post_trade_v3":
            raise ValueError(
                f"{method_contract} requires causal_post_trade_v3 execution"
            )
        if int(plan_basis_dim) < 2:
            raise ValueError(
                f"{method_contract} requires plan_basis_dim >= 2"
            )
    if method_contract == "full_freq_hrl_v3":
        if policy_mode != "freq_hrl":
            raise ValueError("full_freq_hrl_v3 requires policy_mode='freq_hrl'")
        if not (
            float(leakage_scale) > 0.0
            or float(lower_lf_constraint_coef) > 0.0
            or float(lower_lf_dual_lr) > 0.0
        ):
            raise ValueError(
                "full_freq_hrl_v3 requires an active raw leakage penalty or constraint"
            )
    rollout_seed_roots = validate_unique_seeds(
        train_seeds, role="rollout_seed_roots"
    )
    if validation_seeds is None:
        validation_seed_list = [
            derive_seed("freq_hrl_trading_validation_v2", scenario, root)
            for root in rollout_seed_roots
        ]
    else:
        validation_seed_list = list(validation_seeds)
    validation_seed_list, evaluation_seeds = validate_evaluation_seed_roles(
        validation_seed_list, eval_seeds
    )
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    plan_mapper = make_plan_mapper(
        assets=assets,
        plan_basis_dim=plan_basis_dim,
        plan_horizon_s=plan_horizon_s,
        plan_eval_offset_s=plan_eval_offset_s,
        plan_coefficient_scale=plan_coefficient_scale,
        anchor_first_coefficient=(
            execution_timeline_contract == "causal_post_trade_v3"
        ),
    )
    reference_smdp_config = SMDPPPOConfig(
        upper_state_dim=6 * assets + 5,
        lower_state_dim=8 * assets + 1,
        upper_action_dim=plan_mapper.action_dim if plan_mapper is not None else assets,
        lower_action_dim=assets,
        hidden_dim=int(hidden_dim),
        upper_learning_rate=float(learning_rate),
        lower_learning_rate=float(learning_rate),
        epochs=int(ppo_epochs),
        minibatch_size=int(minibatch_size),
        init_log_std=float(init_log_std),
        lower_cost_target=float(lower_lf_constraint_target),
        lower_dual_lr=float(lower_lf_dual_lr),
        lower_lambda_init=max(float(lower_lf_constraint_coef), 0.0),
        lower_max_lambda=20.0,
    )
    target_parameter_count = smdp_parameter_count(reference_smdp_config)
    if policy_mode in FLAT_PPO_MODES:
        recurrent_raw = policy_mode == "flat_gru_ppo"
        raw_history_window = int(steps) if recurrent_raw else RAW_HISTORY_WINDOW
        state_encoder = "causal_gru" if recurrent_raw else "mlp"
        joint_state_dim = raw_lower_state_dim(assets, raw_history_window)
        joint_hidden_dim, joint_count, capacity_ratio = (
            capacity_matched_joint_hidden_dim(
                target_parameter_count=target_parameter_count,
                state_dim=joint_state_dim,
                action_dim=2 * assets,
                requested_hidden_dim=int(hidden_dim),
                state_encoder=state_encoder,
                raw_history_window=raw_history_window,
                raw_feature_dim=assets,
            )
        )
        joint_config = JointPPOConfig(
            state_dim=joint_state_dim,
            action_dim=2 * assets,
            hidden_dim=int(joint_hidden_dim),
            state_encoder=state_encoder,
            raw_history_window=raw_history_window,
            raw_feature_dim=int(assets),
            learning_rate=float(learning_rate),
            epochs=int(ppo_epochs),
            minibatch_size=int(minibatch_size),
            init_log_std=float(init_log_std),
        )
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        joint_model = JointActorCriticPPO(joint_config)
        payload, heldout_rows, joint_model = train_joint_ppo(
            model=joint_model,
            train_seeds=rollout_seed_roots,
            eval_seeds=evaluation_seeds,
            iterations=iterations,
            selection_seeds=validation_seed_list,
            training_seed_fn=(
                (
                    lambda root, iteration: training_rollout_seed(
                        int(seed), root, iteration, domain=f"trading:{scenario}"
                    )
                )
                if resample_training_paths else None
            ),
            rollout_fn=lambda ppo_model, rollout_seed, sample: joint_flat_rollout(
                ppo_model,
                seed=rollout_seed,
                steps=steps,
                assets=assets,
                scenario=scenario,
                sample=sample,
                leakage_scale=0.0,
                lower_lf_effect_filter_window=0,
                lower_lf_raw_recenter_gain=0.0,
                policy_mode=policy_mode,
                reward_scale=float(reward_scale),
                mark_to_market_timing=mark_to_market_timing,
                volume_impact_bps=float(volume_impact_bps),
                execution_timeline_contract=execution_timeline_contract,
                method_contract=method_contract,
            ),
            objective_fn=objective,
            summary_fn=summarize,
            policy=f"{policy_mode}_canonical_joint_action",
            domain="trading",
            metadata={
                "policy_mode": policy_mode,
                "baseline": policy_mode,
                "learned_baseline_implementation_version": (
                    LEARNED_BASELINE_IMPLEMENTATION_VERSION
                ),
                "scenario": scenario,
                "steps": int(steps),
                "assets": int(assets),
                "leakage_scale": 0.0,
                "upper_period": 1,
                "min_upper_duration": 1,
                "upper_observation_contract": "not_applicable_single_flat_state",
                "lower_observation_contract": (
                    f"complete contiguous {raw_history_window}-bar causal raw "
                    "window + position + previous action"
                ),
                "credit_contract": "single task-return GAE for one joint action",
                "frequency_routing_enabled": False,
                "promotion_replanning_enabled": False,
                "handcrafted_frequency_prior": False,
                "capacity_match_contract": (
                    "active parameter count matched to hierarchical PPO within 5%; "
                    "no inactive padding parameters"
                ),
                "capacity_target_parameter_count": int(target_parameter_count),
                "capacity_actual_parameter_count": int(joint_count),
                "capacity_ratio": float(capacity_ratio),
                "capacity_match_status": capacity_match_status(capacity_ratio),
                "requested_hidden_dim": int(hidden_dim),
                "effective_hidden_dim": int(joint_hidden_dim),
                "state_encoder": state_encoder,
                "raw_history_window": int(raw_history_window),
                "raw_history_sampling": "complete_contiguous_oldest_to_newest",
                "training_replicate_seed": int(seed),
                "rollout_seed_roots": list(rollout_seed_roots),
                "validation_seeds": list(validation_seed_list),
                "evaluation_role": evaluation_role,
                "tuning_validation_seeds": (
                    list(evaluation_seeds)
                    if evaluation_role == "tuning_validation" else []
                ),
                "heldout_test_seeds": (
                    list(evaluation_seeds)
                    if evaluation_role == "heldout_test" else []
                ),
                "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
                "training_reward_scale": float(reward_scale),
                "execution_timeline_contract": execution_timeline_contract,
                "method_contract": method_contract,
                "mark_to_market_timing": mark_to_market_timing,
                "volume_impact_bps": float(volume_impact_bps),
                "full_method_implementation_version": (
                    FULL_METHOD_IMPLEMENTATION_VERSION
                    if method_contract == "full_freq_hrl_v3"
                    else "not_applicable"
                ),
                "training_path_protocol": (
                    "fresh_deterministic_path_per_root_and_iteration_v2"
                    if resample_training_paths else "fixed_path_reuse_legacy"
                ),
                "checkpoint_selection_protocol": "disjoint_validation_paths",
                "plan_mode": "primitive_joint_target_execution",
                "plan_basis_dim": 0,
                "plan_horizon_s": 0.0,
                "plan_eval_offset_s": 0.0,
                "plan_coefficient_scale": 0.0,
                "plan_action_dim": int(2 * assets),
                "lower_lf_constraint_coef": 0.0,
                "lower_lf_constraint_target": 0.0,
                "lower_lf_dual_lr": 0.0,
                "lower_lf_objective_weight": 0.0,
                "lower_lf_effect_filter_window": 0,
                "lower_lf_effect_filter_gain": 0.0,
                "lower_lf_raw_recenter_gain": 0.0,
                "lower_lf_raw_recenter_scale": 0.0,
            },
        )
        payload["environment_steps_train"] = int(
            len(rollout_seed_roots) * int(steps) * max(1, int(iterations))
        )
        payload["environment_steps_validation"] = int(
            len(validation_seed_list) * int(steps) * (max(1, int(iterations)) + 1)
        )
        payload["environment_steps_eval"] = int(
            len(evaluation_seeds) * int(steps)
        )
        payload["unique_training_path_count"] = int(
            len(rollout_seed_roots)
            * (max(1, int(iterations)) if resample_training_paths else 1)
        )
        return payload, heldout_rows, joint_model

    smdp_capacity_count = int(target_parameter_count)
    smdp_capacity_ratio = 1.0
    effective_hidden_dim = int(hidden_dim)
    smdp_config = reference_smdp_config
    raw_history_window = RAW_HISTORY_WINDOW
    state_encoder = "mlp"
    if policy_mode in GENERIC_HRL_MODES:
        recurrent_raw = policy_mode == "generic_hrl_gru_ppo"
        raw_history_window = int(steps) if recurrent_raw else RAW_HISTORY_WINDOW
        state_encoder = "causal_gru" if recurrent_raw else "mlp"
        upper_state_dim = raw_upper_state_dim(assets, raw_history_window)
        lower_state_dim = raw_lower_state_dim(assets, raw_history_window)
        effective_hidden_dim, smdp_capacity_count, smdp_capacity_ratio = (
            capacity_matched_smdp_hidden_dim(
                target_parameter_count=target_parameter_count,
                upper_state_dim=upper_state_dim,
                lower_state_dim=lower_state_dim,
                upper_action_dim=reference_smdp_config.upper_action_dim,
                lower_action_dim=reference_smdp_config.lower_action_dim,
                requested_hidden_dim=int(hidden_dim),
                state_encoder=state_encoder,
                raw_history_window=raw_history_window,
                raw_feature_dim=assets,
            )
        )
        smdp_config = replace(
            reference_smdp_config,
            upper_state_dim=upper_state_dim,
            lower_state_dim=lower_state_dim,
            hidden_dim=int(effective_hidden_dim),
            state_encoder=state_encoder,
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(assets),
        )
    smdp_model = FrequencySeparatedActorCriticPPO(smdp_config)
    if policy_mode == "freq_hrl" and bool(use_handcrafted_frequency_prior):
        initialize_smdp_frequency_prior(smdp_model, assets, plan_basis_dim=plan_basis_dim)
    actual_upper_period = int(upper_period)
    actual_min_upper_duration = int(min_upper_duration)
    observation_contract = {
        "freq_hrl": (
            "LF + forecast + uncertainty + compressed HF summaries",
            "current plan + local HF/MF residual context",
        ),
        "generic_hrl_ppo": (
            "complete contiguous 120-bar causal raw window + position + active plan",
            "active plan + position + complete contiguous 120-bar causal raw window",
        ),
        "generic_hrl_gru_ppo": (
            f"causal GRU over the complete {raw_history_window}-bar episode history "
            "+ position + active plan",
            f"active plan + position + causal GRU over the complete "
            f"{raw_history_window}-bar episode history",
        ),
    }[policy_mode]
    credit_contract = "upper strategic PnL; lower execution cost, tracking, and leakage"
    if method_flags["use_additive_frequency_credit"]:
        credit_contract = (
            "exact task-reward conservation: upper plan return; lower execution "
            "deviation return and execution/tracking costs; regularizers separate"
        )
    payload, heldout_rows, smdp_model = train_frequency_separated_ppo(
        model=smdp_model,
        train_seeds=rollout_seed_roots,
        eval_seeds=evaluation_seeds,
        iterations=iterations,
        selection_seeds=validation_seed_list,
        training_seed_fn=(
            (
                lambda root, iteration: training_rollout_seed(
                    int(seed), root, iteration, domain=f"trading:{scenario}"
                )
            )
            if resample_training_paths else None
        ),
        rollout_fn=lambda ppo_model, rollout_seed, sample: smdp_rollout(
            ppo_model,
            seed=rollout_seed,
            steps=steps,
            assets=assets,
            scenario=scenario,
            sample=sample,
            leakage_scale=leakage_scale if sample else 0.0,
            plan_mapper=plan_mapper,
            lower_lf_effect_filter_window=lower_lf_effect_filter_window,
            lower_lf_effect_filter_gain=lower_lf_effect_filter_gain,
            lower_lf_raw_recenter_gain=lower_lf_raw_recenter_gain,
            lower_lf_raw_recenter_scale=lower_lf_raw_recenter_scale,
            policy_mode=policy_mode,
            upper_period=actual_upper_period,
            min_upper_duration=actual_min_upper_duration,
            reward_scale=float(reward_scale),
            mark_to_market_timing=mark_to_market_timing,
            volume_impact_bps=float(volume_impact_bps),
            execute_plan_curve=method_flags["execute_plan_curve"],
            use_additive_frequency_credit=method_flags[
                "use_additive_frequency_credit"
            ],
            constrain_raw_lower_effect=method_flags[
                "constrain_raw_lower_effect"
            ],
            plan_smoothness_weight=float(plan_smoothness_weight),
            execution_timeline_contract=execution_timeline_contract,
            method_contract=method_contract,
        ),
        objective_fn=lambda row: objective(row) - max(float(lower_lf_objective_weight), 0.0) * float(
            row["LowerLFDrift"]
        ),
        summary_fn=summarize,
        policy=f"{policy_mode}_capacity_matched_smdp_ppo",
        domain="trading",
        metadata={
            "policy_mode": policy_mode,
            "baseline": policy_mode,
            "learned_baseline_implementation_version": (
                LEARNED_BASELINE_IMPLEMENTATION_VERSION
            ),
            "scenario": scenario,
            "steps": int(steps),
            "assets": int(assets),
            "leakage_scale": float(leakage_scale),
            "upper_period": int(actual_upper_period),
            "min_upper_duration": int(actual_min_upper_duration),
            "upper_observation_contract": observation_contract[0],
            "lower_observation_contract": observation_contract[1],
            "credit_contract": credit_contract,
            "frequency_routing_enabled": bool(policy_mode == "freq_hrl"),
            "promotion_replanning_enabled": bool(policy_mode == "freq_hrl"),
            "handcrafted_frequency_prior": bool(
                policy_mode == "freq_hrl" and use_handcrafted_frequency_prior
            ),
            "capacity_match_contract": (
                "Freq-HRL reference or active parameter count matched to Freq-HRL "
                "within 5%; equal optimizer, epochs, and rollout seed budget"
            ),
            "capacity_target_parameter_count": int(target_parameter_count),
            "capacity_actual_parameter_count": int(smdp_capacity_count),
            "capacity_ratio": float(smdp_capacity_ratio),
            "capacity_match_status": capacity_match_status(smdp_capacity_ratio),
            "requested_hidden_dim": int(hidden_dim),
            "effective_hidden_dim": int(effective_hidden_dim),
            "state_encoder": state_encoder,
            **({
                "raw_history_window": int(raw_history_window),
                "raw_history_sampling": "complete_contiguous_oldest_to_newest",
            } if policy_mode in GENERIC_HRL_MODES else {}),
            "training_replicate_seed": int(seed),
            "rollout_seed_roots": list(rollout_seed_roots),
            "validation_seeds": list(validation_seed_list),
            "evaluation_role": evaluation_role,
            "tuning_validation_seeds": (
                list(evaluation_seeds)
                if evaluation_role == "tuning_validation" else []
            ),
            "heldout_test_seeds": (
                list(evaluation_seeds)
                if evaluation_role == "heldout_test" else []
            ),
            "selection_objective_version": SELECTION_OBJECTIVE_VERSION,
            "training_reward_scale": float(reward_scale),
            "execution_timeline_contract": execution_timeline_contract,
            "method_contract": method_contract,
            "mark_to_market_timing": mark_to_market_timing,
            "volume_impact_bps": float(volume_impact_bps),
            "full_method_implementation_version": (
                FULL_METHOD_IMPLEMENTATION_VERSION
                if method_contract == "full_freq_hrl_v3"
                else "not_applicable"
            ),
            "executed_plan_curve": bool(method_flags["execute_plan_curve"]),
            "additive_frequency_credit": bool(
                method_flags["use_additive_frequency_credit"]
            ),
            "raw_lower_effect_constraint": bool(
                method_flags["constrain_raw_lower_effect"]
            ),
            "plan_smoothness_weight": float(plan_smoothness_weight),
            "training_path_protocol": (
                "fresh_deterministic_path_per_root_and_iteration_v2"
                if resample_training_paths else "fixed_path_reuse_legacy"
            ),
            "checkpoint_selection_protocol": "disjoint_validation_paths",
            "plan_mode": "learned_bernstein" if plan_mapper is not None else "direct_target",
            "lower_lf_constraint_coef": float(lower_lf_constraint_coef),
            "lower_lf_constraint_target": float(lower_lf_constraint_target),
            "lower_lf_dual_lr": float(lower_lf_dual_lr),
            "lower_lf_objective_weight": float(lower_lf_objective_weight),
            "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
            "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
            "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
            "lower_lf_raw_recenter_scale": float(lower_lf_raw_recenter_scale),
            **(plan_mapper.to_metadata() if plan_mapper is not None else {
                "plan_basis_dim": 0,
                "plan_horizon_s": 0.0,
                "plan_eval_offset_s": 0.0,
                "plan_coefficient_scale": 0.0,
                "plan_action_dim": int(assets),
            }),
        },
    )
    for row in heldout_rows:
        row["baseline"] = policy_mode
        row["policy_mode"] = policy_mode
    payload["environment_steps_train"] = int(
        len(rollout_seed_roots) * int(steps) * max(1, int(iterations))
    )
    payload["environment_steps_validation"] = int(
        len(validation_seed_list) * int(steps) * (max(1, int(iterations)) + 1)
    )
    payload["environment_steps_eval"] = int(len(evaluation_seeds) * int(steps))
    payload["unique_training_path_count"] = int(
        len(rollout_seed_roots)
        * (max(1, int(iterations)) if resample_training_paths else 1)
    )
    return payload, heldout_rows, smdp_model


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
        "# Capacity-Controlled PPO Trading Validation",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- policy mode: `{payload.get('policy_mode', payload.get('baseline', 'freq_hrl'))}`",
        f"- capacity contract: `{payload.get('capacity_match_contract', 'NA')}`",
        f"- optimization reward scale: `{payload.get('training_reward_scale', 1.0)}`",
        f"- observation contract: upper=`{payload.get('upper_observation_contract', 'NA')}`, lower=`{payload.get('lower_observation_contract', 'NA')}`",
        f"- plan mode: `{payload['plan_mode']}`",
        f"- lower LF constraint: coef={payload['lower_lf_constraint_coef']}, target={payload['lower_lf_constraint_target']}, dual_lr={payload['lower_lf_dual_lr']}",
        f"- lower LF effect projector: window={payload['lower_lf_effect_filter_window']}, gain={payload['lower_lf_effect_filter_gain']}",
        f"- raw lower drift recenter: gain={payload['lower_lf_raw_recenter_gain']}, scale={payload['lower_lf_raw_recenter_scale']}",
        f"- scenario: `{payload['scenario']}`",
        f"- train seeds: {payload['train_seeds']}",
        f"- eval seeds: {payload['eval_seeds']}",
        f"- return mean: {summary['total_return_mean']:.4f}",
        f"- Sharpe mean: {summary['sharpe_mean']:.3f}",
        f"- max drawdown mean: {summary['max_drawdown_mean']:.4f}",
        f"- turnover mean: {summary['turnover_mean']:.2f}",
        f"- leakage penalty mean: {summary['leakage_penalty_mean']:.4f}",
        f"- LowerLFDrift mean: {summary['LowerLFDrift_mean']:.4f}",
        f"- LowerLFDriftAbs mean: {summary['LowerLFDriftAbs_mean']:.6f}",
        f"- RawLowerLFDrift mean: {summary['RawLowerLFDrift_mean']:.4f}",
        f"- RawLowerLFDriftAbs mean: {summary['RawLowerLFDriftAbs_mean']:.6f}",
        f"- FocusScore mean: {summary['FocusScore_mean']:.4f}",
        f"- raw recenter boost mean: {summary['raw_recenter_boost_mean_mean']:.4f}",
        f"- plan smoothness mean: {summary['plan_smoothness_mean']:.4f}",
        f"- plan coefficient abs mean: {summary['plan_coeff_abs_mean']:.4f}",
        "",
        "This validates the shared upper/lower PPO actor-critic training core. It uses trading as a domain adapter; the trainer itself only depends on upper/lower states, latent actions, rewards, and done flags.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[31415, 27182, 16180])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--init-log-std", type=float, default=-1.0)
    parser.add_argument(
        "--reward-scale", type=float, default=DEFAULT_TRAINING_REWARD_SCALE
    )
    parser.add_argument("--reuse-fixed-training-paths", action="store_true")
    parser.add_argument("--leakage-scale", type=float, default=0.0)
    parser.add_argument("--plan-basis-dim", type=int, default=0)
    parser.add_argument("--plan-horizon-s", type=float, default=1800.0)
    parser.add_argument("--plan-eval-offset-s", type=float, default=300.0)
    parser.add_argument("--plan-coefficient-scale", type=float, default=0.75)
    parser.add_argument("--lower-lf-constraint-coef", type=float, default=0.0)
    parser.add_argument("--lower-lf-constraint-target", type=float, default=0.0)
    parser.add_argument("--lower-lf-dual-lr", type=float, default=0.0)
    parser.add_argument("--lower-lf-objective-weight", type=float, default=0.0)
    parser.add_argument("--lower-lf-effect-filter-window", type=int, default=0)
    parser.add_argument("--lower-lf-effect-filter-gain", type=float, default=1.0)
    parser.add_argument("--lower-lf-raw-recenter-gain", type=float, default=0.0)
    parser.add_argument("--lower-lf-raw-recenter-scale", type=float, default=0.10)
    parser.add_argument("--policy-mode", choices=POLICY_MODES, default="freq_hrl")
    parser.add_argument("--upper-period", type=int, default=30)
    parser.add_argument("--min-upper-duration", type=int, default=5)
    parser.add_argument("--use-handcrafted-frequency-prior", action="store_true")
    parser.add_argument(
        "--execution-timeline-contract",
        choices=EXECUTION_TIMELINE_CONTRACTS,
        default="legacy_pre_trade_v2",
    )
    parser.add_argument(
        "--method-contract",
        choices=METHOD_CONTRACTS,
        default="routing_core_v2",
    )
    parser.add_argument("--volume-impact-bps", type=float, default=0.0)
    parser.add_argument("--plan-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_ppo_actor_critic"))
    args = parser.parse_args()
    payload, rows, model = train_ppo_actor_critic(
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=args.steps,
        assets=args.assets,
        scenario=args.scenario,
        iterations=args.iterations,
        seed=args.optimizer_seed,
        validation_seeds=(
            None if args.validation_seeds is None else list(args.validation_seeds)
        ),
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        init_log_std=args.init_log_std,
        reward_scale=args.reward_scale,
        resample_training_paths=not args.reuse_fixed_training_paths,
        leakage_scale=args.leakage_scale,
        plan_basis_dim=args.plan_basis_dim,
        plan_horizon_s=args.plan_horizon_s,
        plan_eval_offset_s=args.plan_eval_offset_s,
        plan_coefficient_scale=args.plan_coefficient_scale,
        lower_lf_constraint_coef=args.lower_lf_constraint_coef,
        lower_lf_constraint_target=args.lower_lf_constraint_target,
        lower_lf_dual_lr=args.lower_lf_dual_lr,
        lower_lf_objective_weight=args.lower_lf_objective_weight,
        lower_lf_effect_filter_window=args.lower_lf_effect_filter_window,
        lower_lf_effect_filter_gain=args.lower_lf_effect_filter_gain,
        lower_lf_raw_recenter_gain=args.lower_lf_raw_recenter_gain,
        lower_lf_raw_recenter_scale=args.lower_lf_raw_recenter_scale,
        policy_mode=args.policy_mode,
        upper_period=args.upper_period,
        min_upper_duration=args.min_upper_duration,
        use_handcrafted_frequency_prior=args.use_handcrafted_frequency_prior,
        execution_timeline_contract=args.execution_timeline_contract,
        method_contract=args.method_contract,
        volume_impact_bps=args.volume_impact_bps,
        plan_smoothness_weight=args.plan_smoothness_weight,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_seed.csv", rows)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"model": payload, "per_seed": rows, "summary": payload["summary"]}, f, indent=2)
    torch.save(model.state_dict(), args.output_dir / "ppo_dual_actor_critic.pt")
    write_report(args.output_dir / "report.md", payload)
    print(f"wrote {args.output_dir}")
    print(
        "ppo_dual_actor_critic "
        f"sharpe={payload['summary']['sharpe_mean']:.3f} "
        f"return={payload['summary']['total_return_mean']:.4f} "
        f"LowerLFDrift={payload['summary']['LowerLFDrift_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
