"""PPO-style dual actor-critic validation for Freq-HRL trading."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    FrequencyDiagnostics,
    FrequencyRouter,
    LeakageRegularizer,
    evaluate_rms_leakage_budget,
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
    PromotionRolloutBuilder,
    SMDPPPOConfig,
    TemporalDecisionScheduler,
    causal_gru_actor_parameter_count,
    causal_gru_value_parameter_count,
    concat_hierarchical_batches,
    concat_joint_batches,
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
from .performance_validation import (
    SCENARIOS,
    SUPPORT_MIXTURE_SCENARIO,
    causal_hf_predictability,
    make_synthetic_market,
)


FLAT_PPO_MODES = ("flat_ppo", "flat_gru_ppo")
GENERIC_HRL_MODES = ("generic_hrl_ppo", "generic_hrl_gru_ppo")
POLICY_MODES = ("freq_hrl",) + FLAT_PPO_MODES + GENERIC_HRL_MODES
LEARNED_BASELINE_IMPLEMENTATION_VERSION = (
    "learned_baselines_v5_causal_gru_controls_2026_08_03"
)
FULL_METHOD_IMPLEMENTATION_VERSION = (
    "freq_hrl_full_v4_learned_promotion_credit_plan_leakage_2026_08_03"
)
FULL_METHOD_V5_IMPLEMENTATION_VERSION = (
    "freq_hrl_full_v5_independent_hf_tactical_credit_2026_08_03"
)
FULL_METHOD_V6_IMPLEMENTATION_VERSION = (
    "freq_hrl_full_v6_mixed_regime_counterfactual_control_2026_08_03"
)
FULL_METHOD_V7_IMPLEMENTATION_VERSION = (
    "freq_hrl_full_v7_4_robust_checkpoint_pathwise_promotion_2026_08_08"
)
FULL_METHOD_V3_IMPLEMENTATION_VERSION = (
    "freq_hrl_full_v3_credit_plan_leakage_2026_08_03"
)
EXECUTION_TIMELINE_CONTRACTS = (
    "legacy_pre_trade_v2",
    "causal_post_trade_v3",
)
METHOD_CONTRACTS = (
    "routing_core_v2",
    "curve_credit_control_v3",
    "full_freq_hrl_v3",
    "full_freq_hrl_v4",
    "ablate_promotion_v4",
    "ablate_hf_lower_v4",
    "ablate_leakage_v4",
    "full_freq_hrl_v5",
    "ablate_promotion_v5",
    "ablate_hf_lower_v5",
    "ablate_leakage_v5",
    "full_freq_hrl_v6",
    "ablate_promotion_v6",
    "ablate_hf_lower_v6",
    "ablate_leakage_v6",
    "full_freq_hrl_v7",
    "ablate_promotion_v7",
    "ablate_hf_lower_v7",
    "ablate_leakage_v7",
    "ablate_lf_reference_v7",
    "ablate_upper_residual_v7",
)
V6_METHOD_CONTRACTS = {
    "full_freq_hrl_v6",
    "ablate_promotion_v6",
    "ablate_hf_lower_v6",
    "ablate_leakage_v6",
}
V7_METHOD_CONTRACTS = {
    "full_freq_hrl_v7",
    "ablate_promotion_v7",
    "ablate_hf_lower_v7",
    "ablate_leakage_v7",
    "ablate_lf_reference_v7",
    "ablate_upper_residual_v7",
}
LOWER_OBSERVATION_INTERVENTIONS = (
    "none",
    "zero_residual_frequency",
)
UPPER_PLAN_REFERENCE_MODES = (
    "none",
    "causal_lf",
)

RAW_HISTORY_WINDOW = 120


def resolve_method_contract(method_contract: str) -> dict[str, bool]:
    contract = str(method_contract)
    if contract not in METHOD_CONTRACTS:
        raise ValueError(
            f"unknown method_contract: {contract}; expected one of {METHOD_CONTRACTS}"
        )
    flags = {
        "execute_plan_curve": contract != "routing_core_v2",
        "use_additive_frequency_credit": contract != "routing_core_v2",
        "constrain_raw_lower_effect": contract in {
            "full_freq_hrl_v3",
            "full_freq_hrl_v4",
            "ablate_promotion_v4",
            "ablate_hf_lower_v4",
            "full_freq_hrl_v5",
            "ablate_promotion_v5",
            "ablate_hf_lower_v5",
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "full_freq_hrl_v7",
            "ablate_promotion_v7",
            "ablate_hf_lower_v7",
            "ablate_lf_reference_v7",
            "ablate_upper_residual_v7",
        },
        "learned_promotion_gate": contract in {
            "full_freq_hrl_v4",
            "ablate_hf_lower_v4",
            "ablate_leakage_v4",
            "full_freq_hrl_v5",
            "ablate_hf_lower_v5",
            "ablate_leakage_v5",
            "full_freq_hrl_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
            "full_freq_hrl_v7",
            "ablate_hf_lower_v7",
            "ablate_leakage_v7",
            "ablate_lf_reference_v7",
            "ablate_upper_residual_v7",
        },
        "heuristic_promotion_gate": contract in {
            "routing_core_v2",
            "curve_credit_control_v3",
            "full_freq_hrl_v3",
        },
        "lower_hf_overlay": contract in {
            "full_freq_hrl_v4",
            "ablate_promotion_v4",
            "ablate_leakage_v4",
            "full_freq_hrl_v5",
            "ablate_promotion_v5",
            "ablate_leakage_v5",
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_leakage_v6",
            "full_freq_hrl_v7",
            "ablate_promotion_v7",
            "ablate_leakage_v7",
            "ablate_lf_reference_v7",
            "ablate_upper_residual_v7",
        },
        "separate_hf_tactical": contract in {
            "full_freq_hrl_v5",
            "ablate_promotion_v5",
            "ablate_hf_lower_v5",
            "ablate_leakage_v5",
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
            *V7_METHOD_CONTRACTS,
        },
        "promotion_plan_advantage_credit": contract in {
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
            *V7_METHOD_CONTRACTS,
        },
        "fixed_rms_leakage_budget": contract in {
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
            *V7_METHOD_CONTRACTS,
        },
        "hf_predictability_summary": contract in {
            "full_freq_hrl_v6",
            "ablate_promotion_v6",
            "ablate_hf_lower_v6",
            "ablate_leakage_v6",
            *V7_METHOD_CONTRACTS,
        },
        "causal_lf_plan_reference": contract in (
            V7_METHOD_CONTRACTS - {"ablate_lf_reference_v7"}
        ),
        "hard_hf_budget_projection": contract in {
            "full_freq_hrl_v7",
            "ablate_promotion_v7",
            "ablate_lf_reference_v7",
            "ablate_upper_residual_v7",
        },
        "upper_residual_control": contract != "ablate_upper_residual_v7",
    }
    return flags


def full_method_implementation_version(method_contract: str) -> str:
    if str(method_contract) in V7_METHOD_CONTRACTS:
        return FULL_METHOD_V7_IMPLEMENTATION_VERSION
    if str(method_contract) in V6_METHOD_CONTRACTS:
        return FULL_METHOD_V6_IMPLEMENTATION_VERSION
    if str(method_contract) in {
        "full_freq_hrl_v5",
        "ablate_promotion_v5",
        "ablate_hf_lower_v5",
        "ablate_leakage_v5",
    }:
        return FULL_METHOD_V5_IMPLEMENTATION_VERSION
    if str(method_contract) in {
        "full_freq_hrl_v4",
        "ablate_promotion_v4",
        "ablate_hf_lower_v4",
        "ablate_leakage_v4",
    }:
        return FULL_METHOD_IMPLEMENTATION_VERSION
    if str(method_contract) == "full_freq_hrl_v3":
        return FULL_METHOD_V3_IMPLEMENTATION_VERSION
    return "not_applicable"


def gross_cap(target: np.ndarray, max_gross: float = 1.0) -> np.ndarray:
    out = np.asarray(target, dtype=np.float64).reshape(-1)
    gross = float(np.sum(np.abs(out)))
    if gross > max_gross and gross > 1e-12:
        out = out * (max_gross / gross)
    return out


def causal_lf_plan_reference(
    freq: dict[str, Any],
    assets: int,
    *,
    gain: float = 1.0,
    forecast_blend: float = 0.0,
) -> np.ndarray:
    """Map only causal LF features to an executable portfolio reference."""

    selected_gain = float(gain)
    blend = float(forecast_blend)
    if not np.isfinite(selected_gain) or selected_gain < 0.0:
        raise ValueError("gain must be finite and non-negative")
    if not np.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("forecast_blend must be finite and in [0, 1]")
    dim = int(assets)
    if dim < 1:
        raise ValueError("assets must be positive")
    low = resize(freq.get("x_low", 0.0), dim)
    raw_forecast = np.asarray(
        freq.get("x_low_forecast", low), dtype=np.float64
    )
    forecast = resize(
        raw_forecast[0] if raw_forecast.ndim >= 2 else raw_forecast,
        dim,
    )
    signal = (1.0 - blend) * low + blend * forecast
    return gross_cap(np.tanh(selected_gain * signal / 0.0014))


def resize(value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


def episodic_scenario_seed(root_seed: int, scenario: str) -> int:
    """Derive a stable, disjoint path seed for one reset episode."""

    return derive_seed(
        "freq_hrl_independent_support_episode_v1",
        int(root_seed),
        str(scenario),
    )


def aggregate_episodic_rows(
    rows: Sequence[dict[str, Any]],
    *,
    root_seed: int,
    scenario_label: str,
) -> dict[str, Any]:
    """Aggregate equally weighted reset episodes into one selection row."""

    items = list(rows)
    if not items:
        raise ValueError("at least one episode row is required")
    result: dict[str, Any] = {}
    common_keys = set(items[0]).intersection(*(set(row) for row in items[1:]))
    for key in items[0]:
        if key not in common_keys or key in {"seed", "scenario"}:
            continue
        values = [row[key] for row in items]
        if all(isinstance(value, (bool, int, float, np.number)) for value in values):
            numeric = np.asarray(values, dtype=np.float64)
            if not np.all(np.isfinite(numeric)):
                continue
            result[key] = float(
                np.sum(numeric) if str(key).endswith("_count") else np.mean(numeric)
            )
        elif all(isinstance(value, str) for value in values) and all(
            value == values[0] for value in values[1:]
        ):
            result[key] = values[0]
    result.update({
        "seed": int(root_seed),
        "scenario": str(scenario_label),
        "support_episode_count": int(len(items)),
        "support_episode_scenarios": "|".join(
            str(row["scenario"]) for row in items
        ),
        "training_support_ood_excluded": float(
            all(str(row["scenario"]) != "ood_period" for row in items)
        ),
    })
    return result


def collect_independent_episode_rollouts(
    *,
    root_seed: int,
    sample: bool,
    scenarios: Sequence[str],
    rollout_one: Callable[[int, str], tuple[Any | None, dict[str, Any]]],
    concat_batches: Callable[[Sequence[Any]], Any],
    scenario_label: str,
) -> tuple[Any | None, dict[str, Any]]:
    """Run full reset episodes and combine trajectories only after rollout."""

    batches: list[Any] = []
    rows: list[dict[str, Any]] = []
    for episode_scenario in scenarios:
        batch, row = rollout_one(
            episodic_scenario_seed(int(root_seed), str(episode_scenario)),
            str(episode_scenario),
        )
        if sample:
            if batch is None:
                raise RuntimeError("a sampled independent episode returned no batch")
            batches.append(batch)
        rows.append(row)
    return (
        concat_batches(batches) if sample else None,
        aggregate_episodic_rows(
            rows,
            root_seed=int(root_seed),
            scenario_label=str(scenario_label),
        ),
    )


def make_tracker(
    assets: int,
    *,
    heuristic_promotion: bool = True,
    promotion_adapt_gain: float = 0.05,
) -> TradingFrequencyTracker:
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
        promotion_enable=bool(heuristic_promotion),
        promotion_window_s=30 * 60.0,
        promotion_residual_threshold=0.00035,
        promotion_persistence_ratio=0.50,
        promotion_cooldown_s=10 * 60.0,
        promotion_regime_threshold=3e-05,
        promotion_adapt_low=bool(heuristic_promotion),
        promotion_adapt_gain=float(promotion_adapt_gain),
    )


def frequency_separated_feature_vectors(
    freq: dict[str, Any],
    position: np.ndarray,
    target: np.ndarray | None = None,
    *,
    leakage_feedback: float = 0.0,
    progress: float = 0.0,
    include_heuristic_promotion: bool = True,
    hf_predictability: np.ndarray | None = None,
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
    promote = (
        1.0
        if include_heuristic_promotion and bool(promotion.get("promote", False))
        else 0.0
    )
    promotion_strength = (
        float(promotion.get("promotion_strength", 0.0))
        if include_heuristic_promotion else 0.0
    )
    promotion_shock_age = (
        float(promotion.get("shock_age", 0.0))
        if include_heuristic_promotion else 0.0
    )
    upper_state = np.concatenate([
        resize(upper["x_low"], dim) / scale,
        forecast,
        resize(upper["x_low_uncertainty"], dim) / scale,
        np.tanh(energy),
        np.tanh(resize(upper["x_high_persistence"], dim)),
        np.asarray(position, dtype=np.float64),
        np.asarray([
            promote,
            promotion_strength,
            promotion_shock_age / 30.0,
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
    if hf_predictability is not None:
        predictability = np.clip(
            resize(hf_predictability, dim), 0.0, 1.0
        )
        lower_state = np.concatenate([lower_state, predictability])
    return upper_state.astype(np.float32), lower_state.astype(np.float32)


def promotion_gate_state_dim(assets: int) -> int:
    return 12 * int(assets) + 4


def promotion_gate_feature_vector(
    freq: dict[str, Any],
    *,
    position: np.ndarray,
    target: np.ndarray,
    leakage_feedback: float,
    progress: float,
    elapsed_steps: int,
    upper_period: int,
) -> np.ndarray:
    """Causal gate features without the deterministic promotion decision."""

    position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
    dim = int(position_arr.size)
    target_arr = resize(target, dim)
    scale = 0.0014
    period = max(int(upper_period), 1)
    elapsed_fraction = float(np.clip(int(elapsed_steps) / period, 0.0, 1.0))
    time_to_schedule = float(
        np.clip((period - int(elapsed_steps)) / period, 0.0, 1.0)
    )
    energy = np.sqrt(
        np.maximum(resize(freq.get("x_high_energy", 0.0), dim), 0.0)
    ) / scale
    state = np.concatenate([
        resize(freq.get("x_low", 0.0), dim) / scale,
        resize(freq.get("x_low_forecast", 0.0), dim) / scale,
        resize(freq.get("x_low_uncertainty", 0.0), dim) / scale,
        resize(freq.get("x_mid", 0.0), dim) / scale,
        resize(freq.get("x_high", 0.0), dim) / scale,
        resize(freq.get("x_high_delta", 0.0), dim) / scale,
        np.tanh(energy),
        np.tanh(resize(freq.get("x_high_persistence", 0.0), dim)),
        np.tanh(resize(freq.get("shock_age", 0.0), dim) / 30.0),
        target_arr,
        position_arr,
        target_arr - position_arr,
        np.asarray([
            elapsed_fraction,
            time_to_schedule,
            float(leakage_feedback),
            float(np.clip(progress, 0.0, 1.0)),
        ], dtype=np.float64),
    ])
    expected = promotion_gate_state_dim(dim)
    if state.size != expected:
        raise RuntimeError(
            f"promotion gate state dim mismatch: expected {expected}, got {state.size}"
        )
    return state.astype(np.float32)


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
    include_heuristic_promotion: bool = True,
    hf_predictability: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if str(policy_mode) == "freq_hrl":
        return frequency_separated_feature_vectors(
            freq,
            position,
            target=target,
            leakage_feedback=leakage_feedback,
            progress=progress,
            include_heuristic_promotion=include_heuristic_promotion,
            hf_predictability=hf_predictability,
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


def _bernoulli_actor_parameter_count(state_dim: int, hidden_dim: int) -> int:
    hidden = int(hidden_dim)
    if hidden <= 0:
        return int(state_dim + 1)
    return int(
        hidden * hidden
        + hidden * (int(state_dim) + 3)
        + 1
    )


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

        count = int(
            actor(config.upper_state_dim, config.upper_action_dim)
            + actor(config.lower_state_dim, config.lower_action_dim)
            + value(config.upper_state_dim)
            + 2 * value(config.lower_state_dim)
        )
        if int(config.promotion_state_dim) > 0:
            count += _bernoulli_actor_parameter_count(
                config.promotion_state_dim, config.hidden_dim
            )
            count += _value_parameter_count(
                config.promotion_state_dim, config.hidden_dim
            )
            if float(config.promotion_advantage_coef) > 0.0:
                count += _value_parameter_count(
                    config.promotion_state_dim, config.hidden_dim
                )
        if int(config.hf_state_dim) > 0:
            count += actor(config.hf_state_dim, config.hf_action_dim)
            count += value(config.hf_state_dim)
        return int(count)
    if str(config.state_encoder) != "mlp":
        raise ValueError(f"unknown state_encoder: {config.state_encoder}")
    count = int(
        _actor_parameter_count(
            config.upper_state_dim, config.upper_action_dim, config.hidden_dim
        )
        + _actor_parameter_count(
            config.lower_state_dim, config.lower_action_dim, config.hidden_dim
        )
        + _value_parameter_count(config.upper_state_dim, config.hidden_dim)
        + 2 * _value_parameter_count(config.lower_state_dim, config.hidden_dim)
    )
    if int(config.promotion_state_dim) > 0:
        count += _bernoulli_actor_parameter_count(
            config.promotion_state_dim, config.hidden_dim
        )
        count += _value_parameter_count(
            config.promotion_state_dim, config.hidden_dim
        )
        if float(config.promotion_advantage_coef) > 0.0:
            count += _value_parameter_count(
                config.promotion_state_dim, config.hidden_dim
            )
    if int(config.hf_state_dim) > 0:
        count += _actor_parameter_count(
            config.hf_state_dim, config.hf_action_dim, config.hidden_dim
        )
        count += _value_parameter_count(config.hf_state_dim, config.hidden_dim)
    return int(count)


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
    promotion_state_dim: int = 0,
    hf_state_dim: int = 0,
    hf_action_dim: int = 0,
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
            promotion_state_dim=int(promotion_state_dim),
            hf_state_dim=int(hf_state_dim),
            hf_action_dim=int(hf_action_dim),
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
            promotion_state_dim=int(promotion_state_dim),
            hf_state_dim=int(hf_state_dim),
            hf_action_dim=int(hf_action_dim),
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
        if int(model.config.upper_action_dim) % int(assets) != 0:
            raise ValueError("upper action dimension must be divisible by assets")
        upper_actions_per_asset = int(model.config.upper_action_dim) // int(assets)
        for i in range(assets):
            upper_rows = [
                i * upper_actions_per_asset + k
                for k in range(upper_actions_per_asset)
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


def decode_hierarchical_lower_action(
    latent_action: np.ndarray,
    *,
    assets: int,
    enable_hf_overlay: bool,
    hf_order_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    latent = np.asarray(latent_action, dtype=np.float64).reshape(-1)
    expected = int(assets) * (2 if enable_hf_overlay else 1)
    if latent.size != expected:
        raise ValueError(
            f"expected {expected} hierarchical lower actions, got {latent.size}"
        )
    speed = latent_speed(latent[:assets])
    residual_order = np.zeros(int(assets), dtype=np.float64)
    if enable_hf_overlay:
        residual_order = (
            np.tanh(latent[assets:]) * max(float(hf_order_scale), 0.0)
        )
    return speed, residual_order


def decode_hf_tactical_action(
    latent_action: np.ndarray,
    *,
    assets: int,
    hf_order_scale: float,
) -> np.ndarray:
    latent = np.asarray(latent_action, dtype=np.float64).reshape(-1)
    if latent.size != int(assets):
        raise ValueError(
            f"expected {int(assets)} HF tactical actions, got {latent.size}"
        )
    return np.tanh(latent) * max(float(hf_order_scale), 0.0)


def intervene_lower_observation(
    lower_state: np.ndarray,
    *,
    assets: int,
    policy_mode: str,
    intervention: str,
) -> np.ndarray:
    intervention = str(intervention)
    if intervention not in LOWER_OBSERVATION_INTERVENTIONS:
        raise ValueError(
            f"unknown lower observation intervention: {intervention}"
        )
    state = np.asarray(lower_state, dtype=np.float32).copy()
    if intervention == "none":
        return state
    if str(policy_mode) != "freq_hrl":
        raise ValueError(
            "residual-frequency intervention is only defined for freq_hrl"
        )
    legacy_expected = 8 * int(assets) + 1
    predictability_expected = 9 * int(assets) + 1
    if state.size not in {legacy_expected, predictability_expected}:
        raise ValueError(
            "expected frequency lower state dim "
            f"{legacy_expected} or {predictability_expected}, got {state.size}"
        )
    # Keep plan, position, and gap fixed; remove only MF/HF residual context.
    state[3 * int(assets):8 * int(assets)] = 0.0
    if state.size == predictability_expected:
        state[8 * int(assets) + 1:] = 0.0
    return state


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
    model.reset_recurrent_inference()
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
    learned_promotion_gate: bool = False,
    heuristic_promotion_gate: bool = True,
    promotion_replan_cost: float = 0.0,
    enable_hf_lower: bool = False,
    separate_hf_tactical: bool = False,
    lower_hf_order_scale: float = 0.025,
    lower_observation_intervention: str = "none",
    compute_hf_action_sensitivity: bool = False,
    execution_timeline_contract: str = "legacy_pre_trade_v2",
    method_contract: str = "routing_core_v2",
    promotion_credit_mode: str = "auto",
    leakage_cost_mode: str = "auto",
    lower_lf_budget_rms: float = 0.0025,
    hf_lf_budget_rms: float = 0.00025,
    include_hf_predictability: bool | None = None,
    allow_inactive_mechanism_modules: bool = False,
    upper_plan_reference_mode: str = "none",
    upper_plan_reference_gain: float = 1.0,
    upper_plan_reference_forecast_blend: float = 0.0,
    hard_hf_budget_projection: bool = False,
    promotion_deterministic_threshold: float = 0.5,
    promotion_adapt_gain: float = 0.05,
    promotion_cooldown_steps: int = 0,
    promotion_gate_interval_steps: int = 1,
    promotion_deterministic_mode: str = "actor_probability",
    promotion_advantage_threshold: float = 0.0,
    promotion_advantage_target_threshold: float | None = None,
    promotion_credit_scale: float | None = None,
    upper_residual_action_scale: float = 1.0,
    return_trajectory: bool = False,
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
    upper_plan_reference_mode = str(upper_plan_reference_mode)
    if upper_plan_reference_mode not in UPPER_PLAN_REFERENCE_MODES:
        raise ValueError(
            "unknown upper_plan_reference_mode: "
            f"{upper_plan_reference_mode}"
        )
    if upper_plan_reference_mode != "none" and (
        policy_mode != "freq_hrl" or not execute_plan_curve
    ):
        raise ValueError(
            "a causal upper plan reference requires executable Freq-HRL plans"
        )
    if (
        not np.isfinite(float(upper_plan_reference_gain))
        or float(upper_plan_reference_gain) < 0.0
    ):
        raise ValueError(
            "upper_plan_reference_gain must be finite and non-negative"
        )
    if (
        not np.isfinite(float(upper_plan_reference_forecast_blend))
        or not 0.0 <= float(upper_plan_reference_forecast_blend) <= 1.0
    ):
        raise ValueError(
            "upper_plan_reference_forecast_blend must be finite and in [0, 1]"
        )
    if (
        not np.isfinite(float(promotion_deterministic_threshold))
        or not 0.0 < float(promotion_deterministic_threshold) < 1.0
    ):
        raise ValueError(
            "promotion_deterministic_threshold must be finite and in (0, 1)"
        )
    if (
        not np.isfinite(float(promotion_adapt_gain))
        or float(promotion_adapt_gain) < 0.0
    ):
        raise ValueError("promotion_adapt_gain must be finite and non-negative")
    if int(promotion_cooldown_steps) < 0:
        raise ValueError("promotion_cooldown_steps must be non-negative")
    if int(promotion_gate_interval_steps) < 1:
        raise ValueError("promotion_gate_interval_steps must be positive")
    promotion_deterministic_mode = str(promotion_deterministic_mode)
    if promotion_deterministic_mode not in {
        "actor_probability", "counterfactual_advantage",
    }:
        raise ValueError("unknown deterministic promotion mode")
    if not np.isfinite(float(promotion_advantage_threshold)):
        raise ValueError("promotion_advantage_threshold must be finite")
    resolved_promotion_advantage_target_threshold = (
        float(promotion_advantage_threshold)
        if promotion_advantage_target_threshold is None
        else float(promotion_advantage_target_threshold)
    )
    if not np.isfinite(resolved_promotion_advantage_target_threshold):
        raise ValueError("promotion_advantage_target_threshold must be finite")
    resolved_promotion_credit_scale = (
        float(reward_scale)
        if promotion_credit_scale is None else float(promotion_credit_scale)
    )
    if (
        not np.isfinite(resolved_promotion_credit_scale)
        or resolved_promotion_credit_scale <= 0.0
    ):
        raise ValueError("promotion_credit_scale must be positive and finite")
    if (
        not np.isfinite(float(upper_residual_action_scale))
        or float(upper_residual_action_scale) < 0.0
    ):
        raise ValueError(
            "upper_residual_action_scale must be finite and non-negative"
        )
    method_flags = resolve_method_contract(str(method_contract))
    if str(promotion_credit_mode) == "auto":
        if str(method_contract) in V7_METHOD_CONTRACTS:
            promotion_credit_mode = "paired_plan_advantage"
        elif method_flags["promotion_plan_advantage_credit"]:
            promotion_credit_mode = "incremental_plan_advantage"
        else:
            promotion_credit_mode = "task_return"
    if str(promotion_credit_mode) not in {
        "task_return",
        "incremental_plan_advantage",
        "paired_plan_advantage",
    }:
        raise ValueError("unknown promotion_credit_mode")
    if str(leakage_cost_mode) == "auto":
        leakage_cost_mode = (
            "fixed_rms_budget"
            if method_flags["fixed_rms_leakage_budget"]
            else "spectral_ratio"
        )
    if str(leakage_cost_mode) not in {
        "spectral_ratio",
        "fixed_rms_budget",
    }:
        raise ValueError("unknown leakage_cost_mode")
    include_hf_predictability = (
        bool(method_flags["hf_predictability_summary"])
        if include_hf_predictability is None
        else bool(include_hf_predictability)
    )
    if str(leakage_cost_mode) == "fixed_rms_budget":
        for name, value in (
            ("lower_lf_budget_rms", lower_lf_budget_rms),
            ("hf_lf_budget_rms", hf_lf_budget_rms),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
    if hard_hf_budget_projection and not enable_hf_lower:
        raise ValueError("hard HF budget projection requires an active HF lower")
    if hard_hf_budget_projection and str(leakage_cost_mode) != "fixed_rms_budget":
        raise ValueError(
            "hard HF budget projection requires fixed_rms_budget leakage accounting"
        )
    if (
        str(promotion_credit_mode) in {
            "incremental_plan_advantage",
            "paired_plan_advantage",
        }
        and not execute_plan_curve
    ):
        raise ValueError(
            "plan-advantage promotion credit requires an executable plan curve"
        )
    if use_additive_frequency_credit and str(mark_to_market_timing) != "post_trade":
        raise ValueError(
            "additive frequency credit requires post_trade mark-to-market timing"
        )
    if not np.isfinite(float(promotion_replan_cost)) or float(promotion_replan_cost) < 0.0:
        raise ValueError("promotion_replan_cost must be finite and non-negative")
    if not np.isfinite(float(lower_hf_order_scale)) or float(lower_hf_order_scale) < 0.0:
        raise ValueError("lower_hf_order_scale must be finite and non-negative")
    lower_observation_intervention = str(lower_observation_intervention)
    if lower_observation_intervention not in LOWER_OBSERVATION_INTERVENTIONS:
        raise ValueError(
            "unknown lower_observation_intervention: "
            f"{lower_observation_intervention}"
        )
    if (
        lower_observation_intervention != "none"
        or bool(compute_hf_action_sensitivity)
    ) and policy_mode != "freq_hrl":
        raise ValueError(
            "lower frequency interventions require policy_mode='freq_hrl'"
        )
    use_separate_hf = bool(separate_hf_tactical and enable_hf_lower)
    expected_lower_action_dim = int(assets) * (
        2 if enable_hf_lower and not use_separate_hf else 1
    )
    if int(model.config.lower_action_dim) != expected_lower_action_dim:
        raise ValueError(
            "lower action dim mismatch: "
            f"expected {expected_lower_action_dim}, got {model.config.lower_action_dim}"
        )
    expected_lower_state_dim = (
        (9 if include_hf_predictability else 8) * int(assets) + 1
        if policy_mode == "freq_hrl"
        else int(model.config.lower_state_dim)
    )
    if int(model.config.lower_state_dim) != expected_lower_state_dim:
        raise ValueError(
            "lower state dim mismatch: "
            f"expected {expected_lower_state_dim}, got {model.config.lower_state_dim}"
        )
    expected_hf_state_dim = expected_lower_state_dim if use_separate_hf else 0
    expected_hf_action_dim = int(assets) if use_separate_hf else 0
    configured_hf = (
        int(model.config.hf_state_dim), int(model.config.hf_action_dim)
    )
    expected_hf = (expected_hf_state_dim, expected_hf_action_dim)
    inactive_full_hf = (expected_lower_state_dim, int(assets))
    if configured_hf != expected_hf and not (
        bool(allow_inactive_mechanism_modules)
        and not use_separate_hf
        and configured_hf == inactive_full_hf
    ):
        raise ValueError(
            "HF tactical dimensions mismatch: "
            f"expected state/action {expected_hf_state_dim}/{expected_hf_action_dim}, "
            f"got {model.config.hf_state_dim}/{model.config.hf_action_dim}"
        )
    if separate_hf_tactical and policy_mode != "freq_hrl":
        raise ValueError(
            "separate HF tactical credit requires policy_mode='freq_hrl'"
        )
    if learned_promotion_gate:
        if policy_mode != "freq_hrl":
            raise ValueError("learned promotion gate requires policy_mode='freq_hrl'")
        expected_gate_dim = promotion_gate_state_dim(assets)
        if int(model.config.promotion_state_dim) != expected_gate_dim:
            raise ValueError(
                "learned promotion state dim mismatch: "
                f"expected {expected_gate_dim}, got {model.config.promotion_state_dim}"
            )
    if learned_promotion_gate and heuristic_promotion_gate:
        raise ValueError(
            "learned and heuristic promotion gates cannot both control replanning"
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
    tracker = make_tracker(
        assets,
        heuristic_promotion=bool(heuristic_promotion_gate),
        promotion_adapt_gain=float(promotion_adapt_gain),
    )
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
        reward_penalty_scale=leakage_scale,
        enabled=leakage_scale > 0.0,
    )
    tracking_leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
        reward_penalty_scale=leakage_scale,
        enabled=leakage_scale > 0.0,
    )
    hf_leakage = CausalLeakageRewardShaper(
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
    promotion_builder = (
        PromotionRolloutBuilder(gamma=float(model.config.gamma))
        if learned_promotion_gate else None
    )
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
    raw_tracking_lower_effects: list[np.ndarray] = []
    hf_overlay_effects: list[np.ndarray] = []
    raw_recenter_boosts: list[np.ndarray] = []
    plan_smoothness: list[float] = []
    plan_coeff_abs: list[float] = []
    plan_reference_target_abs: list[float] = []
    plan_reference_coeff_abs: list[float] = []
    plan_residual_coeff_abs: list[float] = []
    upper_credits: list[float] = []
    lower_credits: list[float] = []
    hf_tactical_credits: list[float] = []
    upper_task_credits: list[float] = []
    lower_task_credits: list[float] = []
    hf_tactical_task_credits: list[float] = []
    plan_returns: list[float] = []
    execution_deviation_returns: list[float] = []
    credit_reconstruction_errors: list[float] = []
    upper_leakage_costs: list[float] = []
    lower_leakage_costs: list[float] = []
    hf_leakage_costs: list[float] = []
    decision_reasons: list[str] = []
    promotion_signals = 0
    learned_gate_probabilities: list[float] = []
    learned_gate_advantages: list[float] = []
    learned_gate_actions: list[float] = []
    learned_promotion_absorbed_norm: list[float] = []
    learned_replan_cost_total = 0.0
    promotion_scheduled_boundary_closes = 0
    hf_order_l1: list[float] = []
    hf_overlay_position_l1: list[float] = []
    hf_overlay_returns: list[float] = []
    hf_overlay_incremental_costs: list[float] = []
    hf_overlay_task_effects: list[float] = []
    lower_hf_action_sensitivities: list[float] = []
    lower_hf_overlay_sensitivities: list[float] = []
    promotion_credits: list[float] = []
    promotion_plan_advantages: list[float] = []
    promotion_credit_actions: list[float] = []
    tracking_leakage_budget_ratios: list[float] = []
    hf_leakage_budget_ratios: list[float] = []
    leakage_budget_violations: list[float] = []
    hf_overlay_rms_before_projection: list[float] = []
    hf_overlay_rms_after_projection: list[float] = []
    hf_overlay_projection_scales: list[float] = []
    hf_overlay_projection_events: list[float] = []
    hf_predictability_values: list[np.ndarray] = []
    latest_leakage_feedback = 0.0
    current_target: np.ndarray | None = None
    raw_history: list[np.ndarray] = []
    high_history: list[np.ndarray] = []
    realized_return_history: list[np.ndarray] = []
    promotion_counterfactual_plan: LearnedPlanCurveState | None = None
    promotion_selected_action: float | None = None
    pending_plan_smoothness_cost = 0.0
    last_learned_promotion_step: int | None = None
    last_promotion_gate_step: int | None = None
    promotion_cooldown_blocks = 0
    promotion_gate_interval_blocks = 0

    env.reset()
    model.reset_recurrent_inference()
    for t in range(steps):
        raw_history.append(np.asarray(data["predictor"][t], dtype=np.float64).copy())
        freq = tracker.update_bar(data["predictor"][t], t=float(t * 60.0))
        if plan_state is not None and plan_state.active:
            current_target = gross_cap(plan_state.value_at(float(t * 60.0)))
        # Estimate h_t -> r_{t+1} predictability using completed pairs only.
        # The current h_t is appended after this estimate and r_t after execution.
        hf_predictability = (
            causal_hf_predictability(
                high_history,
                realized_return_history,
                int(assets),
            )
            if include_hf_predictability
            else None
        )
        if hf_predictability is not None:
            hf_predictability_values.append(hf_predictability.copy())
        high_history.append(
            resize(freq.get("x_high", 0.0), int(assets)).copy()
        )
        promotion = dict(freq.get("promotion", {}) or {})
        promote = bool(promotion.get("promote", False))
        learned_replan_cost_this_step = 0.0
        paired_replan_cost_this_step = 0.0
        promotion_candidate_upper_state: np.ndarray | None = None
        promotion_candidate_upper_out: dict[str, np.ndarray | float] | None = None
        forced_reason = scheduler.decision_reason(t, promotion=False)
        if forced_reason is not None:
            if (
                forced_reason == "scheduled"
                and promotion_builder is not None
                and promotion_builder.has_pending
            ):
                promotion_builder.close(done=False)
                promotion_counterfactual_plan = None
                promotion_selected_action = None
                promotion_scheduled_boundary_closes += 1
            reason = forced_reason
        elif learned_promotion_gate:
            if current_target is None or scheduler.last_upper_step is None:
                raise RuntimeError("learned promotion requires an active upper plan")
            elapsed = int(t - scheduler.last_upper_step)
            cooldown_ready = (
                last_learned_promotion_step is None
                or int(t - last_learned_promotion_step)
                >= int(promotion_cooldown_steps)
            )
            gate_interval_ready = (
                last_promotion_gate_step is None
                or int(t - last_promotion_gate_step)
                >= int(promotion_gate_interval_steps)
            )
            if (
                elapsed >= int(min_upper_duration)
                and cooldown_ready
                and gate_interval_ready
            ):
                gate_state = promotion_gate_feature_vector(
                    dict(freq),
                    position=env.position.copy(),
                    target=current_target,
                    leakage_feedback=latest_leakage_feedback,
                    progress=t / max(int(steps) - 1, 1),
                    elapsed_steps=elapsed,
                    upper_period=int(upper_period),
                )
                gate_out = model.act_promotion(
                    gate_state,
                    sample=sample,
                    deterministic_threshold=float(
                        promotion_deterministic_threshold
                    ),
                    deterministic_mode=promotion_deterministic_mode,
                    advantage_threshold=float(
                        promotion_advantage_threshold
                    ),
                )
                gate_action = float(gate_out["action"])
                if promotion_builder is None:
                    raise RuntimeError("learned promotion builder is unavailable")
                promotion_builder.begin(
                    state=gate_state,
                    action=gate_action,
                    logp=float(gate_out["logp"]),
                    value=float(gate_out["value"]),
                )
                promotion_counterfactual_plan = None
                promotion_selected_action = gate_action
                last_promotion_gate_step = int(t)
                learned_gate_probabilities.append(
                    float(gate_out["probability"])
                )
                if bool(gate_out["advantage_head_enabled"]):
                    learned_gate_advantages.append(float(
                        gate_out["predicted_counterfactual_advantage"]
                    ))
                learned_gate_actions.append(gate_action)
                if str(promotion_credit_mode) == "paired_plan_advantage":
                    if plan_state is None or not plan_state.active:
                        raise RuntimeError(
                            "paired promotion credit requires an active old plan"
                        )
                    counterfactual_tracker = tracker.snapshot()
                    promoted_candidate_freq = counterfactual_tracker.promote_residual(
                        strength=1.0
                    )
                    promotion_candidate_upper_state, _ = smdp_policy_feature_vectors(
                        policy_mode=policy_mode,
                        freq=dict(promoted_candidate_freq),
                        raw_history=np.asarray(raw_history, dtype=np.float64),
                        position=env.position.copy(),
                        target=current_target,
                        leakage_feedback=latest_leakage_feedback,
                        progress=t / max(int(steps) - 1, 1),
                        history_window=history_window,
                        include_heuristic_promotion=bool(heuristic_promotion_gate),
                        hf_predictability=hf_predictability,
                    )
                    promotion_candidate_upper_out = model.act_upper(
                        promotion_candidate_upper_state,
                        sample=sample,
                    )
                    if gate_action < 0.5:
                        shadow_plan = plan_state.snapshot()
                        shadow_action = (
                            np.asarray(
                                promotion_candidate_upper_out["action"],
                                dtype=np.float64,
                            )
                            * float(upper_residual_action_scale)
                        )
                        shadow_reference = (
                            causal_lf_plan_reference(
                                dict(promoted_candidate_freq),
                                int(assets),
                                gain=float(upper_plan_reference_gain),
                                forecast_blend=float(
                                    upper_plan_reference_forecast_blend
                                ),
                            )
                            if upper_plan_reference_mode == "causal_lf"
                            else None
                        )
                        shadow_plan.activate(
                            now_s=float(t * 60.0),
                            current_value=current_target,
                            latent_action=shadow_action,
                            reference_target=shadow_reference,
                        )
                        promotion_counterfactual_plan = shadow_plan
                    paired_replan_cost_this_step = float(
                        promotion_replan_cost
                    )
                if gate_action >= 0.5:
                    if str(promotion_credit_mode) in {
                        "incremental_plan_advantage",
                        "paired_plan_advantage",
                    }:
                        if plan_state is None or not plan_state.active:
                            raise RuntimeError(
                                "plan-advantage promotion credit requires an active old plan"
                            )
                        promotion_counterfactual_plan = plan_state.snapshot()
                    promoted_freq = tracker.promote_residual(strength=1.0)
                    learned_info = dict(
                        promoted_freq.get("learned_promotion", {}) or {}
                    )
                    learned_promotion_absorbed_norm.append(
                        float(learned_info.get("absorbed_norm", 0.0))
                    )
                    freq = promoted_freq
                    promotion_signals += 1
                    learned_replan_cost_this_step = float(
                        promotion_replan_cost
                    )
                    learned_replan_cost_total += learned_replan_cost_this_step
                    reason = scheduler.decision_reason(t, promotion=True)
                    if reason != "promotion":
                        raise RuntimeError(
                            "eligible learned gate did not produce a promotion decision"
                        )
                    last_learned_promotion_step = int(t)
                else:
                    reason = None
            else:
                if elapsed >= int(min_upper_duration) and not cooldown_ready:
                    promotion_cooldown_blocks += 1
                if (
                    elapsed >= int(min_upper_duration)
                    and cooldown_ready
                    and not gate_interval_ready
                ):
                    promotion_gate_interval_blocks += 1
                reason = None
        elif heuristic_promotion_gate:
            if promote:
                promotion_signals += 1
            reason = scheduler.decision_reason(
                t,
                promotion=bool(promote and policy_mode == "freq_hrl"),
            )
        else:
            reason = None
        if reason is not None:
            if (
                reason == "promotion"
                and promotion_candidate_upper_state is not None
                and promotion_candidate_upper_out is not None
            ):
                upper_state = promotion_candidate_upper_state
                upper_out = promotion_candidate_upper_out
            else:
                upper_state, _ = smdp_policy_feature_vectors(
                    policy_mode=policy_mode,
                    freq=dict(freq),
                    raw_history=np.asarray(raw_history, dtype=np.float64),
                    position=env.position.copy(),
                    target=current_target,
                    leakage_feedback=latest_leakage_feedback,
                    progress=t / max(int(steps) - 1, 1),
                    history_window=history_window,
                    include_heuristic_promotion=bool(heuristic_promotion_gate),
                    hf_predictability=hf_predictability,
                )
                upper_out = model.act_upper(upper_state, sample=sample)
            executed_upper_action = (
                np.asarray(upper_out["action"], dtype=np.float64)
                * float(upper_residual_action_scale)
            )
            reference_target = (
                causal_lf_plan_reference(
                    dict(freq),
                    int(assets),
                    gain=float(upper_plan_reference_gain),
                    forecast_blend=float(
                        upper_plan_reference_forecast_blend
                    ),
                )
                if upper_plan_reference_mode == "causal_lf"
                else None
            )
            if reference_target is not None:
                plan_reference_target_abs.append(
                    float(np.mean(np.abs(reference_target)))
                )
            if plan_mapper is None:
                current_target = latent_target(executed_upper_action)
            elif plan_state is not None:
                plan = plan_state.activate(
                    now_s=float(t * 60.0),
                    current_value=(
                        env.position.copy()
                        if current_target is None else current_target
                    ),
                    latent_action=executed_upper_action,
                    reference_target=reference_target,
                )
                current_target = gross_cap(plan.target)
                plan_smoothness.append(float(plan.smoothness_penalty))
                plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
                if plan.reference_coefficients is not None:
                    plan_reference_coeff_abs.append(float(np.mean(np.abs(
                        plan.reference_coefficients
                    ))))
                if plan.residual_coefficients is not None:
                    plan_residual_coeff_abs.append(float(np.mean(np.abs(
                        plan.residual_coefficients
                    ))))
                pending_plan_smoothness_cost = max(
                    float(plan_smoothness_weight), 0.0
                ) * float(plan.smoothness_penalty)
            else:
                plan = plan_mapper.target(
                    env.position.copy(),
                    executed_upper_action,
                    reference_target=reference_target,
                )
                current_target = gross_cap(plan.target)
                plan_smoothness.append(float(plan.smoothness_penalty))
                plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
                if plan.reference_coefficients is not None:
                    plan_reference_coeff_abs.append(float(np.mean(np.abs(
                        plan.reference_coefficients
                    ))))
                if plan.residual_coefficients is not None:
                    plan_residual_coeff_abs.append(float(np.mean(np.abs(
                        plan.residual_coefficients
                    ))))
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
            include_heuristic_promotion=bool(heuristic_promotion_gate),
            hf_predictability=hf_predictability,
        )
        factual_lower_state = np.asarray(lower_state, dtype=np.float32).copy()
        if compute_hf_action_sensitivity:
            ablated_state = intervene_lower_observation(
                factual_lower_state,
                assets=assets,
                policy_mode=policy_mode,
                intervention="zero_residual_frequency",
            )
            if use_separate_hf:
                factual_out = model.act_hf(factual_lower_state, sample=False)
                ablated_out = model.act_hf(ablated_state, sample=False)
            else:
                factual_out = model.act_lower(factual_lower_state, sample=False)
                ablated_out = model.act_lower(ablated_state, sample=False)
            factual_action = np.asarray(factual_out["action"], dtype=np.float64)
            ablated_action = np.asarray(ablated_out["action"], dtype=np.float64)
            action_delta = factual_action - ablated_action
            lower_hf_action_sensitivities.append(
                float(np.mean(np.abs(action_delta)))
            )
            if use_separate_hf:
                factual_overlay = decode_hf_tactical_action(
                    factual_action,
                    assets=assets,
                    hf_order_scale=lower_hf_order_scale,
                )
                ablated_overlay = decode_hf_tactical_action(
                    ablated_action,
                    assets=assets,
                    hf_order_scale=lower_hf_order_scale,
                )
                overlay_delta = factual_overlay - ablated_overlay
            else:
                overlay_delta = (
                    action_delta[assets:]
                    if enable_hf_lower else np.zeros(assets, dtype=np.float64)
                )
            lower_hf_overlay_sensitivities.append(
                float(np.mean(np.abs(overlay_delta)))
            )
        hf_state: np.ndarray | None = None
        hf_out: dict[str, np.ndarray | float] | None = None
        if use_separate_hf:
            lower_state = factual_lower_state
            hf_state = intervene_lower_observation(
                factual_lower_state,
                assets=assets,
                policy_mode=policy_mode,
                intervention=lower_observation_intervention,
            )
            lower_out = model.act_lower(lower_state, sample=sample)
            hf_out = model.act_hf(hf_state, sample=sample)
            speed, _ = decode_hierarchical_lower_action(
                np.asarray(lower_out["action"], dtype=np.float64),
                assets=assets,
                enable_hf_overlay=False,
                hf_order_scale=float(lower_hf_order_scale),
            )
            residual_order = decode_hf_tactical_action(
                np.asarray(hf_out["action"], dtype=np.float64),
                assets=assets,
                hf_order_scale=float(lower_hf_order_scale),
            )
        else:
            lower_state = intervene_lower_observation(
                factual_lower_state,
                assets=assets,
                policy_mode=policy_mode,
                intervention=lower_observation_intervention,
            )
            lower_out = model.act_lower(lower_state, sample=sample)
            speed, residual_order = decode_hierarchical_lower_action(
                np.asarray(lower_out["action"], dtype=np.float64),
                assets=assets,
                enable_hf_overlay=bool(enable_hf_lower),
                hf_order_scale=float(lower_hf_order_scale),
            )
        pre_gap = np.asarray(current_target, dtype=np.float64) - env.position.copy()
        raw_recenter_boost = max(float(lower_lf_raw_recenter_gain), 0.0) * np.tanh(
            np.abs(pre_gap) / max(float(lower_lf_raw_recenter_scale), 1e-9)
        )
        if lower_lf_raw_recenter_gain > 0.0:
            speed = np.clip(speed + raw_recenter_boost, 0.05, 1.0)
        env.set_target(current_target)
        lower_env_action: dict[str, Any] = {
            "execution_speed": speed,
            "residual_order": residual_order,
        }
        if hard_hf_budget_projection:
            lower_env_action["hf_overlay_rms_cap"] = float(
                hf_lf_budget_rms
            )
        _, reward, done, info = env.lower_step(lower_env_action)
        hf_overlay_rms_before_projection.append(float(
            info["hf_overlay_rms_before_projection"]
        ))
        hf_overlay_rms_after_projection.append(float(
            info["hf_overlay_rms_after_projection"]
        ))
        hf_overlay_projection_scales.append(float(
            info["hf_overlay_projection_scale"]
        ))
        hf_overlay_projection_events.append(float(
            bool(info["hf_overlay_projected"])
        ))
        raw_lower_effect = np.asarray(info["position"], dtype=np.float64) - np.asarray(
            info["target"], dtype=np.float64
        )
        raw_tracking_lower_effect = np.asarray(
            info["tracking_only_position"], dtype=np.float64
        ) - np.asarray(info["target"], dtype=np.float64)
        hf_overlay_effect = np.asarray(
            info["hf_overlay_position_effect"], dtype=np.float64
        )
        if use_separate_hf:
            tracking_lower_effect = (
                lower_effect_projector.transform(raw_tracking_lower_effect)
                if lower_effect_projector is not None
                else raw_tracking_lower_effect
            )
            lower_effect = tracking_lower_effect + hf_overlay_effect
        else:
            tracking_lower_effect = raw_tracking_lower_effect
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
        tracking_leak_info = tracking_leakage.update(
            upper_effect=current_target,
            lower_effect=(
                raw_tracking_lower_effect
                if constrain_raw_lower_effect else tracking_lower_effect
            ),
        )
        hf_leak_info = hf_leakage.update(
            upper_effect=np.zeros_like(current_target),
            lower_effect=hf_overlay_effect,
        )
        if str(leakage_cost_mode) == "fixed_rms_budget":
            tracking_budget_info = evaluate_rms_leakage_budget(
                float(
                    tracking_leak_info.get("LowerLFDriftAbs", 0.0)
                    if use_separate_hf
                    else leak_info.get("LowerLFDriftAbs", 0.0)
                ),
                float(lower_lf_budget_rms),
            )
            hf_budget_info = evaluate_rms_leakage_budget(
                float(hf_leak_info.get("LowerLFDriftAbs", 0.0)),
                float(hf_lf_budget_rms),
            )
            tracking_budget_ratio = float(
                tracking_budget_info["budget_ratio"]
            )
            hf_budget_ratio = float(hf_budget_info["budget_ratio"])
            tracking_budget_cost = float(
                tracking_budget_info["budget_excess_squared"]
            )
            hf_budget_cost = float(hf_budget_info["budget_excess_squared"])
            lower_constraint_feedback = max(
                tracking_budget_ratio,
                hf_budget_ratio if use_separate_hf else 0.0,
            )
            latest_leakage_feedback = max(
                float(tracking_budget_info["budget_excess"]),
                float(hf_budget_info["budget_excess"])
                if use_separate_hf else 0.0,
            )
        else:
            tracking_budget_ratio = float(
                tracking_leak_info.get("lower_lf_penalty", 0.0)
                if use_separate_hf
                else leak_info.get("lower_lf_penalty", 0.0)
            )
            hf_budget_ratio = float(
                hf_leak_info.get("lower_lf_penalty", 0.0)
            )
            tracking_budget_cost = tracking_budget_ratio
            hf_budget_cost = hf_budget_ratio
            latest_leakage_feedback = float(
                leak_info.get("lower_lf_penalty", 0.0)
            )
            lower_constraint_feedback = tracking_budget_ratio
        if str(method_contract) in {
            "ablate_leakage_v6",
            "ablate_leakage_v7",
        }:
            latest_leakage_feedback = 0.0
            lower_constraint_feedback = 0.0
        tracking_leakage_budget_ratios.append(tracking_budget_ratio)
        hf_leakage_budget_ratios.append(hf_budget_ratio)
        leakage_budget_violations.append(
            float(max(
                tracking_budget_ratio,
                hf_budget_ratio if use_separate_hf else 0.0,
            ) > 1.0)
            if str(leakage_cost_mode) == "fixed_rms_budget"
            else 0.0
        )
        shaped_reward = float(
            leak_info["shaped_reward"] if leak_info["shaped_reward"] is not None else reward
        )
        leakage_reward_penalty = max(float(reward) - shaped_reward, 0.0)
        if use_additive_frequency_credit:
            upper_leakage_cost = max(float(leakage_scale), 0.0) * float(
                leak_info.get("upper_hf_penalty", 0.0)
            )
            if use_separate_hf:
                lower_leakage_cost = max(float(leakage_scale), 0.0) * float(
                    tracking_budget_cost
                )
                hf_leakage_cost = max(float(leakage_scale), 0.0) * float(
                    hf_budget_cost
                )
                tactical_credit = credit_assigner.assign_tactical(
                    info,
                    active_plan=current_target,
                    upper_leakage_cost=upper_leakage_cost,
                    tracking_leakage_cost=lower_leakage_cost,
                    hf_leakage_cost=hf_leakage_cost,
                    plan_smoothness_cost=pending_plan_smoothness_cost,
                )
                upper_credit = float(tactical_credit.upper_training_credit)
                lower_credit = float(tactical_credit.tracking_training_credit)
                hf_tactical_credit = float(tactical_credit.hf_training_credit)
                upper_task_credit = float(tactical_credit.upper_task_credit)
                lower_task_credit = float(tactical_credit.tracking_task_credit)
                hf_tactical_task_credit = float(
                    tactical_credit.hf_task_credit
                )
                plan_return = float(tactical_credit.plan_return)
                execution_deviation_return = float(
                    tactical_credit.tracking_deviation_return
                    + tactical_credit.hf_overlay_return
                )
                reconstruction_error = float(
                    tactical_credit.task_reconstruction_error
                )
            else:
                lower_leakage_cost = max(float(leakage_scale), 0.0) * float(
                    tracking_budget_cost
                )
                hf_leakage_cost = 0.0
                credit = credit_assigner.assign(
                    info,
                    active_plan=current_target,
                    upper_leakage_cost=upper_leakage_cost,
                    lower_leakage_cost=lower_leakage_cost,
                    plan_smoothness_cost=pending_plan_smoothness_cost,
                )
                upper_credit = float(credit.upper_training_credit)
                lower_credit = float(credit.lower_training_credit)
                hf_tactical_credit = 0.0
                upper_task_credit = float(credit.upper_task_credit)
                lower_task_credit = float(credit.lower_task_credit)
                hf_tactical_task_credit = 0.0
                plan_return = float(credit.plan_return)
                execution_deviation_return = float(
                    credit.execution_deviation_return
                )
                reconstruction_error = float(credit.task_reconstruction_error)
            pending_plan_smoothness_cost = 0.0
        else:
            upper_leakage_cost = 0.0
            lower_leakage_cost = leakage_reward_penalty
            hf_leakage_cost = 0.0
            upper_credit = float(info["portfolio_return"]) - float(
                info.get("drawdown_cost", 0.0)
            )
            lower_credit = (
                -float(info["transaction_cost"])
                - float(info.get("inventory_drift_cost", 0.0))
                - leakage_reward_penalty
            )
            hf_tactical_credit = 0.0
            upper_task_credit = upper_credit
            lower_task_credit = lower_credit + leakage_reward_penalty
            hf_tactical_task_credit = 0.0
            plan_return = float(info["portfolio_return"])
            execution_deviation_return = 0.0
            reconstruction_error = float("nan")
        hf_builder_fields: dict[str, Any] = {}
        if use_separate_hf:
            if hf_state is None or hf_out is None:
                raise RuntimeError("HF tactical action was not sampled")
            hf_builder_fields = {
                "hf_state": hf_state,
                "hf_action": np.asarray(hf_out["action"], dtype=np.float32),
                "hf_logp": float(hf_out["logp"]),
                "hf_value": float(hf_out["value"]),
                "hf_reward": float(reward_scale) * hf_tactical_credit,
                "hf_cost": float(hf_budget_ratio),
            }
        builder.add_lower(
            state=lower_state,
            action=np.asarray(lower_out["action"], dtype=np.float32),
            logp=float(lower_out["logp"]),
            value=float(lower_out["value"]),
            reward=float(reward_scale) * lower_credit,
            upper_reward=float(reward_scale) * upper_credit,
            cost=lower_constraint_feedback,
            upper_cost=0.0,
            done=bool(done),
            **hf_builder_fields,
        )
        if promotion_builder is not None and promotion_builder.has_pending:
            promotion_counterfactual_advantage = None
            if str(promotion_credit_mode) == "paired_plan_advantage":
                if (
                    promotion_counterfactual_plan is None
                    or promotion_selected_action is None
                ):
                    raise RuntimeError(
                        "paired promotion credit is missing its alternative plan"
                    )
                alternative_target = gross_cap(
                    promotion_counterfactual_plan.value_at(float(t * 60.0))
                )
                selected_target = np.asarray(current_target, dtype=np.float64)
                if promotion_selected_action >= 0.5:
                    promoted_target = selected_target
                    continue_target = alternative_target
                else:
                    promoted_target = alternative_target
                    continue_target = selected_target
                net_replan_advantage = float(np.dot(
                    promoted_target - continue_target,
                    np.asarray(info["asset_returns"], dtype=np.float64),
                )) - float(paired_replan_cost_this_step)
                promotion_counterfactual_advantage = net_replan_advantage
                promotion_credit = (
                    net_replan_advantage
                    if promotion_selected_action >= 0.5
                    else -net_replan_advantage
                )
                promotion_plan_advantages.append(net_replan_advantage)
            elif str(promotion_credit_mode) == "incremental_plan_advantage":
                promotion_credit = 0.0
                if promotion_counterfactual_plan is not None:
                    old_target = gross_cap(
                        promotion_counterfactual_plan.value_at(float(t * 60.0))
                    )
                    promotion_credit = float(np.dot(
                        np.asarray(current_target, dtype=np.float64) - old_target,
                        np.asarray(info["asset_returns"], dtype=np.float64),
                    ))
                promotion_credit -= float(learned_replan_cost_this_step)
                promotion_plan_advantages.append(float(promotion_credit))
            else:
                promotion_credit = (
                    float(info["task_reward"])
                    - learned_replan_cost_this_step
                )
            promotion_credits.append(float(promotion_credit))
            promotion_credit_actions.append(float(
                promotion_selected_action
                if promotion_selected_action is not None else 0.0
            ))
            promotion_builder.add_reward(
                resolved_promotion_credit_scale * float(promotion_credit),
                counterfactual_advantage=(
                    None
                    if promotion_counterfactual_advantage is None
                    else resolved_promotion_credit_scale
                    * float(promotion_counterfactual_advantage)
                ),
                done=bool(done),
            )

        upper_credits.append(upper_credit)
        lower_credits.append(lower_credit)
        hf_tactical_credits.append(hf_tactical_credit)
        upper_task_credits.append(upper_task_credit)
        lower_task_credits.append(lower_task_credit)
        hf_tactical_task_credits.append(hf_tactical_task_credit)
        plan_returns.append(plan_return)
        execution_deviation_returns.append(execution_deviation_return)
        credit_reconstruction_errors.append(reconstruction_error)
        upper_leakage_costs.append(upper_leakage_cost)
        lower_leakage_costs.append(lower_leakage_cost)
        hf_leakage_costs.append(hf_leakage_cost)
        hf_order_l1.append(float(np.sum(np.abs(residual_order))))
        hf_overlay_position_l1.append(float(np.sum(np.abs(
            np.asarray(info["hf_overlay_position_effect"], dtype=np.float64)
        ))))
        hf_overlay_returns.append(float(info["hf_overlay_return"]))
        hf_overlay_incremental_costs.append(float(
            info["hf_overlay_incremental_transaction_cost"]
            + info["hf_overlay_incremental_inventory_drift_cost"]
            + info["hf_overlay_incremental_drawdown_cost"]
        ))
        hf_overlay_task_effects.append(float(info["hf_overlay_task_effect"]))
        realized_return_history.append(
            np.asarray(info["asset_returns"], dtype=np.float64).copy()
        )
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        lower_effects.append(lower_effect.copy())
        raw_lower_effects.append(raw_lower_effect.copy())
        raw_tracking_lower_effects.append(raw_tracking_lower_effect.copy())
        hf_overlay_effects.append(hf_overlay_effect.copy())
        raw_recenter_boosts.append(np.asarray(raw_recenter_boost, dtype=np.float64).copy())
        if done:
            break

    builder.finish(terminal=True)
    trajectory = builder.build()
    if promotion_builder is not None:
        promotion_builder.finish(terminal=True)
        trajectory.promotion = promotion_builder.build()
    promotion_advantage_targets = (
        np.asarray(
            trajectory.promotion.counterfactual_advantage,
            dtype=np.float64,
        ).reshape(-1)
        if trajectory.promotion is not None
        and trajectory.promotion.counterfactual_advantage is not None
        else np.zeros(0, dtype=np.float64)
    )
    promotion_advantage_predictions = np.asarray(
        learned_gate_advantages, dtype=np.float64
    )
    promotion_advantage_head_enabled = model.promotion_advantage is not None
    promotion_advantage_alignment_valid = (
        promotion_advantage_targets.size > 0
        and promotion_advantage_predictions.size
        == promotion_advantage_targets.size
    )
    promotion_advantage_mae = (
        float(np.mean(np.abs(
            promotion_advantage_predictions - promotion_advantage_targets
        )))
        if promotion_advantage_alignment_valid else 0.0
    )
    promotion_advantage_sign_accuracy = (
        float(np.mean(
            (promotion_advantage_predictions >= float(
                promotion_advantage_threshold
            ))
            == (promotion_advantage_targets >= float(
                resolved_promotion_advantage_target_threshold
            ))
        ))
        if promotion_advantage_alignment_valid else 0.0
    )
    promotion_advantage_correlation = 0.0
    if (
        promotion_advantage_alignment_valid
        and promotion_advantage_targets.size > 1
        and float(np.std(promotion_advantage_targets)) > 1e-12
        and float(np.std(promotion_advantage_predictions)) > 1e-12
    ):
        promotion_advantage_correlation = float(np.corrcoef(
            promotion_advantage_predictions,
            promotion_advantage_targets,
        )[0, 1])
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    reg = LeakageRegularizer(upper_hf_window=6, lower_lf_window=24)
    leak = reg.compute(np.asarray(targets), np.asarray(lower_effects))
    raw_leak = reg.compute(np.asarray(targets), np.asarray(raw_lower_effects))
    tracking_raw_leak = reg.compute(
        np.asarray(targets), np.asarray(raw_tracking_lower_effects)
    )
    hf_overlay_leak = reg.compute(
        np.zeros_like(np.asarray(targets)), np.asarray(hf_overlay_effects)
    )
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
        "promotion_gate_transition_count": int(
            trajectory.promotion.size if trajectory.promotion is not None else 0
        ),
        "promotion_gate_owned_primitive_steps": int(
            np.sum(trajectory.promotion.duration)
            if trajectory.promotion is not None else 0
        ),
        "promotion_gate_mean_duration": float(
            np.mean(trajectory.promotion.duration)
            if trajectory.promotion is not None else 0.0
        ),
        "promotion_gate_probability_mean": float(
            np.mean(learned_gate_probabilities)
        ) if learned_gate_probabilities else 0.0,
        "promotion_gate_probability_std": float(
            np.std(learned_gate_probabilities)
        ) if learned_gate_probabilities else 0.0,
        "promotion_gate_probability_min": float(
            np.min(learned_gate_probabilities)
        ) if learned_gate_probabilities else 0.0,
        "promotion_gate_probability_max": float(
            np.max(learned_gate_probabilities)
        ) if learned_gate_probabilities else 0.0,
        "promotion_gate_action_rate": float(
            np.mean(learned_gate_actions)
        ) if learned_gate_actions else 0.0,
        "promotion_deterministic_mode": promotion_deterministic_mode,
        "promotion_advantage_threshold": float(
            promotion_advantage_threshold
        ),
        "promotion_advantage_target_threshold": float(
            resolved_promotion_advantage_target_threshold
        ),
        "promotion_gate_advantage_head_enabled": float(
            promotion_advantage_head_enabled
        ),
        "promotion_gate_advantage_alignment_valid": float(
            promotion_advantage_alignment_valid
        ),
        "promotion_gate_advantage_prediction_count": int(
            promotion_advantage_predictions.size
        ),
        "promotion_gate_advantage_target_count": int(
            promotion_advantage_targets.size
        ),
        "promotion_gate_advantage_mean": float(
            np.mean(promotion_advantage_predictions)
        ) if promotion_advantage_predictions.size else 0.0,
        "promotion_gate_advantage_std": float(
            np.std(promotion_advantage_predictions)
        ) if promotion_advantage_predictions.size else 0.0,
        "promotion_gate_advantage_min": float(
            np.min(promotion_advantage_predictions)
        ) if promotion_advantage_predictions.size else 0.0,
        "promotion_gate_advantage_max": float(
            np.max(promotion_advantage_predictions)
        ) if promotion_advantage_predictions.size else 0.0,
        "promotion_gate_advantage_target_mean": float(
            np.mean(promotion_advantage_targets)
        ) if promotion_advantage_targets.size else 0.0,
        "promotion_gate_advantage_target_std": float(
            np.std(promotion_advantage_targets)
        ) if promotion_advantage_targets.size else 0.0,
        "promotion_gate_advantage_target_mae": promotion_advantage_mae,
        "promotion_gate_advantage_sign_accuracy": (
            promotion_advantage_sign_accuracy
        ),
        "promotion_gate_advantage_target_correlation": (
            promotion_advantage_correlation
        ),
        "promotion_replan_cost_total": float(learned_replan_cost_total),
        "promotion_scheduled_boundary_close_count": int(
            promotion_scheduled_boundary_closes
        ),
        "promotion_cooldown_block_count": int(promotion_cooldown_blocks),
        "promotion_gate_interval_block_count": int(
            promotion_gate_interval_blocks
        ),
        "promotion_absorbed_norm_mean": float(
            np.mean(learned_promotion_absorbed_norm)
        ) if learned_promotion_absorbed_norm else 0.0,
        "promotion_absorbed_norm_total": float(
            np.sum(learned_promotion_absorbed_norm)
        ) if learned_promotion_absorbed_norm else 0.0,
        "promotion_credit_mode": str(promotion_credit_mode),
        "promotion_credit_scale": float(resolved_promotion_credit_scale),
        "promotion_counterfactual_symmetric": float(
            str(promotion_credit_mode) == "paired_plan_advantage"
        ),
        "promotion_credit_total": float(
            np.sum(promotion_credits)
        ) if promotion_credits else 0.0,
        "promotion_credit_mean": float(
            np.mean(promotion_credits)
        ) if promotion_credits else 0.0,
        "promotion_replan_action_credit_mean": float(np.mean([
            credit for credit, action in zip(
                promotion_credits, promotion_credit_actions
            ) if action >= 0.5
        ])) if any(action >= 0.5 for action in promotion_credit_actions) else 0.0,
        "promotion_continue_action_credit_mean": float(np.mean([
            credit for credit, action in zip(
                promotion_credits, promotion_credit_actions
            ) if action < 0.5
        ])) if any(action < 0.5 for action in promotion_credit_actions) else 0.0,
        "promotion_plan_advantage_total": float(
            np.sum(promotion_plan_advantages)
        ) if (
            promotion_plan_advantages
            and str(promotion_credit_mode) in {
                "incremental_plan_advantage",
                "paired_plan_advantage",
            }
        ) else 0.0,
        "promotion_plan_advantage_mean": float(
            np.mean(promotion_plan_advantages)
        ) if promotion_plan_advantages else 0.0,
        "hf_order_l1_mean": float(np.mean(hf_order_l1)) if hf_order_l1 else 0.0,
        "hf_overlay_position_l1_mean": float(
            np.mean(hf_overlay_position_l1)
        ) if hf_overlay_position_l1 else 0.0,
        "hf_overlay_return_total": float(np.sum(hf_overlay_returns)) if hf_overlay_returns else 0.0,
        "hf_overlay_incremental_cost_total": float(
            np.sum(hf_overlay_incremental_costs)
        ) if hf_overlay_incremental_costs else 0.0,
        "hf_overlay_task_effect_total": float(
            np.sum(hf_overlay_task_effects)
        ) if hf_overlay_task_effects else 0.0,
        "hard_hf_budget_projection": float(bool(hard_hf_budget_projection)),
        "hf_overlay_projection_rate": float(
            np.mean(hf_overlay_projection_events)
        ) if hf_overlay_projection_events else 0.0,
        "hf_overlay_projection_scale_mean": float(
            np.mean(hf_overlay_projection_scales)
        ) if hf_overlay_projection_scales else 1.0,
        "hf_overlay_rms_before_projection_max": float(
            np.max(hf_overlay_rms_before_projection)
        ) if hf_overlay_rms_before_projection else 0.0,
        "hf_overlay_rms_after_projection_max": float(
            np.max(hf_overlay_rms_after_projection)
        ) if hf_overlay_rms_after_projection else 0.0,
        "hf_tactical_transition_count": int(
            trajectory.hf.size if trajectory.hf is not None else 0
        ),
        "lower_hf_action_sensitivity": float(
            np.mean(lower_hf_action_sensitivities)
        ) if lower_hf_action_sensitivities else 0.0,
        "lower_hf_overlay_sensitivity": float(
            np.mean(lower_hf_overlay_sensitivities)
        ) if lower_hf_overlay_sensitivities else 0.0,
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
        "TrackingRawLowerLFDrift": float(
            tracking_raw_leak["LowerLFDrift"]
        ),
        "TrackingRawLowerLFDriftAbs": float(
            tracking_raw_leak["LowerLFDriftAbs"]
        ),
        "HFOverlayLFDrift": float(hf_overlay_leak["LowerLFDrift"]),
        "HFOverlayLFDriftAbs": float(hf_overlay_leak["LowerLFDriftAbs"]),
        "leakage_cost_mode": str(leakage_cost_mode),
        "lower_lf_budget_rms": float(lower_lf_budget_rms),
        "hf_lf_budget_rms": float(hf_lf_budget_rms),
        "tracking_leakage_budget_ratio_mean": float(
            np.mean(tracking_leakage_budget_ratios)
        ) if tracking_leakage_budget_ratios else 0.0,
        "tracking_leakage_budget_ratio_max": float(
            np.max(tracking_leakage_budget_ratios)
        ) if tracking_leakage_budget_ratios else 0.0,
        "hf_leakage_budget_ratio_mean": float(
            np.mean(hf_leakage_budget_ratios)
        ) if hf_leakage_budget_ratios else 0.0,
        "hf_leakage_budget_ratio_max": float(
            np.max(hf_leakage_budget_ratios)
        ) if hf_leakage_budget_ratios else 0.0,
        "leakage_budget_violation_rate": float(
            np.mean(leakage_budget_violations)
        ) if leakage_budget_violations else 0.0,
        "hf_predictability_enabled": float(bool(include_hf_predictability)),
        "hf_predictability_mean": float(np.mean(
            np.asarray(hf_predictability_values, dtype=np.float64)
        )) if hf_predictability_values else 0.0,
        "FocusScore": float(diag["FocusScore"]),
        "upper_low_mi": float(diag.get("upper_low_mi", 0.0)),
        "upper_high_mi": float(diag.get("upper_high_mi", 0.0)),
        "lower_high_mi": float(diag.get("lower_high_mi", 0.0)),
        "lower_low_mi": float(diag.get("lower_low_mi", 0.0)),
        "upper_credit_mean": float(np.mean(upper_credits)) if upper_credits else 0.0,
        "lower_credit_mean": float(np.mean(lower_credits)) if lower_credits else 0.0,
        "hf_tactical_credit_mean": float(
            np.mean(hf_tactical_credits)
        ) if hf_tactical_credits else 0.0,
        "total_lower_credit_mean": float(
            np.mean(np.asarray(lower_credits) + np.asarray(hf_tactical_credits))
        ) if lower_credits else 0.0,
        "upper_task_credit_mean": float(np.mean(upper_task_credits)) if upper_task_credits else 0.0,
        "lower_task_credit_mean": float(np.mean(lower_task_credits)) if lower_task_credits else 0.0,
        "hf_tactical_task_credit_mean": float(
            np.mean(hf_tactical_task_credits)
        ) if hf_tactical_task_credits else 0.0,
        "plan_return_mean": float(np.mean(plan_returns)) if plan_returns else 0.0,
        "execution_deviation_return_mean": float(
            np.mean(execution_deviation_returns)
        ) if execution_deviation_returns else 0.0,
        "upper_leakage_cost_mean": float(np.mean(upper_leakage_costs)) if upper_leakage_costs else 0.0,
        "lower_leakage_cost_mean": float(np.mean(lower_leakage_costs)) if lower_leakage_costs else 0.0,
        "hf_leakage_cost_mean": float(
            np.mean(hf_leakage_costs)
        ) if hf_leakage_costs else 0.0,
        "task_credit_reconstruction_max_abs_error": float(
            np.max(np.abs(finite_reconstruction_errors))
        ) if finite_reconstruction_errors.size else float("nan"),
        "plan_smoothness": float(np.mean(plan_smoothness)) if plan_smoothness else 0.0,
        "plan_coeff_abs": float(np.mean(plan_coeff_abs)) if plan_coeff_abs else 0.0,
        "upper_plan_reference_mode": str(upper_plan_reference_mode),
        "upper_plan_reference_target_abs": float(
            np.mean(plan_reference_target_abs)
        ) if plan_reference_target_abs else 0.0,
        "upper_plan_reference_coeff_abs": float(
            np.mean(plan_reference_coeff_abs)
        ) if plan_reference_coeff_abs else 0.0,
        "upper_plan_residual_coeff_abs": float(
            np.mean(plan_residual_coeff_abs)
        ) if plan_residual_coeff_abs else 0.0,
        "upper_residual_action_scale": float(upper_residual_action_scale),
        "plan_target_step_change_mean": float(
            np.mean(np.abs(target_steps))
        ) if target_steps.size else 0.0,
        "lower_lf_effect_filter_window": int(lower_lf_effect_filter_window),
        "lower_lf_effect_filter_gain": float(lower_lf_effect_filter_gain),
        "lower_lf_raw_recenter_gain": float(lower_lf_raw_recenter_gain),
        "raw_recenter_boost_mean": float(np.mean(raw_recenter_boosts)) if raw_recenter_boosts else 0.0,
        "protocol_valid": 1.0,
        "full_method_implementation_version": (
            full_method_implementation_version(method_contract)
        ),
        "execution_timeline_contract": str(execution_timeline_contract),
        "method_contract": str(method_contract),
        "mark_to_market_timing": str(mark_to_market_timing),
        "volume_impact_bps": float(volume_impact_bps),
        "executed_plan_curve": float(bool(execute_plan_curve)),
        "additive_frequency_credit": float(bool(use_additive_frequency_credit)),
        "raw_lower_effect_constraint": float(bool(constrain_raw_lower_effect)),
        "learned_promotion_gate": float(bool(learned_promotion_gate)),
        "heuristic_promotion_disabled": float(not bool(heuristic_promotion_gate)),
        "heuristic_promotion_gate": float(bool(heuristic_promotion_gate)),
        "promotion_replan_cost": float(promotion_replan_cost),
        "promotion_deterministic_threshold": float(
            promotion_deterministic_threshold
        ),
        "promotion_adapt_gain": float(promotion_adapt_gain),
        "promotion_cooldown_steps": int(promotion_cooldown_steps),
        "promotion_gate_interval_steps": int(promotion_gate_interval_steps),
        "hf_lower_overlay_enabled": float(bool(enable_hf_lower)),
        "hf_tactical_stream_enabled": float(bool(use_separate_hf)),
        "exact_three_way_credit": float(bool(use_separate_hf)),
        "training_support_ood_excluded": float(
            str(scenario) == SUPPORT_MIXTURE_SCENARIO
        ),
        "lower_hf_order_scale": float(lower_hf_order_scale),
        "lower_observation_intervention": lower_observation_intervention,
        "hf_action_sensitivity_computed": float(
            bool(compute_hf_action_sensitivity)
        ),
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
    return (trajectory if sample or bool(return_trajectory) else None), row


def evaluate_hf_lower_intervention(
    model: FrequencySeparatedActorCriticPPO,
    *,
    eval_seeds: list[int],
    rollout_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Paired deterministic evaluation with only lower MF/HF inputs removed.

    Exogenous market paths are identical. Downstream positions and therefore
    later upper/gate states may diverge; outcome deltas are the total system
    effect. ``lower_hf_action_sensitivity`` isolates the immediate policy
    response on the factual control states.
    """

    forbidden = {
        "seed",
        "sample",
        "lower_observation_intervention",
        "compute_hf_action_sensitivity",
    }
    overlap = forbidden.intersection(rollout_kwargs)
    if overlap:
        raise ValueError(
            "rollout_kwargs contains evaluator-owned keys: "
            f"{sorted(overlap)}"
        )
    if not bool(rollout_kwargs.get("enable_hf_lower", False)):
        raise ValueError("HF-lower intervention requires enable_hf_lower=True")
    if str(rollout_kwargs.get("policy_mode", "freq_hrl")) != "freq_hrl":
        raise ValueError("HF-lower intervention requires policy_mode='freq_hrl'")

    rows: list[dict[str, Any]] = []
    for seed in validate_unique_seeds(eval_seeds, role="hf_intervention_eval_seeds"):
        _, control = smdp_rollout(
            model,
            seed=int(seed),
            sample=False,
            lower_observation_intervention="none",
            compute_hf_action_sensitivity=True,
            **rollout_kwargs,
        )
        _, ablated = smdp_rollout(
            model,
            seed=int(seed),
            sample=False,
            lower_observation_intervention="zero_residual_frequency",
            compute_hf_action_sensitivity=False,
            **rollout_kwargs,
        )
        rows.append({
            "seed": int(seed),
            "scenario": str(control["scenario"]),
            "control_intervention": "none",
            "ablated_intervention": "zero_residual_frequency",
            "control_total_return": float(control["total_return"]),
            "ablated_total_return": float(ablated["total_return"]),
            "total_return_delta": float(
                control["total_return"] - ablated["total_return"]
            ),
            "control_sharpe": float(control["sharpe"]),
            "ablated_sharpe": float(ablated["sharpe"]),
            "sharpe_delta": float(control["sharpe"] - ablated["sharpe"]),
            "control_max_drawdown": float(control["max_drawdown"]),
            "ablated_max_drawdown": float(ablated["max_drawdown"]),
            "max_drawdown_reduction": float(
                ablated["max_drawdown"] - control["max_drawdown"]
            ),
            "control_turnover": float(control["turnover"]),
            "ablated_turnover": float(ablated["turnover"]),
            "turnover_delta": float(control["turnover"] - ablated["turnover"]),
            "control_hf_overlay_task_effect": float(
                control["hf_overlay_task_effect_total"]
            ),
            "ablated_hf_overlay_task_effect": float(
                ablated["hf_overlay_task_effect_total"]
            ),
            "lower_hf_action_sensitivity": float(
                control["lower_hf_action_sensitivity"]
            ),
            "lower_hf_overlay_sensitivity": float(
                control["lower_hf_overlay_sensitivity"]
            ),
            "control_upper_decision_count": int(
                control["upper_decision_count"]
            ),
            "ablated_upper_decision_count": int(
                ablated["upper_decision_count"]
            ),
            "control_promotion_replan_count": int(
                control["promotion_replan_count"]
            ),
            "ablated_promotion_replan_count": int(
                ablated["promotion_replan_count"]
            ),
            "paired_exogenous_path_identity": bool(
                int(control["seed"]) == int(ablated["seed"])
                and str(control["scenario"]) == str(ablated["scenario"])
            ),
        })
    return rows


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
        "promotion_gate_transition_count",
        "promotion_gate_owned_primitive_steps",
        "promotion_gate_mean_duration",
        "promotion_gate_probability_mean",
        "promotion_gate_probability_std",
        "promotion_gate_probability_min",
        "promotion_gate_probability_max",
        "promotion_gate_action_rate",
        "promotion_replan_cost_total",
        "promotion_scheduled_boundary_close_count",
        "promotion_gate_interval_block_count",
        "promotion_absorbed_norm_mean",
        "promotion_absorbed_norm_total",
        "promotion_credit_total",
        "promotion_credit_mean",
        "promotion_replan_action_credit_mean",
        "promotion_continue_action_credit_mean",
        "promotion_plan_advantage_total",
        "promotion_plan_advantage_mean",
        "hf_order_l1_mean",
        "hf_overlay_position_l1_mean",
        "hf_overlay_return_total",
        "hf_overlay_incremental_cost_total",
        "hf_overlay_task_effect_total",
        "hf_tactical_transition_count",
        "lower_hf_action_sensitivity",
        "lower_hf_overlay_sensitivity",
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
        "TrackingRawLowerLFDrift",
        "TrackingRawLowerLFDriftAbs",
        "HFOverlayLFDrift",
        "HFOverlayLFDriftAbs",
        "tracking_leakage_budget_ratio_mean",
        "tracking_leakage_budget_ratio_max",
        "hf_leakage_budget_ratio_mean",
        "hf_leakage_budget_ratio_max",
        "leakage_budget_violation_rate",
        "hf_predictability_enabled",
        "hf_predictability_mean",
        "FocusScore",
        "upper_low_mi",
        "upper_high_mi",
        "lower_high_mi",
        "lower_low_mi",
        "upper_credit_mean",
        "lower_credit_mean",
        "hf_tactical_credit_mean",
        "total_lower_credit_mean",
        "upper_task_credit_mean",
        "lower_task_credit_mean",
        "hf_tactical_task_credit_mean",
        "plan_return_mean",
        "execution_deviation_return_mean",
        "upper_leakage_cost_mean",
        "lower_leakage_cost_mean",
        "hf_leakage_cost_mean",
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
        "learned_promotion_gate",
        "heuristic_promotion_disabled",
        "heuristic_promotion_gate",
        "promotion_replan_cost",
        "hf_lower_overlay_enabled",
        "hf_tactical_stream_enabled",
        "exact_three_way_credit",
        "training_support_ood_excluded",
        "lower_hf_order_scale",
        "hf_action_sensitivity_computed",
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
    upper_learning_rate: float | None = None,
    lower_learning_rate: float | None = None,
    hf_learning_rate: float | None = None,
    promotion_learning_rate: float | None = None,
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
    capacity_reference_method_contract: str | None = None,
    volume_impact_bps: float = 0.0,
    plan_smoothness_weight: float = 0.0,
    promotion_replan_cost: float = 0.0,
    promotion_init_logit: float = -2.0,
    promotion_entropy_coef: float | None = None,
    promotion_rate_budget: float = 1.0,
    promotion_rate_coef: float = 0.0,
    promotion_counterfactual_coef: float | None = None,
    promotion_advantage_learning_rate: float | None = None,
    promotion_advantage_coef: float | None = None,
    promotion_advantage_huber_delta: float = 0.1,
    lower_hf_order_scale: float = 0.025,
    promotion_credit_mode: str = "auto",
    promotion_credit_scale: float | None = None,
    leakage_cost_mode: str = "auto",
    lower_lf_budget_rms: float = 0.0025,
    hf_lf_budget_rms: float = 0.00025,
    include_hf_predictability: bool | None = None,
    upper_plan_reference_mode: str = "none",
    upper_plan_reference_gain: float = 1.0,
    upper_plan_reference_forecast_blend: float = 0.0,
    hard_hf_budget_projection: bool = False,
    promotion_deterministic_threshold: float = 0.5,
    promotion_adapt_gain: float = 0.05,
    promotion_cooldown_steps: int = 0,
    promotion_gate_interval_steps: int = 1,
    promotion_deterministic_mode: str = "auto",
    promotion_advantage_threshold: float = 0.0,
    promotion_advantage_target_threshold: float | None = None,
    upper_residual_action_scale: float = 1.0,
    training_scenarios: Sequence[str] | None = None,
    checkpoint_smoothing_window: int = 1,
    checkpoint_min_delta: float = 0.0,
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
    upper_plan_reference_mode = str(upper_plan_reference_mode)
    if upper_plan_reference_mode not in UPPER_PLAN_REFERENCE_MODES:
        raise ValueError(
            "unknown upper_plan_reference_mode: "
            f"{upper_plan_reference_mode}"
        )
    if upper_plan_reference_mode != "none" and (
        policy_mode != "freq_hrl"
        or not method_flags["execute_plan_curve"]
    ):
        raise ValueError(
            "a causal upper plan reference requires executable Freq-HRL plans"
        )
    if hard_hf_budget_projection and not method_flags["lower_hf_overlay"]:
        raise ValueError("hard HF budget projection requires an active HF lower")
    for name, value in (
        ("upper_plan_reference_gain", upper_plan_reference_gain),
        ("promotion_adapt_gain", promotion_adapt_gain),
        ("upper_residual_action_scale", upper_residual_action_scale),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if (
        not np.isfinite(float(upper_plan_reference_forecast_blend))
        or not 0.0 <= float(upper_plan_reference_forecast_blend) <= 1.0
    ):
        raise ValueError(
            "upper_plan_reference_forecast_blend must be finite and in [0, 1]"
        )
    if (
        not np.isfinite(float(promotion_deterministic_threshold))
        or not 0.0 < float(promotion_deterministic_threshold) < 1.0
    ):
        raise ValueError(
            "promotion_deterministic_threshold must be finite and in (0, 1)"
        )
    if int(promotion_cooldown_steps) < 0:
        raise ValueError("promotion_cooldown_steps must be non-negative")
    if int(promotion_gate_interval_steps) < 1:
        raise ValueError("promotion_gate_interval_steps must be positive")
    promotion_deterministic_mode = str(promotion_deterministic_mode)
    if promotion_deterministic_mode not in {
        "auto", "actor_probability", "counterfactual_advantage",
    }:
        raise ValueError("unknown deterministic promotion mode")
    if not np.isfinite(float(promotion_advantage_threshold)):
        raise ValueError("promotion_advantage_threshold must be finite")
    resolved_promotion_advantage_target_threshold = (
        float(promotion_advantage_threshold)
        if promotion_advantage_target_threshold is None
        else float(promotion_advantage_target_threshold)
    )
    if not np.isfinite(resolved_promotion_advantage_target_threshold):
        raise ValueError("promotion_advantage_target_threshold must be finite")
    resolved_promotion_entropy_coef = (
        0.001
        if promotion_entropy_coef is None else float(promotion_entropy_coef)
    )
    if (
        not np.isfinite(resolved_promotion_entropy_coef)
        or resolved_promotion_entropy_coef < 0.0
    ):
        raise ValueError("promotion_entropy_coef must be finite and non-negative")
    if (
        not np.isfinite(float(promotion_rate_budget))
        or not 0.0 <= float(promotion_rate_budget) <= 1.0
    ):
        raise ValueError("promotion_rate_budget must be finite and in [0, 1]")
    if (
        not np.isfinite(float(promotion_rate_coef))
        or float(promotion_rate_coef) < 0.0
    ):
        raise ValueError("promotion_rate_coef must be finite and non-negative")
    resolved_promotion_credit_scale = (
        float(reward_scale)
        if promotion_credit_scale is None else float(promotion_credit_scale)
    )
    if (
        not np.isfinite(resolved_promotion_credit_scale)
        or resolved_promotion_credit_scale <= 0.0
    ):
        raise ValueError("promotion_credit_scale must be positive and finite")
    independent_training_scenarios = (
        None
        if training_scenarios is None
        else tuple(str(value) for value in training_scenarios)
    )
    if independent_training_scenarios is not None:
        if not independent_training_scenarios:
            raise ValueError("training_scenarios cannot be empty")
        if len(set(independent_training_scenarios)) != len(
            independent_training_scenarios
        ):
            raise ValueError("training_scenarios must be unique")
        unknown_training_scenarios = sorted(
            set(independent_training_scenarios) - set(SCENARIOS)
        )
        if unknown_training_scenarios:
            raise ValueError(
                "unknown independent training scenarios: "
                f"{unknown_training_scenarios}"
            )
    episode_multiplier = int(
        len(independent_training_scenarios)
        if independent_training_scenarios is not None else 1
    )
    fixed_ablation_architecture = method_contract in (
        V6_METHOD_CONTRACTS | V7_METHOD_CONTRACTS
    )
    fixed_architecture_reference = (
        "full_freq_hrl_v7"
        if method_contract in V7_METHOD_CONTRACTS else
        "full_freq_hrl_v6"
        if method_contract in V6_METHOD_CONTRACTS else
        method_contract
    )
    capacity_reference_method_contract = str(
        (
            fixed_architecture_reference
            if fixed_ablation_architecture else method_contract
        )
        if capacity_reference_method_contract is None
        else capacity_reference_method_contract
    )
    capacity_reference_flags = resolve_method_contract(
        capacity_reference_method_contract
    )
    if (
        fixed_ablation_architecture
        and capacity_reference_method_contract != fixed_architecture_reference
    ):
        raise ValueError(
            f"{method_contract} requires {fixed_architecture_reference} "
            "as the fixed architecture reference"
        )
    resolved_promotion_credit_mode = str(promotion_credit_mode)
    if resolved_promotion_credit_mode == "auto":
        if method_contract in V7_METHOD_CONTRACTS:
            resolved_promotion_credit_mode = "paired_plan_advantage"
        elif method_flags["promotion_plan_advantage_credit"]:
            resolved_promotion_credit_mode = "incremental_plan_advantage"
        else:
            resolved_promotion_credit_mode = "task_return"
    if resolved_promotion_credit_mode not in {
        "task_return",
        "incremental_plan_advantage",
        "paired_plan_advantage",
    }:
        raise ValueError("unknown promotion_credit_mode")
    if promotion_counterfactual_coef is None:
        resolved_promotion_counterfactual_coef = float(
            resolved_promotion_credit_mode == "paired_plan_advantage"
        )
    else:
        resolved_promotion_counterfactual_coef = float(
            promotion_counterfactual_coef
        )
    if (
        not np.isfinite(resolved_promotion_counterfactual_coef)
        or resolved_promotion_counterfactual_coef < 0.0
    ):
        raise ValueError(
            "promotion_counterfactual_coef must be finite and non-negative"
        )
    if (
        resolved_promotion_counterfactual_coef > 0.0
        and resolved_promotion_credit_mode != "paired_plan_advantage"
    ):
        raise ValueError(
            "promotion_counterfactual_coef requires paired_plan_advantage credit"
        )
    resolved_promotion_advantage_coef = (
        float(method_contract in V7_METHOD_CONTRACTS)
        if promotion_advantage_coef is None
        else float(promotion_advantage_coef)
    )
    if (
        not np.isfinite(resolved_promotion_advantage_coef)
        or resolved_promotion_advantage_coef < 0.0
    ):
        raise ValueError("promotion_advantage_coef must be finite and non-negative")
    if (
        not np.isfinite(float(promotion_advantage_huber_delta))
        or float(promotion_advantage_huber_delta) <= 0.0
    ):
        raise ValueError(
            "promotion_advantage_huber_delta must be positive and finite"
        )
    if (
        resolved_promotion_advantage_coef > 0.0
        and resolved_promotion_credit_mode != "paired_plan_advantage"
    ):
        raise ValueError(
            "promotion advantage learning requires paired_plan_advantage credit"
        )
    if promotion_deterministic_mode == "auto":
        promotion_deterministic_mode = (
            "counterfactual_advantage"
            if resolved_promotion_advantage_coef > 0.0
            else "actor_probability"
        )
    if (
        promotion_deterministic_mode == "counterfactual_advantage"
        and resolved_promotion_advantage_coef <= 0.0
    ):
        raise ValueError(
            "counterfactual-advantage decisions require a learned advantage head"
        )
    resolved_leakage_cost_mode = str(leakage_cost_mode)
    if resolved_leakage_cost_mode == "auto":
        resolved_leakage_cost_mode = (
            "fixed_rms_budget"
            if method_flags["fixed_rms_leakage_budget"]
            else "spectral_ratio"
        )
    if resolved_leakage_cost_mode not in {
        "spectral_ratio",
        "fixed_rms_budget",
    }:
        raise ValueError("unknown leakage_cost_mode")
    if (
        hard_hf_budget_projection
        and resolved_leakage_cost_mode != "fixed_rms_budget"
    ):
        raise ValueError(
            "hard HF budget projection requires fixed_rms_budget leakage accounting"
        )
    resolved_include_hf_predictability = (
        bool(method_flags["hf_predictability_summary"])
        if include_hf_predictability is None
        else bool(include_hf_predictability)
    )
    expected_fixed_promotion_credit = (
        "paired_plan_advantage"
        if method_contract in V7_METHOD_CONTRACTS
        else "incremental_plan_advantage"
    )
    if fixed_ablation_architecture and (
        resolved_promotion_credit_mode != expected_fixed_promotion_credit
        or resolved_leakage_cost_mode != "fixed_rms_budget"
        or not resolved_include_hf_predictability
    ):
        raise ValueError(
            "fixed-architecture contracts require their versioned promotion credit, "
            "fixed RMS leakage budgets, "
            "and the causal HF predictability summary"
        )
    if method_contract in V7_METHOD_CONTRACTS:
        expected_reference_mode = (
            "causal_lf"
            if method_flags["causal_lf_plan_reference"] else "none"
        )
        expected_hard_projection = bool(
            method_flags["hard_hf_budget_projection"]
        )
        expected_residual_scale = (
            1.0 if method_flags["upper_residual_control"] else 0.0
        )
        if upper_plan_reference_mode != expected_reference_mode:
            raise ValueError(
                f"{method_contract} requires upper_plan_reference_mode="
                f"{expected_reference_mode!r}"
            )
        if bool(hard_hf_budget_projection) != expected_hard_projection:
            raise ValueError(
                f"{method_contract} requires hard_hf_budget_projection="
                f"{expected_hard_projection}"
            )
        if not np.isclose(
            float(upper_residual_action_scale), expected_residual_scale
        ):
            raise ValueError(
                f"{method_contract} requires upper_residual_action_scale="
                f"{expected_residual_scale}"
            )
    mark_to_market_timing = (
        "post_trade"
        if execution_timeline_contract == "causal_post_trade_v3"
        else "pre_trade"
    )
    if not np.isfinite(float(volume_impact_bps)) or float(volume_impact_bps) < 0.0:
        raise ValueError("volume_impact_bps must be finite and non-negative")
    for name, value in (
        ("plan_smoothness_weight", plan_smoothness_weight),
        ("promotion_replan_cost", promotion_replan_cost),
        ("lower_hf_order_scale", lower_hf_order_scale),
    ):
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(float(promotion_init_logit)):
        raise ValueError("promotion_init_logit must be finite")
    if resolved_leakage_cost_mode == "fixed_rms_budget":
        for name, value in (
            ("lower_lf_budget_rms", lower_lf_budget_rms),
            ("hf_lf_budget_rms", hf_lf_budget_rms),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
    resolved_learning_rates = {
        "upper": float(
            learning_rate if upper_learning_rate is None else upper_learning_rate
        ),
        "lower": float(
            learning_rate if lower_learning_rate is None else lower_learning_rate
        ),
        "hf": float(learning_rate if hf_learning_rate is None else hf_learning_rate),
        "promotion": float(
            learning_rate
            if promotion_learning_rate is None else promotion_learning_rate
        ),
        "promotion_advantage": float(
            (
                learning_rate
                if promotion_learning_rate is None
                else promotion_learning_rate
            )
            if promotion_advantage_learning_rate is None
            else promotion_advantage_learning_rate
        ),
    }
    for level, value in resolved_learning_rates.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{level}_learning_rate must be positive and finite")
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
    if capacity_reference_method_contract != "routing_core_v2":
        if execution_timeline_contract != "causal_post_trade_v3":
            raise ValueError(
                "non-routing capacity reference requires causal_post_trade_v3"
            )
        if int(plan_basis_dim) < 2:
            raise ValueError(
                "non-routing capacity reference requires plan_basis_dim >= 2"
            )
    frequency_method_contracts = {
        "full_freq_hrl_v3",
        "full_freq_hrl_v4",
        "ablate_promotion_v4",
        "ablate_hf_lower_v4",
        "ablate_leakage_v4",
        "full_freq_hrl_v5",
        "ablate_promotion_v5",
        "ablate_hf_lower_v5",
        "ablate_leakage_v5",
        *V6_METHOD_CONTRACTS,
        *V7_METHOD_CONTRACTS,
    }
    if method_contract in frequency_method_contracts:
        if policy_mode != "freq_hrl":
            raise ValueError(
                f"{method_contract} requires policy_mode='freq_hrl'"
            )
        if method_contract not in {
            "ablate_leakage_v4",
            "ablate_leakage_v5",
            "ablate_leakage_v6",
            "ablate_leakage_v7",
        } and not (
            float(leakage_scale) > 0.0
            or float(lower_lf_constraint_coef) > 0.0
            or float(lower_lf_dual_lr) > 0.0
        ):
            raise ValueError(
                f"{method_contract} requires an active raw leakage penalty or constraint"
            )
    if method_contract in {
        "ablate_leakage_v4",
        "ablate_leakage_v5",
        "ablate_leakage_v6",
        "ablate_leakage_v7",
    } and any(
        float(value) != 0.0
        for value in (
            leakage_scale,
            lower_lf_constraint_coef,
            lower_lf_dual_lr,
            lower_lf_objective_weight,
        )
    ):
        raise ValueError(
            f"{method_contract} requires all leakage training weights to be zero"
        )
    if not method_flags["learned_promotion_gate"] and float(promotion_replan_cost) > 0.0:
        raise ValueError(
            "promotion_replan_cost requires a learned promotion contract"
        )
    if method_contract != "routing_core_v2" and use_handcrafted_frequency_prior:
        raise ValueError(
            "handcrafted frequency priors are forbidden for v3+ method contracts"
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
    architecture_flags = (
        capacity_reference_flags
        if fixed_ablation_architecture else method_flags
    )
    lower_state_dim = (
        (9 if resolved_include_hf_predictability else 8) * assets + 1
    )
    effective_lower_cost_target = (
        1.0
        if resolved_leakage_cost_mode == "fixed_rms_budget"
        else float(lower_lf_constraint_target)
    )
    policy_smdp_config = SMDPPPOConfig(
        upper_state_dim=6 * assets + 5,
        lower_state_dim=lower_state_dim,
        upper_action_dim=plan_mapper.action_dim if plan_mapper is not None else assets,
        lower_action_dim=(
            2 * assets
            if architecture_flags["lower_hf_overlay"]
            and not architecture_flags["separate_hf_tactical"]
            else assets
        ),
        hf_state_dim=(
            lower_state_dim
            if architecture_flags["lower_hf_overlay"]
            and architecture_flags["separate_hf_tactical"]
            else 0
        ),
        hf_action_dim=(
            assets
            if architecture_flags["lower_hf_overlay"]
            and architecture_flags["separate_hf_tactical"]
            else 0
        ),
        promotion_state_dim=(
            promotion_gate_state_dim(assets)
            if architecture_flags["learned_promotion_gate"] else 0
        ),
        hidden_dim=int(hidden_dim),
        upper_learning_rate=resolved_learning_rates["upper"],
        lower_learning_rate=resolved_learning_rates["lower"],
        hf_learning_rate=resolved_learning_rates["hf"],
        promotion_learning_rate=resolved_learning_rates["promotion"],
        epochs=int(ppo_epochs),
        minibatch_size=int(minibatch_size),
        init_log_std=float(init_log_std),
        promotion_init_logit=float(promotion_init_logit),
        promotion_entropy_coef=float(resolved_promotion_entropy_coef),
        promotion_rate_budget=float(promotion_rate_budget),
        promotion_rate_coef=float(promotion_rate_coef),
        promotion_counterfactual_coef=float(
            resolved_promotion_counterfactual_coef
        ),
        promotion_advantage_learning_rate=resolved_learning_rates[
            "promotion_advantage"
        ],
        promotion_advantage_coef=resolved_promotion_advantage_coef,
        promotion_advantage_huber_delta=float(
            promotion_advantage_huber_delta
        ),
        lower_cost_target=float(effective_lower_cost_target),
        lower_dual_lr=float(lower_lf_dual_lr),
        lower_lambda_init=max(float(lower_lf_constraint_coef), 0.0),
        lower_max_lambda=20.0,
    )
    reference_smdp_config = replace(
        policy_smdp_config,
        lower_state_dim=(
            (9 if capacity_reference_flags["hf_predictability_summary"] else 8)
            * assets + 1
        ),
        lower_action_dim=(
            2 * assets
            if capacity_reference_flags["lower_hf_overlay"]
            and not capacity_reference_flags["separate_hf_tactical"]
            else assets
        ),
        hf_state_dim=(
            (9 if capacity_reference_flags["hf_predictability_summary"] else 8)
            * assets + 1
            if capacity_reference_flags["lower_hf_overlay"]
            and capacity_reference_flags["separate_hf_tactical"]
            else 0
        ),
        hf_action_dim=(
            assets
            if capacity_reference_flags["lower_hf_overlay"]
            and capacity_reference_flags["separate_hf_tactical"]
            else 0
        ),
        promotion_state_dim=(
            promotion_gate_state_dim(assets)
            if capacity_reference_flags["learned_promotion_gate"] else 0
        ),
        promotion_advantage_coef=(
            1.0
            if capacity_reference_method_contract == "full_freq_hrl_v7"
            else float(policy_smdp_config.promotion_advantage_coef)
        ),
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

        def joint_protocol_rollout(
            ppo_model: JointActorCriticPPO,
            rollout_seed: int,
            sample: bool,
        ) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
            def rollout_one(
                episode_seed: int,
                episode_scenario: str,
            ) -> tuple[JointTrajectoryBatch | None, dict[str, Any]]:
                return joint_flat_rollout(
                    ppo_model,
                    seed=int(episode_seed),
                    steps=steps,
                    assets=assets,
                    scenario=str(episode_scenario),
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
                )

            if independent_training_scenarios is None:
                return rollout_one(int(rollout_seed), str(scenario))
            return collect_independent_episode_rollouts(
                root_seed=int(rollout_seed),
                sample=bool(sample),
                scenarios=independent_training_scenarios,
                rollout_one=rollout_one,
                concat_batches=concat_joint_batches,
                scenario_label=str(scenario),
            )

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
            rollout_fn=joint_protocol_rollout,
            objective_fn=objective,
            summary_fn=summarize,
            policy=f"{policy_mode}_canonical_joint_action",
            domain="trading",
            checkpoint_smoothing_window=checkpoint_smoothing_window,
            checkpoint_min_delta=checkpoint_min_delta,
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
                "capacity_reference_method_contract": (
                    capacity_reference_method_contract
                ),
                "capacity_reference_implementation_version": (
                    full_method_implementation_version(
                        capacity_reference_method_contract
                    )
                ),
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
                    full_method_implementation_version(method_contract)
                ),
                "training_path_protocol": (
                    "independent_full_episode_support_batch_v1"
                    if independent_training_scenarios is not None else
                    "fresh_deterministic_path_per_root_and_iteration_v2"
                    if resample_training_paths else "fixed_path_reuse_legacy"
                ),
                "training_episode_scenarios": list(
                    independent_training_scenarios or ()
                ),
                "training_episode_count_per_root": int(episode_multiplier),
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
            * episode_multiplier
        )
        payload["environment_steps_validation"] = int(
            len(validation_seed_list) * int(steps) * (max(1, int(iterations)) + 1)
            * episode_multiplier
        )
        payload["environment_steps_eval"] = int(
            len(evaluation_seeds) * int(steps) * episode_multiplier
        )
        payload["unique_training_path_count"] = int(
            len(rollout_seed_roots)
            * (max(1, int(iterations)) if resample_training_paths else 1)
            * episode_multiplier
        )
        return payload, heldout_rows, joint_model

    smdp_capacity_count = smdp_parameter_count(policy_smdp_config)
    smdp_capacity_ratio = float(
        smdp_capacity_count / max(int(target_parameter_count), 1)
    )
    effective_hidden_dim = int(hidden_dim)
    smdp_config = policy_smdp_config
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
                upper_action_dim=policy_smdp_config.upper_action_dim,
                lower_action_dim=policy_smdp_config.lower_action_dim,
                requested_hidden_dim=int(hidden_dim),
                state_encoder=state_encoder,
                raw_history_window=raw_history_window,
                raw_feature_dim=assets,
                promotion_state_dim=policy_smdp_config.promotion_state_dim,
                hf_state_dim=policy_smdp_config.hf_state_dim,
                hf_action_dim=policy_smdp_config.hf_action_dim,
            )
        )
        smdp_config = replace(
            policy_smdp_config,
            upper_state_dim=upper_state_dim,
            lower_state_dim=lower_state_dim,
            hidden_dim=int(effective_hidden_dim),
            state_encoder=state_encoder,
            raw_history_window=int(raw_history_window),
            raw_feature_dim=int(assets),
        )
    elif smdp_capacity_count != int(target_parameter_count):
        effective_hidden_dim, smdp_capacity_count, smdp_capacity_ratio = (
            capacity_matched_smdp_hidden_dim(
                target_parameter_count=target_parameter_count,
                upper_state_dim=policy_smdp_config.upper_state_dim,
                lower_state_dim=policy_smdp_config.lower_state_dim,
                upper_action_dim=policy_smdp_config.upper_action_dim,
                lower_action_dim=policy_smdp_config.lower_action_dim,
                requested_hidden_dim=int(hidden_dim),
                state_encoder=policy_smdp_config.state_encoder,
                raw_history_window=policy_smdp_config.raw_history_window,
                raw_feature_dim=policy_smdp_config.raw_feature_dim,
                promotion_state_dim=policy_smdp_config.promotion_state_dim,
                hf_state_dim=policy_smdp_config.hf_state_dim,
                hf_action_dim=policy_smdp_config.hf_action_dim,
            )
        )
        smdp_config = replace(
            policy_smdp_config,
            hidden_dim=int(effective_hidden_dim),
        )
    smdp_model = FrequencySeparatedActorCriticPPO(smdp_config)
    if policy_mode == "freq_hrl" and bool(use_handcrafted_frequency_prior):
        initialize_smdp_frequency_prior(smdp_model, assets, plan_basis_dim=plan_basis_dim)
    actual_upper_period = int(upper_period)
    actual_min_upper_duration = int(min_upper_duration)
    observation_contract = {
        "freq_hrl": (
            "LF + forecast + uncertainty + compressed HF summaries",
            "current plan + local HF/MF residual context"
            + (
                " + causal next-bar HF predictability summary"
                if resolved_include_hf_predictability else ""
            ),
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
    if (
        method_flags["lower_hf_overlay"]
        and method_flags["separate_hf_tactical"]
    ):
        credit_contract = (
            "exact three-way task-reward conservation: upper plan, tracking "
            "execution, and counterfactual marginal HF tactical credit"
        )

    def smdp_protocol_rollout(
        ppo_model: FrequencySeparatedActorCriticPPO,
        rollout_seed: int,
        sample: bool,
    ) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
        def rollout_one(
            episode_seed: int,
            episode_scenario: str,
        ) -> tuple[HierarchicalTrajectoryBatch | None, dict[str, Any]]:
            return smdp_rollout(
                ppo_model,
                seed=int(episode_seed),
                steps=steps,
                assets=assets,
                scenario=str(episode_scenario),
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
                learned_promotion_gate=method_flags["learned_promotion_gate"],
                heuristic_promotion_gate=method_flags[
                    "heuristic_promotion_gate"
                ],
                promotion_replan_cost=float(promotion_replan_cost),
                enable_hf_lower=method_flags["lower_hf_overlay"],
                separate_hf_tactical=method_flags["separate_hf_tactical"],
                lower_hf_order_scale=float(lower_hf_order_scale),
                execution_timeline_contract=execution_timeline_contract,
                method_contract=method_contract,
                promotion_credit_mode=resolved_promotion_credit_mode,
                promotion_credit_scale=resolved_promotion_credit_scale,
                leakage_cost_mode=resolved_leakage_cost_mode,
                lower_lf_budget_rms=float(lower_lf_budget_rms),
                hf_lf_budget_rms=float(hf_lf_budget_rms),
                include_hf_predictability=resolved_include_hf_predictability,
                allow_inactive_mechanism_modules=fixed_ablation_architecture,
                upper_plan_reference_mode=upper_plan_reference_mode,
                upper_plan_reference_gain=float(upper_plan_reference_gain),
                upper_plan_reference_forecast_blend=float(
                    upper_plan_reference_forecast_blend
                ),
                hard_hf_budget_projection=bool(hard_hf_budget_projection),
                promotion_deterministic_threshold=float(
                    promotion_deterministic_threshold
                ),
                promotion_adapt_gain=float(promotion_adapt_gain),
                promotion_cooldown_steps=int(promotion_cooldown_steps),
                promotion_gate_interval_steps=int(
                    promotion_gate_interval_steps
                ),
                promotion_deterministic_mode=promotion_deterministic_mode,
                promotion_advantage_threshold=float(
                    promotion_advantage_threshold
                ),
                promotion_advantage_target_threshold=float(
                    resolved_promotion_advantage_target_threshold
                ),
                upper_residual_action_scale=float(
                    upper_residual_action_scale
                ),
            )

        if independent_training_scenarios is None:
            return rollout_one(int(rollout_seed), str(scenario))
        return collect_independent_episode_rollouts(
            root_seed=int(rollout_seed),
            sample=bool(sample),
            scenarios=independent_training_scenarios,
            rollout_one=rollout_one,
            concat_batches=concat_hierarchical_batches,
            scenario_label=str(scenario),
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
        rollout_fn=smdp_protocol_rollout,
        objective_fn=lambda row: objective(row) - max(
            float(lower_lf_objective_weight), 0.0
        ) * float(
            row["tracking_leakage_budget_ratio_mean"]
            if resolved_leakage_cost_mode == "fixed_rms_budget"
            else row["LowerLFDrift"]
        ),
        summary_fn=summarize,
        policy=f"{policy_mode}_capacity_matched_smdp_ppo",
        domain="trading",
        checkpoint_smoothing_window=checkpoint_smoothing_window,
        checkpoint_min_delta=checkpoint_min_delta,
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
            "promotion_replanning_enabled": bool(
                method_flags["learned_promotion_gate"]
                or method_flags["heuristic_promotion_gate"]
            ),
            "handcrafted_frequency_prior": bool(
                policy_mode == "freq_hrl" and use_handcrafted_frequency_prior
            ),
            "capacity_match_contract": (
                "identical versioned full-method architecture with inactive "
                "ablated modules"
                if fixed_ablation_architecture else
                "Freq-HRL reference or active parameter count matched to Freq-HRL "
                "within 5%; equal optimizer, epochs, and rollout seed budget"
            ),
            "fixed_ablation_architecture": bool(fixed_ablation_architecture),
            "capacity_target_parameter_count": int(target_parameter_count),
            "capacity_reference_method_contract": (
                capacity_reference_method_contract
            ),
            "capacity_reference_implementation_version": (
                full_method_implementation_version(
                    capacity_reference_method_contract
                )
            ),
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
                full_method_implementation_version(method_contract)
            ),
            "executed_plan_curve": bool(method_flags["execute_plan_curve"]),
            "additive_frequency_credit": bool(
                method_flags["use_additive_frequency_credit"]
            ),
            "raw_lower_effect_constraint": bool(
                method_flags["constrain_raw_lower_effect"]
            ),
            "causal_lf_plan_reference": bool(
                method_flags["causal_lf_plan_reference"]
            ),
            "upper_residual_control": bool(
                method_flags["upper_residual_control"]
            ),
            "learned_promotion_gate": bool(
                method_flags["learned_promotion_gate"]
            ),
            "heuristic_promotion_disabled": bool(
                not method_flags["heuristic_promotion_gate"]
            ),
            "heuristic_promotion_gate": bool(
                method_flags["heuristic_promotion_gate"]
            ),
            "promotion_gate_state_dim": int(
                smdp_config.promotion_state_dim
            ),
            "capacity_reference_promotion_state_dim": int(
                reference_smdp_config.promotion_state_dim
            ),
            "lower_action_dim": int(smdp_config.lower_action_dim),
            "capacity_reference_lower_action_dim": int(
                reference_smdp_config.lower_action_dim
            ),
            "hf_state_dim": int(smdp_config.hf_state_dim),
            "hf_action_dim": int(smdp_config.hf_action_dim),
            "capacity_reference_hf_state_dim": int(
                reference_smdp_config.hf_state_dim
            ),
            "capacity_reference_hf_action_dim": int(
                reference_smdp_config.hf_action_dim
            ),
            "promotion_replan_cost": float(promotion_replan_cost),
            "promotion_init_logit": float(promotion_init_logit),
            "promotion_entropy_coef": float(
                resolved_promotion_entropy_coef
            ),
            "promotion_rate_budget": float(promotion_rate_budget),
            "promotion_rate_coef": float(promotion_rate_coef),
            "promotion_counterfactual_coef": float(
                resolved_promotion_counterfactual_coef
            ),
            "promotion_advantage_learning_rate": resolved_learning_rates[
                "promotion_advantage"
            ],
            "promotion_advantage_coef": resolved_promotion_advantage_coef,
            "promotion_advantage_huber_delta": float(
                promotion_advantage_huber_delta
            ),
            "promotion_deterministic_threshold": float(
                promotion_deterministic_threshold
            ),
            "promotion_adapt_gain": float(promotion_adapt_gain),
            "promotion_cooldown_steps": int(promotion_cooldown_steps),
            "promotion_gate_interval_steps": int(
                promotion_gate_interval_steps
            ),
            "promotion_deterministic_mode": promotion_deterministic_mode,
            "promotion_advantage_threshold": float(
                promotion_advantage_threshold
            ),
            "promotion_advantage_target_threshold": float(
                resolved_promotion_advantage_target_threshold
            ),
            "hf_lower_overlay_enabled": bool(
                method_flags["lower_hf_overlay"]
            ),
            "hf_tactical_stream_enabled": bool(
                method_flags["lower_hf_overlay"]
                and method_flags["separate_hf_tactical"]
            ),
            "exact_three_way_credit": bool(
                method_flags["lower_hf_overlay"]
                and method_flags["separate_hf_tactical"]
            ),
            "promotion_credit_mode": resolved_promotion_credit_mode,
            "promotion_credit_scale": float(
                resolved_promotion_credit_scale
            ),
            "leakage_cost_mode": resolved_leakage_cost_mode,
            "lower_lf_budget_rms": float(lower_lf_budget_rms),
            "hf_lf_budget_rms": float(hf_lf_budget_rms),
            "hard_hf_budget_projection": bool(
                hard_hf_budget_projection
            ),
            "hf_predictability_summary": bool(
                resolved_include_hf_predictability
            ),
            "training_support_components": (
                [
                    "stationary_low_noise",
                    "stationary_high_noise",
                    "localized_burst",
                    "persistent_shift",
                ]
                if scenario == SUPPORT_MIXTURE_SCENARIO else [scenario]
            ),
            "training_support_ood_excluded": bool(
                scenario == SUPPORT_MIXTURE_SCENARIO
            ),
            "lower_hf_order_scale": float(lower_hf_order_scale),
            "upper_learning_rate": resolved_learning_rates["upper"],
            "lower_learning_rate": resolved_learning_rates["lower"],
            "hf_learning_rate": resolved_learning_rates["hf"],
            "promotion_learning_rate": resolved_learning_rates["promotion"],
            "plan_smoothness_weight": float(plan_smoothness_weight),
            "upper_plan_reference_mode": upper_plan_reference_mode,
            "upper_plan_reference_gain": float(
                upper_plan_reference_gain
            ),
            "upper_plan_reference_forecast_blend": float(
                upper_plan_reference_forecast_blend
            ),
            "upper_residual_action_scale": float(
                upper_residual_action_scale
            ),
            "training_path_protocol": (
                "independent_full_episode_support_batch_v1"
                if independent_training_scenarios is not None else
                "fresh_mixed_support_path_per_root_and_iteration_ood_excluded_v3"
                if scenario == SUPPORT_MIXTURE_SCENARIO else
                "fresh_deterministic_path_per_root_and_iteration_v2"
                if resample_training_paths else "fixed_path_reuse_legacy"
            ),
            "training_episode_scenarios": list(
                independent_training_scenarios or ()
            ),
            "training_episode_count_per_root": int(episode_multiplier),
            "checkpoint_selection_protocol": "disjoint_validation_paths",
            "plan_mode": "learned_bernstein" if plan_mapper is not None else "direct_target",
            "lower_lf_constraint_coef": float(lower_lf_constraint_coef),
            "lower_lf_constraint_target": float(
                effective_lower_cost_target
            ),
            "requested_lower_lf_constraint_target": float(
                lower_lf_constraint_target
            ),
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
    if fixed_ablation_architecture:
        architecture_label = (
            "full-v7" if method_contract in V7_METHOD_CONTRACTS else "full-v6"
        )
        payload["trajectory_contract"]["hf"] = (
            payload["trajectory_contract"]["hf"]
            if method_flags["lower_hf_overlay"]
            else f"{architecture_label} HF module retained but inactive; "
            "no HF transitions"
        )
        payload["trajectory_contract"]["promotion"] = (
            payload["trajectory_contract"]["promotion"]
            if method_flags["learned_promotion_gate"]
            else f"{architecture_label} promotion module retained but inactive; "
            "no gate transitions"
        )
    for row in heldout_rows:
        row["baseline"] = policy_mode
        row["policy_mode"] = policy_mode
    payload["environment_steps_train"] = int(
        len(rollout_seed_roots) * int(steps) * max(1, int(iterations))
        * episode_multiplier
    )
    payload["environment_steps_validation"] = int(
        len(validation_seed_list) * int(steps) * (max(1, int(iterations)) + 1)
        * episode_multiplier
    )
    payload["environment_steps_eval"] = int(
        len(evaluation_seeds) * int(steps) * episode_multiplier
    )
    payload["unique_training_path_count"] = int(
        len(rollout_seed_roots)
        * (max(1, int(iterations)) if resample_training_paths else 1)
        * episode_multiplier
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
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, SUPPORT_MIXTURE_SCENARIO),
        default="persistent_shift",
    )
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
    parser.add_argument(
        "--capacity-reference-method-contract",
        choices=METHOD_CONTRACTS,
        default=None,
    )
    parser.add_argument("--volume-impact-bps", type=float, default=0.0)
    parser.add_argument("--plan-smoothness-weight", type=float, default=0.0)
    parser.add_argument("--promotion-replan-cost", type=float, default=0.0)
    parser.add_argument("--promotion-init-logit", type=float, default=-2.0)
    parser.add_argument("--promotion-entropy-coef", type=float, default=None)
    parser.add_argument("--promotion-rate-budget", type=float, default=1.0)
    parser.add_argument("--promotion-rate-coef", type=float, default=0.0)
    parser.add_argument("--promotion-counterfactual-coef", type=float, default=None)
    parser.add_argument(
        "--promotion-deterministic-threshold", type=float, default=0.5
    )
    parser.add_argument("--promotion-adapt-gain", type=float, default=0.05)
    parser.add_argument("--promotion-cooldown-steps", type=int, default=0)
    parser.add_argument("--promotion-gate-interval-steps", type=int, default=1)
    parser.add_argument("--lower-hf-order-scale", type=float, default=0.025)
    parser.add_argument(
        "--promotion-credit-mode",
        choices=(
            "auto",
            "task_return",
            "incremental_plan_advantage",
            "paired_plan_advantage",
        ),
        default="auto",
    )
    parser.add_argument("--promotion-credit-scale", type=float, default=None)
    parser.add_argument(
        "--leakage-cost-mode",
        choices=("auto", "spectral_ratio", "fixed_rms_budget"),
        default="auto",
    )
    parser.add_argument("--lower-lf-budget-rms", type=float, default=0.0025)
    parser.add_argument("--hf-lf-budget-rms", type=float, default=0.00025)
    parser.add_argument(
        "--hard-hf-budget-projection", action="store_true"
    )
    parser.add_argument(
        "--upper-plan-reference-mode",
        choices=UPPER_PLAN_REFERENCE_MODES,
        default="none",
    )
    parser.add_argument("--upper-plan-reference-gain", type=float, default=1.0)
    parser.add_argument(
        "--upper-plan-reference-forecast-blend", type=float, default=0.0
    )
    parser.add_argument("--upper-residual-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--include-hf-predictability",
        action="store_true",
        default=None,
    )
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
        capacity_reference_method_contract=(
            args.capacity_reference_method_contract
        ),
        volume_impact_bps=args.volume_impact_bps,
        plan_smoothness_weight=args.plan_smoothness_weight,
        promotion_replan_cost=args.promotion_replan_cost,
        promotion_init_logit=args.promotion_init_logit,
        promotion_entropy_coef=args.promotion_entropy_coef,
        promotion_rate_budget=args.promotion_rate_budget,
        promotion_rate_coef=args.promotion_rate_coef,
        promotion_counterfactual_coef=args.promotion_counterfactual_coef,
        promotion_deterministic_threshold=(
            args.promotion_deterministic_threshold
        ),
        promotion_adapt_gain=args.promotion_adapt_gain,
        promotion_cooldown_steps=args.promotion_cooldown_steps,
        promotion_gate_interval_steps=args.promotion_gate_interval_steps,
        lower_hf_order_scale=args.lower_hf_order_scale,
        promotion_credit_mode=args.promotion_credit_mode,
        promotion_credit_scale=args.promotion_credit_scale,
        leakage_cost_mode=args.leakage_cost_mode,
        lower_lf_budget_rms=args.lower_lf_budget_rms,
        hf_lf_budget_rms=args.hf_lf_budget_rms,
        include_hf_predictability=args.include_hf_predictability,
        hard_hf_budget_projection=args.hard_hf_budget_projection,
        upper_plan_reference_mode=args.upper_plan_reference_mode,
        upper_plan_reference_gain=args.upper_plan_reference_gain,
        upper_plan_reference_forecast_blend=(
            args.upper_plan_reference_forecast_blend
        ),
        upper_residual_action_scale=args.upper_residual_action_scale,
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
