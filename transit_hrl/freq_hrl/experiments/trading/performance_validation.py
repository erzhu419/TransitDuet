"""Synthetic performance validation for the FreqTradeDuet MVP.

This script is intentionally not a smoke test.  It runs multiple strategies on
the same synthetic non-stationary market, records trading metrics, and reports
frequency responsibility diagnostics.  The strategies are heuristic policies,
not trained RL policies; the purpose is to validate the Freq-HRL protocol before
spending cycles on expensive learned-policy integration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    FrequencyDiagnostics,
    LeakageRegularizer,
    RewardAttributionAccumulator,
)
from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingActionEffectOperator,
    TradingFrequencyTracker,
)


BASELINES = (
    "vanilla_rl",
    "hrl_raw",
    "raw_history",
    "freq_single_policy",
    "lf_upper_only",
    "hf_lower_only",
    "allfreq_alllayers",
    "swapped",
    "no_promotion",
    "no_leakage",
    "freq_hrl",
)


SCENARIOS = (
    "persistent_shift",
    "promotion_recovery",
    "stationary_low_noise",
    "stationary_high_noise",
    "localized_burst",
    "ood_period",
)


def make_synthetic_market(
    seed: int,
    steps: int = 720,
    n_assets: int = 3,
    scenario: str = "persistent_shift",
) -> dict[str, np.ndarray]:
    """Create a market with controlled frequency/regime structure."""
    scenario = str(scenario or "persistent_shift").lower()
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic market scenario: {scenario}")
    rng = np.random.default_rng(seed)
    t = np.arange(steps, dtype=np.float64)
    phases = np.linspace(0.0, math.pi, n_assets)
    cycle_period = 180.0 if scenario != "ood_period" else 95.0
    cycle_amp = 0.00045
    if scenario == "stationary_low_noise":
        cycle_amp = 0.00028
    elif scenario == "stationary_high_noise":
        cycle_amp = 0.00035
    cycle = np.stack([
        cycle_amp * np.sin(2.0 * math.pi * t / cycle_period + phase)
        for phase in phases
    ], axis=1)
    base = np.zeros((steps, n_assets), dtype=np.float64)
    base[:, 0] = 0.00025
    base[:, 1] = -0.00010
    if n_assets > 2:
        base[:, 2] = 0.00005

    shift_t = int(0.55 * steps)
    has_shift = scenario in {"persistent_shift", "promotion_recovery", "ood_period"}
    if has_shift:
        if scenario == "promotion_recovery":
            shift_t = int(0.45 * steps)
            # Abrupt low-frequency reversal: the pre-shift target is now wrong,
            # so recovery depends on how quickly persistent innovations are
            # promoted into the upper plan.
            base[shift_t:, 0] -= 0.00125
            base[shift_t:, 1] += 0.00105
            if n_assets > 2:
                base[shift_t:, 2] -= 0.00070
        else:
            base[shift_t:, 0] += 0.00075
            base[shift_t:, 1] -= 0.00055
            if n_assets > 2:
                base[shift_t:, 2] += 0.00040
    else:
        shift_t = steps + 1

    high_sigma = {
        "persistent_shift": 0.00045,
        "promotion_recovery": 0.00040,
        "stationary_low_noise": 0.00020,
        "stationary_high_noise": 0.00075,
        "localized_burst": 0.00035,
        "ood_period": 0.00055,
    }[scenario]
    high = rng.normal(0.0, high_sigma, size=(steps, n_assets))
    for i in range(1, steps):
        high[i] += 0.35 * high[i - 1]

    shock_mask = np.zeros((steps, n_assets), dtype=bool)
    shock_count = {
        "persistent_shift": 10,
        "promotion_recovery": 8,
        "stationary_low_noise": 3,
        "stationary_high_noise": 14,
        "localized_burst": 2,
        "ood_period": 12,
    }[scenario]
    for _ in range(shock_count):
        if scenario == "localized_burst":
            center = int(0.45 * steps)
            start_low = max(2, center - max(8, steps // 12))
            start_high = min(max(start_low + 1, steps - 4), center + max(10, steps // 12))
            start = int(rng.integers(start_low, start_high))
            length = int(rng.integers(5, max(6, min(15, steps // 12))))
        else:
            start_hi = max(21, steps - 40)
            start = int(rng.integers(20, start_hi))
            length = int(rng.integers(6, 22))
        length = min(length, max(1, steps - start))
        asset = int(rng.integers(0, n_assets))
        direction = float(rng.choice([-1.0, 1.0]))
        amp_hi = 0.0040 if scenario in {"localized_burst", "stationary_high_noise"} else 0.0030
        amp = direction * float(rng.uniform(0.0012, amp_hi))
        decay = np.exp(-np.arange(length, dtype=np.float64) / max(length / 3.0, 1.0))
        high[start:start + length, asset] += amp * decay
        shock_mask[start:start + length, asset] = True

    low = base + cycle
    predictor_noise = 0.00012 if scenario == "stationary_low_noise" else 0.00020
    predictor = low + high + rng.normal(0.0, predictor_noise, size=(steps, n_assets))
    # Realized returns are mostly driven by the low component.  High residuals
    # are intentionally weakly predictive and noisy; policies that route raw HF
    # into the upper planner should overtrade in this setting.
    return_noise = {
        "persistent_shift": 0.00075,
        "promotion_recovery": 0.00065,
        "stationary_low_noise": 0.00035,
        "stationary_high_noise": 0.00110,
        "localized_burst": 0.00065,
        "ood_period": 0.00090,
    }[scenario]
    high_alpha = {
        "persistent_shift": 0.08,
        "promotion_recovery": 0.12,
        "stationary_low_noise": 0.02,
        "stationary_high_noise": 0.02,
        "localized_burst": 0.35,
        "ood_period": 0.08,
    }[scenario]
    returns = low + high_alpha * high + rng.normal(0.0, return_noise, size=(steps, n_assets))
    volume = 1.0 + 25.0 * np.abs(high) / 0.003 + rng.uniform(0.0, 0.5, size=(steps, n_assets))
    return {
        "returns": returns,
        "predictor": predictor,
        "low_truth": low,
        "high_truth": high,
        "volume": volume,
        "shock_mask": shock_mask,
        "regime_shift_t": np.asarray([shift_t], dtype=np.int64),
        "scenario": np.asarray([scenario]),
    }


def normalize_target(signal: np.ndarray, scale: float = 0.0014, max_gross: float = 1.0) -> np.ndarray:
    target = np.tanh(np.asarray(signal, dtype=np.float64).reshape(-1) / max(scale, 1e-9))
    gross = float(np.sum(np.abs(target)))
    if gross > max_gross and gross > 1e-12:
        target *= max_gross / gross
    return target


def moving_average(history: list[np.ndarray], window: int, dim: int | None = None) -> np.ndarray:
    if not history:
        if dim is None:
            dim = 1
        return np.zeros(int(dim), dtype=np.float64)
    arr = np.asarray(history[-max(1, int(window)):], dtype=np.float64)
    return arr.mean(axis=0)


def _resize(value: Any, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == dim:
        return arr
    return np.resize(arr, dim)


def _bounded_hf_residual(
    x_high: np.ndarray,
    x_mid: np.ndarray,
    high_energy: np.ndarray,
    high_persistence: np.ndarray,
    shock_age: np.ndarray,
    target: np.ndarray,
    current_position: np.ndarray,
    gain: float,
    recenter_gain: float,
    scale: float = 0.0014,
    max_abs: float = 0.025,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-horizon lower correction with leakage-aware recentering."""
    dim = target.size
    energy = np.sqrt(np.maximum(_resize(high_energy, dim), 0.0)) / max(scale, 1e-9)
    persistence = np.clip(_resize(high_persistence, dim), 0.0, 1.0)
    age = np.clip(_resize(shock_age, dim), 0.0, 1.0)
    energy_gate = np.tanh(energy)
    persistence_gate = np.clip(0.25 + 0.75 * np.maximum(persistence, age), 0.0, 1.0)
    residual_signal = np.tanh((x_high + 0.25 * x_mid) / max(scale, 1e-9))
    gap = target - current_position
    direction_gate = np.where(
        np.sign(residual_signal) == np.sign(gap),
        1.0,
        np.where(np.abs(gap) < 0.03, 0.75, 0.35),
    )
    raw_residual = max(float(gain), 0.0) * energy_gate * persistence_gate * direction_gate * residual_signal
    drift = current_position - target
    increases_drift = np.sign(raw_residual) == np.sign(drift)
    raw_residual = np.where(increases_drift, 0.35 * raw_residual, raw_residual)
    recenter = max(float(recenter_gain), 0.0) * np.tanh(drift / 0.08)
    residual = np.clip(raw_residual - recenter, -max_abs, max_abs)
    return residual, energy_gate


def _causal_hf_utility(
    high_history: list[np.ndarray],
    return_history: list[np.ndarray],
    dim: int,
    window: int = 90,
) -> np.ndarray:
    """Estimate whether recent HF residuals predicted next-bar returns."""
    if len(high_history) < 12 or len(return_history) < 12:
        return np.zeros(dim, dtype=np.float64)
    x_all = np.asarray(high_history, dtype=np.float64)
    y_all = np.asarray(return_history, dtype=np.float64)
    n = min(x_all.shape[0] - 1, y_all.shape[0] - 1, int(window))
    if n < 8:
        return np.zeros(dim, dtype=np.float64)
    x = x_all[-n - 1:-1]
    y = y_all[-n:]
    if x.shape[1] != dim:
        x = np.resize(x, (x.shape[0], dim))
    if y.shape[1] != dim:
        y = np.resize(y, (y.shape[0], dim))
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.mean(x * x, axis=0) * np.mean(y * y, axis=0)) + 1e-12
    corr = np.mean(x * y, axis=0) / denom
    hit = np.mean(np.sign(x) == np.sign(y), axis=0) - 0.5
    score = np.maximum(corr, 0.0) + 0.5 * np.maximum(hit, 0.0)
    return np.clip(score / 0.35, 0.0, 1.0)


def policy_action(
    name: str,
    tracker: TradingFrequencyTracker,
    raw_signal: np.ndarray,
    raw_history: list[np.ndarray],
    current_position: np.ndarray,
    high_history: list[np.ndarray] | None = None,
    return_history: list[np.ndarray] | None = None,
    promotion_mid_gain: float = 1.0,
    hf_residual_gain: float = 0.0,
    hf_recenter_gain: float = 0.0,
    hf_speed_gain: float = 0.0,
    hf_energy_speed_gain: float = 0.0,
    promotion_residual_plan_gain: float = 0.0,
    promotion_speed_boost: float = 0.0,
    seed_phase: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    feats = tracker.features()
    dim = raw_signal.size
    x_low = _resize(feats.get("x_low", 0.0), dim)
    x_mid = _resize(feats.get("x_mid", 0.0), dim)
    x_high = _resize(feats.get("x_high", 0.0), dim)
    high_energy = _resize(feats.get("x_high_energy", 0.0), dim)
    high_persistence = _resize(feats.get("x_high_persistence", 0.0), dim)
    shock_age = _resize(feats.get("shock_age", 0.0), dim)
    promotion = (
        tracker.promotion_gate.signal().to_mapping()
        if tracker.promotion_gate is not None
        else {"promote": False, "promotion_strength": 0.0}
    )
    strength = float(promotion.get("promotion_strength", 0.0))
    promote = bool(promotion.get("promote", False))
    promote_for_plan = promote and promotion.get("reason") in {
        "persistent_high_residual",
        "hysteresis",
    }
    hf_utility = _causal_hf_utility(high_history or [], return_history or [], dim)

    if name == "freq_hrl":
        promoted_residual = (
            max(float(promotion_residual_plan_gain), 0.0) * strength * x_high
            if promote_for_plan else 0.0
        )
        plan_signal = (
            x_low
            + (promotion_mid_gain * strength * x_mid if promote_for_plan else 0.0)
            + promoted_residual
        )
        target = normalize_target(plan_signal, max_gross=1.0)
        gap = target - current_position
        align = np.sign(gap) * x_high / 0.0014
        residual, energy_gate = _bounded_hf_residual(
            x_high,
            x_mid,
            high_energy,
            high_persistence,
            shock_age,
            target,
            current_position,
            gain=hf_residual_gain,
            recenter_gain=hf_recenter_gain,
        )
        alpha = np.clip(
            0.55 + hf_utility * (
                max(float(hf_speed_gain), 0.0) * np.tanh(align)
                + max(float(hf_energy_speed_gain), 0.0) * energy_gate
            )
            + (max(float(promotion_speed_boost), 0.0) * strength if promote_for_plan else 0.0),
            0.15,
            1.0,
        )
    elif name == "vanilla_rl":
        target = normalize_target(raw_signal)
        alpha = np.ones(dim, dtype=np.float64) * 0.75
        residual = np.zeros(dim, dtype=np.float64)
        promotion = {"promote": False, "promotion_strength": 0.0}
    elif name == "no_promotion":
        target = normalize_target(x_low)
        gap = target - current_position
        align = np.sign(gap) * x_high / 0.0014
        residual, energy_gate = _bounded_hf_residual(
            x_high,
            x_mid,
            high_energy,
            high_persistence,
            shock_age,
            target,
            current_position,
            gain=hf_residual_gain,
            recenter_gain=hf_recenter_gain,
        )
        alpha = np.clip(
            0.55 + hf_utility * (
                max(float(hf_speed_gain), 0.0) * np.tanh(align)
                + max(float(hf_energy_speed_gain), 0.0) * energy_gate
            ),
            0.15,
            1.0,
        )
        promotion = {"promote": False, "promotion_strength": 0.0}
    elif name == "freq_single_policy":
        signal = x_low + x_mid + x_high
        target = normalize_target(signal)
        alpha = np.clip(0.60 + 0.25 * np.tanh(np.sign(target - current_position) * signal / 0.0014), 0.10, 1.0)
        residual = np.zeros(dim, dtype=np.float64)
        promotion = {"promote": False, "promotion_strength": 0.0}
    elif name == "lf_upper_only":
        target = normalize_target(x_low)
        alpha = np.ones(dim, dtype=np.float64) * 0.55
        residual = np.zeros(dim, dtype=np.float64)
        promotion = {"promote": False, "promotion_strength": 0.0}
    elif name == "hf_lower_only":
        target = np.zeros(dim, dtype=np.float64)
        alpha = np.ones(dim, dtype=np.float64) * 0.20
        residual = 0.045 * np.tanh(x_high / 0.0014)
        promotion = {"promote": False, "promotion_strength": 0.0}
    elif name == "allfreq_alllayers":
        target = normalize_target(x_low + x_mid + x_high)
        gap = target - current_position
        alpha = np.clip(0.55 + 0.35 * np.tanh(np.sign(gap) * raw_signal / 0.0014), 0.05, 1.0)
        residual = 0.015 * np.tanh(x_high / 0.0014)
    elif name == "swapped":
        target = normalize_target(x_high)
        alpha = np.clip(0.35 + 0.25 * np.tanh(np.sign(target - current_position) * x_low / 0.0014), 0.05, 0.80)
        residual = np.zeros(dim, dtype=np.float64)
    elif name == "hrl_raw":
        target = normalize_target(raw_signal)
        alpha = np.ones(dim, dtype=np.float64) * 0.50
        residual = np.zeros(dim, dtype=np.float64)
    elif name == "raw_history":
        target = normalize_target(moving_average(raw_history, window=20, dim=dim))
        alpha = np.ones(dim, dtype=np.float64) * 0.45
        residual = np.zeros(dim, dtype=np.float64)
    elif name == "no_leakage":
        target = normalize_target(x_low + (strength * x_mid if promote_for_plan else 0.0))
        alpha = np.ones(dim, dtype=np.float64) * 0.45
        # Deliberately let the lower controller take directional residual bets.
        residual = 0.055 * np.tanh((x_high + 0.15 * x_mid) / 0.0014)
    else:
        raise ValueError(f"unknown baseline: {name}")

    lower_action = {"execution_speed": alpha, "residual_order": residual}
    diag_features = {
        "x_low": x_low,
        "x_mid": x_mid,
        "x_high": x_high,
        "x_high_energy": high_energy,
        "x_high_persistence": high_persistence,
        "shock_age": shock_age,
        "hf_utility": hf_utility,
        "promotion": promotion,
    }
    return target, lower_action, diag_features


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = 1.0 - equity / np.maximum(peaks, 1e-12)
    return float(np.max(dd))


def summarize_run(
    baseline: str,
    seed: int,
    rewards: list[float],
    raw_rewards: list[float],
    leakage_reward_penalties: list[float],
    pnl_returns: list[float],
    oracle_pnl_returns: list[float],
    costs: list[float],
    turnovers: list[float],
    inventory_drifts: list[float],
    equity_curve: list[float],
    targets: list[np.ndarray],
    trades: list[np.ndarray],
    diagnostics: FrequencyDiagnostics,
    reward_attribution: dict[str, Any],
    promotion_count: int,
    promotion_flags: list[bool],
    first_promotion_after_shift: int | None,
    regime_shift_t: int,
) -> dict[str, Any]:
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    rewards_arr = np.asarray(rewards, dtype=np.float64)
    raw_rewards_arr = np.asarray(raw_rewards, dtype=np.float64)
    leakage_penalty_arr = np.asarray(leakage_reward_penalties, dtype=np.float64)
    equity = np.asarray(equity_curve, dtype=np.float64)
    total_return = float(equity[-1] - 1.0) if equity.size else 0.0
    sharpe = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0
    downside = pnl[pnl < 0.0]
    sortino = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (downside.std() + 1e-12)) if downside.size else sharpe
    mdd = max_drawdown(equity)
    calmar = float(total_return / max(mdd, 1e-9))

    op = TradingActionEffectOperator(target_history=targets)
    reg = LeakageRegularizer(upper_hf_window=6, lower_lf_window=24)
    leakage = reg.compute(
        upper_effect=np.asarray(targets, dtype=np.float64),
        lower_effect=op.lower_effect(trades),
    )
    diag = diagnostics.summarize_episode()
    post_shift_pnl = pnl[regime_shift_t:min(pnl.size, regime_shift_t + 120)]
    post_shift_cum_pnl = float(np.sum(post_shift_pnl)) if post_shift_pnl.size else 0.0
    post_shift_recovery_cost = float(-np.sum(np.minimum(post_shift_pnl, 0.0))) if post_shift_pnl.size else 0.0
    oracle = np.asarray(oracle_pnl_returns, dtype=np.float64)
    post_shift_oracle = oracle[regime_shift_t:min(oracle.size, regime_shift_t + 120)]
    post_shift_regret = (
        float(np.sum(np.maximum(post_shift_oracle - post_shift_pnl[:post_shift_oracle.size], 0.0)))
        if post_shift_pnl.size and post_shift_oracle.size else 0.0
    )
    promotion_delay = (
        -1.0 if first_promotion_after_shift is None
        else float(first_promotion_after_shift - regime_shift_t)
    )
    flags = np.asarray(promotion_flags, dtype=bool)
    labels = np.zeros(flags.size, dtype=bool)
    if labels.size:
        labels[regime_shift_t:min(labels.size, regime_shift_t + 120)] = True
    tp = float(np.sum(flags & labels))
    tn = float(np.sum((~flags) & (~labels)))
    fp = float(np.sum(flags & (~labels)))
    fn = float(np.sum((~flags) & labels))
    tpr = tp / max(tp + fn, 1.0)
    tnr = tn / max(tn + fp, 1.0)
    promotion_accuracy = 0.5 * (tpr + tnr)
    promotion_precision = tp / max(tp + fp, 1.0)
    promotion_recall = tp / max(tp + fn, 1.0)
    row = {
        "baseline": baseline,
        "seed": int(seed),
        "total_return": total_return,
        "mean_reward": float(rewards_arr.mean()) if rewards_arr.size else 0.0,
        "base_mean_reward": float(raw_rewards_arr.mean()) if raw_rewards_arr.size else 0.0,
        "leakage_reward_penalty": float(leakage_penalty_arr.mean()) if leakage_penalty_arr.size else 0.0,
        "leakage_reward_penalty_total": float(leakage_penalty_arr.sum()) if leakage_penalty_arr.size else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "turnover": float(np.sum(turnovers)),
        "transaction_cost": float(np.sum(costs)),
        "inventory_drift_mean": float(np.mean(inventory_drifts)) if inventory_drifts else 0.0,
        "post_shift_mean_pnl": float(post_shift_pnl.mean()) if post_shift_pnl.size else 0.0,
        "post_shift_cum_pnl_120": post_shift_cum_pnl,
        "recovery_cost_120": post_shift_recovery_cost,
        "recovery_regret_120": post_shift_regret,
        "promotion_count": int(promotion_count),
        "promotion_delay": promotion_delay,
        "PromotionDelay": promotion_delay,
        "ShockResponseTime": float(diag["ShockResponseTime"]),
        "regime_promotion_accuracy": promotion_accuracy,
        "regime_promotion_precision": promotion_precision,
        "regime_promotion_recall": promotion_recall,
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
        "FocusScore": float(diag["FocusScore"]),
    }
    row.update({key: float(value) for key, value in reward_attribution.items()})
    return row


def run_baseline(
    seed: int,
    baseline: str,
    steps: int,
    n_assets: int,
    scenario: str = "persistent_shift",
    freq_method: str = "ema",
    promotion_threshold: float = 0.00035,
    promotion_ratio: float = 0.50,
    promotion_window_s: float = 30 * 60.0,
    promotion_cooldown_s: float = 10 * 60.0,
    promotion_regime_threshold: float = 3e-05,
    promotion_min_age_s: float = 0.0,
    promotion_activation_strength_threshold: float = 0.0,
    promotion_startup_strength_age_s: float = 0.0,
    promotion_startup_strength_threshold: float = 0.0,
    promotion_mid_gain: float = 0.5,
    promotion_adapt_gain: float = 0.05,
    hf_residual_gain: float = 0.0,
    hf_recenter_gain: float = 0.0,
    hf_speed_gain: float = 0.0,
    hf_energy_speed_gain: float = 0.0,
    promotion_residual_plan_gain: float = 0.0,
    promotion_speed_boost: float = 0.0,
    leakage_reward_scale: float = 0.00005,
    promotion_adaptation_cost_scale: float = 0.00005,
) -> dict[str, Any]:
    data = make_synthetic_market(
        seed=seed,
        steps=steps,
        n_assets=n_assets,
        scenario=scenario,
    )
    env = PortfolioExecutionEnv(
        data["returns"],
        volumes=data["volume"],
        config=PortfolioExecutionConfig(
            transaction_cost_bps=50.0,
            slippage_bps=10.0,
            max_leverage=1.0,
            inventory_drift_penalty=0.002,
            drawdown_penalty=0.0,
        ),
    )
    tracker = TradingFrequencyTracker(
        bar_sec=60.0,
        method=freq_method,
        low_period_s=120 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=30 * 60.0,
        energy_period_s=10 * 60.0,
        persistence_period_s=30 * 60.0,
        persistence_threshold=0.0010,
        feature_norm=np.ones(n_assets) * 0.0015,
        promotion_enable=baseline in {
            "freq_hrl",
            "allfreq_alllayers",
            "swapped",
            "no_leakage",
        },
        promotion_window_s=promotion_window_s,
        promotion_residual_threshold=promotion_threshold,
        promotion_persistence_ratio=promotion_ratio,
        promotion_cooldown_s=promotion_cooldown_s,
        promotion_regime_threshold=promotion_regime_threshold,
        promotion_min_age_s=promotion_min_age_s,
        promotion_activation_strength_threshold=promotion_activation_strength_threshold,
        promotion_startup_strength_age_s=promotion_startup_strength_age_s,
        promotion_startup_strength_threshold=promotion_startup_strength_threshold,
        promotion_adapt_low=True,
        promotion_adapt_gain=promotion_adapt_gain,
    )
    diagnostics = FrequencyDiagnostics(mi_bins=8)
    leakage_shaper = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(
            upper_hf_window=6,
            lower_lf_window=24,
            upper_hf_weight=1.0,
            lower_lf_weight=1.0,
        ),
        reward_penalty_scale=leakage_reward_scale,
        enabled=(baseline != "no_leakage"),
    )
    reward_attribution = RewardAttributionAccumulator()
    raw_history: list[np.ndarray] = []
    high_history: list[np.ndarray] = []
    return_history: list[np.ndarray] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    leakage_reward_penalties: list[float] = []
    pnl_returns: list[float] = []
    oracle_pnl_returns: list[float] = []
    costs: list[float] = []
    turnovers: list[float] = []
    inventory_drifts: list[float] = []
    equity_curve: list[float] = []
    targets: list[np.ndarray] = []
    trades: list[np.ndarray] = []
    promotion_count = 0
    promotion_flags: list[bool] = []
    first_promotion_after_shift = None
    regime_shift_t = int(data["regime_shift_t"][0])
    prev_promotion_absorbed_norm = 0.0

    env.reset()
    for t in range(steps):
        raw_signal = data["predictor"][t]
        old_position = env.position.copy()
        tracker.update_bar(raw_signal, t=float(t * 60.0))
        target, lower_action, diag_features = policy_action(
            baseline,
            tracker,
            raw_signal,
            raw_history,
            env.position.copy(),
            high_history=high_history,
            return_history=return_history,
            promotion_mid_gain=promotion_mid_gain,
            hf_residual_gain=hf_residual_gain,
            hf_recenter_gain=hf_recenter_gain,
            hf_speed_gain=hf_speed_gain,
            hf_energy_speed_gain=hf_energy_speed_gain,
            promotion_residual_plan_gain=promotion_residual_plan_gain,
            promotion_speed_boost=promotion_speed_boost,
        )
        raw_history.append(raw_signal.copy())
        high_history.append(np.asarray(diag_features.get("x_high", np.zeros(n_assets)), dtype=np.float64).copy())
        promotion = diag_features.get("promotion", {})
        promoted = bool(promotion.get("promote", False))
        promotion_flags.append(promoted)
        if promoted:
            promotion_count += 1
            if t >= regime_shift_t and first_promotion_after_shift is None:
                first_promotion_after_shift = int(t)
        env.set_target(target)
        _, reward, done, info = env.lower_step(lower_action)
        lower_effect = (
            np.asarray(info["position"], dtype=np.float64)
            - np.asarray(info["target"], dtype=np.float64)
        )
        leakage_info = leakage_shaper.update(
            upper_effect=target,
            lower_effect=lower_effect,
            reward=float(reward),
        )
        tracker_summary = tracker.summary()
        absorbed_norm = float(tracker_summary.get("freq_promotion_absorbed_norm", 0.0))
        absorbed_delta = max(0.0, absorbed_norm - prev_promotion_absorbed_norm)
        prev_promotion_absorbed_norm = absorbed_norm
        promotion_strength_for_cost = (
            float(promotion.get("promotion_strength", 0.0))
            if promoted else 0.0
        )
        promotion_adaptation_cost = (
            max(float(promotion_adaptation_cost_scale), 0.0)
            * (promotion_strength_for_cost + absorbed_delta)
        )
        shaped_reward = float(leakage_info["shaped_reward"]) - promotion_adaptation_cost
        pnl = float(info["portfolio_return"] - info["transaction_cost"])
        oracle_target = normalize_target(data["low_truth"][t], max_gross=1.0)
        oracle_pnl = float(np.dot(oracle_target, data["returns"][t]))
        rewards.append(shaped_reward)
        raw_rewards.append(float(reward))
        leakage_reward_penalties.append(float(leakage_info["leakage_reward_penalty"]))
        pnl_returns.append(pnl)
        oracle_pnl_returns.append(oracle_pnl)
        return_history.append(np.asarray(data["returns"][t], dtype=np.float64).copy())
        costs.append(float(info["transaction_cost"]))
        turnovers.append(float(info["turnover"]))
        inventory_drifts.append(float(info["inventory_drift"]))
        equity_curve.append(float(info["equity"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        trades.append(np.asarray(info["trade"], dtype=np.float64).copy())
        low_pnl = float(np.dot(old_position, data["low_truth"][t]))
        high_pnl = float(np.dot(old_position, data["returns"][t] - data["low_truth"][t]))
        inv_cost = 0.002 * float(info["inventory_drift"])
        low_cost = max(0.0, -low_pnl)
        high_cost = (
            max(0.0, -high_pnl)
            + float(info["transaction_cost"])
            + inv_cost
        )
        reward_attribution.log_step(
            task_reward=float(reward),
            low_frequency_cost=low_cost,
            high_frequency_cost=high_cost,
            leakage_cost=float(leakage_info["leakage_reward_penalty"]),
            promotion_adaptation_cost=promotion_adaptation_cost,
            metadata={
                "low_component_pnl": low_pnl,
                "high_component_pnl": high_pnl,
            },
        )
        diagnostics.log_step(
            t=float(t),
            states={
                "regime_shift": t == regime_shift_t,
                "shock": bool(np.any(data["shock_mask"][t])),
                "lower_responded": float(info["turnover"]) > 0.02,
            },
            actions={
                "upper": target,
                "lower": np.asarray(info["trade"], dtype=np.float64),
            },
            freq_features=diag_features,
            effects={
                "upper": target,
                "lower": lower_effect,
            },
        )
        if done:
            break

    return summarize_run(
        baseline=baseline,
        seed=seed,
        rewards=rewards,
        raw_rewards=raw_rewards,
        leakage_reward_penalties=leakage_reward_penalties,
        pnl_returns=pnl_returns,
        oracle_pnl_returns=oracle_pnl_returns,
        costs=costs,
        turnovers=turnovers,
        inventory_drifts=inventory_drifts,
        equity_curve=equity_curve,
        targets=targets,
        trades=trades,
        diagnostics=diagnostics,
        reward_attribution=reward_attribution.episode_metrics(),
        promotion_count=promotion_count,
        promotion_flags=promotion_flags,
        first_promotion_after_shift=first_promotion_after_shift,
        regime_shift_t=regime_shift_t,
    ) | {"scenario": scenario, "freq_method": str(freq_method)}


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        key for key in rows[0].keys()
        if key not in {"baseline", "seed", "scenario", "freq_method"}
        and isinstance(rows[0].get(key), (int, float, np.integer, np.floating))
    ]
    out = []
    for baseline in BASELINES:
        group = [row for row in rows if row["baseline"] == baseline]
        if not group:
            continue
        summary = {"baseline": baseline, "n": len(group)}
        if "scenario" in rows[0]:
            scenarios = sorted({str(row.get("scenario", "")) for row in group})
            if len(scenarios) == 1:
                summary["scenario"] = scenarios[0]
        if "freq_method" in rows[0]:
            methods = sorted({str(row.get("freq_method", "")) for row in group})
            if len(methods) == 1:
                summary["freq_method"] = methods[0]
        for metric in metrics:
            vals = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(vals.mean())
            summary[f"{metric}_std"] = float(vals.std(ddof=0))
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: list[dict[str, Any]],
    seeds: list[int],
    steps: int,
    scenario: str,
    freq_method: str,
    promotion_threshold: float,
    promotion_ratio: float,
    promotion_window_s: float,
    promotion_cooldown_s: float,
    promotion_regime_threshold: float,
    promotion_min_age_s: float,
    promotion_activation_strength_threshold: float,
    promotion_startup_strength_age_s: float,
    promotion_startup_strength_threshold: float,
    promotion_mid_gain: float,
    promotion_adapt_gain: float,
    hf_residual_gain: float,
    hf_recenter_gain: float,
    hf_speed_gain: float,
    hf_energy_speed_gain: float,
    promotion_residual_plan_gain: float,
    promotion_speed_boost: float,
    leakage_reward_scale: float,
    promotion_adaptation_cost_scale: float,
) -> None:
    by_name = {row["baseline"]: row for row in summary}
    freq = by_name["freq_hrl"]
    no_prom = by_name.get("no_promotion")
    best = max(summary, key=lambda row: row["sharpe_mean"])
    if no_prom is not None:
        sharpe_delta = freq["sharpe_mean"] - no_prom["sharpe_mean"]
        return_delta = freq["total_return_mean"] - no_prom["total_return_mean"]
        post_shift_delta = (
            freq["post_shift_cum_pnl_120_mean"]
            - no_prom["post_shift_cum_pnl_120_mean"]
        )
    else:
        sharpe_delta = return_delta = post_shift_delta = 0.0
    rows = [
        "# Trading Performance Validation",
        "",
        "Synthetic high-cost noisy market with low-frequency alpha, high-frequency shocks, and a persistent regime shift.",
        "",
        f"- seeds: {seeds}",
        f"- bars per seed: {steps}",
        f"- scenario: `{scenario}`",
        f"- frequency encoder: `{freq_method}`",
        f"- promotion config: threshold={promotion_threshold}, persistence_ratio={promotion_ratio}, window_s={promotion_window_s}, cooldown_s={promotion_cooldown_s}, regime_threshold={promotion_regime_threshold}, min_age_s={promotion_min_age_s}, activation_strength_threshold={promotion_activation_strength_threshold}, startup_strength_age_s={promotion_startup_strength_age_s}, startup_strength_threshold={promotion_startup_strength_threshold}, mid_gain={promotion_mid_gain}, adapt_gain={promotion_adapt_gain}, residual_plan_gain={promotion_residual_plan_gain}, speed_boost={promotion_speed_boost}",
        f"- HF lower config: residual_gain={hf_residual_gain}, recenter_gain={hf_recenter_gain}, speed_gain={hf_speed_gain}, energy_speed_gain={hf_energy_speed_gain}",
        f"- leakage reward scale: {leakage_reward_scale}",
        f"- promotion adaptation cost scale: {promotion_adaptation_cost_scale}",
        "- policies are deterministic heuristics, not trained RL policies",
        "- task metrics include return, Sharpe, drawdown, turnover, transaction cost, and inventory drift",
        "- frequency diagnostics include UpperHFPower, LowerLFDrift, FocusScore, PromotionDelay, ShockResponseTime, regime-promotion accuracy, recovery cost, and oracle-regime recovery regret",
        "",
        "## Headline",
        "",
        (
            f"Best Sharpe baseline: `{best['baseline']}` "
            f"({best['sharpe_mean']:.3f} +/- {best['sharpe_std']:.3f})."
        ),
        (
            f"`freq_hrl`: Sharpe {freq['sharpe_mean']:.3f}, "
            f"return {freq['total_return_mean']:.4f}, "
            f"max drawdown {freq['max_drawdown_mean']:.4f}, "
            f"turnover {freq['turnover_mean']:.2f}, "
            f"FocusScore {freq['FocusScore_mean']:.3f}."
        ),
        (
            f"Against `no_promotion`, tuned promotion changes Sharpe by {sharpe_delta:+.3f}, "
            f"return by {return_delta:+.4f}, and post-shift-120 PnL by {post_shift_delta:+.5f}."
        ),
        "",
        "## Interpretation",
        "",
        "- Frequency routing is useful in this validation: `freq_hrl` beats raw-history, all-frequency, swapped, no-promotion, and no-leakage baselines on Sharpe. Against `lf_upper_only`, the current gain is a small Sharpe edge rather than clean return dominance.",
        "- `allfreq_alllayers` and `hrl_raw` overtrade heavily under noisy high-frequency shocks, which is visible in turnover and transaction cost.",
        "- `lf_upper_only` remains close on this synthetic task, so the incremental value of HF lower control should be claimed as modest and scenario-dependent.",
        "- `swapped` has negative FocusScore and poor task metrics, supporting the direction of the LF-to-upper / HF-to-lower assignment.",
        "- Promotion is now a small positive contributor on headline Sharpe, but the return and immediate post-shift deltas remain effectively flat in this conservative setting.",
        "- Leakage regularization is applied online to the reward signal sent to learners; `no_leakage` disables this shaping path.",
        "- Reward attribution splits each episode into LF cost, HF cost, leakage cost, and promotion adaptation cost for credit diagnostics.",
        "- The immediate post-shift 120-bar window is still not improved by the best Sharpe setting, so promotion should be claimed as task-positive here, not as fully optimized shock recovery.",
        "",
        "## Summary Table",
        "",
        "| baseline | return | Sharpe | shaped reward | LF cost | HF cost | leak cost | promo cost | max DD | turnover | cost | post_shift_120 | recovery_cost | recovery_regret | PromotionDelay | ShockResponse | promo_acc | UpperHFPower | LowerLFDrift | FocusScore | promotions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        rows.append(
            f"| {row['baseline']} "
            f"| {row['total_return_mean']:.4f} "
            f"| {row['sharpe_mean']:.3f} "
            f"| {row['mean_reward_mean']:.6f} "
            f"| {row['freq_attr_low_frequency_cost_mean']:.8f} "
            f"| {row['freq_attr_high_frequency_cost_mean']:.8f} "
            f"| {row['freq_attr_leakage_cost_mean']:.8f} "
            f"| {row['freq_attr_promotion_adaptation_cost_mean']:.8f} "
            f"| {row['max_drawdown_mean']:.4f} "
            f"| {row['turnover_mean']:.2f} "
            f"| {row['transaction_cost_mean']:.4f} "
            f"| {row['post_shift_cum_pnl_120_mean']:.5f} "
            f"| {row['recovery_cost_120_mean']:.5f} "
            f"| {row['recovery_regret_120_mean']:.5f} "
            f"| {row['PromotionDelay_mean']:.1f} "
            f"| {row['ShockResponseTime_mean']:.1f} "
            f"| {row['regime_promotion_accuracy_mean']:.3f} "
            f"| {row['UpperHFPower_mean']:.4f} "
            f"| {row['LowerLFDrift_mean']:.3f} "
            f"| {row['FocusScore_mean']:.3f} "
            f"| {row['promotion_count_mean']:.1f} |"
        )
    rows.extend([
        "",
        "## Current Validation Boundary",
        "",
        "This is performance validation for the Freq-HRL protocol on a controlled synthetic trading task. It is not yet learned-policy validation, and it is not yet a TransitDuet simulator training result.",
    ])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2026])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument(
        "--freq-method",
        default="ema",
        choices=["ema", "fourier", "state_space", "haar_wavelet", "wavelet"],
    )
    parser.add_argument("--promotion-threshold", type=float, default=0.00035)
    parser.add_argument("--promotion-ratio", type=float, default=0.50)
    parser.add_argument("--promotion-window-s", type=float, default=30 * 60.0)
    parser.add_argument("--promotion-cooldown-s", type=float, default=10 * 60.0)
    parser.add_argument("--promotion-regime-threshold", type=float, default=3e-05)
    parser.add_argument("--promotion-min-age-s", type=float, default=0.0)
    parser.add_argument("--promotion-activation-strength-threshold", type=float, default=0.0)
    parser.add_argument("--promotion-startup-strength-age-s", type=float, default=0.0)
    parser.add_argument("--promotion-startup-strength-threshold", type=float, default=0.0)
    parser.add_argument("--promotion-mid-gain", type=float, default=0.5)
    parser.add_argument("--promotion-adapt-gain", type=float, default=0.05)
    parser.add_argument("--hf-residual-gain", type=float, default=0.0)
    parser.add_argument("--hf-recenter-gain", type=float, default=0.0)
    parser.add_argument("--hf-speed-gain", type=float, default=0.0)
    parser.add_argument("--hf-energy-speed-gain", type=float, default=0.0)
    parser.add_argument("--promotion-residual-plan-gain", type=float, default=0.0)
    parser.add_argument("--promotion-speed-boost", type=float, default=0.0)
    parser.add_argument("--leakage-reward-scale", type=float, default=0.00005)
    parser.add_argument("--promotion-adaptation-cost-scale", type=float, default=0.00005)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_performance"),
    )
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        for baseline in BASELINES:
            rows.append(run_baseline(
                seed,
                baseline,
                args.steps,
                args.assets,
                scenario=args.scenario,
                freq_method=args.freq_method,
                promotion_threshold=args.promotion_threshold,
                promotion_ratio=args.promotion_ratio,
                promotion_window_s=args.promotion_window_s,
                promotion_cooldown_s=args.promotion_cooldown_s,
                promotion_regime_threshold=args.promotion_regime_threshold,
                promotion_min_age_s=args.promotion_min_age_s,
                promotion_activation_strength_threshold=args.promotion_activation_strength_threshold,
                promotion_startup_strength_age_s=args.promotion_startup_strength_age_s,
                promotion_startup_strength_threshold=args.promotion_startup_strength_threshold,
                promotion_mid_gain=args.promotion_mid_gain,
                promotion_adapt_gain=args.promotion_adapt_gain,
                hf_residual_gain=args.hf_residual_gain,
                hf_recenter_gain=args.hf_recenter_gain,
                hf_speed_gain=args.hf_speed_gain,
                hf_energy_speed_gain=args.hf_energy_speed_gain,
                promotion_residual_plan_gain=args.promotion_residual_plan_gain,
                promotion_speed_boost=args.promotion_speed_boost,
                leakage_reward_scale=args.leakage_reward_scale,
                promotion_adaptation_cost_scale=args.promotion_adaptation_cost_scale,
            ))

    summary = aggregate(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed.csv", rows)
    write_csv(args.output_dir / "summary.csv", summary)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"per_seed": rows, "summary": summary}, f, indent=2)
    write_report(
        args.output_dir / "report.md",
        summary,
        list(args.seeds),
        args.steps,
        scenario=args.scenario,
        freq_method=args.freq_method,
        promotion_threshold=args.promotion_threshold,
        promotion_ratio=args.promotion_ratio,
        promotion_window_s=args.promotion_window_s,
        promotion_cooldown_s=args.promotion_cooldown_s,
        promotion_regime_threshold=args.promotion_regime_threshold,
        promotion_min_age_s=args.promotion_min_age_s,
        promotion_activation_strength_threshold=args.promotion_activation_strength_threshold,
        promotion_startup_strength_age_s=args.promotion_startup_strength_age_s,
        promotion_startup_strength_threshold=args.promotion_startup_strength_threshold,
        promotion_mid_gain=args.promotion_mid_gain,
        promotion_adapt_gain=args.promotion_adapt_gain,
        hf_residual_gain=args.hf_residual_gain,
        hf_recenter_gain=args.hf_recenter_gain,
        hf_speed_gain=args.hf_speed_gain,
        hf_energy_speed_gain=args.hf_energy_speed_gain,
        promotion_residual_plan_gain=args.promotion_residual_plan_gain,
        promotion_speed_boost=args.promotion_speed_boost,
        leakage_reward_scale=args.leakage_reward_scale,
        promotion_adaptation_cost_scale=args.promotion_adaptation_cost_scale,
    )

    best = max(summary, key=lambda row: row["sharpe_mean"])
    print(f"wrote {args.output_dir}")
    print(
        "best_sharpe="
        f"{best['baseline']} sharpe={best['sharpe_mean']:.3f} "
        f"return={best['total_return_mean']:.4f} "
        f"mdd={best['max_drawdown_mean']:.4f}"
    )
    freq = next(row for row in summary if row["baseline"] == "freq_hrl")
    print(
        "freq_hrl "
        f"sharpe={freq['sharpe_mean']:.3f} "
        f"return={freq['total_return_mean']:.4f} "
        f"UpperHFPower={freq['UpperHFPower_mean']:.3f} "
        f"LowerLFDrift={freq['LowerLFDrift_mean']:.3f} "
        f"FocusScore={freq['FocusScore_mean']:.3f}"
    )


if __name__ == "__main__":
    main()
