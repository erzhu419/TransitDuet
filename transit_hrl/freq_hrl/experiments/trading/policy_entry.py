"""Train/evaluate entry point for pluggable trading policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from freq_hrl.core import CausalLeakageRewardShaper, LeakageRegularizer
from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingFrequencyTracker,
)
from freq_hrl.policies import (
    ActorCriticTradingController,
    ActorCriticTradingParams,
    ActorCriticTradingPlanner,
    FrequencyTradingController,
    FrequencyTradingPlanner,
    LinearFrequencyTradingController,
    LinearFrequencyTradingPlanner,
    LinearTradingParams,
    PolicyGradientTradingController,
    PolicyGradientTradingParams,
    PolicyGradientTradingPlanner,
)

from .performance_validation import SCENARIOS, make_synthetic_market, max_drawdown


def make_policy(
    policy: str,
    params: LinearTradingParams | PolicyGradientTradingParams | ActorCriticTradingParams | None = None,
    sample: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[Any, Any]:
    if policy == "heuristic":
        return FrequencyTradingPlanner(promotion_mid_gain=0.5), FrequencyTradingController()
    if policy == "linear":
        params = params if isinstance(params, LinearTradingParams) else LinearTradingParams()
        return LinearFrequencyTradingPlanner(params), LinearFrequencyTradingController(params)
    if policy == "pg_linear":
        params = params if isinstance(params, PolicyGradientTradingParams) else PolicyGradientTradingParams()
        return (
            PolicyGradientTradingPlanner(params, sample=sample, rng=rng),
            PolicyGradientTradingController(params, sample=sample, rng=rng),
        )
    if policy == "ac_linear":
        params = params if isinstance(params, ActorCriticTradingParams) else ActorCriticTradingParams()
        return (
            ActorCriticTradingPlanner(params, sample=sample, rng=rng),
            ActorCriticTradingController(params, sample=sample, rng=rng),
        )
    raise ValueError(f"unknown policy: {policy}")


def run_eval(
    seed: int,
    steps: int,
    assets: int,
    policy: str = "heuristic",
    params: LinearTradingParams | PolicyGradientTradingParams | ActorCriticTradingParams | None = None,
    scenario: str = "persistent_shift",
) -> dict[str, float]:
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
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
    planner, controller = make_policy(policy, params)
    env.reset()
    pnl_returns = []
    equity = []
    turnover = []
    targets = []
    lower_effects = []
    promotion_count = 0
    for t in range(steps):
        raw_signal = data["predictor"][t]
        freq = tracker.update_bar(raw_signal, t=float(t * 60.0))
        if bool(dict(freq.get("promotion", {}) or {}).get("promote", False)):
            promotion_count += 1
        obs = {
            "raw_signal": raw_signal,
            "position": env.position.copy(),
            "t": t,
        }
        upper = planner.plan(obs, tracker.upper_features(), context={"frequency": freq, "n_assets": assets})
        env.set_target(upper.action)
        lower = controller.act(obs, tracker.lower_features(upper.action, env.position), upper, context={"frequency": freq})
        _, _, done, info = env.lower_step(lower.action)
        lower_effect = (
            np.asarray(info["position"], dtype=np.float64)
            - np.asarray(info["target"], dtype=np.float64)
        )
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        lower_effects.append(lower_effect.copy())
        if done:
            break
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    sharpe = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0
    leakage = LeakageRegularizer(
        upper_hf_window=6,
        lower_lf_window=24,
        upper_hf_weight=1.0,
        lower_lf_weight=1.0,
    ).compute(
        upper_effect=np.asarray(targets, dtype=np.float64),
        lower_effect=np.asarray(lower_effects, dtype=np.float64),
    ) if targets else {
        "leakage_penalty": 0.0,
        "UpperHFPower": 0.0,
        "LowerLFDrift": 0.0,
    }
    return {
        "seed": int(seed),
        "scenario": scenario,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotion_count),
        "leakage_penalty": float(leakage["leakage_penalty"]),
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
    }


def objective(
    row: dict[str, float],
    leakage_policy_loss_scale: float = 0.0,
    leakage_constraint_threshold: float = 0.0,
    leakage_lagrange_multiplier: float = 0.0,
    lower_lf_policy_loss_scale: float = 0.0,
    lower_lf_constraint_threshold: float = 0.0,
    lower_lf_lagrange_multiplier: float = 0.0,
) -> float:
    leakage = float(row.get("leakage_penalty", 0.0))
    violation = max(0.0, leakage - max(float(leakage_constraint_threshold), 0.0))
    lower_lf = float(row.get("LowerLFDrift", 0.0))
    lower_violation = max(0.0, lower_lf - max(float(lower_lf_constraint_threshold), 0.0))
    leakage_penalty = (
        max(float(leakage_policy_loss_scale), 0.0) * leakage
        + max(float(leakage_lagrange_multiplier), 0.0) * violation
        + max(float(lower_lf_policy_loss_scale), 0.0) * lower_lf
        + max(float(lower_lf_lagrange_multiplier), 0.0) * lower_violation
    )
    return (
        float(row["total_return"])
        + 0.01 * float(row["sharpe"])
        - 0.25 * float(row["max_drawdown"])
        - 0.0005 * float(row["turnover"])
        - leakage_penalty
    )


def _arr(value: Any, dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if dim is not None and arr.size != dim:
        arr = np.resize(arr, dim)
    return arr


def upper_critic_features(
    observation: dict[str, Any],
    freq: dict[str, Any],
    assets: int,
    scale: float = 0.0014,
) -> np.ndarray:
    x_low = _arr(freq.get("x_low", np.zeros(assets)), assets) / max(scale, 1e-9)
    x_mid = _arr(freq.get("x_mid", np.zeros(assets)), assets) / max(scale, 1e-9)
    promotion = dict(freq.get("promotion", {}) or {})
    strength = float(promotion.get("promotion_strength", 0.0)) if promotion.get("promote", False) else 0.0
    position = _arr(observation.get("position", np.zeros(assets)), assets)
    return np.asarray([
        float(np.mean(x_low)),
        float(np.mean(np.abs(x_low))),
        float(np.mean(x_mid)),
        float(np.mean(np.abs(x_mid))),
        strength,
        float(np.mean(np.abs(position))),
    ], dtype=np.float64)


def lower_critic_features(
    observation: dict[str, Any],
    freq: dict[str, Any],
    target: np.ndarray,
    assets: int,
    scale: float = 0.0014,
) -> np.ndarray:
    position = _arr(observation.get("position", np.zeros(assets)), assets)
    target = _arr(target, assets)
    x_high = _arr(freq.get("x_high", np.zeros(assets)), assets) / max(scale, 1e-9)
    energy = np.sqrt(np.maximum(_arr(freq.get("x_high_energy", np.zeros(assets)), assets), 0.0)) / max(scale, 1e-9)
    gap = target - position
    return np.asarray([
        float(np.mean(np.abs(gap))),
        float(np.mean(np.sign(gap) * x_high)),
        float(np.mean(np.abs(x_high))),
        float(np.mean(energy)),
        1.0,
    ], dtype=np.float64)


def evaluate_params(
    vector: np.ndarray,
    seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
) -> tuple[float, list[dict[str, float]]]:
    params = LinearTradingParams.from_vector(vector)
    rows = [
        run_eval(seed, steps, assets, policy="linear", params=params, scenario=scenario)
        for seed in seeds
    ]
    score = float(np.mean([objective(row) for row in rows]))
    return score, rows


def score_candidate(case: tuple[list[float], list[int], int, int, str]) -> tuple[float, list[float]]:
    vector, train_seeds, steps, assets, scenario = case
    score, _ = evaluate_params(np.asarray(vector, dtype=np.float64), train_seeds, steps, assets, scenario)
    return score, list(vector)


def run_pg_episode(
    params: PolicyGradientTradingParams,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    rng_seed: int,
    discount: float = 0.995,
    leakage_policy_loss_scale: float = 0.0,
    leakage_constraint_threshold: float = 0.0,
    leakage_lagrange_multiplier: float = 0.0,
    lower_lf_policy_loss_scale: float = 0.0,
    lower_lf_constraint_threshold: float = 0.0,
    lower_lf_lagrange_multiplier: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
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
    rng = np.random.default_rng(rng_seed)
    planner, controller = make_policy("pg_linear", params=params, sample=True, rng=rng)
    grads: list[np.ndarray] = []
    objective_rewards: list[float] = []
    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    leakage_loss_penalties: list[float] = []
    leakage_constraint_violations: list[float] = []
    lower_lf_loss_penalties: list[float] = []
    lower_lf_constraint_violations: list[float] = []
    promotion_count = 0
    leakage_shaper = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(
            upper_hf_window=6,
            lower_lf_window=24,
            upper_hf_weight=1.0,
            lower_lf_weight=1.0,
        ),
        reward_penalty_scale=0.0,
        enabled=True,
    )
    env.reset()
    for t in range(steps):
        raw_signal = data["predictor"][t]
        freq = tracker.update_bar(raw_signal, t=float(t * 60.0))
        if bool(dict(freq.get("promotion", {}) or {}).get("promote", False)):
            promotion_count += 1
        obs = {
            "raw_signal": raw_signal,
            "position": env.position.copy(),
            "t": t,
        }
        upper = planner.plan(obs, tracker.upper_features(), context={"frequency": freq, "n_assets": assets})
        env.set_target(upper.action)
        lower = controller.act(obs, tracker.lower_features(upper.action, env.position), upper, context={"frequency": freq})
        _, reward, done, info = env.lower_step(lower.action)
        step_reward = float(info["portfolio_return"] - info["transaction_cost"])
        lower_effect = (
            np.asarray(info["position"], dtype=np.float64)
            - np.asarray(info["target"], dtype=np.float64)
        )
        leakage_info = leakage_shaper.update(
            upper_effect=np.asarray(info["target"], dtype=np.float64),
            lower_effect=lower_effect,
            reward=None,
        )
        raw_leakage = float(leakage_info["leakage_penalty"])
        violation = max(0.0, raw_leakage - max(float(leakage_constraint_threshold), 0.0))
        lower_lf = float(leakage_info.get("LowerLFDrift", 0.0))
        lower_violation = max(0.0, lower_lf - max(float(lower_lf_constraint_threshold), 0.0))
        lower_lf_loss_penalty = (
            max(float(lower_lf_policy_loss_scale), 0.0) * lower_lf
            + max(float(lower_lf_lagrange_multiplier), 0.0) * lower_violation
        )
        leakage_loss_penalty = (
            max(float(leakage_policy_loss_scale), 0.0) * raw_leakage
            + max(float(leakage_lagrange_multiplier), 0.0) * violation
            + lower_lf_loss_penalty
        )
        objective_rewards.append(step_reward - leakage_loss_penalty)
        leakage_loss_penalties.append(leakage_loss_penalty)
        leakage_constraint_violations.append(violation)
        lower_lf_loss_penalties.append(lower_lf_loss_penalty)
        lower_lf_constraint_violations.append(lower_violation)
        pnl_returns.append(step_reward)
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        grad = np.asarray(upper.metadata.get("policy_grad_logp", 0.0), dtype=np.float64)
        grad = grad + np.asarray(lower.metadata.get("policy_grad_logp", 0.0), dtype=np.float64)
        grads.append(grad)
        if done:
            break
    rewards_arr = np.asarray(objective_rewards, dtype=np.float64)
    returns = np.zeros_like(rewards_arr)
    running = 0.0
    gamma = float(np.clip(discount, 0.0, 1.0))
    for idx in range(rewards_arr.size - 1, -1, -1):
        running = rewards_arr[idx] + gamma * running
        returns[idx] = running
    if returns.size:
        advantages = returns - float(np.mean(returns))
        std = float(np.std(advantages))
        if std > 1e-12:
            advantages = advantages / std
    else:
        advantages = returns
    grad_arr = np.asarray(grads, dtype=np.float64)
    if grad_arr.size:
        grad_estimate = np.mean(grad_arr * advantages[:, None], axis=0)
    else:
        grad_estimate = np.zeros_like(params.to_vector())
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    sharpe = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0
    row = {
        "seed": int(seed),
        "scenario": scenario,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotion_count),
        "policy_loss_leakage_penalty": float(np.mean(leakage_loss_penalties)) if leakage_loss_penalties else 0.0,
        "policy_loss_leakage_penalty_total": float(np.sum(leakage_loss_penalties)) if leakage_loss_penalties else 0.0,
        "leakage_constraint_violation": float(np.mean(leakage_constraint_violations)) if leakage_constraint_violations else 0.0,
        "leakage_constraint_violation_total": float(np.sum(leakage_constraint_violations)) if leakage_constraint_violations else 0.0,
        "lower_lf_policy_loss_penalty": float(np.mean(lower_lf_loss_penalties)) if lower_lf_loss_penalties else 0.0,
        "lower_lf_constraint_violation": float(np.mean(lower_lf_constraint_violations)) if lower_lf_constraint_violations else 0.0,
        "leakage_penalty": float(leakage_shaper.last_info.get("leakage_penalty", 0.0)),
        "UpperHFPower": float(leakage_shaper.last_info.get("UpperHFPower", 0.0)),
        "LowerLFDrift": float(leakage_shaper.last_info.get("LowerLFDrift", 0.0)),
    }
    return grad_estimate, row


def run_actor_critic_episode(
    params: ActorCriticTradingParams,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    rng_seed: int,
    discount: float = 0.995,
    leakage_policy_loss_scale: float = 0.0,
    leakage_constraint_threshold: float = 0.0,
    leakage_lagrange_multiplier: float = 0.0,
    lower_lf_policy_loss_scale: float = 0.0,
    lower_lf_constraint_threshold: float = 0.0,
    lower_lf_lagrange_multiplier: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets, scenario=scenario)
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
    rng = np.random.default_rng(rng_seed)
    planner, controller = make_policy("ac_linear", params=params, sample=True, rng=rng)
    actor_grads: list[np.ndarray] = []
    upper_features_rows: list[np.ndarray] = []
    lower_features_rows: list[np.ndarray] = []
    objective_rewards: list[float] = []
    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    leakage_loss_penalties: list[float] = []
    leakage_constraint_violations: list[float] = []
    lower_lf_loss_penalties: list[float] = []
    lower_lf_constraint_violations: list[float] = []
    promotion_count = 0
    leakage_shaper = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(
            upper_hf_window=6,
            lower_lf_window=24,
            upper_hf_weight=1.0,
            lower_lf_weight=1.0,
        ),
        reward_penalty_scale=0.0,
        enabled=True,
    )
    env.reset()
    for t in range(steps):
        raw_signal = data["predictor"][t]
        freq = tracker.update_bar(raw_signal, t=float(t * 60.0))
        freq_dict = dict(freq)
        if bool(dict(freq.get("promotion", {}) or {}).get("promote", False)):
            promotion_count += 1
        obs = {
            "raw_signal": raw_signal,
            "position": env.position.copy(),
            "t": t,
        }
        upper_features = upper_critic_features(obs, freq_dict, assets)
        upper = planner.plan(obs, tracker.upper_features(), context={"frequency": freq, "n_assets": assets})
        env.set_target(upper.action)
        lower_features = lower_critic_features(obs, freq_dict, upper.action, assets)
        lower = controller.act(obs, tracker.lower_features(upper.action, env.position), upper, context={"frequency": freq})
        _, _, done, info = env.lower_step(lower.action)
        step_reward = float(info["portfolio_return"] - info["transaction_cost"])
        lower_effect = (
            np.asarray(info["position"], dtype=np.float64)
            - np.asarray(info["target"], dtype=np.float64)
        )
        leakage_info = leakage_shaper.update(
            upper_effect=np.asarray(info["target"], dtype=np.float64),
            lower_effect=lower_effect,
            reward=None,
        )
        raw_leakage = float(leakage_info["leakage_penalty"])
        violation = max(0.0, raw_leakage - max(float(leakage_constraint_threshold), 0.0))
        lower_lf = float(leakage_info.get("LowerLFDrift", 0.0))
        lower_violation = max(0.0, lower_lf - max(float(lower_lf_constraint_threshold), 0.0))
        lower_lf_loss_penalty = (
            max(float(lower_lf_policy_loss_scale), 0.0) * lower_lf
            + max(float(lower_lf_lagrange_multiplier), 0.0) * lower_violation
        )
        leakage_loss_penalty = (
            max(float(leakage_policy_loss_scale), 0.0) * raw_leakage
            + max(float(leakage_lagrange_multiplier), 0.0) * violation
            + lower_lf_loss_penalty
        )
        objective_rewards.append(step_reward - leakage_loss_penalty)
        leakage_loss_penalties.append(leakage_loss_penalty)
        leakage_constraint_violations.append(violation)
        lower_lf_loss_penalties.append(lower_lf_loss_penalty)
        lower_lf_constraint_violations.append(lower_violation)
        pnl_returns.append(step_reward)
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        grad = np.asarray(upper.metadata.get("policy_grad_logp", 0.0), dtype=np.float64)
        grad = grad + np.asarray(lower.metadata.get("policy_grad_logp", 0.0), dtype=np.float64)
        actor_grads.append(grad)
        upper_features_rows.append(upper_features)
        lower_features_rows.append(lower_features)
        if done:
            break

    rewards = np.asarray(objective_rewards, dtype=np.float64)
    upper_arr = np.asarray(upper_features_rows, dtype=np.float64)
    lower_arr = np.asarray(lower_features_rows, dtype=np.float64)
    upper_w = params.upper_value_vector()
    lower_w = params.lower_value_vector()
    if rewards.size:
        values = upper_arr @ upper_w + lower_arr @ lower_w
        next_values = np.concatenate([values[1:], np.zeros(1, dtype=np.float64)])
        td_errors = rewards + float(discount) * next_values - values
        actor_adv = td_errors.copy()
        adv_std = float(np.std(actor_adv))
        if adv_std > 1e-12:
            actor_adv = (actor_adv - float(np.mean(actor_adv))) / adv_std
        actor_grad = np.mean(np.asarray(actor_grads, dtype=np.float64) * actor_adv[:, None], axis=0)
        upper_grad = np.mean(td_errors[:, None] * upper_arr, axis=0)
        lower_grad = np.mean(td_errors[:, None] * lower_arr, axis=0)
    else:
        td_errors = np.zeros(0, dtype=np.float64)
        actor_grad = np.zeros_like(params.actor_vector())
        upper_grad = np.zeros_like(params.upper_value_vector())
        lower_grad = np.zeros_like(params.lower_value_vector())

    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    sharpe = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0
    row = {
        "seed": int(seed),
        "scenario": scenario,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotion_count),
        "td_error_mean": float(np.mean(td_errors)) if td_errors.size else 0.0,
        "td_error_abs_mean": float(np.mean(np.abs(td_errors))) if td_errors.size else 0.0,
        "critic_value_loss": float(np.mean(td_errors * td_errors)) if td_errors.size else 0.0,
        "policy_loss_leakage_penalty": float(np.mean(leakage_loss_penalties)) if leakage_loss_penalties else 0.0,
        "policy_loss_leakage_penalty_total": float(np.sum(leakage_loss_penalties)) if leakage_loss_penalties else 0.0,
        "leakage_constraint_violation": float(np.mean(leakage_constraint_violations)) if leakage_constraint_violations else 0.0,
        "leakage_constraint_violation_total": float(np.sum(leakage_constraint_violations)) if leakage_constraint_violations else 0.0,
        "lower_lf_policy_loss_penalty": float(np.mean(lower_lf_loss_penalties)) if lower_lf_loss_penalties else 0.0,
        "lower_lf_constraint_violation": float(np.mean(lower_lf_constraint_violations)) if lower_lf_constraint_violations else 0.0,
        "leakage_penalty": float(leakage_shaper.last_info.get("leakage_penalty", 0.0)),
        "UpperHFPower": float(leakage_shaper.last_info.get("UpperHFPower", 0.0)),
        "LowerLFDrift": float(leakage_shaper.last_info.get("LowerLFDrift", 0.0)),
    }
    return actor_grad, upper_grad, lower_grad, row


def effective_workers(requested: int, case_count: int) -> int:
    workers = max(1, min(int(requested), max(1, int(case_count))))
    if os.name == "nt":
        workers = min(workers, 61)
    return workers


def train_linear_policy(
    train_seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    generations: int,
    population: int,
    elite_frac: float,
    seed: int,
    workers: int = 1,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mean = LinearTradingParams().to_vector()
    std = np.asarray([0.50, 0.50, 0.40, 0.25, 0.20, 0.30, 0.50], dtype=np.float64)
    elite_n = max(1, int(round(population * elite_frac)))
    history = []
    best_score = -float("inf")
    best_vector = mean.copy()
    for gen in range(max(1, int(generations))):
        candidates = [mean.copy()]
        for _ in range(max(0, int(population) - 1)):
            candidates.append(mean + rng.normal(0.0, std))
        workers = effective_workers(workers, len(candidates))
        if workers == 1:
            scored = [
                (evaluate_params(vec, train_seeds, steps, assets, scenario)[0], vec)
                for vec in candidates
            ]
        else:
            scored = []
            cases = [
                (vec.tolist(), list(train_seeds), int(steps), int(assets), str(scenario))
                for vec in candidates
            ]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(score_candidate, case) for case in cases]
                for fut in as_completed(futures):
                    score, vec = fut.result()
                    scored.append((score, np.asarray(vec, dtype=np.float64)))
        scored.sort(key=lambda item: item[0], reverse=True)
        elite = np.asarray([vec for _, vec in scored[:elite_n]], dtype=np.float64)
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), 0.03)
        if scored[0][0] > best_score:
            best_score = float(scored[0][0])
            best_vector = scored[0][1].copy()
        history.append({
            "generation": gen,
            "best_score": float(scored[0][0]),
            "mean_score": float(np.mean([score for score, _ in scored])),
            "std": std.tolist(),
            "best_params": LinearTradingParams.from_vector(scored[0][1]).to_mapping(),
        })
    return {
        "policy": "linear",
        "scenario": scenario,
        "train_seeds": list(train_seeds),
        "steps": int(steps),
        "assets": int(assets),
        "best_score": float(best_score),
        "params": LinearTradingParams.from_vector(best_vector).to_mapping(),
        "history": history,
    }


def train_policy_gradient(
    train_seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    iterations: int,
    learning_rate: float,
    seed: int,
    discount: float = 0.995,
    leakage_policy_loss_scale: float = 0.0,
    leakage_constraint_threshold: float = 0.0,
    leakage_lagrange_lr: float = 0.0,
    lower_lf_policy_loss_scale: float = 0.0,
    lower_lf_constraint_threshold: float = 0.0,
    lower_lf_lagrange_lr: float = 0.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    params = PolicyGradientTradingParams()
    vector = params.to_vector()
    best_vector = vector.copy()
    best_score = -float("inf")
    history = []
    lr = max(float(learning_rate), 0.0)
    lagrange_multiplier = 0.0
    lower_lf_lagrange_multiplier = 0.0
    for iteration in range(max(1, int(iterations))):
        grads = []
        sampled_rows = []
        for train_seed in train_seeds:
            grad, row = run_pg_episode(
                PolicyGradientTradingParams.from_vector(vector, template=params),
                seed=int(train_seed),
                steps=steps,
                assets=assets,
                scenario=scenario,
                rng_seed=int(rng.integers(0, 2**31 - 1)),
                discount=discount,
                leakage_policy_loss_scale=leakage_policy_loss_scale,
                leakage_constraint_threshold=leakage_constraint_threshold,
                leakage_lagrange_multiplier=lagrange_multiplier,
                lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
            )
            grads.append(grad)
            sampled_rows.append(row)
        grad_mean = np.mean(np.asarray(grads, dtype=np.float64), axis=0)
        grad_norm = float(np.linalg.norm(grad_mean))
        if grad_norm > 10.0:
            grad_mean = grad_mean * (10.0 / grad_norm)
        vector = vector + lr * grad_mean
        vector = np.clip(vector, -4.0, 4.0)
        current_params = PolicyGradientTradingParams.from_vector(vector, template=params)
        eval_rows = [
            run_eval(seed=int(eval_seed), steps=steps, assets=assets, policy="pg_linear", params=current_params, scenario=scenario)
            for eval_seed in train_seeds
        ]
        score = float(np.mean([
            objective(
                row,
                leakage_policy_loss_scale=leakage_policy_loss_scale,
                leakage_constraint_threshold=leakage_constraint_threshold,
                leakage_lagrange_multiplier=lagrange_multiplier,
                lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
            )
            for row in eval_rows
        ]))
        if score > best_score:
            best_score = score
            best_vector = vector.copy()
        mean_violation = float(np.mean([
            row.get("leakage_constraint_violation", 0.0)
            for row in sampled_rows
        ]))
        lagrange_multiplier = max(
            0.0,
            lagrange_multiplier + max(float(leakage_lagrange_lr), 0.0) * mean_violation,
        )
        mean_lower_violation = float(np.mean([
            row.get("lower_lf_constraint_violation", 0.0)
            for row in sampled_rows
        ]))
        lower_lf_lagrange_multiplier = max(
            0.0,
            lower_lf_lagrange_multiplier
            + max(float(lower_lf_lagrange_lr), 0.0) * mean_lower_violation,
        )
        history.append({
            "iteration": int(iteration),
            "sampled_objective": float(np.mean([
                objective(
                    row,
                    leakage_policy_loss_scale=leakage_policy_loss_scale,
                    leakage_constraint_threshold=leakage_constraint_threshold,
                    leakage_lagrange_multiplier=lagrange_multiplier,
                    lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                    lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                    lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
                )
                for row in sampled_rows
            ])),
            "deterministic_objective": score,
            "deterministic_task_objective": float(np.mean([objective(row) for row in eval_rows])),
            "deterministic_sharpe": float(np.mean([row["sharpe"] for row in eval_rows])),
            "grad_norm": float(np.linalg.norm(grad_mean)),
            "policy_loss_leakage_penalty": float(np.mean([
                row.get("policy_loss_leakage_penalty", 0.0)
                for row in sampled_rows
            ])),
            "leakage_constraint_violation": mean_violation,
            "leakage_lagrange_multiplier": float(lagrange_multiplier),
            "lower_lf_policy_loss_penalty": float(np.mean([
                row.get("lower_lf_policy_loss_penalty", 0.0)
                for row in sampled_rows
            ])),
            "lower_lf_constraint_violation": mean_lower_violation,
            "lower_lf_lagrange_multiplier": float(lower_lf_lagrange_multiplier),
            "params": current_params.to_mapping(),
        })
    best_params = PolicyGradientTradingParams.from_vector(best_vector, template=params)
    return {
        "policy": "pg_linear",
        "trainer": (
            "on_policy_reinforce_leakage_constrained"
            if leakage_policy_loss_scale > 0.0 or leakage_lagrange_lr > 0.0
            or lower_lf_policy_loss_scale > 0.0 or lower_lf_lagrange_lr > 0.0
            else "on_policy_reinforce"
        ),
        "scenario": scenario,
        "train_seeds": list(train_seeds),
        "steps": int(steps),
        "assets": int(assets),
        "best_score": float(best_score),
        "leakage_policy_loss_scale": float(leakage_policy_loss_scale),
        "leakage_constraint_threshold": float(leakage_constraint_threshold),
        "leakage_lagrange_lr": float(leakage_lagrange_lr),
        "lower_lf_policy_loss_scale": float(lower_lf_policy_loss_scale),
        "lower_lf_constraint_threshold": float(lower_lf_constraint_threshold),
        "lower_lf_lagrange_lr": float(lower_lf_lagrange_lr),
        "params": best_params.to_mapping(),
        "history": history,
    }


def train_actor_critic(
    train_seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    iterations: int,
    actor_learning_rate: float,
    critic_learning_rate: float,
    seed: int,
    discount: float = 0.995,
    leakage_policy_loss_scale: float = 0.0,
    leakage_constraint_threshold: float = 0.0,
    leakage_lagrange_lr: float = 0.0,
    lower_lf_policy_loss_scale: float = 0.0,
    lower_lf_constraint_threshold: float = 0.0,
    lower_lf_lagrange_lr: float = 0.0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    params = ActorCriticTradingParams()
    actor_vector = params.actor_vector()
    upper_value = params.upper_value_vector()
    lower_value = params.lower_value_vector()
    best_params = params
    best_score = -float("inf")
    history = []
    actor_lr = max(float(actor_learning_rate), 0.0)
    critic_lr = max(float(critic_learning_rate), 0.0)
    lagrange_multiplier = 0.0
    lower_lf_lagrange_multiplier = 0.0
    for iteration in range(max(1, int(iterations))):
        current_params = params.with_vectors(actor_vector, upper_value, lower_value)
        actor_grads = []
        upper_grads = []
        lower_grads = []
        sampled_rows = []
        for train_seed in train_seeds:
            actor_grad, upper_grad, lower_grad, row = run_actor_critic_episode(
                current_params,
                seed=int(train_seed),
                steps=steps,
                assets=assets,
                scenario=scenario,
                rng_seed=int(rng.integers(0, 2**31 - 1)),
                discount=discount,
                leakage_policy_loss_scale=leakage_policy_loss_scale,
                leakage_constraint_threshold=leakage_constraint_threshold,
                leakage_lagrange_multiplier=lagrange_multiplier,
                lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
            )
            actor_grads.append(actor_grad)
            upper_grads.append(upper_grad)
            lower_grads.append(lower_grad)
            sampled_rows.append(row)

        actor_grad_mean = np.mean(np.asarray(actor_grads, dtype=np.float64), axis=0)
        upper_grad_mean = np.mean(np.asarray(upper_grads, dtype=np.float64), axis=0)
        lower_grad_mean = np.mean(np.asarray(lower_grads, dtype=np.float64), axis=0)
        actor_norm = float(np.linalg.norm(actor_grad_mean))
        if actor_norm > 10.0:
            actor_grad_mean = actor_grad_mean * (10.0 / actor_norm)
        critic_norm = float(np.linalg.norm(np.concatenate([upper_grad_mean, lower_grad_mean])))
        if critic_norm > 10.0:
            scale = 10.0 / critic_norm
            upper_grad_mean = upper_grad_mean * scale
            lower_grad_mean = lower_grad_mean * scale

        actor_vector = np.clip(actor_vector + actor_lr * actor_grad_mean, -4.0, 4.0)
        upper_value = np.clip(upper_value + critic_lr * upper_grad_mean, -10.0, 10.0)
        lower_value = np.clip(lower_value + critic_lr * lower_grad_mean, -10.0, 10.0)
        current_params = params.with_vectors(actor_vector, upper_value, lower_value)
        eval_rows = [
            run_eval(seed=int(eval_seed), steps=steps, assets=assets, policy="ac_linear", params=current_params, scenario=scenario)
            for eval_seed in train_seeds
        ]
        score = float(np.mean([
            objective(
                row,
                leakage_policy_loss_scale=leakage_policy_loss_scale,
                leakage_constraint_threshold=leakage_constraint_threshold,
                leakage_lagrange_multiplier=lagrange_multiplier,
                lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
            )
            for row in eval_rows
        ]))
        if score > best_score:
            best_score = score
            best_params = current_params
        mean_violation = float(np.mean([
            row.get("leakage_constraint_violation", 0.0)
            for row in sampled_rows
        ]))
        lagrange_multiplier = max(
            0.0,
            lagrange_multiplier + max(float(leakage_lagrange_lr), 0.0) * mean_violation,
        )
        mean_lower_violation = float(np.mean([
            row.get("lower_lf_constraint_violation", 0.0)
            for row in sampled_rows
        ]))
        lower_lf_lagrange_multiplier = max(
            0.0,
            lower_lf_lagrange_multiplier
            + max(float(lower_lf_lagrange_lr), 0.0) * mean_lower_violation,
        )
        history.append({
            "iteration": int(iteration),
            "sampled_objective": float(np.mean([
                objective(
                    row,
                    leakage_policy_loss_scale=leakage_policy_loss_scale,
                    leakage_constraint_threshold=leakage_constraint_threshold,
                    leakage_lagrange_multiplier=lagrange_multiplier,
                    lower_lf_policy_loss_scale=lower_lf_policy_loss_scale,
                    lower_lf_constraint_threshold=lower_lf_constraint_threshold,
                    lower_lf_lagrange_multiplier=lower_lf_lagrange_multiplier,
                )
                for row in sampled_rows
            ])),
            "deterministic_objective": score,
            "deterministic_task_objective": float(np.mean([objective(row) for row in eval_rows])),
            "deterministic_sharpe": float(np.mean([row["sharpe"] for row in eval_rows])),
            "td_error_abs_mean": float(np.mean([row["td_error_abs_mean"] for row in sampled_rows])),
            "critic_value_loss": float(np.mean([row["critic_value_loss"] for row in sampled_rows])),
            "actor_grad_norm": float(np.linalg.norm(actor_grad_mean)),
            "critic_grad_norm": float(np.linalg.norm(np.concatenate([upper_grad_mean, lower_grad_mean]))),
            "policy_loss_leakage_penalty": float(np.mean([
                row.get("policy_loss_leakage_penalty", 0.0)
                for row in sampled_rows
            ])),
            "leakage_constraint_violation": mean_violation,
            "leakage_lagrange_multiplier": float(lagrange_multiplier),
            "lower_lf_policy_loss_penalty": float(np.mean([
                row.get("lower_lf_policy_loss_penalty", 0.0)
                for row in sampled_rows
            ])),
            "lower_lf_constraint_violation": mean_lower_violation,
            "lower_lf_lagrange_multiplier": float(lower_lf_lagrange_multiplier),
            "params": current_params.to_mapping(),
        })

    return {
        "policy": "ac_linear",
        "trainer": (
            "td0_actor_critic_leakage_constrained"
            if leakage_policy_loss_scale > 0.0 or leakage_lagrange_lr > 0.0
            or lower_lf_policy_loss_scale > 0.0 or lower_lf_lagrange_lr > 0.0
            else "td0_actor_critic"
        ),
        "scenario": scenario,
        "train_seeds": list(train_seeds),
        "steps": int(steps),
        "assets": int(assets),
        "best_score": float(best_score),
        "discount": float(discount),
        "actor_learning_rate": float(actor_learning_rate),
        "critic_learning_rate": float(critic_learning_rate),
        "leakage_policy_loss_scale": float(leakage_policy_loss_scale),
        "leakage_constraint_threshold": float(leakage_constraint_threshold),
        "leakage_lagrange_lr": float(leakage_lagrange_lr),
        "lower_lf_policy_loss_scale": float(lower_lf_policy_loss_scale),
        "lower_lf_constraint_threshold": float(lower_lf_constraint_threshold),
        "lower_lf_lagrange_lr": float(lower_lf_lagrange_lr),
        "params": best_params.to_mapping(),
        "history": history,
    }


def summarize(rows: list[dict[str, float]], mode: str, policy: str) -> dict[str, Any]:
    summary = {
        "mode": mode,
        "policy": policy,
        "n": len(rows),
        "total_return_mean": float(np.mean([r["total_return"] for r in rows])),
        "sharpe_mean": float(np.mean([r["sharpe"] for r in rows])),
        "max_drawdown_mean": float(np.mean([r["max_drawdown"] for r in rows])),
        "turnover_mean": float(np.mean([r["turnover"] for r in rows])),
        "promotion_count_mean": float(np.mean([r["promotion_count"] for r in rows])),
    }
    for key in ["leakage_penalty", "UpperHFPower", "LowerLFDrift"]:
        if rows and key in rows[0]:
            summary[f"{key}_mean"] = float(np.mean([r[key] for r in rows]))
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_linear_params(path: Path | None) -> LinearTradingParams:
    if path is None:
        return LinearTradingParams()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return LinearTradingParams.from_mapping(payload.get("params", payload))


def load_pg_params(path: Path | None) -> PolicyGradientTradingParams:
    if path is None:
        return PolicyGradientTradingParams()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return PolicyGradientTradingParams.from_mapping(payload.get("params", payload))


def load_ac_params(path: Path | None) -> ActorCriticTradingParams:
    if path is None:
        return ActorCriticTradingParams()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return ActorCriticTradingParams.from_mapping(payload.get("params", payload))


def flatten_params(value: Mapping[str, Any], prefix: str = "") -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for key, val in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(val, Mapping):
            rows.extend(flatten_params(val, prefix=name))
        elif isinstance(val, (list, tuple)):
            for idx, item in enumerate(val):
                rows.append((f"{name}[{idx}]", float(item)))
        else:
            rows.append((name, float(val)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--policy", choices=["heuristic", "linear", "pg_linear", "ac_linear"], default="heuristic")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--pg-iterations", type=int, default=12)
    parser.add_argument("--pg-learning-rate", type=float, default=0.05)
    parser.add_argument("--pg-discount", type=float, default=0.995)
    parser.add_argument("--pg-leakage-policy-loss-scale", type=float, default=0.0)
    parser.add_argument("--pg-leakage-constraint-threshold", type=float, default=0.0)
    parser.add_argument("--pg-leakage-lagrange-lr", type=float, default=0.0)
    parser.add_argument("--pg-lower-lf-policy-loss-scale", type=float, default=0.0)
    parser.add_argument("--pg-lower-lf-constraint-threshold", type=float, default=0.0)
    parser.add_argument("--pg-lower-lf-lagrange-lr", type=float, default=0.0)
    parser.add_argument("--ac-iterations", type=int, default=20)
    parser.add_argument("--ac-actor-learning-rate", type=float, default=0.03)
    parser.add_argument("--ac-critic-learning-rate", type=float, default=0.08)
    parser.add_argument("--ac-discount", type=float, default=0.995)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_policy_entry"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_payload = None
    params = None
    policy = args.policy
    eval_seeds = args.eval_seeds if args.eval_seeds is not None else args.seeds

    if args.mode == "train":
        if policy == "pg_linear":
            model_payload = train_policy_gradient(
                train_seeds=list(args.train_seeds),
                steps=args.steps,
                assets=args.assets,
                scenario=args.scenario,
                iterations=args.pg_iterations,
                learning_rate=args.pg_learning_rate,
                seed=args.optimizer_seed,
                discount=args.pg_discount,
                leakage_policy_loss_scale=args.pg_leakage_policy_loss_scale,
                leakage_constraint_threshold=args.pg_leakage_constraint_threshold,
                leakage_lagrange_lr=args.pg_leakage_lagrange_lr,
                lower_lf_policy_loss_scale=args.pg_lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=args.pg_lower_lf_constraint_threshold,
                lower_lf_lagrange_lr=args.pg_lower_lf_lagrange_lr,
            )
            model_path = args.model_path or (args.output_dir / "pg_linear_policy.json")
            params = PolicyGradientTradingParams.from_mapping(model_payload["params"])
        elif policy == "ac_linear":
            model_payload = train_actor_critic(
                train_seeds=list(args.train_seeds),
                steps=args.steps,
                assets=args.assets,
                scenario=args.scenario,
                iterations=args.ac_iterations,
                actor_learning_rate=args.ac_actor_learning_rate,
                critic_learning_rate=args.ac_critic_learning_rate,
                seed=args.optimizer_seed,
                discount=args.ac_discount,
                leakage_policy_loss_scale=args.pg_leakage_policy_loss_scale,
                leakage_constraint_threshold=args.pg_leakage_constraint_threshold,
                leakage_lagrange_lr=args.pg_leakage_lagrange_lr,
                lower_lf_policy_loss_scale=args.pg_lower_lf_policy_loss_scale,
                lower_lf_constraint_threshold=args.pg_lower_lf_constraint_threshold,
                lower_lf_lagrange_lr=args.pg_lower_lf_lagrange_lr,
            )
            model_path = args.model_path or (args.output_dir / "ac_linear_policy.json")
            params = ActorCriticTradingParams.from_mapping(model_payload["params"])
        else:
            policy = "linear"
            model_payload = train_linear_policy(
                train_seeds=list(args.train_seeds),
                steps=args.steps,
                assets=args.assets,
                scenario=args.scenario,
                generations=args.generations,
                population=args.population,
                elite_frac=args.elite_frac,
                seed=args.optimizer_seed,
                workers=args.workers,
            )
            model_path = args.model_path or (args.output_dir / "linear_policy.json")
            params = LinearTradingParams.from_mapping(model_payload["params"])
        with model_path.open("w", encoding="utf-8") as f:
            json.dump(model_payload, f, indent=2)
    elif policy == "linear":
        params = load_linear_params(args.model_path)
    elif policy == "pg_linear":
        params = load_pg_params(args.model_path)
    elif policy == "ac_linear":
        params = load_ac_params(args.model_path)

    rows = [
        run_eval(seed, args.steps, args.assets, policy=policy, params=params, scenario=args.scenario)
        for seed in eval_seeds
    ]
    write_rows(args.output_dir / "per_seed.csv", rows)
    summary = summarize(rows, args.mode, policy)
    payload = {"summary": summary, "per_seed": rows}
    if model_payload is not None:
        payload["model"] = model_payload
    elif params is not None:
        payload["params"] = params.to_mapping()
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    report = [
        "# Trading Policy Entry",
        "",
        f"- mode: `{args.mode}`",
        f"- policy: `{policy}`",
        f"- scenario: `{args.scenario}`",
        f"- eval seeds: {list(eval_seeds)}",
        f"- total return mean: {summary['total_return_mean']:.4f}",
        f"- Sharpe mean: {summary['sharpe_mean']:.3f}",
        f"- max drawdown mean: {summary['max_drawdown_mean']:.4f}",
        f"- turnover mean: {summary['turnover_mean']:.2f}",
        f"- leakage penalty mean: {summary.get('leakage_penalty_mean', 0.0):.4f}",
        "",
        "The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. The `pg_linear` policy is trained by on-policy Gaussian REINFORCE over upper targets and lower execution speeds. The `ac_linear` policy uses separated upper low-frequency and lower high-frequency TD(0) critics to train the same actor with bootstrapped advantages. Optional leakage flags add policy-loss penalties and Lagrange-style constraints for both total causal action-effect leakage and lower LF drift.",
    ]
    if model_payload is not None:
        report.extend([
            "",
            "## Learned Parameters",
            "",
            "| parameter | value |",
            "|---|---:|",
        ])
        for key, value in flatten_params(model_payload["params"]):
            report.append(f"| {key} | {float(value):+.4f} |")
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}")
    print(f"policy_entry mode={args.mode} policy={policy} sharpe={summary['sharpe_mean']:.3f} return={summary['total_return_mean']:.4f}")


if __name__ == "__main__":
    main()
