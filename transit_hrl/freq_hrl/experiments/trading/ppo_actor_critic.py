"""PPO-style dual actor-critic validation for Freq-HRL trading."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.core import CausalLeakageRewardShaper, LeakageRegularizer
from freq_hrl.domains.trading import PortfolioExecutionConfig, PortfolioExecutionEnv, TradingFrequencyTracker
from freq_hrl.policies import BernsteinPlanCurve
from freq_hrl.rl import (
    DualActorCriticPPO,
    DualPPOConfig,
    LearnedPlanActionMapper,
    TrajectoryBatch,
    summarize_numeric_rows,
    train_dual_ppo,
)

from .performance_validation import SCENARIOS, make_synthetic_market, max_drawdown


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


def feature_vectors(freq: dict[str, Any], position: np.ndarray, target: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    dim = int(position.size)
    scale = 0.0014
    x_low = resize(freq.get("x_low", np.zeros(dim)), dim) / scale
    x_mid = resize(freq.get("x_mid", np.zeros(dim)), dim) / scale
    x_high = resize(freq.get("x_high", np.zeros(dim)), dim) / scale
    promotion = dict(freq.get("promotion", {}) or {})
    strength = float(promotion.get("promotion_strength", 0.0)) if promotion.get("promote", False) else 0.0
    upper_state = np.concatenate([
        x_low,
        x_mid,
        x_high,
        strength * x_mid,
        np.asarray(position, dtype=np.float64),
        np.ones(1, dtype=np.float64),
    ])
    if target is None:
        target = np.zeros(dim, dtype=np.float64)
    gap = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    energy = np.sqrt(np.maximum(resize(freq.get("x_high_energy", np.zeros(dim)), dim), 0.0)) / scale
    align = np.tanh(np.sign(gap) * x_high)
    lower_state = np.concatenate([
        gap,
        align,
        np.tanh(energy),
        np.ones(1, dtype=np.float64),
    ])
    return upper_state.astype(np.float32), lower_state.astype(np.float32)


def initialize_frequency_prior(model: DualActorCriticPPO, assets: int, plan_basis_dim: int = 0) -> None:
    """Initialize the linear actors near the existing frequency-routing prior."""
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
                upper_linear.weight[row, i] = 0.75 * ramp
                upper_linear.weight[row, assets + i] = 0.18 * ramp
                upper_linear.weight[row, 3 * assets + i] = 0.32 * ramp
                upper_linear.weight[row, 4 * assets + i] = -0.10 * ramp
            lower_linear.weight[i, assets + i] = 0.20
            lower_linear.weight[i, 2 * assets + i] = 0.02
            lower_linear.bias[i] = 0.20


def latent_target(latent: np.ndarray) -> np.ndarray:
    return gross_cap(np.tanh(np.asarray(latent, dtype=np.float64)))


def make_plan_mapper(
    assets: int,
    plan_basis_dim: int,
    plan_horizon_s: float,
    plan_eval_offset_s: float,
    plan_coefficient_scale: float,
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
    )


def latent_speed(latent: np.ndarray) -> np.ndarray:
    return np.clip(0.05 + 0.95 / (1.0 + np.exp(-np.asarray(latent, dtype=np.float64))), 0.05, 1.0)


def rollout(
    model: DualActorCriticPPO,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    sample: bool,
    leakage_scale: float = 0.0,
    plan_mapper: LearnedPlanActionMapper | None = None,
) -> tuple[TrajectoryBatch | None, dict[str, float]]:
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
    tracker = make_tracker(assets)
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
        reward_penalty_scale=leakage_scale,
        enabled=leakage_scale > 0.0,
    )
    upper_states: list[np.ndarray] = []
    lower_states: list[np.ndarray] = []
    upper_actions: list[np.ndarray] = []
    lower_actions: list[np.ndarray] = []
    old_upper_logp: list[float] = []
    old_lower_logp: list[float] = []
    old_upper_value: list[float] = []
    old_lower_value: list[float] = []
    rewards: list[float] = []
    dones: list[float] = []
    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    targets: list[np.ndarray] = []
    lower_effects: list[np.ndarray] = []
    plan_smoothness: list[float] = []
    plan_coeff_abs: list[float] = []
    promotions = 0
    env.reset()
    for t in range(steps):
        freq = tracker.update_bar(data["predictor"][t], t=float(t * 60.0))
        if bool(dict(freq.get("promotion", {}) or {}).get("promote", False)):
            promotions += 1
        upper_state, lower_state_probe = feature_vectors(dict(freq), env.position.copy())
        upper_out = model.act_upper(upper_state, sample=sample)
        if plan_mapper is None:
            target = latent_target(np.asarray(upper_out["action"], dtype=np.float64))
        else:
            plan = plan_mapper.target(env.position.copy(), np.asarray(upper_out["action"], dtype=np.float64))
            target = gross_cap(plan.target)
            plan_smoothness.append(float(plan.smoothness_penalty))
            plan_coeff_abs.append(float(np.mean(np.abs(plan.coefficients))))
        _, lower_state = feature_vectors(dict(freq), env.position.copy(), target=target)
        lower_out = model.act_lower(lower_state, sample=sample)
        speed = latent_speed(np.asarray(lower_out["action"], dtype=np.float64))
        env.set_target(target)
        _, reward, done, info = env.lower_step({
            "execution_speed": speed,
            "residual_order": np.zeros(assets, dtype=np.float64),
        })
        lower_effect = np.asarray(info["position"], dtype=np.float64) - np.asarray(info["target"], dtype=np.float64)
        leak_info = leakage.update(upper_effect=target, lower_effect=lower_effect, reward=float(reward))
        step_reward = float(leak_info["shaped_reward"] if leak_info["shaped_reward"] is not None else reward)
        upper_states.append(upper_state)
        lower_states.append(lower_state)
        upper_actions.append(np.asarray(upper_out["action"], dtype=np.float32))
        lower_actions.append(np.asarray(lower_out["action"], dtype=np.float32))
        old_upper_logp.append(float(upper_out["logp"]))
        old_lower_logp.append(float(lower_out["logp"]))
        old_upper_value.append(float(upper_out["value"]))
        old_lower_value.append(float(lower_out["value"]))
        rewards.append(step_reward)
        dones.append(float(done))
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        targets.append(np.asarray(info["target"], dtype=np.float64).copy())
        lower_effects.append(lower_effect.copy())
        if done:
            break
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    reg = LeakageRegularizer(upper_hf_window=6, lower_lf_window=24)
    leak = reg.compute(np.asarray(targets, dtype=np.float64), np.asarray(lower_effects, dtype=np.float64)) if targets else {
        "leakage_penalty": 0.0,
        "UpperHFPower": 0.0,
        "LowerLFDrift": 0.0,
    }
    row = {
        "seed": int(seed),
        "scenario": scenario,
        "total_return": float(eq[-1] - 1.0) if eq.size else 0.0,
        "sharpe": float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotions),
        "leakage_penalty": float(leak["leakage_penalty"]),
        "UpperHFPower": float(leak["UpperHFPower"]),
        "LowerLFDrift": float(leak["LowerLFDrift"]),
        "plan_smoothness": float(np.mean(plan_smoothness)) if plan_smoothness else 0.0,
        "plan_coeff_abs": float(np.mean(plan_coeff_abs)) if plan_coeff_abs else 0.0,
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
    )
    return batch, row


def objective(row: dict[str, float]) -> float:
    return float(row["total_return"]) + 0.01 * float(row["sharpe"]) - 0.25 * float(row["max_drawdown"]) - 0.0005 * float(row["turnover"])


def summarize(rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = [
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover",
        "promotion_count",
        "leakage_penalty",
        "UpperHFPower",
        "LowerLFDrift",
        "plan_smoothness",
        "plan_coeff_abs",
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
    leakage_scale: float = 0.0,
    plan_basis_dim: int = 0,
    plan_horizon_s: float = 1800.0,
    plan_eval_offset_s: float = 300.0,
    plan_coefficient_scale: float = 0.75,
) -> tuple[dict[str, Any], list[dict[str, float]], DualActorCriticPPO]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    plan_mapper = make_plan_mapper(
        assets=assets,
        plan_basis_dim=plan_basis_dim,
        plan_horizon_s=plan_horizon_s,
        plan_eval_offset_s=plan_eval_offset_s,
        plan_coefficient_scale=plan_coefficient_scale,
    )
    config = DualPPOConfig(
        upper_state_dim=5 * assets + 1,
        lower_state_dim=3 * assets + 1,
        upper_action_dim=plan_mapper.action_dim if plan_mapper is not None else assets,
        lower_action_dim=assets,
        hidden_dim=0,
        learning_rate=0.003,
        epochs=4,
        minibatch_size=512,
        init_log_std=-2.5,
    )
    model = DualActorCriticPPO(config)
    initialize_frequency_prior(model, assets, plan_basis_dim=plan_basis_dim)
    payload, heldout_rows, model = train_dual_ppo(
        model=model,
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        iterations=iterations,
        rollout_fn=lambda ppo_model, rollout_seed, sample: rollout(
            ppo_model,
            seed=rollout_seed,
            steps=steps,
            assets=assets,
            scenario=scenario,
            sample=sample,
            leakage_scale=leakage_scale if sample else 0.0,
            plan_mapper=plan_mapper,
        ),
        objective_fn=objective,
        summary_fn=summarize,
        policy="ppo_dual_actor_critic",
        trainer="shared_dual_level_ppo",
        domain="trading",
        metadata={
            "scenario": scenario,
            "steps": int(steps),
            "assets": int(assets),
            "leakage_scale": float(leakage_scale),
            "plan_mode": "learned_bernstein" if plan_mapper is not None else "direct_target",
            **(plan_mapper.to_metadata() if plan_mapper is not None else {
                "plan_basis_dim": 0,
                "plan_horizon_s": 0.0,
                "plan_eval_offset_s": 0.0,
                "plan_coefficient_scale": 0.0,
                "plan_action_dim": int(assets),
            }),
        },
    )
    return payload, heldout_rows, model


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# PPO Dual Actor-Critic Trading Validation",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- plan mode: `{payload['plan_mode']}`",
        f"- scenario: `{payload['scenario']}`",
        f"- train seeds: {payload['train_seeds']}",
        f"- eval seeds: {payload['eval_seeds']}",
        f"- return mean: {summary['total_return_mean']:.4f}",
        f"- Sharpe mean: {summary['sharpe_mean']:.3f}",
        f"- max drawdown mean: {summary['max_drawdown_mean']:.4f}",
        f"- turnover mean: {summary['turnover_mean']:.2f}",
        f"- leakage penalty mean: {summary['leakage_penalty_mean']:.4f}",
        f"- LowerLFDrift mean: {summary['LowerLFDrift_mean']:.4f}",
        f"- plan smoothness mean: {summary['plan_smoothness_mean']:.4f}",
        f"- plan coefficient abs mean: {summary['plan_coeff_abs_mean']:.4f}",
        "",
        "This validates the shared upper/lower PPO actor-critic training core. It uses trading as a domain adapter; the trainer itself only depends on upper/lower states, latent actions, rewards, and done flags.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[31415, 27182, 16180])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", choices=SCENARIOS, default="persistent_shift")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--leakage-scale", type=float, default=0.0)
    parser.add_argument("--plan-basis-dim", type=int, default=0)
    parser.add_argument("--plan-horizon-s", type=float, default=1800.0)
    parser.add_argument("--plan-eval-offset-s", type=float, default=300.0)
    parser.add_argument("--plan-coefficient-scale", type=float, default=0.75)
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
        leakage_scale=args.leakage_scale,
        plan_basis_dim=args.plan_basis_dim,
        plan_horizon_s=args.plan_horizon_s,
        plan_eval_offset_s=args.plan_eval_offset_s,
        plan_coefficient_scale=args.plan_coefficient_scale,
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
