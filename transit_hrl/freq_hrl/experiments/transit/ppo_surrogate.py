"""Transit-domain adapter for the shared dual PPO Freq-HRL trainer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from freq_hrl.core import CausalLeakageRewardShaper, LeakageRegularizer
from freq_hrl.domains.transit import TransitFrequencyTracker
from freq_hrl.rl import DualActorCriticPPO, DualPPOConfig, TrajectoryBatch, summarize_numeric_rows, train_dual_ppo

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


def make_tracker() -> TransitFrequencyTracker:
    return TransitFrequencyTracker(
        update_interval_s=60.0,
        bin_sec=60.0,
        method="ema",
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
        promotion_residual_threshold=1.5,
        promotion_persistence_ratio=0.35,
        promotion_cooldown_s=20 * 60.0,
    )


def feature_vectors(
    tracker: TransitFrequencyTracker,
    service_gap: np.ndarray,
    target_delta_s: np.ndarray | None = None,
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
    return upper.astype(np.float32), lower.astype(np.float32)


def latent_headway_delta(latent: np.ndarray, max_delta_s: float = 30.0) -> np.ndarray:
    return np.tanh(np.asarray(latent, dtype=np.float64)) * float(max_delta_s)


def latent_hold(latent: np.ndarray, max_hold_s: float = 45.0) -> np.ndarray:
    return float(max_hold_s) / (1.0 + np.exp(-np.asarray(latent, dtype=np.float64)))


def initialize_transit_prior(model: DualActorCriticPPO, corridors: int) -> None:
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
            upper_linear.weight[i, 0] = -0.75
            upper_linear.weight[i, 2] = -0.35
            lower_linear.bias[i] = -1.6


def rollout(
    model: DualActorCriticPPO,
    seed: int,
    steps: int,
    corridors: int,
    scenario: str,
    sample: bool,
    leakage_scale: float = 0.0,
) -> tuple[TrajectoryBatch | None, dict[str, Any]]:
    demand = make_synthetic_transit_demand(seed, steps, corridors, scenario)
    tracker = make_tracker()
    leakage = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=5, lower_lf_window=20),
        reward_penalty_scale=leakage_scale,
        enabled=leakage_scale > 0.0,
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
    rewards: list[float] = []
    dones: list[float] = []
    wait_proxy: list[float] = []
    cv_proxy: list[float] = []
    hold_trace: list[np.ndarray] = []
    target_trace: list[np.ndarray] = []
    promotions = 0
    for t in range(int(steps)):
        arrivals = {(i, True): float(demand[t, i]) for i in range(corridors)}
        tracker.update(arrivals)
        if tracker.summary()["freq_promotion_flag"] > 0.0:
            promotions += 1
        upper_state, _ = feature_vectors(tracker, service_gap)
        upper_out = model.act_upper(upper_state, sample=sample)
        target_delta = latent_headway_delta(np.asarray(upper_out["action"], dtype=np.float64))
        _, lower_state = feature_vectors(tracker, service_gap, target_delta)
        lower_out = model.act_lower(lower_state, sample=sample)
        hold_s = latent_hold(np.asarray(lower_out["action"], dtype=np.float64))

        demand_ema = 0.97 * demand_ema + 0.03 * demand[t]
        crowding = demand[t] - demand_ema
        service_gap = 0.82 * service_gap + 2.4 * crowding + 0.28 * target_delta + 0.14 * hold_s
        cv_gap = 0.75 * cv_gap + 0.20 * (service_gap - float(np.mean(service_gap))) - 0.06 * (hold_s - float(np.mean(hold_s)))
        wait = 4.0 + 0.018 * float(np.mean(demand[t])) + 0.012 * float(np.mean(np.maximum(service_gap, 0.0))) + 0.006 * float(np.mean(hold_s))
        cv = float(np.std(service_gap) / 120.0 + np.std(cv_gap) / 180.0)
        overshoot = float(np.mean(np.maximum(np.abs(target_delta) - 24.0, 0.0)) / 24.0)
        reward = -(wait + 3.0 * cv + 0.18 * overshoot + 0.004 * float(np.mean(hold_s)))
        leak_info = leakage.update(
            upper_effect=target_delta / 30.0,
            lower_effect=hold_s / 45.0,
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
        rewards.append(step_reward)
        dones.append(float(done))
        wait_proxy.append(wait)
        cv_proxy.append(cv)
        hold_trace.append(hold_s.copy() / 45.0)
        target_trace.append(target_delta.copy() / 30.0)
    reg = LeakageRegularizer(upper_hf_window=5, lower_lf_window=20)
    leak = reg.compute(np.asarray(target_trace, dtype=np.float64), np.asarray(hold_trace, dtype=np.float64))
    row = {
        "seed": int(seed),
        "scenario": scenario,
        "total_reward": float(np.sum(rewards)),
        "reward_mean": float(np.mean(rewards)),
        "wait_proxy": float(np.mean(wait_proxy)),
        "headway_cv": float(np.mean(cv_proxy)),
        "hold_mean": float(np.mean(np.asarray(hold_trace) * 45.0)),
        "promotion_count": int(promotions),
        "leakage_penalty": float(leak["leakage_penalty"]),
        "UpperHFPower": float(leak["UpperHFPower"]),
        "LowerLFDrift": float(leak["LowerLFDrift"]),
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
) -> tuple[dict[str, Any], list[dict[str, Any]], DualActorCriticPPO]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    probe = make_tracker()
    upper_dim = int(probe.upper_features("low_mid").size + corridors + 1)
    lower_dim = int(corridors * probe.lower_features(0, True, "high_mid").size + 2 * corridors + 1)
    config = DualPPOConfig(
        upper_state_dim=upper_dim,
        lower_state_dim=lower_dim,
        upper_action_dim=corridors,
        lower_action_dim=corridors,
        hidden_dim=0,
        learning_rate=0.002,
        epochs=3,
        minibatch_size=256,
        init_log_std=-2.0,
    )
    model = DualActorCriticPPO(config)
    initialize_transit_prior(model, corridors)
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
        ),
        objective_fn=objective,
        summary_fn=summarize,
        policy="ppo_dual_actor_critic",
        trainer="shared_dual_level_ppo",
        domain="transit_surrogate",
        metadata={
            "scenario": scenario,
            "steps": int(steps),
            "corridors": int(corridors),
            "leakage_scale": float(leakage_scale),
        },
    )


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
        "# Transit Surrogate PPO Validation",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- domain: `{payload['domain']}`",
        f"- scenario: `{payload['scenario']}`",
        f"- train seeds: {payload['train_seeds']}",
        f"- eval seeds: {payload['eval_seeds']}",
        f"- reward mean: {summary['reward_mean_mean']:.4f}",
        f"- wait proxy mean: {summary['wait_proxy_mean']:.4f}",
        f"- headway CV mean: {summary['headway_cv_mean']:.4f}",
        f"- hold mean: {summary['hold_mean_mean']:.2f}",
        f"- leakage penalty mean: {summary['leakage_penalty_mean']:.4f}",
        f"- LowerLFDrift mean: {summary['LowerLFDrift_mean']:.4f}",
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
