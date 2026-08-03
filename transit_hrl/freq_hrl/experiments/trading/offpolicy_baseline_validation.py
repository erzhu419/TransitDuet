"""Flat SAC/TD3 baselines on the shared trading environment and metric contract."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from freq_hrl.core import FrequencyDiagnostics, LeakageRegularizer
from freq_hrl.domains.trading import PortfolioExecutionConfig, PortfolioExecutionEnv
from freq_hrl.experiments.reproducibility import (
    derive_seed,
    training_rollout_seed,
    validate_evaluation_seed_roles,
    validate_unique_seeds,
)
from freq_hrl.rl import FlatOffPolicyActorCritic, OffPolicyConfig, ReplayBuffer

from .metrics import (
    METRIC_CONTRACT_VERSION,
    periods_per_year_from_bar_seconds,
    summarize_pnl_series,
)
from .performance_validation import make_synthetic_market
from .ppo_actor_critic import bounded_speed, gross_cap, make_tracker, objective, summarize


OFFPOLICY_MODES = ("flat_sac", "flat_td3")


def flat_state(
    raw_signal: np.ndarray,
    position: np.ndarray,
    target: np.ndarray,
    *,
    progress: float,
) -> np.ndarray:
    position_arr = np.asarray(position, dtype=np.float64).reshape(-1)
    target_arr = np.asarray(target, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_signal, dtype=np.float64).reshape(-1) / 0.0014
    if raw.size != position_arr.size:
        raw = np.resize(raw, position_arr.size)
    if target_arr.size != position_arr.size:
        target_arr = np.resize(target_arr, position_arr.size)
    return np.concatenate([
        raw,
        np.tanh(raw),
        np.tanh(np.abs(raw)),
        raw * position_arr,
        position_arr,
        target_arr,
        target_arr - position_arr,
        np.asarray([float(np.clip(progress, 0.0, 1.0))]),
    ]).astype(np.float32)


def decode_flat_action(action: np.ndarray, assets: int) -> tuple[np.ndarray, np.ndarray]:
    bounded = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    if bounded.size != 2 * int(assets):
        raise ValueError(f"expected {2 * int(assets)} flat actions, got {bounded.size}")
    target = gross_cap(bounded[:assets])
    speed = bounded_speed(bounded[assets:])
    return target, speed


def run_offpolicy_episode(
    agent: FlatOffPolicyActorCritic,
    *,
    seed: int,
    steps: int,
    assets: int,
    scenario: str,
    policy_mode: str,
    training: bool,
    replay: ReplayBuffer | None = None,
    replay_rng: np.random.Generator | None = None,
    global_step: int = 0,
    warmup_steps: int = 256,
    batch_size: int = 64,
    updates_per_step: int = 1,
) -> tuple[dict[str, Any], int, list[dict[str, float]]]:
    if policy_mode not in OFFPOLICY_MODES:
        raise ValueError(f"unknown off-policy mode: {policy_mode}")
    if training and (replay is None or replay_rng is None):
        raise ValueError("training requires replay and replay_rng")
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
    diagnostics = FrequencyDiagnostics(mi_bins=8)
    pnl_returns: list[float] = []
    equity: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    targets: list[np.ndarray] = []
    lower_effects: list[np.ndarray] = []
    updates: list[dict[str, float]] = []
    current_target = np.zeros(assets, dtype=np.float64)
    env.reset()
    for t in range(int(steps)):
        raw_signal = np.asarray(data["predictor"][t], dtype=np.float64)
        freq = tracker.update_bar(raw_signal, t=float(t * 60.0))
        state = flat_state(
            raw_signal,
            env.position.copy(),
            current_target,
            progress=t / max(int(steps) - 1, 1),
        )
        if training and global_step < int(warmup_steps):
            action = replay_rng.uniform(-1.0, 1.0, size=2 * assets).astype(np.float32)  # type: ignore[union-attr]
        else:
            action = agent.act(state, sample=training)
        target, speed = decode_flat_action(action, assets)
        env.set_target(target)
        _, reward, done, info = env.lower_step({
            "execution_speed": speed,
            "residual_order": np.zeros(assets, dtype=np.float64),
        })
        current_target = np.asarray(info["target"], dtype=np.float64).copy()
        next_index = min(t + 1, int(steps) - 1)
        next_state = flat_state(
            np.asarray(data["predictor"][next_index], dtype=np.float64),
            env.position.copy(),
            current_target,
            progress=(t + 1) / max(int(steps) - 1, 1),
        )
        if training:
            replay.add(state, action, float(reward), next_state, bool(done))  # type: ignore[union-attr]
            global_step += 1
            if global_step >= int(warmup_steps) and replay.size >= int(batch_size):  # type: ignore[union-attr]
                for _ in range(max(1, int(updates_per_step))):
                    batch = replay.sample(  # type: ignore[union-attr]
                        int(batch_size), replay_rng, agent.device  # type: ignore[arg-type]
                    )
                    updates.append(agent.update(batch))
        lower_effect = np.asarray(info["position"], dtype=np.float64) - current_target
        diagnostics.log_step(
            t=float(t * 60.0),
            states={
                "regime_shift": t == int(data["regime_shift_t"][0]),
                "shock": bool(np.any(data["shock_mask"][t])),
                "lower_responded": float(info["turnover"]) > 0.02,
            },
            actions={
                "upper": current_target,
                "lower": np.asarray(info["trade"], dtype=np.float64),
            },
            freq_features=dict(freq),
            effects={"upper": current_target, "lower": lower_effect},
        )
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        costs.append(float(info["transaction_cost"]))
        targets.append(current_target.copy())
        lower_effects.append(lower_effect.copy())
        if done:
            break

    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    financial = summarize_pnl_series(
        pnl,
        eq,
        periods_per_year=periods_per_year_from_bar_seconds(60.0),
    )
    leakage = LeakageRegularizer(
        upper_hf_window=6,
        lower_lf_window=24,
    ).compute(np.asarray(targets), np.asarray(lower_effects))
    diag = diagnostics.summarize_episode()
    row = {
        "baseline": policy_mode,
        "policy_mode": policy_mode,
        "seed": int(seed),
        "scenario": str(scenario),
        **financial,
        "turnover": float(np.sum(turnover)),
        "transaction_cost": float(np.sum(costs)),
        "promotion_count": 0,
        "upper_decision_count": int(len(pnl_returns)),
        "lower_decision_count": int(len(pnl_returns)),
        "upper_mean_duration": 1.0,
        "upper_to_lower_ratio": 1.0,
        "leakage_penalty": float(leakage["leakage_penalty"]),
        "UpperHFPower": float(leakage["UpperHFPower"]),
        "LowerLFDrift": float(leakage["LowerLFDrift"]),
        "LowerLFDriftAbs": float(leakage["LowerLFDriftAbs"]),
        "RawLowerLFDrift": float(leakage["LowerLFDrift"]),
        "RawLowerLFDriftAbs": float(leakage["LowerLFDriftAbs"]),
        "FocusScore": float(diag["FocusScore"]),
        "protocol_valid": 1.0,
        "routing_contract": "raw_history",
        "temporal_contract": "single_level_flat_joint_action",
        "replay_size": int(replay.size) if replay is not None else 0,
        "gradient_updates": int(len(updates)),
    }
    return row, int(global_step), updates


def _mean_update_metrics(
    rows: list[dict[str, float]],
    *,
    algorithm: str,
) -> dict[str, float]:
    if not rows:
        return {
            "critic_loss": 0.0,
            "actor_loss": 0.0,
            "alpha": 0.0,
            "gradient_updates": 0.0,
            "actor_optimizer_steps": 0.0,
            "critic_optimizer_steps": 0.0,
            "temperature_optimizer_steps": 0.0,
        }
    metrics = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ("critic_loss", "actor_loss", "alpha")
    }
    actor_steps = float(sum(float(row["actor_updated"]) for row in rows))
    critic_steps = float(len(rows))
    temperature_steps = float(len(rows) if algorithm == "sac" else 0)
    return {
        **metrics,
        "gradient_updates": actor_steps + critic_steps + temperature_steps,
        "actor_optimizer_steps": actor_steps,
        "critic_optimizer_steps": critic_steps,
        "temperature_optimizer_steps": temperature_steps,
    }


def train_flat_offpolicy_baseline(
    *,
    policy_mode: str,
    train_seeds: list[int],
    eval_seeds: list[int],
    steps: int,
    assets: int,
    scenario: str,
    iterations: int,
    seed: int,
    validation_seeds: list[int] | None = None,
    resample_training_paths: bool = True,
    hidden_dim: int = 64,
    replay_capacity: int = 100_000,
    warmup_steps: int = 256,
    batch_size: int = 64,
    updates_per_step: int = 1,
    objective_fn: Callable[[dict[str, Any]], float] = objective,
) -> tuple[dict[str, Any], list[dict[str, Any]], FlatOffPolicyActorCritic]:
    if policy_mode not in OFFPOLICY_MODES:
        raise ValueError(f"unknown policy_mode: {policy_mode}")
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
    validation_seed_list, heldout_test_seeds = validate_evaluation_seed_roles(
        validation_seed_list, eval_seeds
    )
    algorithm = "sac" if policy_mode == "flat_sac" else "td3"
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    replay_rng = np.random.default_rng(int(seed) + 104729)
    state_dim = 7 * int(assets) + 1
    action_dim = 2 * int(assets)
    config = OffPolicyConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        algorithm=algorithm,
        hidden_dim=int(hidden_dim),
    )
    agent = FlatOffPolicyActorCritic(config)
    replay = ReplayBuffer(
        capacity=int(replay_capacity),
        state_dim=state_dim,
        action_dim=action_dim,
    )
    initial_rows = [
        run_offpolicy_episode(
            agent,
            seed=int(eval_seed),
            steps=int(steps),
            assets=int(assets),
            scenario=scenario,
            policy_mode=policy_mode,
            training=False,
        )[0]
        for eval_seed in validation_seed_list
    ]
    best_score = float(np.mean([objective_fn(row) for row in initial_rows]))
    best_state = copy.deepcopy(agent.state_dict())
    history = [{
        "iteration": -1,
        "score": best_score,
        **summarize(initial_rows),
        "critic_loss": 0.0,
        "actor_loss": 0.0,
        "alpha": float(agent.alpha.item()),
        "gradient_updates": 0.0,
    }]
    global_step = 0
    actor_optimizer_steps = 0
    critic_optimizer_steps = 0
    temperature_optimizer_steps = 0
    for iteration in range(max(1, int(iterations))):
        iteration_updates: list[dict[str, float]] = []
        iteration_train_seeds = [
            training_rollout_seed(
                int(seed), root, iteration, domain=f"trading:{scenario}"
            )
            if resample_training_paths else int(root)
            for root in rollout_seed_roots
        ]
        for train_seed in iteration_train_seeds:
            _, global_step, updates = run_offpolicy_episode(
                agent,
                seed=int(train_seed),
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                policy_mode=policy_mode,
                training=True,
                replay=replay,
                replay_rng=replay_rng,
                global_step=global_step,
                warmup_steps=int(warmup_steps),
                batch_size=int(batch_size),
                updates_per_step=int(updates_per_step),
            )
            iteration_updates.extend(updates)
        actor_optimizer_steps += int(sum(
            float(row["actor_updated"]) for row in iteration_updates
        ))
        critic_optimizer_steps += len(iteration_updates)
        if algorithm == "sac":
            temperature_optimizer_steps += len(iteration_updates)
        eval_rows = [
            run_offpolicy_episode(
                agent,
                seed=int(eval_seed),
                steps=int(steps),
                assets=int(assets),
                scenario=scenario,
                policy_mode=policy_mode,
                training=False,
            )[0]
            for eval_seed in validation_seed_list
        ]
        score = float(np.mean([objective_fn(row) for row in eval_rows]))
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(agent.state_dict())
        history.append({
            "iteration": int(iteration),
            "training_rollout_seeds": iteration_train_seeds,
            "score": score,
            **summarize(eval_rows),
            **_mean_update_metrics(iteration_updates, algorithm=algorithm),
            "replay_size": int(replay.size),
            "environment_steps": int(global_step),
        })
    agent.load_state_dict(best_state)
    heldout_rows = [
        run_offpolicy_episode(
            agent,
            seed=int(eval_seed),
            steps=int(steps),
            assets=int(assets),
            scenario=scenario,
            policy_mode=policy_mode,
            training=False,
        )[0]
        for eval_seed in heldout_test_seeds
    ]
    payload = {
        "policy": policy_mode,
        "trainer": f"flat_{algorithm}_twin_q_v1",
        "domain": "trading",
        "policy_mode": policy_mode,
        "baseline": policy_mode,
        "scenario": scenario,
        "train_seeds": list(rollout_seed_roots),
        "rollout_seed_roots": list(rollout_seed_roots),
        "validation_seeds": list(validation_seed_list),
        "selection_seeds": list(validation_seed_list),
        "eval_seeds": list(heldout_test_seeds),
        "heldout_test_seeds": list(heldout_test_seeds),
        "steps": int(steps),
        "assets": int(assets),
        "iterations": int(iterations),
        "best_score": best_score,
        "config": config.to_dict(),
        "history": history,
        "summary": summarize(heldout_rows),
        "replay_capacity": int(replay_capacity),
        "warmup_steps": int(warmup_steps),
        "batch_size": int(batch_size),
        "updates_per_step": int(updates_per_step),
        "environment_steps_train": int(global_step),
        "environment_steps_validation": int(
            len(validation_seed_list) * int(steps) * (max(1, int(iterations)) + 1)
        ),
        "environment_steps_eval": int(len(heldout_test_seeds) * int(steps)),
        "unique_training_path_count": int(
            len(rollout_seed_roots)
            * (max(1, int(iterations)) if resample_training_paths else 1)
        ),
        "training_replicate_seed": int(seed),
        "training_path_protocol": (
            "fresh_deterministic_path_per_root_and_iteration_v2"
            if resample_training_paths else "fixed_path_reuse_legacy"
        ),
        "checkpoint_selection_protocol": "disjoint_validation_paths",
        "actor_optimizer_steps_train": int(actor_optimizer_steps),
        "critic_optimizer_steps_train": int(critic_optimizer_steps),
        "temperature_optimizer_steps_train": int(temperature_optimizer_steps),
        "gradient_updates_train": int(
            actor_optimizer_steps
            + critic_optimizer_steps
            + temperature_optimizer_steps
        ),
        "observation_contract": (
            "raw signal, tanh/absolute transforms, raw-position interaction, "
            "position, target, gap, and progress"
        ),
        "action_contract": "joint target weights and execution speeds every primitive step",
        "metric_contract_version": METRIC_CONTRACT_VERSION,
    }
    return payload, heldout_rows, agent


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    rows: list[dict[str, Any]],
    agent: FlatOffPolicyActorCritic,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "per_seed.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"model": payload, "per_seed": rows}, f, indent=2)
    torch.save({
        "model_state_dict": agent.state_dict(),
        "config": agent.config.to_dict(),
        "policy_mode": payload["policy_mode"],
        "metric_contract_version": payload["metric_contract_version"],
    }, output_dir / "checkpoint.pt")
    summary = payload["summary"]
    lines = [
        "# Flat Off-Policy Learned Baseline",
        "",
        f"- trainer: `{payload['trainer']}`",
        f"- policy mode: `{payload['policy_mode']}`",
        f"- scenario: `{payload['scenario']}`",
        f"- train environment steps: `{payload['environment_steps_train']}`",
        f"- gradient updates: `{payload['gradient_updates_train']}`",
        f"- metric contract: `{payload['metric_contract_version']}`",
        f"- return mean: `{summary.get('total_return_mean', float('nan')):.6f}`",
        f"- episode information ratio mean: "
        f"`{summary.get('episode_information_ratio_mean', float('nan')):.6f}`",
        "",
        "This is a raw-observation, single-level joint-action baseline. Frequency "
        "diagnostics are computed after evaluation and are never policy inputs.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-mode", choices=OFFPOLICY_MODES, default="flat_sac")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[31415, 27182, 16180])
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--scenario", default="persistent_shift")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--optimizer-seed", type=int, default=2026)
    parser.add_argument("--reuse-fixed-training-paths", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/trading_offpolicy_baseline"),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    payload, rows, agent = train_flat_offpolicy_baseline(
        policy_mode=args.policy_mode,
        train_seeds=list(args.train_seeds),
        eval_seeds=list(args.eval_seeds),
        steps=int(args.steps),
        assets=int(args.assets),
        scenario=str(args.scenario),
        iterations=int(args.iterations),
        seed=int(args.optimizer_seed),
        validation_seeds=(
            None if args.validation_seeds is None else list(args.validation_seeds)
        ),
        resample_training_paths=not args.reuse_fixed_training_paths,
        hidden_dim=int(args.hidden_dim),
        replay_capacity=int(args.replay_capacity),
        warmup_steps=int(args.warmup_steps),
        batch_size=int(args.batch_size),
        updates_per_step=int(args.updates_per_step),
    )
    write_outputs(args.output_dir, payload, rows, agent)
    print(
        f"offpolicy_baseline mode={args.policy_mode} rows={len(rows)} "
        f"updates={payload['gradient_updates_train']} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
