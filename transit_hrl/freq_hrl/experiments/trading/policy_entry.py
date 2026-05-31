"""Train/evaluate entry point for pluggable trading policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingFrequencyTracker,
)
from freq_hrl.policies import (
    FrequencyTradingController,
    FrequencyTradingPlanner,
    LinearFrequencyTradingController,
    LinearFrequencyTradingPlanner,
    LinearTradingParams,
)

from .performance_validation import SCENARIOS, make_synthetic_market, max_drawdown


def make_policy(policy: str, params: LinearTradingParams | None = None) -> tuple[Any, Any]:
    if policy == "heuristic":
        return FrequencyTradingPlanner(promotion_mid_gain=0.5), FrequencyTradingController()
    if policy == "linear":
        params = params or LinearTradingParams()
        return LinearFrequencyTradingPlanner(params), LinearFrequencyTradingController(params)
    raise ValueError(f"unknown policy: {policy}")


def run_eval(
    seed: int,
    steps: int,
    assets: int,
    policy: str = "heuristic",
    params: LinearTradingParams | None = None,
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
        promotion_persistence_ratio=0.40,
        promotion_cooldown_s=60 * 60.0,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.25,
    )
    planner, controller = make_policy(policy, params)
    env.reset()
    pnl_returns = []
    equity = []
    turnover = []
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
        pnl_returns.append(float(info["portfolio_return"] - info["transaction_cost"]))
        equity.append(float(info["equity"]))
        turnover.append(float(info["turnover"]))
        if done:
            break
    pnl = np.asarray(pnl_returns, dtype=np.float64)
    eq = np.asarray(equity, dtype=np.float64)
    total_return = float(eq[-1] - 1.0) if eq.size else 0.0
    sharpe = float(np.sqrt(max(pnl.size, 1)) * pnl.mean() / (pnl.std() + 1e-12)) if pnl.size else 0.0
    return {
        "seed": int(seed),
        "scenario": scenario,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(eq),
        "turnover": float(np.sum(turnover)),
        "promotion_count": int(promotion_count),
    }


def objective(row: dict[str, float]) -> float:
    return (
        float(row["total_return"])
        + 0.01 * float(row["sharpe"])
        - 0.25 * float(row["max_drawdown"])
        - 0.0005 * float(row["turnover"])
    )


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


def summarize(rows: list[dict[str, float]], mode: str, policy: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "policy": policy,
        "n": len(rows),
        "total_return_mean": float(np.mean([r["total_return"] for r in rows])),
        "sharpe_mean": float(np.mean([r["sharpe"] for r in rows])),
        "max_drawdown_mean": float(np.mean([r["max_drawdown"] for r in rows])),
        "turnover_mean": float(np.mean([r["turnover"] for r in rows])),
        "promotion_count_mean": float(np.mean([r["promotion_count"] for r in rows])),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_params(path: Path | None) -> LinearTradingParams:
    if path is None:
        return LinearTradingParams()
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return LinearTradingParams.from_mapping(payload.get("params", payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train"], default="eval")
    parser.add_argument("--policy", choices=["heuristic", "linear"], default="heuristic")
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
        with model_path.open("w", encoding="utf-8") as f:
            json.dump(model_payload, f, indent=2)
        params = LinearTradingParams.from_mapping(model_payload["params"])
    elif policy == "linear":
        params = load_params(args.model_path)

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
        "",
        "The `linear` policy is trained by cross-entropy policy search over shared frequency-routing coefficients. It is a lightweight learned-policy validation path for the Freq-HRL protocol, not a full SAC/PPO implementation.",
    ]
    if model_payload is not None:
        report.extend([
            "",
            "## Learned Parameters",
            "",
            "| parameter | value |",
            "|---|---:|",
        ])
        for key, value in model_payload["params"].items():
            report.append(f"| {key} | {float(value):+.4f} |")
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}")
    print(f"policy_entry mode={args.mode} policy={policy} sharpe={summary['sharpe_mean']:.3f} return={summary['total_return_mean']:.4f}")


if __name__ == "__main__":
    main()
