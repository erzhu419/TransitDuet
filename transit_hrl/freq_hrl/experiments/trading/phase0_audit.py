"""Phase-0 logging-only audit for the trading validation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    LeakageRegularizer,
    Phase0TraceLogger,
    load_phase0_records,
    validate_phase0_record_schema,
)
from freq_hrl.domains.trading import (
    PortfolioExecutionConfig,
    PortfolioExecutionEnv,
    TradingFrequencyTracker,
)

from .performance_validation import make_synthetic_market, policy_action


def make_tracker(n_assets: int) -> TradingFrequencyTracker:
    return TradingFrequencyTracker(
        bar_sec=60.0,
        method="ema",
        low_period_s=120 * 60.0,
        fast_period_s=5 * 60.0,
        mid_period_s=30 * 60.0,
        energy_period_s=10 * 60.0,
        persistence_period_s=30 * 60.0,
        persistence_threshold=0.0010,
        feature_norm=np.ones(n_assets) * 0.0015,
        promotion_enable=True,
        promotion_window_s=30 * 60.0,
        promotion_residual_threshold=0.00035,
        promotion_persistence_ratio=0.40,
        promotion_cooldown_s=60 * 60.0,
        promotion_adapt_low=True,
        promotion_adapt_gain=0.25,
    )


def _arr(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def run_phase0_log(seed: int, steps: int, assets: int, path: Path) -> dict[str, Any]:
    data = make_synthetic_market(seed=seed, steps=steps, n_assets=assets)
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
    leakage_shaper = CausalLeakageRewardShaper(
        regularizer=LeakageRegularizer(upper_hf_window=6, lower_lf_window=24),
        reward_penalty_scale=0.00005,
        enabled=True,
    )
    raw_history: list[np.ndarray] = []
    env.reset()
    with Phase0TraceLogger(path) as logger:
        for t in range(steps):
            raw_signal = data["predictor"][t]
            timestamp = float(t * 60.0)
            z_t = tracker.update_bar(raw_signal, t=timestamp)
            target, lower_action, _ = policy_action(
                "freq_hrl",
                tracker,
                raw_signal,
                raw_history,
                env.position.copy(),
                promotion_mid_gain=0.5,
            )
            raw_history.append(raw_signal.copy())
            env.set_target(target)
            _, reward, done, info = env.lower_step(lower_action)
            lower_effect = (
                np.asarray(info["position"], dtype=np.float64)
                - np.asarray(info["target"], dtype=np.float64)
            )
            leak_info = leakage_shaper.update(
                upper_effect=target,
                lower_effect=lower_effect,
                reward=float(reward),
            )
            logger.write({
                "t": int(t),
                "domain": "trading",
                "entity_id": "market",
                "x_raw": raw_signal,
                "x_bin": {
                    "timestamp": timestamp,
                    "entity_id": "market",
                    "x_raw": raw_signal,
                },
                "z_t": z_t,
                "a_U": target,
                "a_L": lower_action,
                "plan_curve": {
                    "type": "target_weights",
                    "target": target,
                },
                "action_effects": {
                    "upper": target,
                    "lower": lower_effect,
                },
                "reward": {
                    "task_reward": float(reward),
                    "shaped_reward": float(leak_info["shaped_reward"]),
                    "leakage_cost": float(leak_info["leakage_reward_penalty"]),
                    "portfolio_return": float(info["portfolio_return"]),
                    "transaction_cost": float(info["transaction_cost"]),
                },
                "metadata": {
                    "seed": int(seed),
                    "baseline": "freq_hrl",
                    "bar_sec": 60.0,
                },
            })
            if done:
                break
    return {"records": steps, "path": str(path)}


def reconstruct_from_bins(path: Path, assets: int) -> dict[str, Any]:
    records = load_phase0_records(path)
    tracker = make_tracker(assets)
    max_abs_error = 0.0
    checked = 0
    keys = ["x_low", "x_mid", "x_high", "x_high_energy", "x_high_persistence"]
    for record in records:
        validate_phase0_record_schema(record)
        x_bin = record["x_bin"]
        feats = tracker.update_bar(x_bin["x_raw"], t=float(x_bin["timestamp"]))
        z_t = record["z_t"]
        for key in keys:
            logged = _arr(z_t.get(key, []))
            replayed = _arr(feats.get(key, []))
            n = min(logged.size, replayed.size)
            if n:
                max_abs_error = max(
                    max_abs_error,
                    float(np.max(np.abs(logged[:n] - replayed[:n]))),
                )
                checked += 1
    return {
        "records": len(records),
        "checked_arrays": checked,
        "max_abs_reconstruction_error": max_abs_error,
        "passed": bool(max_abs_error <= 1e-12 and checked > 0),
    }


def write_report(path: Path, log_path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Trading Phase-0 Logging Audit",
        "",
        f"- log: `{log_path}`",
        f"- records: {result['records']}",
        f"- checked arrays: {result['checked_arrays']}",
        f"- max abs reconstruction error: {result['max_abs_reconstruction_error']:.3e}",
        f"- passed: {result['passed']}",
        "",
        "The audit records `x_raw`, causal `x_bin`, frequency state `z_t`, upper action `a_U`, lower action `a_L`, plan target, action effects, rewards, and entity IDs. The reconstruction check replays only logged `x_bin` values through the tracker and compares the reconstructed frequency state with logged `z_t`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--assets", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("transit_hrl/results/trading_phase0_audit"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "phase0_trace.jsonl"
    run_phase0_log(args.seed, args.steps, args.assets, log_path)
    result = reconstruct_from_bins(log_path, args.assets)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    write_report(args.output_dir / "report.md", log_path, result)
    print(f"wrote {args.output_dir}")
    print(
        "phase0_reconstruction "
        f"passed={result['passed']} "
        f"max_abs_error={result['max_abs_reconstruction_error']:.3e}"
    )


if __name__ == "__main__":
    main()
