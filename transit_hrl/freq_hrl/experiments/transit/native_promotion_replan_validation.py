"""Native Transit promotion-replan validation with shared PPO loop."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.statistics import claim_status, paired_delta_stats
from freq_hrl.experiments.transit.native_shared_ppo import (
    TRANSIT_DUET_ROOT,
    run_native_shared_ppo_episode_loop,
)


COMMON_OVERRIDES: dict[str, Any] = {
    "frequency": {
        "promotion": {
            "enable": True,
            "state_features": True,
            "residual_threshold": 0.60,
            "persistence_ratio": 0.35,
            "cooldown_min": 10.0,
            "adapt_low": True,
            "adapt_gain": 0.08,
            "adapt_strength_min": 0.20,
            "adapt_local": True,
        },
    },
    "upper": {
        "timetable_planner": {
            "action_ema_alpha": 1.0,
            "replan_interval_s": 1200.0,
            "promotion_replan_strength_min": 0.80,
        },
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "interval_only": {
        "upper": {"timetable_planner": {"promotion_replan": False}},
    },
    "native_promotion_replan": {
        "upper": {"timetable_planner": {"promotion_replan": True}},
    },
}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _variant_overrides(override: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(COMMON_OVERRIDES))
    return _merge_dict(merged, dict(override))


def _row_from_payload(seed: int, variant: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    rows = payload.get("rows", [])
    last = rows[-1] if rows else {}
    return {
        "seed": int(seed),
        "variant": str(variant),
        "status": payload.get("status", "missing"),
        "ep_reward": float(summary.get("ep_reward_mean", last.get("ep_reward", 0.0))),
        "avg_wait_min": float(summary.get("avg_wait_min_mean", last.get("avg_wait_min", 0.0))),
        "headway_cv": float(summary.get("headway_cv_mean", last.get("headway_cv", 0.0))),
        "score": float(summary.get("score_mean", 0.0)),
        "upper_plan_decisions": float(summary.get("upper_plan_decisions_mean", 0.0)),
        "upper_plan_reuse_ratio": float(summary.get("upper_plan_reuse_ratio_mean", 0.0)),
        "freq_promotion_strength": float(summary.get("freq_promotion_strength_mean", 0.0)),
        "shared_ppo_lower_samples": float(last.get("shared_ppo_lower_samples", 0.0)),
        "shared_ppo_loss": float(last.get("shared_ppo_loss", 0.0)),
    }


def paired_checks(rows: list[dict[str, Any]], min_pairs: int = 5) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for metric, lower_is_better in [
        ("ep_reward", False),
        ("avg_wait_min", True),
        ("score", False),
        ("upper_plan_decisions", False),
    ]:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric=metric,
            treatment="native_promotion_replan",
            control="interval_only",
            lower_is_better=lower_is_better,
        )
        checks.append({
            "check": f"native_promotion_replan_vs_interval_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=int(min_pairs)),
        })
    return checks


def run_validation(
    output_dir: Path,
    config_path: Path,
    seeds: list[int],
    episodes: int,
    device: str,
    min_pairs: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for variant, overrides in VARIANTS.items():
        payloads[variant] = {}
        for seed in seeds:
            run_dir = output_dir / variant / f"seed_{int(seed)}"
            payload = run_native_shared_ppo_episode_loop(
                output_dir=run_dir,
                config_path=config_path,
                seed=int(seed),
                episodes=int(episodes),
                device=str(device),
                config_overrides=_variant_overrides(overrides),
            )
            payloads[variant][str(seed)] = {
                "summary": payload.get("summary", {}),
                "status": payload.get("status", "missing"),
                "rows": payload.get("rows", []),
            }
            rows.append(_row_from_payload(int(seed), variant, payload))
    checks = paired_checks(rows, min_pairs=int(min_pairs))
    summary = summarize(rows)
    payload = {
        "config_path": str(config_path),
        "seeds": [int(seed) for seed in seeds],
        "episodes": int(episodes),
        "min_pairs": int(min_pairs),
        "variants": list(VARIANTS.keys()),
        "summary": summary,
        "rows": rows,
        "paired_checks": checks,
        "payloads": payloads,
    }
    write_outputs(output_dir, payload)
    return payload


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for variant in VARIANTS:
        vrows = [row for row in rows if row["variant"] == variant]
        for metric in ["ep_reward", "avg_wait_min", "headway_cv", "score", "upper_plan_decisions"]:
            values = np.asarray([float(row[metric]) for row in vrows], dtype=np.float64)
            summary[f"{variant}_{metric}_mean"] = float(np.mean(values)) if values.size else 0.0
    return summary


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    rows = payload["rows"]
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    checks = payload["paired_checks"]
    if checks:
        with (output_dir / "paired_checks.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(checks)
    write_report(output_dir / "report.md", payload)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Native Transit Promotion Replan Validation",
        "",
        "This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.",
        "",
        "| variant | seed | reward | wait | cv | score | upper decisions | promotion strength | samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} "
            f"| {int(row['seed'])} "
            f"| {row['ep_reward']:.3f} "
            f"| {row['avg_wait_min']:.4f} "
            f"| {row['headway_cv']:.4f} "
            f"| {row['score']:.4f} "
            f"| {row['upper_plan_decisions']:.1f} "
            f"| {row['freq_promotion_strength']:.4f} "
            f"| {row['shared_ppo_lower_samples']:.0f} |"
        )
    lines.extend([
        "",
        "| check | status | metric | n | delta | CI95 low | CI95 high | win rate |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["paired_checks"]:
        lines.append(
            f"| {row['check']} "
            f"| {row['status']} "
            f"| {row['metric']} "
            f"| {row['n_common']} "
            f"| {row['delta_mean']:+.4f} "
            f"| {row['delta_ci95_low']:+.4f} "
            f"| {row['delta_ci95_high']:+.4f} "
            f"| {row['win_rate']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=TRANSIT_DUET_ROOT / "configs_freqduet" / "T_freqhrl_native_full.yaml",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 51, 61, 71, 81, 91, 101])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_native_promotion_replan"),
    )
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        config_path=args.config,
        seeds=list(args.seeds),
        episodes=int(args.episodes),
        device=str(args.device),
        min_pairs=int(args.min_pairs),
    )
    reward_check = next(
        row for row in payload["paired_checks"]
        if row["check"] == "native_promotion_replan_vs_interval_ep_reward"
    )
    print(f"wrote {args.output_dir}")
    print(
        "native_promotion_replan "
        f"reward_delta={reward_check['delta_mean']:+.4f} "
        f"status={reward_check['status']}"
    )


if __name__ == "__main__":
    main()
