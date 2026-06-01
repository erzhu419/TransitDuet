"""Native Transit wait-credit validation with the shared PPO loop."""

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
        "promotion": {"enable": False},
    },
    "upper": {
        "timetable_planner": {
            "promotion_replan": False,
            "action_ema_alpha": 1.0,
        },
    },
}

VARIANTS: dict[str, dict[str, Any]] = {
    "no_wait_credit": {
        "reward_attribution": {"enable": False},
    },
    "native_wait_credit": {
        "reward_attribution": {
            "enable": True,
            "upper_wait_weight": 0.45,
            "lower_wait_weight": 0.20,
            "lower_board_credit_weight": 0.08,
            "lower_share_source": "local_low",
            "lower_positive_high_only": True,
            "lower_high_share_cap": 0.75,
            "wait_norm_s": 600.0,
            "wait_clip": 2.0,
            "low_share_floor": 0.05,
            "normalize_upper_credit": True,
        },
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


def _score(row: dict[str, Any]) -> float:
    return -float(row.get("avg_wait_min", 0.0)) - 2.0 * float(row.get("headway_cv", 0.0))


def _row_from_payload(seed: int, variant: str, payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    summary = payload.get("summary", {})
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    return {
        "seed": int(seed),
        "variant": str(variant),
        "status": payload.get("status", "missing"),
        "episodes": int(payload.get("episodes", len(rows))),
        "avg_wait_min_mean": float(summary.get("avg_wait_min_mean", 0.0)),
        "headway_cv_mean": float(summary.get("headway_cv_mean", 0.0)),
        "ep_reward_mean": float(summary.get("ep_reward_mean", 0.0)),
        "score_mean": float(summary.get("score_mean", 0.0)),
        "final_avg_wait_min": float(last.get("avg_wait_min", 0.0)),
        "final_headway_cv": float(last.get("headway_cv", 0.0)),
        "final_ep_reward": float(last.get("ep_reward", 0.0)),
        "final_score": _score(last),
        "wait_improvement": float(first.get("avg_wait_min", 0.0)) - float(last.get("avg_wait_min", 0.0)),
        "reward_improvement": float(last.get("ep_reward", 0.0)) - float(first.get("ep_reward", 0.0)),
        "freq_wait_lower_net_mean": float(summary.get("freq_wait_lower_net_mean_mean", 0.0)),
        "freq_wait_upper_credit_mean": float(summary.get("freq_wait_upper_credit_mean_mean", 0.0)),
        "freq_wait_upper_credit_std": float(summary.get("freq_wait_upper_credit_std_mean", 0.0)),
        "freq_wait_low_share_mean": float(summary.get("freq_wait_low_share_mean_mean", 0.0)),
        "freq_wait_lower_high_share_mean": float(summary.get("freq_wait_lower_high_share_mean_mean", 0.0)),
        "freq_wait_boarded_pax": float(summary.get("freq_wait_boarded_pax_mean", 0.0)),
        "shared_ppo_lower_samples": float(last.get("shared_ppo_lower_samples", 0.0)),
    }


def paired_checks(rows: list[dict[str, Any]], min_pairs: int = 5) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metrics = [
        ("final_avg_wait_min", True),
        ("avg_wait_min_mean", True),
        ("final_ep_reward", False),
        ("final_score", False),
        ("wait_improvement", False),
        ("freq_wait_upper_credit_std", False),
    ]
    for metric, lower_is_better in metrics:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric=metric,
            treatment="native_wait_credit",
            control="no_wait_credit",
            lower_is_better=lower_is_better,
        )
        checks.append({
            "check": f"native_wait_credit_vs_no_wait_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=int(min_pairs)),
        })
    return checks


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows)}
    for variant in VARIANTS:
        vrows = [row for row in rows if row["variant"] == variant]
        for metric in [
            "final_avg_wait_min",
            "avg_wait_min_mean",
            "final_ep_reward",
            "final_score",
            "wait_improvement",
            "freq_wait_upper_credit_std",
            "freq_wait_boarded_pax",
        ]:
            values = np.asarray([float(row[metric]) for row in vrows], dtype=np.float64)
            summary[f"{variant}_{metric}_mean"] = float(np.mean(values)) if values.size else 0.0
    return summary


def run_validation(
    output_dir: Path,
    config_path: Path,
    seeds: list[int],
    episodes: int,
    device: str,
    learning_rate: float = 1e-3,
    min_pairs: int = 5,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for variant, overrides in VARIANTS.items():
        payloads[variant] = {}
        for seed in seeds:
            payload = run_native_shared_ppo_episode_loop(
                output_dir=output_dir / variant / f"seed_{int(seed)}",
                config_path=config_path,
                seed=int(seed),
                episodes=int(episodes),
                device=str(device),
                learning_rate=float(learning_rate),
                config_overrides=_variant_overrides(overrides),
            )
            payloads[variant][str(seed)] = {
                "summary": payload.get("summary", {}),
                "status": payload.get("status", "missing"),
                "rows": payload.get("rows", []),
            }
            rows.append(_row_from_payload(int(seed), variant, payload))
    checks = paired_checks(rows, min_pairs=int(min_pairs))
    payload = {
        "config_path": str(config_path),
        "seeds": [int(seed) for seed in seeds],
        "episodes": int(episodes),
        "learning_rate": float(learning_rate),
        "min_pairs": int(min_pairs),
        "variants": list(VARIANTS.keys()),
        "summary": summarize(rows),
        "rows": rows,
        "paired_checks": checks,
        "payloads": payloads,
    }
    write_outputs(output_dir, payload)
    return payload


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
        "# Native Transit Wait-Credit Validation",
        "",
        "This compares the native shared-PPO episode loop with and without frequency-attributed passenger wait credit.",
        "",
        "| variant | seed | final wait | mean wait | final reward | final score | wait improvement | upper credit std | pax | samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} "
            f"| {int(row['seed'])} "
            f"| {row['final_avg_wait_min']:.4f} "
            f"| {row['avg_wait_min_mean']:.4f} "
            f"| {row['final_ep_reward']:.3f} "
            f"| {row['final_score']:.4f} "
            f"| {row['wait_improvement']:+.4f} "
            f"| {row['freq_wait_upper_credit_std']:.4f} "
            f"| {row['freq_wait_boarded_pax']:.1f} "
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[31, 41, 51, 61, 71])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-pairs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/transit_native_wait_credit"),
    )
    args = parser.parse_args()
    payload = run_validation(
        output_dir=args.output_dir,
        config_path=args.config,
        seeds=list(args.seeds),
        episodes=int(args.episodes),
        device=str(args.device),
        learning_rate=float(args.learning_rate),
        min_pairs=int(args.min_pairs),
    )
    wait_check = next(
        row for row in payload["paired_checks"]
        if row["check"] == "native_wait_credit_vs_no_wait_final_avg_wait_min"
    )
    print(f"wrote {args.output_dir}")
    print(
        "native_wait_credit "
        f"final_wait_delta={wait_check['delta_mean']:+.4f} "
        f"status={wait_check['status']}"
    )


if __name__ == "__main__":
    main()
