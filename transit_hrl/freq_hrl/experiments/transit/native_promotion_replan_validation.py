"""Native Transit promotion-replan validation with shared PPO loop."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from freq_hrl.experiments.statistics import (
    claim_status,
    noninferiority_status,
    paired_delta_stats,
)
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
            "terminal_shift_min_s": -45.0,
            "terminal_shift_max_s": 45.0,
        },
    },
}
DEFAULT_LOWER_HF_WAIT_ACTION_GAIN_S = 45.0

VARIANTS: dict[str, dict[str, Any]] = {
    "interval_only": {
        "upper": {"timetable_planner": {"promotion_replan": False}},
    },
    "native_promotion_replan": {
        "upper": {"timetable_planner": {"promotion_replan": True}},
    },
    "native_learned_gate": {
        "_learned_promotion_gate": True,
        "_promotion_gate_threshold": 0.92,
        "_promotion_gate_strength_min": 0.95,
        "_promotion_gate_age_min": 1.0,
        "_promotion_gate_min_elapsed_s": 900.0,
        "_promotion_gate_cooldown_s": 900.0,
        "_promotion_gate_preselect_action": True,
        "_promotion_gate_plan_blend": 0.0,
        "_promotion_gate_low_signal_min": 0.10,
        "_promotion_gate_max_hf_to_lf_ratio": 8.0,
        "_promotion_gate_max_replans": 1,
        "upper": {"timetable_planner": {"promotion_replan": False}},
    },
    "native_wait_aware_replan": {
        "_learned_promotion_gate": True,
        "_promotion_gate_threshold": 0.92,
        "_promotion_gate_strength_min": 0.95,
        "_promotion_gate_age_min": 1.0,
        "_promotion_gate_min_elapsed_s": 900.0,
        "_promotion_gate_cooldown_s": 900.0,
        "_promotion_gate_preselect_action": True,
        "_promotion_gate_plan_blend": 0.0,
        "_promotion_gate_low_signal_min": 0.10,
        "_promotion_gate_max_hf_to_lf_ratio": 8.0,
        "_promotion_gate_max_replans": 1,
        "_promotion_replan_policy": "learned_wait_aware",
        "_promotion_replan_wait_gain_s": 8.0,
        "_promotion_replan_max_shift_s": 2.0,
        "_promotion_replan_state_wait_weight": 0.85,
        "_promotion_replan_frequency_weight": 0.15,
        "_promotion_replan_min_pressure": 0.25,
        "_promotion_replan_require_shift": True,
        "_promotion_replan_hold_guard_weight": 0.85,
        "_promotion_replan_same_wait_min": 0.70,
        "_promotion_replan_gap_guard_min_ratio": 0.998,
        "_promotion_replan_gap_guard_max_ratio": 1.30,
        "_promotion_replan_base_action": "actor",
        "_promotion_replan_actor_base_trust_s": 2.0,
        "frequency": {
            "hold_feedback": {
                "enable": True,
                "window": 512,
                "wait_norm_s": 600.0,
                "wait_clip": 2.0,
                "board_norm": 8.0,
                "high_threshold": 0.0,
            },
        },
        "upper": {
            "timetable_planner": {
                "promotion_replan": False,
            },
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
    return _merge_dict(merged, {
        key: value for key, value in dict(override).items()
        if not str(key).startswith("_")
    })


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
        "upper_plan_target_mean": float(summary.get("upper_plan_target_mean_mean", last.get("upper_plan_target_mean", 0.0))),
        "upper_plan_target_std": float(summary.get("upper_plan_target_std_mean", last.get("upper_plan_target_std", 0.0))),
        "terminal_launch_shift_mean": float(summary.get("terminal_launch_shift_mean_mean", last.get("terminal_launch_shift_mean", 0.0))),
        "terminal_launch_shift_std": float(summary.get("terminal_launch_shift_std_mean", last.get("terminal_launch_shift_std", 0.0))),
        "freq_promotion_strength": float(summary.get("freq_promotion_strength_mean", 0.0)),
        "shared_ppo_lower_samples": float(last.get("shared_ppo_lower_samples", 0.0)),
        "shared_ppo_gate_evaluations": float(last.get("shared_ppo_gate_evaluations", 0.0)),
        "shared_ppo_gate_replans": float(last.get("shared_ppo_gate_replans", 0.0)),
        "shared_ppo_gate_value_mean": float(last.get("shared_ppo_gate_value_mean", 0.0)),
        "shared_ppo_wait_replan_count": float(last.get("shared_ppo_wait_replan_count", 0.0)),
        "shared_ppo_wait_replan_pressure_mean": float(last.get("shared_ppo_wait_replan_pressure_mean", 0.0)),
        "shared_ppo_wait_replan_shift_pressure_mean": float(last.get("shared_ppo_wait_replan_shift_pressure_mean", 0.0)),
        "shared_ppo_wait_replan_gap_ratio_mean": float(last.get("shared_ppo_wait_replan_gap_ratio_mean", 0.0)),
        "shared_ppo_wait_replan_same_hold_mean": float(last.get("shared_ppo_wait_replan_same_hold_mean", 0.0)),
        "shared_ppo_wait_replan_same_wait_mean": float(last.get("shared_ppo_wait_replan_same_wait_mean", 0.0)),
        "shared_ppo_wait_replan_shift_mean_s": float(last.get("shared_ppo_wait_replan_shift_mean_s", 0.0)),
        "shared_ppo_wait_replan_shift_abs_mean_s": float(last.get("shared_ppo_wait_replan_shift_abs_mean_s", 0.0)),
        "shared_ppo_wait_replan_actor_base_used_mean": float(last.get("shared_ppo_wait_replan_actor_base_used_mean", 0.0)),
        "shared_ppo_wait_replan_base_delta_abs_mean_s": float(last.get("shared_ppo_wait_replan_base_delta_abs_mean_s", 0.0)),
        "shared_ppo_wait_replan_final_delta_abs_mean_s": float(last.get("shared_ppo_wait_replan_final_delta_abs_mean_s", 0.0)),
        "shared_ppo_loss": float(last.get("shared_ppo_loss", 0.0)),
    }


def paired_checks(
    rows: list[dict[str, Any]],
    min_pairs: int = 5,
    treatment: str = "native_promotion_replan",
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metrics = [
        ("ep_reward", False),
        ("avg_wait_min", True),
        ("score", False),
        ("upper_plan_decisions", False),
    ]
    if treatment in {"native_learned_gate", "native_wait_aware_replan"}:
        metrics.append(("shared_ppo_gate_replans", False))
    if treatment == "native_wait_aware_replan":
        metrics.extend([
            ("shared_ppo_wait_replan_count", False),
            ("shared_ppo_wait_replan_pressure_mean", False),
            ("shared_ppo_wait_replan_shift_pressure_mean", False),
            ("shared_ppo_wait_replan_gap_ratio_mean", False),
            ("shared_ppo_wait_replan_same_hold_mean", False),
            ("shared_ppo_wait_replan_same_wait_mean", False),
            ("shared_ppo_wait_replan_shift_abs_mean_s", False),
            ("shared_ppo_wait_replan_shift_mean_s", True),
            ("shared_ppo_wait_replan_actor_base_used_mean", False),
            ("shared_ppo_wait_replan_base_delta_abs_mean_s", False),
            ("shared_ppo_wait_replan_final_delta_abs_mean_s", False),
            ("upper_plan_target_mean", True),
            ("terminal_launch_shift_mean", True),
        ])
    for metric, lower_is_better in metrics:
        stats = paired_delta_stats(
            rows,
            variant_key="variant",
            pair_keys=("seed",),
            metric=metric,
            treatment=treatment,
            control="interval_only",
            lower_is_better=lower_is_better,
        )
        checks.append({
            "check": f"{treatment}_vs_interval_{metric}",
            **stats,
            "status": claim_status(stats, min_pairs=int(min_pairs)),
        })
    if treatment in {"native_learned_gate", "native_wait_aware_replan"}:
        for metric, lower_is_better, margin in [
            ("ep_reward", False, 15.0),
            ("avg_wait_min", True, 0.01),
        ]:
            stats = paired_delta_stats(
                rows,
                variant_key="variant",
                pair_keys=("seed",),
                metric=metric,
                treatment=treatment,
                control="interval_only",
                lower_is_better=lower_is_better,
            )
            checks.append({
                "check": f"{treatment}_vs_interval_{metric}_noninferiority",
                **stats,
                "noninferiority_margin": float(margin),
                "status": noninferiority_status(
                    stats,
                    max_loss=float(margin),
                    min_pairs=int(min_pairs),
                ),
            })
    return checks


def _run_variant_seed_job(job: dict[str, Any]) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
    variant = str(job["variant"])
    seed = int(job["seed"])
    overrides = dict(job["overrides"])
    variant_lower_gain = float(
        overrides.get("_lower_hf_wait_action_gain_s", job["lower_hf_wait_action_gain_s"])
    )
    payload = run_native_shared_ppo_episode_loop(
        output_dir=Path(job["output_dir"]) / variant / f"seed_{seed}",
        config_path=Path(job["config_path"]),
        seed=seed,
        episodes=int(job["episodes"]),
        device=str(job["device"]),
        config_overrides=_variant_overrides(overrides),
        learned_promotion_gate=bool(overrides.get("_learned_promotion_gate", False)),
        promotion_gate_threshold=float(overrides.get("_promotion_gate_threshold", 0.62)),
        promotion_gate_strength_min=float(overrides.get("_promotion_gate_strength_min", 0.0)),
        promotion_gate_age_min=float(overrides.get("_promotion_gate_age_min", 0.0)),
        promotion_gate_min_elapsed_s=float(overrides.get("_promotion_gate_min_elapsed_s", 0.0)),
        promotion_gate_cooldown_s=float(overrides.get("_promotion_gate_cooldown_s", 0.0)),
        promotion_gate_preselect_action=bool(overrides.get("_promotion_gate_preselect_action", False)),
        promotion_gate_plan_blend=float(overrides.get("_promotion_gate_plan_blend", 0.0)),
        promotion_gate_low_signal_min=float(overrides.get("_promotion_gate_low_signal_min", 0.0)),
        promotion_gate_max_hf_to_lf_ratio=float(overrides.get("_promotion_gate_max_hf_to_lf_ratio", 0.0)),
        promotion_gate_max_replans=int(overrides.get("_promotion_gate_max_replans", 0)),
        promotion_gate_max_total_replans=int(overrides.get("_promotion_gate_max_total_replans", 0)),
        promotion_replan_policy=str(overrides.get("_promotion_replan_policy", "actor")),
        promotion_replan_wait_gain_s=float(overrides.get("_promotion_replan_wait_gain_s", 0.0)),
        promotion_replan_max_shift_s=float(overrides.get("_promotion_replan_max_shift_s", 30.0)),
        promotion_replan_state_wait_weight=float(overrides.get("_promotion_replan_state_wait_weight", 1.0)),
        promotion_replan_frequency_weight=float(overrides.get("_promotion_replan_frequency_weight", 1.0)),
        promotion_replan_min_pressure=float(overrides.get("_promotion_replan_min_pressure", 0.0)),
        promotion_replan_require_shift=bool(overrides.get("_promotion_replan_require_shift", False)),
        promotion_replan_hold_guard_weight=float(overrides.get("_promotion_replan_hold_guard_weight", 0.0)),
        promotion_replan_same_wait_min=float(overrides.get("_promotion_replan_same_wait_min", 0.0)),
        promotion_replan_gap_guard_min_ratio=float(overrides.get("_promotion_replan_gap_guard_min_ratio", 0.0)),
        promotion_replan_gap_guard_max_ratio=float(overrides.get("_promotion_replan_gap_guard_max_ratio", 0.0)),
        promotion_replan_base_action=str(overrides.get("_promotion_replan_base_action", "active")),
        promotion_replan_actor_base_trust_s=float(overrides.get("_promotion_replan_actor_base_trust_s", 0.0)),
        lower_hf_wait_action_gain_s=variant_lower_gain,
        offpolicy_replay_updates=int(job["offpolicy_replay_updates"]),
    )
    row = _row_from_payload(seed, variant, payload)
    compact = {
        "summary": payload.get("summary", {}),
        "status": payload.get("status", "missing"),
        "rows": payload.get("rows", []),
    }
    return variant, str(seed), compact, row


def run_validation(
    output_dir: Path,
    config_path: Path,
    seeds: list[int],
    episodes: int,
    device: str,
    min_pairs: int = 5,
    lower_hf_wait_action_gain_s: float = DEFAULT_LOWER_HF_WAIT_ACTION_GAIN_S,
    offpolicy_replay_updates: int = 1,
    workers: int = 1,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    jobs: list[dict[str, Any]] = []
    for variant, overrides in VARIANTS.items():
        payloads[variant] = {}
        for seed in seeds:
            jobs.append({
                "variant": str(variant),
                "overrides": dict(overrides),
                "seed": int(seed),
                "output_dir": str(output_dir),
                "config_path": str(config_path),
                "episodes": int(episodes),
                "device": str(device),
                "lower_hf_wait_action_gain_s": float(lower_hf_wait_action_gain_s),
                "offpolicy_replay_updates": int(offpolicy_replay_updates),
            })
    if int(workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as executor:
            futures = [executor.submit(_run_variant_seed_job, job) for job in jobs]
            for future in as_completed(futures):
                variant, seed_key, compact, row = future.result()
                payloads.setdefault(variant, {})[seed_key] = compact
                rows.append(row)
    else:
        for job in jobs:
            variant, seed_key, compact, row = _run_variant_seed_job(job)
            payloads.setdefault(variant, {})[seed_key] = compact
            rows.append(row)
    variant_rank = {variant: idx for idx, variant in enumerate(VARIANTS)}
    rows.sort(key=lambda row: (variant_rank.get(str(row["variant"]), 999), int(row["seed"])))
    checks = paired_checks(rows, min_pairs=int(min_pairs))
    for treatment in ("native_learned_gate", "native_wait_aware_replan"):
        if any(row.get("variant") == treatment for row in rows):
            checks.extend(paired_checks(
                rows,
                min_pairs=int(min_pairs),
                treatment=treatment,
            ))
    summary = summarize(rows)
    payload = {
        "config_path": str(config_path),
        "seeds": [int(seed) for seed in seeds],
        "episodes": int(episodes),
        "min_pairs": int(min_pairs),
        "lower_hf_wait_action_gain_s": float(lower_hf_wait_action_gain_s),
        "offpolicy_replay_updates": int(max(1, int(offpolicy_replay_updates))),
        "workers": int(max(1, int(workers))),
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
        for metric in [
            "ep_reward",
            "avg_wait_min",
            "headway_cv",
            "score",
            "upper_plan_decisions",
            "upper_plan_target_mean",
            "upper_plan_target_std",
            "terminal_launch_shift_mean",
            "terminal_launch_shift_std",
            "shared_ppo_gate_evaluations",
            "shared_ppo_gate_replans",
            "shared_ppo_gate_value_mean",
            "shared_ppo_wait_replan_count",
            "shared_ppo_wait_replan_pressure_mean",
            "shared_ppo_wait_replan_shift_mean_s",
            "shared_ppo_wait_replan_shift_abs_mean_s",
            "shared_ppo_wait_replan_actor_base_used_mean",
            "shared_ppo_wait_replan_base_delta_abs_mean_s",
            "shared_ppo_wait_replan_final_delta_abs_mean_s",
        ]:
            values = np.asarray([float(row.get(metric, 0.0)) for row in vrows], dtype=np.float64)
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
            fieldnames = list(checks[0].keys())
            for row in checks[1:]:
                for key in row:
                    if key not in fieldnames:
                        fieldnames.append(key)
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(checks)
    write_report(output_dir / "report.md", payload)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Native Transit Promotion Replan Validation",
        "",
        "This runs the native Transit episode loop through the shared PPO adapter and toggles native promotion-triggered timetable replanning.",
        f"All variants use lower HF wait action prior gain `{payload.get('lower_hf_wait_action_gain_s', 0.0):.1f}s` so promotion is validated inside the full Freq-HRL lower-control loop.",
        f"Each native batch uses `{payload.get('offpolicy_replay_updates', 1)}` shared-PPO replay update(s).",
        f"Runner workers: `{payload.get('workers', 1)}`.",
        "",
        "| variant | seed | reward | wait | cv | score | upper decisions | launch shift | gate replans | wait replans | shift | gate | promotion strength | samples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
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
            f"| {row.get('terminal_launch_shift_mean', 0.0):+.2f} "
            f"| {row['shared_ppo_gate_replans']:.1f} "
            f"| {row.get('shared_ppo_wait_replan_count', 0.0):.1f} "
            f"| {row.get('shared_ppo_wait_replan_shift_mean_s', 0.0):.2f} "
            f"| {row['shared_ppo_gate_value_mean']:.3f} "
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
        "--lower-hf-wait-action-gain-s",
        type=float,
        default=DEFAULT_LOWER_HF_WAIT_ACTION_GAIN_S,
    )
    parser.add_argument("--offpolicy-replay-updates", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
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
        lower_hf_wait_action_gain_s=float(args.lower_hf_wait_action_gain_s),
        offpolicy_replay_updates=int(args.offpolicy_replay_updates),
        workers=int(args.workers),
    )
    reward_check = next(
        row for row in payload["paired_checks"]
        if row["check"] == "native_promotion_replan_vs_interval_ep_reward"
    )
    learned_reward = next(
        (
            row for row in payload["paired_checks"]
            if row["check"] == "native_learned_gate_vs_interval_ep_reward"
        ),
        None,
    )
    print(f"wrote {args.output_dir}")
    print(
        "native_promotion_replan "
        f"reward_delta={reward_check['delta_mean']:+.4f} "
        f"status={reward_check['status']}"
    )
    if learned_reward is not None:
        print(
            "native_learned_gate "
            f"reward_delta={learned_reward['delta_mean']:+.4f} "
            f"status={learned_reward['status']}"
        )


if __name__ == "__main__":
    main()
