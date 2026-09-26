#!/usr/bin/env python3
"""Paired Stage-10 development analysis against frozen Stage-9 candidate rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import pointmaze_learned_termination_stage10_spec as spec  # noqa: E402
from scripts.analyze_pointmaze_budgeted_trigger_stage9 import (  # noqa: E402
    analyze as analyze_stage9,
)
from scripts.analyze_pointmaze_plan_validity_stage8b import _interval  # noqa: E402
from freq_hrl.experiments.pointmaze_learned_termination import (  # noqa: E402
    ALGORITHM_PATH,
    PROTOCOL_VERSION,
)


def _by_root(paths: Iterable[Path]) -> dict[int, dict[str, Any]]:
    result = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or len(payload.get("cells", [])) != 1:
            raise ValueError(f"incomplete Stage-10 result: {path}")
        root = int(payload["cells"][0]["optimizer_seed"])
        if root in result:
            raise ValueError(f"duplicate Stage-10 optimizer root: {root}")
        result[root] = payload
    return result


def _root_summary(
    baseline: dict[str, Any],
    reference: dict[str, Any],
    *,
    preflight: bool,
) -> dict[str, float]:
    cell = baseline["cells"][0]
    old = reference["cells"][0]
    protocol = baseline["protocol"]
    root = int(cell["optimizer_seed"])
    options = spec.cell_options(root, preflight=preflight)
    if (
        protocol["protocol_version"] != PROTOCOL_VERSION
        or protocol["algorithm_path"] != ALGORITHM_PATH
        or protocol["optimizer_seed"] != root
        or protocol["evidence_role"] != "learned_termination_development"
        or cell["protocol_version"] != PROTOCOL_VERSION
        or cell["algorithm_path"] != ALGORITHM_PATH
        or cell["controller_parameter_count"] != 267018
        or cell["controller_gradient_updates_train"] <= 0
        or cell["termination_optimizer_steps"] <= 0
        or cell["termination_actor_weight_change_norm"] <= 0
        or cell["termination_state_feature_count"] != 38
        or any(
            cell["runtime_versions"].get(name) != version
            for name, version in spec.RUNTIME_EXPECTATIONS.items()
        )
    ):
        raise ValueError(f"Stage-10 learned training contract changed: {root}")
    for key in (
        "iterations", "horizon", "upper_period_seconds", "history_seconds",
        "fast_period_seconds", "maximum_subgoal_delta", "reference_hidden_dim",
        "learning_rate", "checkpoint_evaluation_interval", "max_offset_steps",
        "check_stride_steps", "termination_iterations", "termination_hidden_dim",
        "termination_learning_rate",
    ):
        if protocol[key] != options[key]:
            raise ValueError(f"Stage-10 frozen option changed: {(root, key)}")
    for role in ("train", "selection", "branch_fit", "trigger_eval"):
        field = f"{role}_seeds"
        if (
            tuple(protocol[field]) != options[role]
            or tuple(cell[field]) != options[role]
            or tuple(old[field]) != options[role]
        ):
            raise ValueError(f"Stage-10 seed role changed: {(root, role)}")
    if (
        protocol["task_options"] != reference["protocol"]["task_options"]
        or cell["controller_selected_iteration"]
        != old["selected_checkpoint_iteration"]
        or cell["termination_training_iterations"]
        != options["termination_iterations"]
    ):
        raise ValueError(f"Stage-10 controller replay contract changed: {root}")
    expected_rollouts = (
        len(options["branch_fit"]) * (1 + int(options["termination_iterations"]))
        + len(options["selection"]) * len(cell["termination_validation_history"])
    )
    if (
        cell["termination_rollouts_before_eval"] != expected_rollouts
        or cell["termination_extra_primitive_steps"]
        != expected_rollouts * int(options["horizon"])
        or cell["termination_extra_primitive_steps"]
        > old["branch_fit_primitive_steps_replayed"]
    ):
        raise ValueError(f"Stage-10 primitive-step budget changed: {root}")
    eval_seeds = set(options["trigger_eval"])
    old_rows = {
        (int(row["seed"]), row["mode"]): row
        for row in old["trigger_evaluation_rows"]
    }
    fixed = {int(row["seed"]): row for row in cell["fixed_replay_rows"]}
    learned = {
        int(row["seed"]): row
        for row in cell["learned_termination_evaluation_rows"]
    }
    if set(fixed) != eval_seeds or set(learned) != eval_seeds:
        raise ValueError(f"Stage-10 paired eval paths changed: {root}")
    horizon = int(options["horizon"])
    period = int(round(float(options["upper_period_seconds"]) / 0.01))
    for seed in eval_seeds:
        prior_fixed = old_rows[(seed, "fixed")]
        replay = fixed[seed]
        row = learned[seed]
        decisions = row["decision_steps"]
        durations = np.diff([*decisions, horizon])
        if (
            replay["decision_steps"] != prior_fixed["decision_steps"]
            or abs(replay["tracking_squared_error_integral"]
                   - prior_fixed["tracking_squared_error_integral"]) > 1e-8
            or abs(replay["episode_return"] - prior_fixed["episode_return"]) > 1e-8
            or row["mode"] != "learned_termination_ppo"
            or not row["protocol_valid"]
            or row["has_privileged_regime_input"]
            or not row["planner_called_only_on_decision"]
            or row["episode_length"] != horizon
            or row["upper_decision_count"] != horizon // period
            or decisions[0] != 0
            or [step // period for step in decisions]
            != list(range(horizon // period))
            or min(durations) < period - int(options["max_offset_steps"])
            or max(durations) > period + int(options["max_offset_steps"])
            or row["trigger_score_checks"] < 1
            or not np.isfinite(row["tracking_squared_error_integral"])
            or not np.isfinite(row["episode_return"])
        ):
            raise ValueError(f"Stage-10 paired episode invalid: {(root, seed)}")
    candidate = "causal_validity_interactions"
    return {
        "baseline_minus_fixed_episode_tracking_ise": float(np.mean([
            old_rows[(seed, "fixed")]["tracking_squared_error_integral"]
            - learned[seed]["tracking_squared_error_integral"]
            for seed in eval_seeds
        ])),
        "candidate_minus_learned_termination_episode_tracking_ise": float(np.mean([
            learned[seed]["tracking_squared_error_integral"]
            - old_rows[(seed, candidate)]["tracking_squared_error_integral"]
            for seed in eval_seeds
        ])),
        "candidate_minus_learned_termination_episode_return": float(np.mean([
            old_rows[(seed, candidate)]["episode_return"]
            - learned[seed]["episode_return"]
            for seed in eval_seeds
        ])),
        "learned_termination_early_calls_per_episode": float(np.mean([
            learned[seed]["trigger_early_calls"] for seed in eval_seeds
        ])),
    }


def analyze(
    baseline_paths: Iterable[Path],
    reference_paths: Iterable[Path],
) -> dict[str, Any]:
    baseline_inputs = tuple(baseline_paths)
    reference_inputs = tuple(reference_paths)
    baseline = _by_root(baseline_inputs)
    reference = _by_root(reference_inputs)
    roots = set(baseline)
    if roots == set(spec.PREFLIGHT_OPTIMIZER_SEEDS):
        preflight = True
    elif roots == set(spec.OPTIMIZER_SEEDS):
        preflight = False
    else:
        raise ValueError("Stage-10 root matrix is incomplete")
    if roots != set(reference):
        raise ValueError("Stage-10 baseline/reference roots differ")
    analyze_stage9(reference_inputs)
    summaries = {
        str(root): _root_summary(
            baseline[root], reference[root], preflight=preflight
        )
        for root in sorted(roots)
    }
    intervals = {
        name: _interval(
            [summary[name] for summary in summaries.values()], confidence=0.95
        )
        for name in next(iter(summaries.values()))
    }
    checks = {
        "baseline_learns_vs_fixed": (
            intervals["baseline_minus_fixed_episode_tracking_ise"]["status"]
            == "supported"
        ),
        "candidate_beats_learned_termination": (
            intervals["candidate_minus_learned_termination_episode_tracking_ise"]
            ["status"] == "supported"
        ),
    }
    return {
        "analysis_version": "pointmaze_learned_termination_stage10_analysis_v1",
        "protocol_version": PROTOCOL_VERSION,
        "matrix": "preflight" if preflight else "formal_development",
        "statistical_unit": "optimizer_seed_root",
        "root_summaries": summaries,
        "intervals": intervals,
        "qualification_checks": checks,
        "decision": (
            "stage10_preflight_software_only" if preflight else
            "stage10_development_gate_passed" if all(checks.values()) else
            "stage10_development_gate_failed"
        ),
        "claim_boundary": (
            "Development comparison against revealed Stage-9 confirmation paths; "
            "fresh-seed confirmation and task transfer remain required"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_inputs", nargs="+", type=Path)
    parser.add_argument("--stage9-inputs", nargs="+", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = analyze(args.baseline_inputs, args.stage9_inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# PointMaze Stage-10 Learned Termination",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "| Quantity | Root mean [95% CI] | Status |",
        "|---|---:|---|",
    ]
    for name, item in result["intervals"].items():
        lines.append(
            f"| {name} | {item['mean']:.6f} "
            f"[{item['ci_lower']:.6f}, {item['ci_upper']:.6f}] "
            f"| {item['status']} |"
        )
    lines.extend(("", result["claim_boundary"] + ".", ""))
    (args.output_dir / "report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
