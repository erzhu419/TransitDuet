#!/usr/bin/env python3
"""Analyze the frozen v14.29 restoration portfolio confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from freq_hrl.rl.restoration_portfolio import (
    restoration_snapshot_eligible,
    select_guarded_restoration_portfolio,
)
from scripts import mujoco_v14_29_portfolio_confirmatory_spec as spec
from scripts.analyze_mujoco_v14_26_robust_paired_fd_preflight import (
    _metric,
    cell_relative_dir,
)


ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not 0 <= int(successes) <= int(total) or int(total) < 1:
        raise ValueError("Wilson interval requires 0 <= successes <= total")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("Wilson confidence must be in (0, 1)")
    n = float(total)
    p = float(successes) / n
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = (
        z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_direction_gate(payload: dict[str, Any]) -> bool:
    critics = payload.get("critic_metrics")
    directions = payload.get("gradient_metrics")
    if not isinstance(critics, dict) or set(critics) != {"upper", "lower"}:
        raise ValueError("v14.29 payload lacks both critic blocks")
    gate = payload.get("gradient_error") is None
    for level in ("upper", "lower"):
        gate = gate and (
            float(critics[level]["ensemble_holdout_r2"])
            > spec.CRITIC_MINIMUM_HOLDOUT_R2
            and float(critics[level]["action_permutation_mse_increase_fraction"])
            > spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
        )
        if payload.get("gradient_error") is not None:
            continue
        if not isinstance(directions, dict) or level not in directions:
            raise ValueError("v14.29 payload lacks paired-direction metrics")
        block = directions[level]
        for role, expected_count in (
            ("train", spec.EXPECTED_PAIRED_TRAIN_DIRECTION_COUNT),
            ("holdout", spec.EXPECTED_PAIRED_HOLDOUT_DIRECTION_COUNT),
        ):
            values = block.get(role)
            if not isinstance(values, dict):
                raise ValueError(f"v14.29 payload lacks {role} direction block")
            parameter_count = int(values["parameter_count"])
            ranks = values.get("per_mode_design_rank")
            if (
                values.get("estimator") != spec.PAIRED_DIRECTION_ESTIMATOR
                or int(values["path_count"]) != expected_count
                or int(values["global_design_rank"]) != parameter_count
                or not isinstance(ranks, dict)
                or len(ranks) != 4
                or any(int(rank) != parameter_count for rank in ranks.values())
            ):
                raise ValueError("v14.29 paired direction design drifted")
        mode_cosines = block.get("holdout_mode_direction_cosines")
        if not isinstance(mode_cosines, dict) or len(mode_cosines) != 4:
            raise ValueError("v14.29 payload lacks four mode cosines")
        minimum_mode = min(float(value) for value in mode_cosines.values())
        gate = gate and (
            float(block["holdout_direction_cosine"])
            > spec.MINIMUM_PAIRED_HOLDOUT_COSINE
            and minimum_mode > spec.MINIMUM_PAIRED_HOLDOUT_COSINE
        )
    return bool(gate)


def _trace_invariant(diagnostics: Any, expected_paths: int) -> bool:
    return bool(
        isinstance(diagnostics, dict)
        and diagnostics.get("contract")
        == "paired_exact_action_reward_and_latent_trace_invariance_v1"
        and int(diagnostics.get("path_count", -1)) == int(expected_paths)
        and int(diagnostics.get("executed_action_trace_match_count", -1))
        == int(expected_paths)
        and int(diagnostics.get("reward_trace_match_count", -1))
        == int(expected_paths)
        and int(diagnostics.get("latent_policy_trace_match_count", -1))
        == int(expected_paths)
        and float(diagnostics.get("maximum_reward_mean_absolute_delta", -1.0))
        == 0.0
        and float(diagnostics.get("maximum_episode_return_absolute_delta", -1.0))
        == 0.0
        and diagnostics.get("all_traces_invariant") is True
    )


def _validate_payload(payload: dict[str, Any], environment: str, seed: int) -> None:
    expected = {
        "probe_version": spec.PROBE_VERSION,
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "critic_train_roots": list(spec.CRITIC_TRAIN_ROOTS),
        "critic_holdout_roots": list(spec.CRITIC_HOLDOUT_ROOTS),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "critic_train_base_path_count": spec.EXPECTED_CRITIC_TRAIN_BASE_PATH_COUNT,
        "critic_holdout_base_path_count": spec.EXPECTED_CRITIC_HOLDOUT_BASE_PATH_COUNT,
        "critic_train_path_count": spec.EXPECTED_CRITIC_TRAIN_PATH_COUNT,
        "critic_holdout_path_count": spec.EXPECTED_CRITIC_HOLDOUT_PATH_COUNT,
        "design_path_count": spec.EXPECTED_DESIGN_PATH_COUNT,
        "validation_path_count": spec.EXPECTED_VALIDATION_PATH_COUNT,
        "critic_collection_mode": spec.CRITIC_COLLECTION_MODE,
        "critic_intervention_bias_rms": spec.CRITIC_INTERVENTION_BIAS_RMS,
        "critic_intervention_variants": list(spec.CRITIC_INTERVENTION_VARIANTS),
        "critic_intervention_direction_scheme": (
            spec.CRITIC_INTERVENTION_DIRECTION_SCHEME
        ),
        "critic_intervention_hadamard_order": (
            spec.CRITIC_INTERVENTION_HADAMARD_ORDER
        ),
        "paired_direction_estimator": spec.PAIRED_DIRECTION_ESTIMATOR,
        "actor_update_scope": spec.ACTOR_UPDATE_SCOPE,
        "actor_direction_source": spec.ACTOR_DIRECTION_SOURCE,
        "minimum_paired_holdout_cosine": spec.MINIMUM_PAIRED_HOLDOUT_COSINE,
        "baseline_router_strength": spec.BASELINE_ROUTER_STRENGTH,
        "router_strength_values": list(spec.ROUTER_STRENGTH_VALUES),
        "design_fold_count": spec.DESIGN_FOLD_COUNT,
        "critic_seeds": list(spec.CRITIC_ENSEMBLE_SEEDS),
        "critic_hidden_dim": spec.CRITIC_HIDDEN_DIM,
        "critic_epochs": spec.CRITIC_EPOCHS,
        "critic_minibatch_size": spec.CRITIC_MINIBATCH_SIZE,
        "critic_learning_rate": spec.CRITIC_LEARNING_RATE,
        "cost_return_horizon_decisions": {
            "upper": spec.UPPER_COST_RETURN_HORIZON_DECISIONS,
            "lower": spec.LOWER_COST_RETURN_HORIZON_DECISIONS,
        },
        "critic_minimum_holdout_r2": spec.CRITIC_MINIMUM_HOLDOUT_R2,
        "critic_minimum_action_permutation_mse_increase": (
            spec.CRITIC_MINIMUM_ACTION_PERMUTATION_MSE_INCREASE
        ),
        "minimum_gradient_median_cosine": spec.MINIMUM_GRADIENT_MEDIAN_COSINE,
        "actor_state_limit": spec.ACTOR_STATE_LIMIT_PER_LEVEL,
        "actor_step_rms_values": list(spec.ACTOR_STEP_RMS_VALUES),
        "minimum_reduction": spec.MINIMUM_REDUCTION,
        "funnel_multiplier": spec.FUNNEL_MULTIPLIER,
        "workers": spec.WORKERS,
        "risk_mode": spec.RISK_MODE,
        "cvar_alpha": spec.CVAR_ALPHA,
        "episode_horizon": spec.EPISODE_HORIZON,
        "leakage_cost_mode": spec.LEAKAGE_COST_MODE,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v14.29 payload contract mismatch: {mismatches}")
    fold_baselines = payload.get("design_fold_baselines")
    if not isinstance(fold_baselines, list) or len(fold_baselines) != 2:
        raise ValueError("v14.29 payload lacks two fold baselines")
    direction_gate = _validate_direction_gate(payload)
    if bool(payload.get("critic_gate_pass")) != direction_gate:
        raise ValueError("v14.29 direction gate does not reproduce")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("v14.29 payload lacks candidate registry")
    expected_count = spec.EXPECTED_ROUTER_CANDIDATE_COUNT + (
        spec.EXPECTED_ACTOR_CANDIDATE_COUNT if direction_gate else 0
    )
    if len(candidates) != expected_count or int(
        payload.get("candidate_count", -1)
    ) != expected_count:
        raise ValueError("v14.29 candidate count violates the direction gate")
    actor_source = (
        f"{spec.ACTOR_DIRECTION_SOURCE}:{spec.PAIRED_DIRECTION_ESTIMATOR}"
    )
    expected_registry = (
        [
            (actor_source, float(step), float(spec.BASELINE_ROUTER_STRENGTH))
            for step in spec.ACTOR_STEP_RMS_VALUES
        ]
        if direction_gate else []
    ) + [
        ("function_preserving_router_adapter", 0.0, float(strength))
        for strength in spec.ROUTER_STRENGTH_VALUES
    ]
    actual_registry = [
        (
            str(candidate.get("source")),
            float(candidate.get("step_rms", float("nan"))),
            float(candidate.get("router_strength", float("nan"))),
        )
        for candidate in candidates
    ]
    if actual_registry != expected_registry:
        raise ValueError("v14.29 candidate registry or order drifted")
    replay_candidates = []
    for candidate in candidates:
        fold_snapshots = candidate.get("design_fold_snapshots")
        fold_flags = candidate.get("design_fold_eligible")
        if (
            not isinstance(fold_snapshots, list) or len(fold_snapshots) != 2
            or not isinstance(fold_flags, list) or len(fold_flags) != 2
        ):
            raise ValueError("v14.29 candidate lacks two-fold evidence")
        reproduced_folds = [
            restoration_snapshot_eligible(
                snapshot,
                baseline,
                minimum_reduction=spec.MINIMUM_REDUCTION,
                funnel_multiplier=spec.FUNNEL_MULTIPLIER,
            )
            for snapshot, baseline in zip(
                fold_snapshots, fold_baselines, strict=True
            )
        ]
        pooled = restoration_snapshot_eligible(
            candidate["snapshot"],
            payload["design_baseline"],
            minimum_reduction=spec.MINIMUM_REDUCTION,
            funnel_multiplier=spec.FUNNEL_MULTIPLIER,
        )
        router = candidate["source"] == "function_preserving_router_adapter"
        trace_gate = (
            _trace_invariant(
                candidate.get("trace_invariance"),
                spec.EXPECTED_DESIGN_PATH_COUNT,
            )
            if router else candidate.get("trace_invariance") is None
        )
        if bool(candidate.get("requires_trace_invariance")) != router:
            raise ValueError("v14.29 trace requirement does not match source")
        if bool(candidate.get("trace_invariance_eligible")) != trace_gate:
            raise ValueError("v14.29 design trace gate does not reproduce")
        if list(map(bool, fold_flags)) != reproduced_folds:
            raise ValueError("v14.29 fold flags do not reproduce")
        if bool(candidate["design_eligible"]) != bool(
            pooled and all(reproduced_folds) and trace_gate
        ):
            raise ValueError("v14.29 design eligibility does not reproduce")
        expected_priority = [
            float(candidate["step_rms"]),
            float(candidate["router_strength"]),
        ]
        if candidate.get("selection_priority") != expected_priority:
            raise ValueError("v14.29 candidate selection priority drifted")
        replay_candidate = {
            **candidate,
            "fold_snapshots": fold_snapshots,
        }
        if router and isinstance(replay_candidate.get("trace_invariance"), dict):
            replay_candidate["trace_invariance"] = {
                **replay_candidate["trace_invariance"],
                "all_traces_invariant": bool(trace_gate),
            }
        replay_candidates.append(replay_candidate)

    decision = select_guarded_restoration_portfolio(
        replay_candidates,
        baseline=payload["design_baseline"],
        fold_baselines=fold_baselines,
        minimum_reduction=spec.MINIMUM_REDUCTION,
        funnel_multiplier=spec.FUNNEL_MULTIPLIER,
    )
    if int(payload.get("design_eligible_candidate_count", -1)) != len(
        decision.eligible_indices
    ):
        raise ValueError("v14.29 eligible candidate count does not reproduce")
    selected = payload.get("selected_design_candidate")
    expected_selected = (
        None
        if decision.selected_index is None
        else candidates[decision.selected_index]
    )
    if selected != expected_selected:
        raise ValueError("v14.29 selected design candidate does not reproduce")
    validation_candidate = payload.get("validation_candidate")
    if selected is None:
        if validation_candidate is not None or payload.get("validation_supported"):
            raise ValueError("v14.29 validation exists without design selection")
        return
    if validation_candidate is None or payload.get("validation_baseline") is None:
        raise ValueError("v14.29 selected candidate lacks validation")
    router = selected["source"] == "function_preserving_router_adapter"
    validation_trace_gate = (
        _trace_invariant(
            payload.get("validation_trace_invariance"),
            spec.EXPECTED_VALIDATION_PATH_COUNT,
        )
        if router else payload.get("validation_trace_invariance") is None
    )
    if router and selected["parameter_sha256"] != payload["baseline_parameter_sha256"]:
        raise ValueError("v14.29 router candidate changed actor parameters")
    validation_gate = restoration_snapshot_eligible(
        validation_candidate,
        payload["validation_baseline"],
        minimum_reduction=spec.MINIMUM_REDUCTION,
        funnel_multiplier=spec.FUNNEL_MULTIPLIER,
    ) and validation_trace_gate
    if bool(payload.get("validation_supported")) != bool(validation_gate):
        raise ValueError("v14.29 validation gate does not reproduce")


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v14.29 probe result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_payload(payload, environment, seed)
            selected = payload.get("selected_design_candidate")
            validation_baseline = payload.get("validation_baseline")
            validation_candidate = payload.get("validation_candidate")
            baseline_merit = _metric(validation_baseline, "frequency_violation_merit")
            candidate_merit = _metric(validation_candidate, "frequency_violation_merit")
            reduction = (
                None
                if baseline_merit is None or candidate_merit is None or baseline_merit <= 0
                else (baseline_merit - candidate_merit) / baseline_merit
            )
            rows.append({
                "environment": str(environment),
                "optimizer_seed": int(seed),
                "critic_direction_gate_pass": bool(payload["critic_gate_pass"]),
                "design_eligible_candidate_count": int(
                    payload["design_eligible_candidate_count"]
                ),
                "selected_source": None if selected is None else selected["source"],
                "selected_step_rms": None if selected is None else float(selected["step_rms"]),
                "selected_router_strength": (
                    None if selected is None else float(selected["router_strength"])
                ),
                "validation_relative_merit_reduction": reduction,
                "validation_reward_violation_count": _metric(
                    validation_candidate, "reward_violation_count"
                ),
                "validation_frequency_violation_count": _metric(
                    validation_candidate, "frequency_violation_count"
                ),
                "selected_router_trace_invariant": (
                    None
                    if selected is None or selected["source"]
                    != "function_preserving_router_adapter"
                    else _trace_invariant(
                        payload.get("validation_trace_invariance"),
                        spec.EXPECTED_VALIDATION_PATH_COUNT,
                    )
                ),
                "validation_supported": bool(payload["validation_supported"]),
            })

    environment_results = []
    for environment in spec.ENVIRONMENTS:
        selected_rows = [row for row in rows if row["environment"] == environment]
        successes = sum(row["validation_supported"] for row in selected_rows)
        lower, upper = wilson_interval(
            successes, len(selected_rows), confidence=spec.CONFIDENCE_LEVEL
        )
        router_rows = [
            row for row in selected_rows
            if row["selected_source"] == "function_preserving_router_adapter"
        ]
        trace_gate = all(
            row["selected_router_trace_invariant"] is True
            for row in router_rows
        )
        environment_results.append({
            "environment": str(environment),
            "optimizer_seed_count": len(selected_rows),
            "supported_count": successes,
            "abstention_or_failure_count": len(selected_rows) - successes,
            "success_rate": successes / len(selected_rows),
            "success_rate_wilson_lower": lower,
            "success_rate_wilson_upper": upper,
            "router_selection_count": len(router_rows),
            "actor_selection_count": sum(
                row["selected_source"] not in {None, "function_preserving_router_adapter"}
                for row in selected_rows
            ),
            "all_selected_router_traces_invariant": trace_gate,
            "confirmatory_gate_pass": bool(
                lower > spec.SUCCESS_RATE_NULL and trace_gate
            ),
        })

    supported = all(row["confirmatory_gate_pass"] for row in environment_results)
    result = {
        "analysis_version": "mujoco_v14_29_portfolio_confirmatory_v1",
        "status": (
            "mechanism_portfolio_confirmed"
            if supported else "mechanism_portfolio_not_confirmed"
        ),
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "anchor_run_name": spec.ANCHOR_RUN_NAME,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "statistical_unit": "optimizer_seed",
        "inference_scope": (
            "fresh_optimizer_seeds_conditional_on_frozen_validation_path_panel"
        ),
        "confidence_level": spec.CONFIDENCE_LEVEL,
        "success_rate_null": spec.SUCCESS_RATE_NULL,
        "cell_count": len(rows),
        "supported_cell_count": sum(row["validation_supported"] for row in rows),
        "environment_results": environment_results,
        "cells": rows,
    }
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "portfolio_confirmatory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "portfolio_confirmatory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "supported_cell_count": result["supported_cell_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
