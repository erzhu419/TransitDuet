#!/usr/bin/env python3
"""Probe causal raw-policy responsibility distillation on frozen anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.experiments.mujoco.control_validation import (
    _model_parameter_sha256,
    paired_relative_frequency_feasibility_diagnostics,
)
from freq_hrl.experiments.reproducibility import derive_seed
from freq_hrl.rl import (
    distill_hierarchical_actor_heads,
    restoration_snapshot_eligible,
)
from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as spec
from scripts.probe_mujoco_action_cost_critic_restoration import (
    _parallel_map,
    _rollout_path,
)
from scripts.probe_mujoco_radial_restoration import (
    _evaluate_rows,
    _load_model,
    _v14_17_anchor_profile,
)
from scripts.probe_mujoco_zeroth_order_actor_restoration import (
    _paths_for_roots,
    _snapshot_fn,
)


PROBE_VERSION = "mujoco_raw_policy_responsibility_distillation_probe_v2"
FREQUENCY_ENDPOINTS = (
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
)


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not parsed or len(set(parsed)) != len(parsed) or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("roots must be unique non-negative integers")
    return parsed


def _complete_endpoint_diagnostics(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    risk_mode: str,
    cvar_alpha: float,
) -> dict[str, Any]:
    diagnostics = paired_relative_frequency_feasibility_diagnostics(
        rows,
        baseline_rows=baseline_rows,
        expected_modes=spec.DISTURBANCE_MODES,
        lower_reduction_fraction=float(
            summary["lower_deployment_frequency_reference_reduction_fraction"]
        ),
        upper_reduction_fraction=float(
            summary["upper_deployment_frequency_reference_reduction_fraction"]
        ),
        lower_power_floor=float(
            summary["lower_deployment_frequency_rms_budget"]
        ) ** 2,
        upper_power_floor=float(
            summary["upper_deployment_frequency_rms_budget"]
        ) ** 2,
        risk_mode=str(risk_mode),
        cvar_alpha=float(cvar_alpha),
    )
    tolerance = 1e-10
    endpoint_maximums = {
        endpoint: max(
            float(item["normalized_violation"])
            for item in diagnostics["constraints"]
            if str(item["endpoint"]) == endpoint
        )
        for endpoint in FREQUENCY_ENDPOINTS
    }
    reward_maximum = max(
        float(item["normalized_violation"])
        for item in diagnostics["constraints"]
        if str(item["endpoint"]) == "reward_mean"
    )
    return {
        "contract": "all_five_frequency_endpoints_and_reward_floor_v1",
        "complete": bool(
            reward_maximum <= tolerance
            and all(value <= tolerance for value in endpoint_maximums.values())
        ),
        "reward_maximum_normalized_violation": reward_maximum,
        "frequency_endpoint_maximum_normalized_violations": endpoint_maximums,
        "worst_constraint": diagnostics["worst_constraint"],
    }


def _collect_trajectories(
    checkpoint_path: Path,
    summary: dict[str, Any],
    paths: Iterable[dict[str, Any]],
    *,
    episode_horizon: int,
    leakage_cost_mode: str,
) -> list[Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint)
    source_hash = _model_parameter_sha256(model)
    trajectories = []
    for path in paths:
        policy_seed = derive_seed(
            "mujoco_raw_policy_distillation_teacher_v1",
            str(summary["environment"]),
            int(path["seed"]),
            str(path["disturbance_mode"]),
        )
        torch.manual_seed(int(policy_seed))
        np.random.seed(int(policy_seed) % (2**32 - 1))
        batch, _ = _rollout_path(
            model,
            summary=summary,
            path=path,
            sample=False,
            collect_trajectory=True,
            episode_horizon=int(episode_horizon),
            leakage_cost_mode=str(leakage_cost_mode),
        )
        if batch is None:
            raise RuntimeError("distillation teacher did not return a trajectory")
        trajectories.append(batch)
    if _model_parameter_sha256(model) != source_hash:
        raise RuntimeError("distillation teacher mutated the frozen actor")
    return trajectories


def _candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    torch.set_num_threads(1)
    checkpoint = torch.load(
        job["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint)
    config = dict(job["config"])
    distillation_config = {
        key: value for key, value in config.items()
        if key != "router_strength"
    }
    router_strength = float(config.get(
        "router_strength", job["summary"]["lower_action_router_strength"]
    ))
    diagnostics = distill_hierarchical_actor_heads(
        model,
        job["trajectories"],
        upper_action_scale=float(job["summary"]["upper_action_scale"]),
        lower_action_scale=float(job["summary"]["lower_action_scale"]),
        lower_action_context_start=int(job["lower_action_context_start"]),
        **distillation_config,
    )
    rows = _evaluate_rows(
        model,
        summary=job["summary"],
        paths=job["paths"],
        episode_horizon=int(job["episode_horizon"]),
        leakage_cost_mode=str(job["leakage_cost_mode"]),
        router_strength=router_strength,
    )
    return {
        "candidate_index": int(job["candidate_index"]),
        "config": config,
        "parameter_sha256": _model_parameter_sha256(model),
        "distillation": diagnostics,
        "evaluation_router_strength": router_strength,
        "rows": rows,
    }


def _evaluate_selected(
    checkpoint_path: Path,
    summary: dict[str, Any],
    trajectories: list[Any],
    config: dict[str, Any],
    paths: list[dict[str, Any]],
    *,
    episode_horizon: int,
    leakage_cost_mode: str,
    lower_action_context_start: int,
) -> tuple[str, list[dict[str, Any]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _load_model(checkpoint)
    distillation_config = {
        key: value for key, value in config.items()
        if key != "router_strength"
    }
    router_strength = float(config.get(
        "router_strength", summary["lower_action_router_strength"]
    ))
    distill_hierarchical_actor_heads(
        model,
        trajectories,
        upper_action_scale=float(summary["upper_action_scale"]),
        lower_action_scale=float(summary["lower_action_scale"]),
        lower_action_context_start=int(lower_action_context_start),
        **distillation_config,
    )
    return _model_parameter_sha256(model), _evaluate_rows(
        model,
        summary=summary,
        paths=paths,
        episode_horizon=int(episode_horizon),
        leakage_cost_mode=str(leakage_cost_mode),
        router_strength=router_strength,
    )


def run_probe(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    output_path: Path,
    distill_roots: tuple[int, ...],
    design_roots: tuple[int, ...],
    validation_roots: tuple[int, ...],
    candidates: tuple[dict[str, float], ...],
    design_fold_count: int,
    episode_horizon: int,
    leakage_cost_mode: str,
    risk_mode: str,
    cvar_alpha: float,
    minimum_merit_reduction: float,
    funnel_multiplier: float,
    workers: int,
) -> dict[str, Any]:
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary, _ = _v14_17_anchor_profile(raw_summary)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline_model = _load_model(checkpoint)
    baseline_parameter_sha256 = _model_parameter_sha256(baseline_model)
    action_dim = int(baseline_model.config.lower_action_dim)
    lower_action_context_start = (
        int(baseline_model.config.lower_state_dim)
        - 3 * action_dim
        - int(bool(checkpoint.get("lower_action_router_observe_strength", False)))
    )
    if (
        action_dim != int(baseline_model.config.upper_action_dim)
        or lower_action_context_start < 0
    ):
        raise ValueError("v15 requires aligned MuJoCo upper/lower action contexts")
    root_roles = (distill_roots, design_roots, validation_roots)
    flattened = tuple(root for role in root_roles for root in role)
    if len(flattened) != len(set(flattened)):
        raise ValueError("distillation, design, and validation roots must be disjoint")
    fold_count = int(design_fold_count)
    if fold_count < 1 or len(design_roots) % fold_count:
        raise ValueError("design roots must divide into nonempty folds")

    distill_paths = _paths_for_roots(summary["environment"], distill_roots)
    design_paths = _paths_for_roots(summary["environment"], design_roots)
    validation_paths = _paths_for_roots(summary["environment"], validation_roots)
    trajectories = _collect_trajectories(
        checkpoint_path,
        summary,
        distill_paths,
        episode_horizon=episode_horizon,
        leakage_cost_mode=leakage_cost_mode,
    )
    design_baseline_rows = _evaluate_rows(
        baseline_model,
        summary=summary,
        paths=design_paths,
        episode_horizon=episode_horizon,
        leakage_cost_mode=leakage_cost_mode,
        router_strength=float(summary["lower_action_router_strength"]),
    )
    snapshot_fn = _snapshot_fn(
        summary,
        design_baseline_rows,
        risk_mode=risk_mode,
        cvar_alpha=cvar_alpha,
    )
    design_baseline = snapshot_fn(design_baseline_rows)
    paths_per_root = len(spec.DISTURBANCE_MODES)
    roots_per_fold = len(design_roots) // fold_count
    fold_slices = [
        slice(
            index * roots_per_fold * paths_per_root,
            (index + 1) * roots_per_fold * paths_per_root,
        )
        for index in range(fold_count)
    ]
    fold_baseline_rows = [design_baseline_rows[item] for item in fold_slices]
    fold_snapshot_fns = [
        _snapshot_fn(
            summary,
            rows,
            risk_mode=risk_mode,
            cvar_alpha=cvar_alpha,
        )
        for rows in fold_baseline_rows
    ]
    fold_baselines = [
        function(rows)
        for function, rows in zip(
            fold_snapshot_fns, fold_baseline_rows, strict=True
        )
    ]

    jobs = [
        {
            "candidate_index": index,
            "checkpoint_path": str(checkpoint_path),
            "summary": summary,
            "trajectories": trajectories,
            "config": config,
            "paths": design_paths,
            "episode_horizon": int(episode_horizon),
            "leakage_cost_mode": str(leakage_cost_mode),
            "lower_action_context_start": int(lower_action_context_start),
        }
        for index, config in enumerate(candidates)
    ]
    results = _parallel_map(_candidate_job, jobs, workers=int(workers))
    if [item["candidate_index"] for item in results] != list(range(len(candidates))):
        raise RuntimeError("distillation candidate order changed")

    public_candidates = []
    eligible_indices = []
    for result in results:
        rows = result.pop("rows")
        snapshot = snapshot_fn(rows)
        folds = [rows[item] for item in fold_slices]
        fold_snapshots = [
            function(fold_rows)
            for function, fold_rows in zip(
                fold_snapshot_fns, folds, strict=True
            )
        ]
        complete = _complete_endpoint_diagnostics(
            rows, design_baseline_rows, summary,
            risk_mode=risk_mode, cvar_alpha=cvar_alpha,
        )
        fold_complete = [
            _complete_endpoint_diagnostics(
                fold_rows,
                baseline_rows,
                summary,
                risk_mode=risk_mode,
                cvar_alpha=cvar_alpha,
            )
            for fold_rows, baseline_rows in zip(
                folds, fold_baseline_rows, strict=True
            )
        ]
        merit_gate = restoration_snapshot_eligible(
            snapshot,
            design_baseline,
            minimum_reduction=minimum_merit_reduction,
            funnel_multiplier=funnel_multiplier,
        )
        fold_merit_gates = [
            restoration_snapshot_eligible(
                fold_snapshot,
                fold_baseline,
                minimum_reduction=minimum_merit_reduction,
                funnel_multiplier=funnel_multiplier,
            )
            for fold_snapshot, fold_baseline in zip(
                fold_snapshots, fold_baselines, strict=True
            )
        ]
        eligible = bool(
            merit_gate
            and all(fold_merit_gates)
            and complete["complete"]
            and all(item["complete"] for item in fold_complete)
        )
        index = int(result["candidate_index"])
        if eligible:
            eligible_indices.append(index)
        public_candidates.append({
            **result,
            "snapshot": snapshot,
            "design_complete_endpoint_gate": complete,
            "design_fold_complete_endpoint_gates": fold_complete,
            "design_merit_gate": bool(merit_gate),
            "design_fold_merit_gates": list(map(bool, fold_merit_gates)),
            "design_eligible": eligible,
        })

    def rank(index: int) -> tuple[float, ...]:
        candidate = public_candidates[index]
        snapshot = candidate["snapshot"]
        config = candidate["config"]
        return (
            float(snapshot["frequency_violation_merit"]),
            float(snapshot["worst_frequency_violation"]),
            float(config["blend"]),
            float(config["transfer_strength"]),
            float(config["slow_alpha"]),
            float(config.get("head_delta_rms_limit", float("inf"))),
            float(config.get("raw_target_limit", float("inf"))),
            float(config.get(
                "router_strength", summary["lower_action_router_strength"]
            )),
            float(index),
        )

    selected_index = min(eligible_indices, key=rank) if eligible_indices else None
    validation = None
    validation_supported = False
    if selected_index is not None:
        selected = public_candidates[selected_index]
        validation_baseline_rows = _evaluate_rows(
            baseline_model,
            summary=summary,
            paths=validation_paths,
            episode_horizon=episode_horizon,
            leakage_cost_mode=leakage_cost_mode,
            router_strength=float(summary["lower_action_router_strength"]),
        )
        parameter_sha256, validation_rows = _evaluate_selected(
            checkpoint_path,
            summary,
            trajectories,
            selected["config"],
            validation_paths,
            episode_horizon=episode_horizon,
            leakage_cost_mode=leakage_cost_mode,
            lower_action_context_start=lower_action_context_start,
        )
        if parameter_sha256 != selected["parameter_sha256"]:
            raise RuntimeError("selected distillation parameters did not replay")
        validation_snapshot_fn = _snapshot_fn(
            summary,
            validation_baseline_rows,
            risk_mode=risk_mode,
            cvar_alpha=cvar_alpha,
        )
        baseline_snapshot = validation_snapshot_fn(validation_baseline_rows)
        candidate_snapshot = validation_snapshot_fn(validation_rows)
        complete = _complete_endpoint_diagnostics(
            validation_rows,
            validation_baseline_rows,
            summary,
            risk_mode=risk_mode,
            cvar_alpha=cvar_alpha,
        )
        merit = restoration_snapshot_eligible(
            candidate_snapshot,
            baseline_snapshot,
            minimum_reduction=minimum_merit_reduction,
            funnel_multiplier=funnel_multiplier,
        )
        validation_supported = bool(merit and complete["complete"])
        validation = {
            "baseline_snapshot": baseline_snapshot,
            "candidate_snapshot": candidate_snapshot,
            "complete_endpoint_gate": complete,
            "merit_gate": bool(merit),
            "supported": validation_supported,
        }

    payload = {
        "probe_version": PROBE_VERSION,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "evidence_role": spec.EVIDENCE_ROLE,
        "environment": str(summary["environment"]),
        "optimizer_seed": int(raw_summary["optimizer_seed"]),
        "checkpoint": str(checkpoint_path),
        "baseline_parameter_sha256": baseline_parameter_sha256,
        "lower_action_context_start": int(lower_action_context_start),
        "distill_roots": list(map(int, distill_roots)),
        "design_roots": list(map(int, design_roots)),
        "validation_roots": list(map(int, validation_roots)),
        "candidate_count": len(candidates),
        "design_eligible_candidate_count": len(eligible_indices),
        "selected_index": selected_index,
        "selected_candidate": (
            None if selected_index is None else public_candidates[selected_index]
        ),
        "validation": validation,
        "validation_supported": validation_supported,
        "status": (
            "raw_policy_distillation_preflight_supported"
            if validation_supported else "no_complete_raw_policy_candidate"
        ),
        "selection_contract": spec.SELECTION_CONTRACT,
        "candidates": public_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--distill-roots", type=_parse_ints,
        default=spec.DISTILL_ROOTS,
    )
    parser.add_argument(
        "--design-roots", type=_parse_ints,
        default=spec.DESIGN_ROOTS,
    )
    parser.add_argument(
        "--validation-roots", type=_parse_ints,
        default=spec.VALIDATION_ROOTS,
    )
    parser.add_argument("--design-fold-count", type=int, default=spec.DESIGN_FOLD_COUNT)
    parser.add_argument("--episode-horizon", type=int, default=spec.EPISODE_HORIZON)
    parser.add_argument("--leakage-cost-mode", default=spec.LEAKAGE_COST_MODE)
    parser.add_argument("--risk-mode", default=spec.RISK_MODE)
    parser.add_argument("--cvar-alpha", type=float, default=spec.CVAR_ALPHA)
    parser.add_argument(
        "--minimum-merit-reduction",
        type=float,
        default=spec.MINIMUM_MERIT_REDUCTION,
    )
    parser.add_argument(
        "--funnel-multiplier", type=float, default=spec.FUNNEL_MULTIPLIER
    )
    parser.add_argument("--workers", type=int, default=spec.WORKERS)
    args = parser.parse_args()
    payload = run_probe(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        output_path=args.output,
        distill_roots=args.distill_roots,
        design_roots=args.design_roots,
        validation_roots=args.validation_roots,
        candidates=spec.CANDIDATES,
        design_fold_count=args.design_fold_count,
        episode_horizon=args.episode_horizon,
        leakage_cost_mode=args.leakage_cost_mode,
        risk_mode=args.risk_mode,
        cvar_alpha=args.cvar_alpha,
        minimum_merit_reduction=args.minimum_merit_reduction,
        funnel_multiplier=args.funnel_multiplier,
        workers=args.workers,
    )
    print(json.dumps({
        "status": payload["status"],
        "design_eligible_candidate_count": payload[
            "design_eligible_candidate_count"
        ],
        "selected_index": payload["selected_index"],
        "validation_supported": payload["validation_supported"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
