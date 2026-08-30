#!/usr/bin/env python3
"""Probe deployment-aligned zeroth-order restoration of MuJoCo actor heads."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from freq_hrl.experiments.mujoco.control_validation import (
    _model_parameter_sha256,
    crossed_deployment_frequency_guard_paths,
    paired_closed_loop_guard_snapshot,
)
from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as v14_17_spec
from scripts.probe_mujoco_radial_restoration import (
    _actor_output_head,
    _evaluate_rows,
    _load_model,
    _v14_17_anchor_profile,
)


LEGACY_PROBE_VERSION = "mujoco_zeroth_order_actor_restoration_probe_v1"
DISTRIBUTIONAL_PROBE_VERSION = "mujoco_zeroth_order_actor_restoration_probe_v2"
PROBE_VERSIONS = {LEGACY_PROBE_VERSION, DISTRIBUTIONAL_PROBE_VERSION}


def _parse_positive_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if (
        not values
        or len(set(values)) != len(values)
        or any(not np.isfinite(item) or item <= 0.0 for item in values)
    ):
        raise ValueError("positive float registry must be finite, unique, and nonempty")
    return values


def actor_output_head_vector(model: Any) -> np.ndarray:
    tensors = []
    for actor in (model.upper_actor, model.lower_actor):
        head = _actor_output_head(actor)
        tensors.append(head.weight.detach().cpu().numpy().reshape(-1))
        if head.bias is not None:
            tensors.append(head.bias.detach().cpu().numpy().reshape(-1))
    return np.concatenate(tensors).astype(np.float64, copy=True)


def apply_actor_output_head_delta(model: Any, delta: np.ndarray) -> None:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    expected = actor_output_head_vector(model).size
    if values.size != expected or not np.all(np.isfinite(values)):
        raise ValueError("actor output-head delta must be finite and aligned")
    offset = 0
    with torch.no_grad():
        for actor in (model.upper_actor, model.lower_actor):
            head = _actor_output_head(actor)
            for parameter in (head.weight, head.bias):
                if parameter is None:
                    continue
                count = parameter.numel()
                update = torch.as_tensor(
                    values[offset:offset + count].reshape(parameter.shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                parameter.add_(update)
                offset += count
    if offset != values.size:
        raise RuntimeError("actor output-head delta application was incomplete")


def antithetic_directions(
    dimension: int, count: int, seed: int
) -> list[np.ndarray]:
    if int(dimension) < 1 or int(count) < 1:
        raise ValueError("zeroth-order directions require positive dimensions")
    rng = np.random.default_rng(int(seed))
    return [
        rng.choice((-1.0, 1.0), size=int(dimension)).astype(np.float64)
        for _ in range(int(count))
    ]


def _fitness_key(snapshot: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(snapshot["reward_violation_count"]),
        float(snapshot["frequency_violation_merit"]),
        float(snapshot["worst_frequency_violation"]),
        float(snapshot["frequency_violation_count"]),
    )


def ranked_antithetic_gradient(
    directions: list[np.ndarray],
    pair_snapshots: list[tuple[dict[str, Any], dict[str, Any]]],
) -> np.ndarray:
    if not directions or len(directions) != len(pair_snapshots):
        raise ValueError("antithetic direction and snapshot registries must align")
    dimension = directions[0].size
    if any(direction.size != dimension for direction in directions):
        raise ValueError("antithetic directions must share one dimension")
    flattened = [
        snapshot for pair in pair_snapshots for snapshot in pair
    ]
    order = sorted(range(len(flattened)), key=lambda index: _fitness_key(
        flattened[index]
    ))
    ranks = np.empty(len(flattened), dtype=np.float64)
    denominator = max(1, len(flattened) - 1)
    for rank, index in enumerate(order):
        ranks[index] = float(rank) / float(denominator)
    gradient = np.zeros(dimension, dtype=np.float64)
    for index, direction in enumerate(directions):
        plus_rank = ranks[2 * index]
        minus_rank = ranks[2 * index + 1]
        gradient += (plus_rank - minus_rank) * direction
    gradient /= float(len(directions))
    rms = float(np.sqrt(np.mean(np.square(gradient))))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("antithetic rank gradient is degenerate")
    return gradient / rms


def _paths_for_roots(env_id: str, roots: Iterable[int]) -> list[dict[str, Any]]:
    seeds, modes = crossed_deployment_frequency_guard_paths(
        tuple(map(int, roots)),
        v14_17_spec.TRAINING_DISTURBANCE_MODES,
        env_id=str(env_id),
    )
    return [
        {"seed": int(seed), "disturbance_mode": str(modes[int(seed)])}
        for seed in seeds
    ]


def _evaluate_delta_job(job: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one actor-head delta on all paths assigned to a worker."""
    torch.set_num_threads(1)
    checkpoint = torch.load(
        job["checkpoint_path"], map_location="cpu", weights_only=False
    )
    model = _load_model(checkpoint)
    apply_actor_output_head_delta(
        model, np.asarray(job["delta"], dtype=np.float64)
    )
    rows = _evaluate_rows(
        model,
        summary=job["summary"],
        paths=job["paths"],
        episode_horizon=int(job["episode_horizon"]),
        leakage_cost_mode=str(job["leakage_cost_mode"]),
        router_strength=float(job["router_strength"]),
    )
    return {
        "parameter_sha256": _model_parameter_sha256(model),
        "rows": rows,
    }


def _snapshot_fn(
    summary: dict[str, Any],
    baseline_rows: list[dict[str, Any]],
    *,
    risk_mode: str,
    cvar_alpha: float,
):
    modes = tuple(v14_17_spec.TRAINING_DISTURBANCE_MODES)

    def snapshot(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return paired_closed_loop_guard_snapshot(
            rows,
            baseline_rows=baseline_rows,
            expected_modes=modes,
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

    return snapshot


def _eligible(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    minimum_reduction: float,
    funnel_multiplier: float,
) -> bool:
    return bool(
        int(candidate["reward_violation_count"]) == 0
        and float(candidate["frequency_violation_merit"])
        <= float(baseline["frequency_violation_merit"])
        * (1.0 - float(minimum_reduction))
        and float(candidate["worst_frequency_violation"])
        <= float(baseline["worst_frequency_violation"])
        * float(funnel_multiplier)
    )


def run_probe(
    *,
    checkpoint_path: Path,
    summary_path: Path,
    output_path: Path,
    direction_count: int,
    direction_seed: int,
    perturb_rms: float,
    step_rms_values: tuple[float, ...],
    design_roots: tuple[int, ...],
    validation_roots: tuple[int, ...],
    minimum_reduction: float,
    funnel_multiplier: float,
    episode_horizon: int,
    leakage_cost_mode: str,
    workers: int = 1,
    risk_mode: str | None = None,
    cvar_alpha: float | None = None,
    probe_version: str = LEGACY_PROBE_VERSION,
) -> dict[str, Any]:
    if int(workers) < 1:
        raise ValueError("zeroth-order probe requires at least one worker")
    if str(probe_version) not in PROBE_VERSIONS:
        raise ValueError(f"unsupported zeroth-order probe version: {probe_version}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary, _ = _v14_17_anchor_profile(raw_summary)
    baseline_model = _load_model(checkpoint)
    baseline_hash = _model_parameter_sha256(baseline_model)
    if baseline_hash != str(checkpoint["frozen_parameter_sha256"]):
        raise RuntimeError("zeroth-order probe did not reconstruct the frozen actor")
    resolved_risk_mode = str(
        risk_mode
        if risk_mode is not None
        else summary["deployment_frequency_closed_loop_risk_mode"]
    )
    resolved_cvar_alpha = float(
        cvar_alpha
        if cvar_alpha is not None
        else summary["deployment_frequency_closed_loop_cvar_alpha"]
    )
    design_paths = _paths_for_roots(summary["environment"], design_roots)
    validation_paths = _paths_for_roots(
        summary["environment"], validation_roots
    )
    if {
        (row["disturbance_mode"], row["seed"]) for row in design_paths
    } & {
        (row["disturbance_mode"], row["seed"]) for row in validation_paths
    }:
        raise RuntimeError("zeroth-order design and validation paths overlap")

    dimension = actor_output_head_vector(baseline_model).size
    directions = antithetic_directions(dimension, direction_count, direction_seed)
    checkpoint_absolute = str(checkpoint_path.resolve())

    executor = None
    if int(workers) > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=multiprocessing.get_context("spawn"),
        )

    def evaluate_deltas(
        deltas: list[np.ndarray], paths: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        jobs = [{
            "checkpoint_path": checkpoint_absolute,
            "summary": summary,
            "paths": paths,
            "episode_horizon": int(episode_horizon),
            "leakage_cost_mode": str(leakage_cost_mode),
            "router_strength": float(summary["lower_action_router_strength"]),
            "delta": np.asarray(delta, dtype=np.float64),
        } for delta in deltas]
        if executor is None:
            return [_evaluate_delta_job(job) for job in jobs]
        return list(executor.map(_evaluate_delta_job, jobs))

    try:
        direct_specs = [
            (index, sign, sign * float(perturb_rms) * direction)
            for index, direction in enumerate(directions)
            for sign in (1.0, -1.0)
        ]
        design_results = evaluate_deltas(
            [np.zeros(dimension, dtype=np.float64)]
            + [row[2] for row in direct_specs],
            design_paths,
        )
        if design_results[0]["parameter_sha256"] != baseline_hash:
            raise RuntimeError("parallel baseline reconstruction changed the actor")
        design_baseline_rows = design_results[0]["rows"]
        design_snapshot = _snapshot_fn(
            summary,
            design_baseline_rows,
            risk_mode=resolved_risk_mode,
            cvar_alpha=resolved_cvar_alpha,
        )
        design_baseline = design_snapshot(design_baseline_rows)
        direct_candidates = []
        pair_snapshots = []
        for pair_index in range(len(directions)):
            pair = []
            for orientation_index in range(2):
                result_index = 1 + 2 * pair_index + orientation_index
                index, orientation, delta = direct_specs[
                    2 * pair_index + orientation_index
                ]
                result = design_results[result_index]
                snapshot = design_snapshot(result["rows"])
                pair.append(snapshot)
                direct_candidates.append({
                    "source": "antithetic_direction",
                    "direction_index": int(index),
                    "orientation": float(orientation),
                    "step_rms": float(perturb_rms),
                    "parameter_sha256": result["parameter_sha256"],
                    "snapshot": snapshot,
                    "_delta": delta,
                })
            pair_snapshots.append((pair[0], pair[1]))
        gradient = ranked_antithetic_gradient(directions, pair_snapshots)
        gradient_specs = [
            (orientation, step_rms, orientation * float(step_rms) * gradient)
            for step_rms in step_rms_values
            for orientation in (-1.0, 1.0)
        ]
        gradient_results = evaluate_deltas(
            [row[2] for row in gradient_specs], design_paths
        )
        gradient_candidates = []
        for (orientation, step_rms, delta), result in zip(
            gradient_specs, gradient_results, strict=True
        ):
            gradient_candidates.append({
                "source": "ranked_antithetic_gradient",
                "direction_index": None,
                "orientation": float(orientation),
                "step_rms": float(step_rms),
                "parameter_sha256": result["parameter_sha256"],
                "snapshot": design_snapshot(result["rows"]),
                "_delta": delta,
            })
        candidates = direct_candidates + gradient_candidates
        for candidate in candidates:
            candidate["design_eligible"] = _eligible(
                candidate["snapshot"],
                design_baseline,
                minimum_reduction=minimum_reduction,
                funnel_multiplier=funnel_multiplier,
            )
        eligible = [
            candidate for candidate in candidates
            if candidate["design_eligible"]
        ]
        eligible.sort(key=lambda candidate: (
            _fitness_key(candidate["snapshot"]),
            float(candidate["step_rms"]),
            str(candidate["source"]),
            float(candidate["orientation"]),
        ))
        selected = eligible[0] if eligible else None
        validation_baseline = None
        validation_candidate = None
        validation_supported = False
        if selected is not None:
            validation_results = evaluate_deltas(
                [
                    np.zeros(dimension, dtype=np.float64),
                    np.asarray(selected["_delta"], dtype=np.float64),
                ],
                validation_paths,
            )
            if validation_results[0]["parameter_sha256"] != baseline_hash:
                raise RuntimeError(
                    "parallel validation baseline changed the actor"
                )
            if validation_results[1]["parameter_sha256"] != str(
                selected["parameter_sha256"]
            ):
                raise RuntimeError(
                    "selected zeroth-order actor delta was not reconstructed exactly"
                )
            validation_baseline_rows = validation_results[0]["rows"]
            validation_snapshot = _snapshot_fn(
                summary,
                validation_baseline_rows,
                risk_mode=resolved_risk_mode,
                cvar_alpha=resolved_cvar_alpha,
            )
            validation_baseline = validation_snapshot(
                validation_baseline_rows
            )
            validation_candidate = validation_snapshot(
                validation_results[1]["rows"]
            )
            validation_supported = _eligible(
                validation_candidate,
                validation_baseline,
                minimum_reduction=minimum_reduction,
                funnel_multiplier=funnel_multiplier,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    public_candidates = [
        {key: value for key, value in candidate.items() if key != "_delta"}
        for candidate in candidates
    ]
    public_selected = None
    if selected is not None:
        selected_index = next(
            index for index, candidate in enumerate(candidates)
            if candidate is selected
        )
        public_selected = public_candidates[selected_index]
    payload = {
        "probe_version": str(probe_version),
        "checkpoint": str(checkpoint_path),
        "summary": str(summary_path),
        "environment": str(summary["environment"]),
        "optimizer_seed": int(summary["optimizer_seed"]),
        "baseline_parameter_sha256": baseline_hash,
        "actor_output_head_dimension": int(dimension),
        "direction_count": int(direction_count),
        "direction_seed": int(direction_seed),
        "perturb_rms": float(perturb_rms),
        "step_rms_values": list(step_rms_values),
        "design_roots": list(design_roots),
        "validation_roots": list(validation_roots),
        "design_path_count": len(design_paths),
        "validation_path_count": len(validation_paths),
        "minimum_reduction": float(minimum_reduction),
        "funnel_multiplier": float(funnel_multiplier),
        "workers": int(workers),
        "risk_mode": resolved_risk_mode,
        "cvar_alpha": resolved_cvar_alpha,
        "design_baseline": design_baseline,
        "candidate_count": len(candidates),
        "design_eligible_candidate_count": len(eligible),
        "selected_design_candidate": public_selected,
        "validation_baseline": validation_baseline,
        "validation_candidate": validation_candidate,
        "validation_supported": bool(validation_supported),
        "candidates": public_candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--direction-count", type=int, default=8)
    parser.add_argument("--direction-seed", type=int, required=True)
    parser.add_argument("--perturb-rms", type=float, default=1e-6)
    parser.add_argument(
        "--step-rms-values", default="1e-8,3e-8,1e-7,3e-7,1e-6"
    )
    parser.add_argument("--design-roots", required=True)
    parser.add_argument("--validation-roots", required=True)
    parser.add_argument("--minimum-reduction", type=float, default=1e-4)
    parser.add_argument("--funnel-multiplier", type=float, default=3.0)
    parser.add_argument("--episode-horizon", type=int, default=1000)
    parser.add_argument("--leakage-cost-mode", default="power_excess")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--risk-mode")
    parser.add_argument("--cvar-alpha", type=float)
    parser.add_argument(
        "--probe-version", choices=sorted(PROBE_VERSIONS),
        default=LEGACY_PROBE_VERSION,
    )
    args = parser.parse_args()
    roots = lambda value: tuple(  # noqa: E731
        int(item.strip()) for item in value.split(",") if item.strip()
    )
    payload = run_probe(
        checkpoint_path=args.checkpoint,
        summary_path=args.summary,
        output_path=args.output,
        direction_count=args.direction_count,
        direction_seed=args.direction_seed,
        perturb_rms=args.perturb_rms,
        step_rms_values=_parse_positive_floats(args.step_rms_values),
        design_roots=roots(args.design_roots),
        validation_roots=roots(args.validation_roots),
        minimum_reduction=args.minimum_reduction,
        funnel_multiplier=args.funnel_multiplier,
        episode_horizon=args.episode_horizon,
        leakage_cost_mode=args.leakage_cost_mode,
        workers=args.workers,
        risk_mode=args.risk_mode,
        cvar_alpha=args.cvar_alpha,
        probe_version=args.probe_version,
    )
    print(json.dumps({
        "output": str(args.output),
        "design_eligible_candidate_count": payload[
            "design_eligible_candidate_count"
        ],
        "validation_supported": payload["validation_supported"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
