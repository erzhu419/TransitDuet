#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v14.16 crossed-restoration mechanism screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v14_16_crossed_restoration_mechanism_screen_spec as spec,
)


ANALYSIS_VERSION = "mujoco_v14_16_mechanism_screen_analysis_v1"
FREQUENCY_METRICS = (
    "LowerLFDriftAbs",
    "RawLowerLFDriftAbs",
    "LatentLowerLFDriftAbs",
    "UpperHFPowerAbs",
    "LatentUpperHFPowerAbs",
)
EXPECTED_MERGED_MANIFEST_STATUS = "development_screen_complete_unanalyzed"
RETURN_THRESHOLD = -0.02
FREQUENCY_THRESHOLD = -math.log(0.95)
EPSILON = 1e-12


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"evaluation rows are empty: {path}")
    return rows


def _index_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {
        (str(row["disturbance_mode"]), int(row["seed"])): row
        for row in rows
    }
    if len(indexed) != len(rows):
        raise ValueError("evaluation paths must be unique")
    return indexed


def _paired_path_effects(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = _index_rows(candidate_rows)
    baseline = _index_rows(baseline_rows)
    if set(candidate) != set(baseline):
        raise ValueError("candidate and comparator paths do not align")
    effects: list[dict[str, Any]] = []
    for mode, seed in sorted(candidate):
        left = candidate[mode, seed]
        right = baseline[mode, seed]
        baseline_return = float(right["episode_return"])
        candidate_return = float(left["episode_return"])
        row: dict[str, Any] = {
            "disturbance_mode": mode,
            "seed": int(seed),
            "normalized_episode_return": (
                (candidate_return - baseline_return)
                / max(abs(baseline_return), 1.0)
            ),
        }
        for metric in FREQUENCY_METRICS:
            baseline_value = float(right[metric])
            candidate_value = float(left[metric])
            if (
                not math.isfinite(baseline_value)
                or not math.isfinite(candidate_value)
                or baseline_value < 0.0
                or candidate_value < 0.0
            ):
                raise ValueError("frequency endpoints must be finite and non-negative")
            row[metric] = math.log(
                (baseline_value + EPSILON) / (candidate_value + EPSILON)
            )
        effects.append(row)
    return effects


def _pooled_effects(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = ("normalized_episode_return", *FREQUENCY_METRICS)
    if not rows:
        raise ValueError("paired effects cannot be empty")
    pooled = {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metrics
    }
    if not np.all(np.isfinite(list(pooled.values()))):
        raise ValueError("pooled paired effects must be finite")
    return pooled


def _effect_gate(effects: dict[str, float]) -> dict[str, Any]:
    endpoint_pass = {
        "normalized_episode_return": bool(
            effects["normalized_episode_return"] >= RETURN_THRESHOLD
        ),
        **{
            metric: bool(effects[metric] >= FREQUENCY_THRESHOLD)
            for metric in FREQUENCY_METRICS
        },
    }
    return {
        "endpoint_pass": endpoint_pass,
        "pass_count": int(sum(endpoint_pass.values())),
        "complete": bool(all(endpoint_pass.values())),
    }


def _cell_dir(
    run: Path,
    *,
    environment: str,
    arm: str,
    optimizer_seed: int,
) -> Path:
    return (
        run / "cells" / environment / arm
        / f"replicate_{int(optimizer_seed)}"
    )


def _input_sha256(run: Path, cell_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    paths = [
        run / "preregistration.json",
        run / "merged" / "cell_manifest.json",
        run / "merged" / "run_scoped_result_sync.json",
    ]
    for cell in sorted(cell_paths):
        paths.extend((
            cell / "cell_summary.json",
            cell / "evaluation_rows.csv",
            cell / "training_history.json",
        ))
    for path in paths:
        relative = str(path.relative_to(run)).encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def analyze(run_dir: Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    preregistration = _read_json(run / "preregistration.json")
    manifest = _read_json(run / "merged" / "cell_manifest.json")
    if not (
        preregistration.get("development_protocol_version")
        == spec.DEVELOPMENT_PROTOCOL_VERSION
        and preregistration.get("frozen_algorithm_revision")
        == spec.FROZEN_ALGORITHM_REVISION
        and preregistration.get("frozen_source_manifest_sha256")
        == spec.FROZEN_SOURCE_MANIFEST_SHA256
        and preregistration.get("dispatched_environment_subset")
        == list(spec.ENVIRONMENTS)
        and preregistration.get("dispatched_optimizer_seed_subset")
        == list(spec.OPTIMIZER_SEEDS)
        and preregistration.get("arms")
        == json.loads(json.dumps(spec.ARMS))
    ):
        raise ValueError("v14.16 preregistration identity mismatch")
    if manifest.get("status") != EXPECTED_MERGED_MANIFEST_STATUS:
        raise ValueError("v14.16 merged cell manifest is not valid")

    expected_cells = (
        len(spec.ENVIRONMENTS)
        * len(spec.OPTIMIZER_SEEDS)
        * (len(spec.ARMS) + 1)
    )
    if int(manifest.get("cell_count", -1)) != expected_cells:
        raise ValueError("v14.16 merged cell count is incomplete")

    replicate_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            baseline_path = _cell_dir(
                run,
                environment=environment,
                arm=spec.MATCHED_COMPARATOR_ARM,
                optimizer_seed=optimizer_seed,
            )
            baseline_rows = _read_rows(
                baseline_path / "evaluation_rows.csv"
            )
            for arm in spec.LEARNED_ARMS:
                path = _cell_dir(
                    run,
                    environment=environment,
                    arm=arm,
                    optimizer_seed=optimizer_seed,
                )
                summary = _read_json(path / "cell_summary.json")
                effects = _paired_path_effects(
                    _read_rows(path / "evaluation_rows.csv"),
                    baseline_rows,
                )
                pooled = _pooled_effects(effects)
                gate = _effect_gate(pooled)
                selected_reward_violations = int(summary.get(
                    "deployment_frequency_closed_loop_guard_selected_reward_"
                    "violation_count",
                    -1,
                ))
                selected_frequency_violations = int(summary.get(
                    "deployment_frequency_closed_loop_guard_selected_frequency_"
                    "violation_count",
                    -1,
                ))
                engineering_pass = bool(
                    selected_reward_violations == 0
                    and int(summary.get(
                        "deployment_frequency_closed_loop_guard_effective_"
                        "update_count",
                        0,
                    )) >= 1
                    and summary.get("protocol_version")
                    == spec.FROZEN_CORE_PROTOCOL_VERSION
                    and summary.get("protocol_version_selection")
                    == spec.FROZEN_CORE_PROTOCOL_VERSION
                )
                replicate_rows.append({
                    "environment": environment,
                    "optimizer_seed": int(optimizer_seed),
                    "arm": arm,
                    **pooled,
                    "endpoint_pass_count": int(gate["pass_count"]),
                    "complete_effect_gate": bool(gate["complete"]),
                    "engineering_pass": engineering_pass,
                    "selected_checkpoint_iteration": int(summary.get(
                        "selected_checkpoint_iteration", -2
                    )),
                    "selected_reward_violation_count": (
                        selected_reward_violations
                    ),
                    "selected_frequency_violation_count": (
                        selected_frequency_violations
                    ),
                    "closed_loop_effective_update_count": int(summary.get(
                        "deployment_frequency_closed_loop_guard_effective_"
                        "update_count",
                        0,
                    )),
                    "restoration_reward_actor_frozen": bool(summary.get(
                        "deployment_frequency_restoration_freeze_reward_actor",
                        False,
                    )),
                    "pathwise_robust": bool(summary.get(
                        "deployment_frequency_pathwise_robust", False
                    )),
                    "anchor_replay_path_count": int(summary.get(
                        "deployment_frequency_anchor_state_replay_path_count",
                        -1,
                    )),
                })
                for row in effects:
                    path_rows.append({
                        "environment": environment,
                        "optimizer_seed": int(optimizer_seed),
                        "arm": arm,
                        **row,
                    })

    arm_rows: list[dict[str, Any]] = []
    metrics = ("normalized_episode_return", *FREQUENCY_METRICS)
    for arm in spec.LEARNED_ARMS:
        selected = [row for row in replicate_rows if row["arm"] == arm]
        means = {
            metric: float(np.mean([float(row[metric]) for row in selected]))
            for metric in metrics
        }
        environment_gates = {}
        for environment in spec.ENVIRONMENTS:
            environment_rows = [
                row for row in selected if row["environment"] == environment
            ]
            environment_gates[environment] = _effect_gate({
                metric: float(np.mean([
                    float(row[metric]) for row in environment_rows
                ]))
                for metric in metrics
            })
        engineering_count = sum(
            bool(row["engineering_pass"]) for row in selected
        )
        complete_count = sum(
            bool(row["complete_effect_gate"]) for row in selected
        )
        arm_rows.append({
            "arm": arm,
            **means,
            "replicate_count": len(selected),
            "engineering_pass_count": engineering_count,
            "complete_effect_gate_count": complete_count,
            "environment_complete_count": sum(
                bool(gate["complete"])
                for gate in environment_gates.values()
            ),
            "environment_gates": environment_gates,
            "mean_endpoint_margin": float(np.mean([
                means["normalized_episode_return"] - RETURN_THRESHOLD,
                *(
                    means[metric] - FREQUENCY_THRESHOLD
                    for metric in FREQUENCY_METRICS
                ),
            ])),
        })

    arm_rows.sort(
        key=lambda row: (
            int(row["environment_complete_count"]),
            int(row["complete_effect_gate_count"]),
            int(row["engineering_pass_count"]),
            float(row["mean_endpoint_margin"]),
        ),
        reverse=True,
    )
    primary = next(
        row for row in arm_rows
        if row["arm"] == spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM
    )
    primary_ready = bool(
        int(primary["environment_complete_count"]) == len(spec.ENVIRONMENTS)
        and int(primary["engineering_pass_count"])
        == len(spec.ENVIRONMENTS) * len(spec.OPTIMIZER_SEEDS)
    )
    status = (
        "primary_mechanism_ready_for_fresh_multiseed_development"
        if primary_ready else "primary_mechanism_not_ready"
    )
    all_cell_paths = [
        run / "anchors" / environment / f"replicate_{optimizer_seed}"
        for environment in spec.ENVIRONMENTS
        for optimizer_seed in spec.OPTIMIZER_SEEDS
    ] + [
        _cell_dir(
            run,
            environment=environment,
            arm=arm,
            optimizer_seed=optimizer_seed,
        )
        for environment in spec.ENVIRONMENTS
        for optimizer_seed in spec.OPTIMIZER_SEEDS
        for arm in spec.ARMS
    ]
    return {
        "analysis_version": ANALYSIS_VERSION,
        "status": status,
        "evidence_role": "mechanism_screen_development_not_confirmation",
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": (
            spec.FROZEN_SOURCE_MANIFEST_SHA256
        ),
        "statistical_unit": "optimizer_seed",
        "optimizer_seed_count": len(spec.OPTIMIZER_SEEDS),
        "heldout_paths_are_not_replicates": True,
        "return_noninferiority_threshold": RETURN_THRESHOLD,
        "frequency_log_reduction_threshold": FREQUENCY_THRESHOLD,
        "primary_candidate_arm": spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM,
        "primary_ready": primary_ready,
        "arm_ranking": arm_rows,
        "replicate_rows": replicate_rows,
        "path_rows": path_rows,
        "input_sha256": _input_sha256(run, all_cell_paths),
        "claim_boundary": (
            "Three optimizer seeds provide development diagnostics only. "
            "Held-out environment paths are paired observations, not "
            "independent statistical replicates. Any nominated mechanism "
            "requires a frozen larger multiseed development screen followed "
            "by fresh confirmation seeds."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append({
            key: (
                json.dumps(value, sort_keys=True)
                if isinstance(value, (dict, list)) else value
            )
            for key, value in row.items()
        })
    fields = list(flattened[0]) if flattened else []
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(flattened)


def write_analysis(output_dir: Path, decision: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    decision_path = output / "decision.json"
    if decision_path.exists() and decision_path.read_text(
        encoding="utf-8"
    ) != rendered:
        raise RuntimeError("existing v14.16 decision differs")
    decision_path.write_text(rendered, encoding="utf-8")
    _write_csv(output / "arm_ranking.csv", decision["arm_ranking"])
    _write_csv(output / "replicate_rows.csv", decision["replicate_rows"])
    _write_csv(output / "path_rows.csv", decision["path_rows"])
    lines = [
        "# MuJoCo v14.16 Crossed Restoration Mechanism Screen",
        "",
        f"- Status: `{decision['status']}`",
        f"- Primary arm: `{decision['primary_candidate_arm']}`",
        f"- Optimizer seeds: `{decision['optimizer_seed_count']}`",
        "- Statistical unit: optimizer seed; held-out paths are paired only.",
        "",
        "| rank | arm | env complete | cell complete | engineering | return | mean margin |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(decision["arm_ranking"], start=1):
        lines.append(
            f"| {rank} | {row['arm']} | "
            f"{row['environment_complete_count']}/{len(spec.ENVIRONMENTS)} | "
            f"{row['complete_effect_gate_count']}/{row['replicate_count']} | "
            f"{row['engineering_pass_count']}/{row['replicate_count']} | "
            f"{row['normalized_episode_return']:.6f} | "
            f"{row['mean_endpoint_margin']:.6f} |"
        )
    lines.extend(("", decision["claim_boundary"], ""))
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = analyze(args.run_dir)
    write_analysis(args.output_dir, decision)
    print(
        f"mujoco_v14_16_mechanism status={decision['status']} "
        f"primary_ready={decision['primary_ready']}"
    )


if __name__ == "__main__":
    main()
