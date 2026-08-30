#!/usr/bin/env python3
"""Analyze the frozen v14.21 distributional actor preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts import mujoco_v14_21_distributional_actor_preflight_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _validate_payload(payload: dict[str, Any], environment: str, seed: int) -> None:
    expected = {
        "probe_version": spec.PROBE_VERSION,
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "direction_count": spec.DIRECTION_COUNT,
        "direction_seed": spec.DIRECTION_SEED,
        "perturb_rms": spec.PERTURB_RMS,
        "step_rms_values": list(spec.STEP_RMS_VALUES),
        "design_roots": list(spec.DESIGN_ROOTS),
        "validation_roots": list(spec.VALIDATION_ROOTS),
        "minimum_reduction": spec.MINIMUM_REDUCTION,
        "funnel_multiplier": spec.FUNNEL_MULTIPLIER,
        "design_path_count": spec.EXPECTED_PATH_COUNT,
        "validation_path_count": spec.EXPECTED_PATH_COUNT,
        "candidate_count": 2 * spec.DIRECTION_COUNT + 2 * len(
            spec.STEP_RMS_VALUES
        ),
        "workers": spec.WORKERS,
        "risk_mode": spec.RISK_MODE,
        "cvar_alpha": spec.CVAR_ALPHA,
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v14.21 payload contract mismatch: {mismatches}")


def _metric(snapshot: dict[str, Any] | None, key: str) -> float | int | None:
    if snapshot is None:
        return None
    value = snapshot[key]
    if key.endswith("count"):
        return int(value)
    return float(value)


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    rows = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v14.21 probe result: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _validate_payload(payload, environment, seed)
            selected = payload.get("selected_design_candidate")
            design_candidate = None if selected is None else selected["snapshot"]
            validation_baseline = payload.get("validation_baseline")
            validation_candidate = payload.get("validation_candidate")
            design_baseline_merit = float(
                payload["design_baseline"]["frequency_violation_merit"]
            )
            validation_baseline_merit = _metric(
                validation_baseline, "frequency_violation_merit"
            )
            validation_candidate_merit = _metric(
                validation_candidate, "frequency_violation_merit"
            )
            relative_reduction = None
            if (
                validation_baseline_merit is not None
                and validation_candidate_merit is not None
                and float(validation_baseline_merit) > 0.0
            ):
                relative_reduction = (
                    float(validation_baseline_merit)
                    - float(validation_candidate_merit)
                ) / float(validation_baseline_merit)
            rows.append({
                "environment": environment,
                "optimizer_seed": int(seed),
                "design_eligible_candidate_count": int(
                    payload["design_eligible_candidate_count"]
                ),
                "selected_source": None if selected is None else selected["source"],
                "selected_step_rms": None if selected is None else selected["step_rms"],
                "selected_orientation": (
                    None if selected is None else selected["orientation"]
                ),
                "design_baseline_merit": design_baseline_merit,
                "design_candidate_merit": _metric(
                    design_candidate, "frequency_violation_merit"
                ),
                "validation_baseline_merit": validation_baseline_merit,
                "validation_candidate_merit": validation_candidate_merit,
                "validation_relative_merit_reduction": relative_reduction,
                "validation_reward_violation_count": _metric(
                    validation_candidate, "reward_violation_count"
                ),
                "validation_frequency_violation_count": _metric(
                    validation_candidate, "frequency_violation_count"
                ),
                "validation_worst_frequency_violation": _metric(
                    validation_candidate, "worst_frequency_violation"
                ),
                "validation_supported": bool(payload["validation_supported"]),
            })
    supported = sum(row["validation_supported"] for row in rows)
    result = {
        "analysis_version": "mujoco_v14_21_distributional_actor_preflight_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(rows),
        "supported_cell_count": supported,
        "status": (
            "distributional_actor_restoration_preflight_supported"
            if supported == spec.EXPECTED_CELL_COUNT
            else "distributional_actor_restoration_preflight_not_supported"
        ),
        "cells": rows,
    }
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "distributional_actor_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "distributional_actor_preflight.csv").open(
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
