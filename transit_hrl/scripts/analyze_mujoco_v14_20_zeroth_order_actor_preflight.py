#!/usr/bin/env python3
"""Analyze the frozen v14.20 zeroth-order actor restoration preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scripts import mujoco_v14_20_zeroth_order_actor_preflight_spec as spec


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _validate_payload(payload: dict[str, Any], environment: str, seed: int) -> None:
    expected = {
        "probe_version": "mujoco_zeroth_order_actor_restoration_probe_v1",
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
        "design_path_count": 8,
        "validation_path_count": 8,
        "candidate_count": 2 * spec.DIRECTION_COUNT + 2 * len(
            spec.STEP_RMS_VALUES
        ),
    }
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items() if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v14.20 payload contract mismatch: {mismatches}")


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    rows = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v14.20 probe result: {path}")
            payload = json.loads(path.read_text())
            _validate_payload(payload, environment, seed)
            selected = payload.get("selected_design_candidate")
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
                "design_baseline_merit": float(
                    payload["design_baseline"]["frequency_violation_merit"]
                ),
                "design_candidate_merit": (
                    None if selected is None else float(
                        selected["snapshot"]["frequency_violation_merit"]
                    )
                ),
                "validation_baseline_merit": (
                    None if payload["validation_baseline"] is None else float(
                        payload["validation_baseline"][
                            "frequency_violation_merit"
                        ]
                    )
                ),
                "validation_candidate_merit": (
                    None if payload["validation_candidate"] is None else float(
                        payload["validation_candidate"][
                            "frequency_violation_merit"
                        ]
                    )
                ),
                "validation_supported": bool(payload["validation_supported"]),
            })
    supported = sum(row["validation_supported"] for row in rows)
    result = {
        "analysis_version": "mujoco_v14_20_zeroth_order_actor_preflight_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(rows),
        "supported_cell_count": supported,
        "status": (
            "zeroth_order_actor_restoration_preflight_supported"
            if supported == spec.EXPECTED_CELL_COUNT
            else "zeroth_order_actor_restoration_preflight_not_supported"
        ),
        "cells": rows,
    }
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "zeroth_order_actor_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "zeroth_order_actor_preflight.csv").open(
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
