#!/usr/bin/env python3
"""Analyze the development-only MuJoCo v15 raw-policy preflight."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from scripts import mujoco_v15_raw_policy_distillation_preflight_spec as spec
from scripts.probe_mujoco_raw_policy_distillation import (
    FREQUENCY_ENDPOINTS,
    PROBE_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]


def cell_relative_dir(run_name: str, environment: str, seed: int) -> Path:
    return Path("results") / run_name / "cells" / environment / f"replicate_{seed}"


def _validated_cell(
    environment: str, seed: int, payload: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("probe_version") != PROBE_VERSION:
        raise ValueError("v15 probe version does not match the frozen preflight")
    if payload.get("development_protocol_version") != spec.DEVELOPMENT_PROTOCOL_VERSION:
        raise ValueError("v15 development protocol version drifted")
    if payload.get("evidence_role") != spec.EVIDENCE_ROLE:
        raise ValueError("v15 evidence role drifted")
    if (
        str(payload.get("environment")) != str(environment)
        or int(payload.get("optimizer_seed", -1)) != int(seed)
    ):
        raise ValueError("v15 payload cell identity does not match its path")
    for key, expected in (
        ("distill_roots", spec.DISTILL_ROOTS),
        ("design_roots", spec.DESIGN_ROOTS),
        ("validation_roots", spec.VALIDATION_ROOTS),
    ):
        if tuple(map(int, payload.get(key, ()))) != tuple(expected):
            raise ValueError(f"v15 {key} drifted")
    if int(payload.get("candidate_count", -1)) != len(spec.CANDIDATES):
        raise ValueError("v15 candidate count drifted")
    candidates = list(payload.get("candidates", ()))
    if (
        len(candidates) != len(spec.CANDIDATES)
        or [int(item["candidate_index"]) for item in candidates]
        != list(range(len(spec.CANDIDATES)))
    ):
        raise ValueError("v15 candidate registry is incomplete or reordered")

    supported = bool(payload.get("validation_supported", False))
    validation = payload.get("validation")
    selected_index = payload.get("selected_index")
    selected = payload.get("selected_candidate")
    if supported:
        if selected_index is None or not isinstance(validation, dict):
            raise ValueError("supported v15 cell omits its selected validation")
        if not bool(validation.get("supported", False)):
            raise ValueError("v15 top-level and validation support disagree")
        complete = dict(validation.get("complete_endpoint_gate") or {})
        endpoint_values = dict(
            complete.get("frequency_endpoint_maximum_normalized_violations")
            or {}
        )
        if (
            not bool(complete.get("complete", False))
            or set(endpoint_values) != set(FREQUENCY_ENDPOINTS)
            or any(float(value) > 1e-10 for value in endpoint_values.values())
            or float(complete.get("reward_maximum_normalized_violation", 1.0))
            > 1e-10
            or not bool(validation.get("merit_gate", False))
        ):
            raise ValueError("supported v15 cell does not satisfy its complete gate")
    elif payload.get("status") == "raw_policy_distillation_preflight_supported":
        raise ValueError("unsupported v15 cell has a supported status")

    return {
        "environment": str(environment),
        "optimizer_seed": int(seed),
        "status": str(payload.get("status")),
        "validation_supported": supported,
        "design_eligible_candidate_count": int(
            payload.get("design_eligible_candidate_count", 0)
        ),
        "selected_index": None if selected_index is None else int(selected_index),
        "selected_config": None if selected is None else dict(selected["config"]),
        "validation_frequency_violation_merit": (
            None if not isinstance(validation, dict)
            else float(validation["candidate_snapshot"]["frequency_violation_merit"])
        ),
        "validation_worst_frequency_violation": (
            None if not isinstance(validation, dict)
            else float(validation["candidate_snapshot"]["worst_frequency_violation"])
        ),
        "validation_reward_maximum_normalized_violation": (
            None if not isinstance(validation, dict)
            else float(validation["complete_endpoint_gate"][
                "reward_maximum_normalized_violation"
            ])
        ),
    }


def analyze_payloads(
    payloads: Iterable[tuple[str, int, dict[str, Any]]],
) -> dict[str, Any]:
    cells = []
    seen: set[tuple[str, int]] = set()
    for environment, seed, payload in payloads:
        key = (str(environment), int(seed))
        if key in seen:
            raise ValueError(f"duplicate v15 cell: {key}")
        seen.add(key)
        cells.append(_validated_cell(*key, payload))
    expected = {
        (environment, int(seed))
        for environment in spec.ENVIRONMENTS
        for seed in spec.OPTIMIZER_SEEDS
    }
    if seen != expected:
        raise ValueError(
            "v15 cell registry mismatch: "
            f"missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    support_count = sum(bool(cell["validation_supported"]) for cell in cells)
    return {
        "analysis_version": "mujoco_v15_raw_policy_distillation_analysis_v1",
        "evidence_role": spec.EVIDENCE_ROLE,
        "selection_contract": spec.SELECTION_CONTRACT,
        "cell_count": len(cells),
        "validation_supported_count": support_count,
        "status": (
            "raw_policy_distillation_preflight_supported_all_environments"
            if support_count == spec.EXPECTED_CELL_COUNT
            else "raw_policy_distillation_preflight_not_supported"
        ),
        "cells": sorted(
            cells, key=lambda item: (item["environment"], item["optimizer_seed"])
        ),
    }


def analyze_run(run_name: str, output_dir: Path | None = None) -> dict[str, Any]:
    payloads = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            path = ROOT / cell_relative_dir(run_name, environment, seed) / "probe.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing v15 probe result: {path}")
            payloads.append((environment, seed, json.loads(path.read_text())))
    result = analyze_payloads(payloads)
    target = output_dir or (ROOT / "results" / run_name / "analysis")
    target.mkdir(parents=True, exist_ok=True)
    (target / "raw_policy_distillation_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (target / "raw_policy_distillation_cells.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "environment", "optimizer_seed", "status",
            "validation_supported", "design_eligible_candidate_count",
            "selected_index", "selected_config",
            "validation_frequency_violation_merit",
            "validation_worst_frequency_violation",
            "validation_reward_maximum_normalized_violation",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["cells"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = analyze_run(args.run_name, args.output_dir)
    print(json.dumps({
        "status": result["status"],
        "cell_count": result["cell_count"],
        "validation_supported_count": result["validation_supported_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
