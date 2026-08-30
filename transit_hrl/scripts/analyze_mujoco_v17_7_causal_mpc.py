#!/usr/bin/env python3
"""Analyze the frozen MuJoCo v17.7 causal MPC development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import (  # noqa: E402
    mujoco_v17_7_causal_mpc_diagnostic_spec as spec,
)


def path_relative_dir(
    run_name: str,
    environment: str,
    mode: str,
    seed: int,
) -> Path:
    return (
        Path("results") / str(run_name) / "paths" / str(environment)
        / str(mode) / f"seed_{int(seed)}"
    )


def expected_path_files(run_name: str) -> list[Path]:
    return [
        ROOT / path_relative_dir(run_name, environment, mode, seed)
        / "causal_mpc_path.json"
        for environment in spec.ENVIRONMENTS
        for mode in spec.DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    ]


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def summarize_candidate(
    rows: list[dict[str, Any]],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    by_environment: dict[str, Any] = {}
    for environment in spec.ENVIRONMENTS:
        selected = [
            row for row in rows if row["environment"] == environment
        ]
        candidate_rows = [
            row["candidates"][candidate_id] for row in selected
        ]
        recoverable = [
            row for row in selected
            if not row["baseline"]["joint_feasible"]
            and row["oracle"]["joint_feasible"]
        ]
        baseline_feasible = [
            row for row in selected if row["baseline"]["joint_feasible"]
        ]
        by_environment[environment] = {
            "path_count": len(selected),
            "baseline_lower_power_mean": _mean([
                float(row["baseline"]["lower_power"]) for row in selected
            ]),
            "candidate_lower_power_mean": _mean([
                float(row["candidates"][candidate_id]["lower_power"])
                for row in selected
            ]),
            "upper_budget_path_count": sum(
                int(row["upper_budget_pass"]) for row in candidate_rows
            ),
            "joint_budget_path_count": sum(
                int(row["joint_budget_pass"]) for row in candidate_rows
            ),
            "oracle_recoverable_path_count": len(recoverable),
            "recovered_path_count": sum(
                int(row["candidates"][candidate_id][
                    "recovers_oracle_recoverable_failure"
                ])
                for row in recoverable
            ),
            "baseline_feasible_path_count": len(baseline_feasible),
            "preserved_baseline_feasible_path_count": sum(
                int(row["candidates"][candidate_id][
                    "preserves_baseline_feasible_path"
                ])
                for row in baseline_feasible
            ),
        }

    candidates = [row["candidates"][candidate_id] for row in rows]
    numerical_valid_count = sum(
        int(
            float(row["reconstruction_error_max"])
            <= spec.RECONSTRUCTION_TOLERANCE
            and float(row["bound_violation_max"])
            <= spec.BOUND_TOLERANCE
        )
        for row in candidates
    )
    oracle_infeasible_rows = [
        row for row in rows if not row["oracle"]["joint_feasible"]
    ]
    oracle_feasible_rows = [
        row for row in rows if row["oracle"]["joint_feasible"]
    ]
    actor_positive = lambda row: bool(
        float(row["candidates"][candidate_id][
            "actor_floor_power_excess_max"
        ]) > 0.0
    )
    return {
        "candidate_id": str(candidate_id),
        "config": dict(spec.CANDIDATES[candidate_id]),
        "path_count": len(rows),
        "numerically_valid_path_count": numerical_valid_count,
        "upper_budget_path_count": sum(
            int(row["upper_budget_pass"]) for row in candidates
        ),
        "joint_budget_path_count": sum(
            int(row["joint_budget_pass"]) for row in candidates
        ),
        "recoverable_path_count": sum(
            int(row["recovers_oracle_recoverable_failure"])
            for row in candidates
        ),
        "preserved_baseline_feasible_path_count": sum(
            int(row["preserves_baseline_feasible_path"])
            for row in candidates
        ),
        "lower_power_mean": _mean([
            float(row["lower_power"]) for row in candidates
        ]),
        "upper_power_mean": _mean([
            float(row["upper_power"]) for row in candidates
        ]),
        "runtime_seconds_mean": _mean([
            float(row["runtime_seconds"]) for row in candidates
        ]),
        "prefix_feasible_rate_mean": _mean([
            float(row["prefix_upper_budget_feasible_rate"])
            for row in candidates
        ]),
        "actor_floor_path_positive_count": sum(
            int(actor_positive(row)) for row in rows
        ),
        "actor_floor_oracle_infeasible_true_positive_count": sum(
            int(actor_positive(row)) for row in oracle_infeasible_rows
        ),
        "actor_floor_oracle_feasible_false_positive_count": sum(
            int(actor_positive(row)) for row in oracle_feasible_rows
        ),
        "environment_results": by_environment,
    }


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != spec.EXPECTED_PATH_COUNT:
        raise ValueError("v17.7 path manifest is incomplete")
    path_keys = {
        (
            str(row["environment"]),
            str(row["disturbance_mode"]),
            int(row["evaluation_seed"]),
        )
        for row in rows
    }
    expected_keys = {
        (environment, mode, int(seed))
        for environment in spec.ENVIRONMENTS
        for mode in spec.DISTURBANCE_MODES
        for seed in spec.EVALUATION_SEEDS
    }
    if path_keys != expected_keys or len(path_keys) != len(rows):
        raise ValueError("v17.7 path keys are missing or duplicated")
    if any(
        row.get("status") != "causal_mpc_path_complete"
        or row.get("development_protocol_version")
        != spec.DEVELOPMENT_PROTOCOL_VERSION
        or row.get("frozen_core_revision") != spec.FROZEN_CORE_REVISION
        or row.get("frozen_source_manifest_sha256")
        != spec.FROZEN_SOURCE_MANIFEST_SHA256
        or row.get("source_identity", {}).get("source_identity_status")
        != "verified"
        or row.get("legacy_replay_audit", {}).get("exact") is not True
        or set(row.get("candidates", {})) != set(spec.CANDIDATES)
        for row in rows
    ):
        raise ValueError("v17.7 path identity or integrity validation failed")
    baseline_feasible_count = sum(
        int(row["baseline"]["joint_feasible"]) for row in rows
    )
    oracle_feasible_count = sum(
        int(row["oracle"]["joint_feasible"]) for row in rows
    )
    oracle_recoverable_count = sum(
        int(
            not row["baseline"]["joint_feasible"]
            and row["oracle"]["joint_feasible"]
        )
        for row in rows
    )
    if (
        baseline_feasible_count != 32
        or oracle_feasible_count != 113
        or oracle_recoverable_count != 81
    ):
        raise ValueError("v17.7 oracle dependency no longer matches v17.6")

    candidates = {
        candidate_id: summarize_candidate(
            rows, candidate_id=candidate_id
        )
        for candidate_id in spec.CANDIDATES
    }
    selected_id = max(
        spec.CANDIDATES,
        key=lambda candidate_id: (
            candidates[candidate_id]["numerically_valid_path_count"],
            candidates[candidate_id]["upper_budget_path_count"],
            candidates[candidate_id]["joint_budget_path_count"],
            -candidates[candidate_id]["lower_power_mean"],
            -candidates[candidate_id]["runtime_seconds_mean"],
        ),
    )
    selected = candidates[selected_id]
    lower_no_worse = all(
        values["candidate_lower_power_mean"]
        <= values["baseline_lower_power_mean"] + spec.POWER_TOLERANCE
        for values in selected["environment_results"].values()
    )
    recovery_by_environment_pass = all(
        selected["environment_results"][environment][
            "recovered_path_count"
        ] >= threshold
        for environment, threshold
        in spec.RECOVERY_GATE_BY_ENVIRONMENT.items()
    )
    gate = {
        "numerical_validity": bool(
            selected["numerically_valid_path_count"]
            == spec.EXPECTED_PATH_COUNT
        ),
        "upper_budget_all_paths": bool(
            selected["upper_budget_path_count"] == spec.EXPECTED_PATH_COUNT
        ),
        "total_recovery": bool(
            selected["recoverable_path_count"] >= spec.RECOVERY_GATE_TOTAL
        ),
        "recovery_by_environment": recovery_by_environment_pass,
        "preserve_baseline_feasible_walker": bool(
            selected["environment_results"]["Walker2d-v5"][
                "preserved_baseline_feasible_path_count"
            ] >= spec.PRESERVE_BASELINE_FEASIBLE_WALKER_GATE
        ),
        "mean_lower_no_worse_each_environment": lower_no_worse,
    }
    advances = bool(all(gate.values()))
    return {
        "status": (
            "causal_mpc_advances_to_fresh_training"
            if advances else "causal_mpc_not_advanced"
        ),
        "integrity_status": "valid",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_core_revision": spec.FROZEN_CORE_REVISION,
        "frozen_source_manifest_sha256": (
            spec.FROZEN_SOURCE_MANIFEST_SHA256
        ),
        "path_count": len(rows),
        "baseline_joint_feasible_path_count": baseline_feasible_count,
        "oracle_joint_feasible_path_count": oracle_feasible_count,
        "oracle_recoverable_path_count": oracle_recoverable_count,
        "selected_candidate_id": selected_id,
        "advancement_gate": gate,
        "eligible_for_fresh_training": advances,
        "candidates": candidates,
        "claim_boundary": (
            "candidate selection on reused rejected paths; no fresh reward or "
            "generalization claim"
        ),
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MuJoCo v17.7 Causal MPC Outcome",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| Candidate | Valid | Upper pass | Joint pass | Recovered | Mean lower | Runtime/path (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, row in summary["candidates"].items():
        lines.append(
            f"| {candidate_id} | {row['numerically_valid_path_count']} | "
            f"{row['upper_budget_path_count']} | "
            f"{row['joint_budget_path_count']} | "
            f"{row['recoverable_path_count']} | "
            f"{row['lower_power_mean']:.8f} | "
            f"{row['runtime_seconds_mean']:.2f} |"
        )
    lines.extend([
        "",
        f"Selected candidate: `{summary['selected_candidate_id']}`",
        "",
        "## Advancement Gate",
        "",
    ])
    for name, passed in summary["advancement_gate"].items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend([
        "",
        "This is a causal mechanism-selection diagnostic on reused rejected ",
        "paths. It is not fresh learned-policy or publication evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    files = expected_path_files(args.run_name)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} v17.7 path files; first={missing[0]}"
        )
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    summary = analyze(rows)
    output = ROOT / "results" / args.run_name / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "causal_mpc.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "causal_mpc.md").write_text(
        _render_markdown(summary), encoding="utf-8"
    )
    print(
        f"v17.7 causal-mpc status={summary['status']} "
        f"selected={summary['selected_candidate_id']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
