#!/usr/bin/env python3
"""Aggregate the v17.6 full-horizon oracle path matrix."""

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

from scripts import mujoco_v17_6_full_horizon_oracle_spec as spec  # noqa: E402


def path_relative_dir(
    run_name: str, environment: str, mode: str, seed: int
) -> Path:
    return (
        Path("results") / str(run_name) / "paths" / str(environment)
        / str(mode) / f"seed_{int(seed)}"
    )


def summarize_paths(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_feasible = sum(bool(row["baseline"]["joint_feasible"]) for row in rows)
    oracle_feasible = sum(bool(row["oracle"]["joint_feasible"]) for row in rows)
    recoverable = sum(bool(row["recoverable_by_responsibility_split"]) for row in rows)
    intrinsic = len(rows) - oracle_feasible
    upper_infeasible = sum(
        not bool(row["oracle"]["upper_constraint_feasible"]) for row in rows
    )
    if recoverable and intrinsic:
        status = "mixed_online_router_and_total_action_limits"
    elif recoverable:
        status = "full_horizon_oracle_supports_online_router_rebuild"
    elif intrinsic:
        status = "full_horizon_oracle_supports_actor_feasibility_constraint"
    else:
        status = "all_replayed_paths_already_jointly_feasible"
    return {
        "status": status,
        "path_count": len(rows),
        "baseline_joint_feasible_path_count": baseline_feasible,
        "oracle_joint_feasible_path_count": oracle_feasible,
        "recoverable_path_count": recoverable,
        "oracle_infeasible_path_count": intrinsic,
        "upper_budget_physically_infeasible_path_count": upper_infeasible,
        "baseline_upper_power_mean": float(np.mean([
            float(row["baseline"]["upper_power"]) for row in rows
        ])),
        "baseline_lower_power_mean": float(np.mean([
            float(row["baseline"]["lower_power"]) for row in rows
        ])),
        "oracle_upper_power_mean": float(np.mean([
            float(row["oracle"]["upper_power"]) for row in rows
        ])),
        "oracle_lower_power_mean": float(np.mean([
            float(row["oracle"]["lower_power"]) for row in rows
        ])),
        "solver_optimality_max": float(max(
            float(row["oracle"]["solver_optimality_max"]) for row in rows
        )),
        "kkt_residual_max": float(max(
            float(row["oracle"]["kkt_residual_inf"]) for row in rows
        )),
        "bound_violation_max": float(max(
            float(row["oracle"]["bound_violation_max"]) for row in rows
        )),
        "reconstruction_error_max": float(max(
            float(row["oracle"]["reconstruction_error_max"]) for row in rows
        )),
    }


def analyze(run_name: str, *, root: Path = ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for environment in spec.ENVIRONMENTS:
        for mode in spec.DISTURBANCE_MODES:
            for seed in spec.EVALUATION_SEEDS:
                path = (
                    Path(root)
                    / path_relative_dir(run_name, environment, mode, seed)
                    / "oracle_path.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(path)
                row = json.loads(path.read_text(encoding="utf-8"))
                expected = (environment, mode, int(seed))
                observed = (
                    row.get("environment"),
                    row.get("disturbance_mode"),
                    int(row.get("evaluation_seed", -1)),
                )
                if observed != expected:
                    raise ValueError("v17.6 oracle path identity mismatch")
                rows.append(row)
    mechanics_valid = bool(
        len(rows) == spec.EXPECTED_PATH_COUNT
        and all(row["legacy_replay_audit"]["exact"] for row in rows)
        and all(
            float(row["oracle"]["solver_optimality_max"])
            <= spec.SOLVER_OPTIMALITY_TOLERANCE
            and float(row["oracle"]["kkt_residual_inf"])
            <= spec.KKT_RESIDUAL_TOLERANCE
            and float(row["oracle"]["bound_violation_max"])
            <= spec.BOUND_TOLERANCE
            and float(row["oracle"]["reconstruction_error_max"])
            <= spec.RECONSTRUCTION_TOLERANCE
            for row in rows
        )
    )
    if not mechanics_valid:
        raise ValueError("v17.6 oracle mechanics certificate failed")
    overall = summarize_paths(rows)
    by_environment = {
        environment: summarize_paths([
            row for row in rows if row["environment"] == environment
        ])
        for environment in spec.ENVIRONMENTS
    }
    return {
        "status": overall["status"],
        "integrity_status": "valid",
        "evidence_role": spec.EVIDENCE_ROLE,
        "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
        "frozen_oracle_revision": spec.FROZEN_ORACLE_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "mechanics_valid": True,
        "overall": overall,
        "environment_results": by_environment,
        "claim_boundary": (
            "reused rejected v17.4 paths and an acausal oracle; development "
            "mechanism diagnosis only"
        ),
    }


def _report(result: dict[str, Any]) -> str:
    overall = result["overall"]
    lines = [
        "# MuJoCo v17.6 Full-Horizon Oracle Outcome",
        "",
        f"Status: `{result['status']}`",
        "",
        "| Environment | Paths | Baseline feasible | Oracle feasible | Recoverable | Oracle infeasible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for environment, row in result["environment_results"].items():
        lines.append(
            f"| {environment} | {row['path_count']} | "
            f"{row['baseline_joint_feasible_path_count']} | "
            f"{row['oracle_joint_feasible_path_count']} | "
            f"{row['recoverable_path_count']} | "
            f"{row['oracle_infeasible_path_count']} |"
        )
    lines.extend([
        "",
        f"Across {overall['path_count']} reused paths, the oracle recovered "
        f"{overall['recoverable_path_count']} paths that the online v17.4 "
        "split did not make jointly feasible. "
        f"{overall['oracle_infeasible_path_count']} paths remained infeasible "
        "even for the acausal bounded full-horizon split.",
        "",
        "This is development-only mechanism diagnosis. The acausal oracle is "
        "not an online policy and these accessed paths cannot support a fresh "
        "performance or generalization claim.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    result = analyze(args.run_name)
    output = ROOT / "results" / args.run_name / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "full_horizon_oracle.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "full_horizon_oracle.md").write_text(
        _report(result), encoding="utf-8"
    )
    print(
        f"v17.6 oracle status={result['status']} "
        f"paths={result['overall']['path_count']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
