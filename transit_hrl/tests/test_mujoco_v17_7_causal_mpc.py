from types import SimpleNamespace

import numpy as np

from scripts import mujoco_v17_7_causal_mpc_diagnostic_spec as spec
from scripts.analyze_mujoco_v17_7_causal_mpc import analyze
from scripts.run_mujoco_v17_7_causal_mpc_path import evaluate_candidate
from scripts.submit_mujoco_v17_7_causal_mpc_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
    oracle_relative_path,
    selected_paths,
)


def _candidate(*, joint: bool, lower: float, runtime: float):
    return {
        "upper_budget_pass": True,
        "joint_budget_pass": bool(joint),
        "recovers_oracle_recoverable_failure": bool(joint),
        "preserves_baseline_feasible_path": bool(joint),
        "upper_power": 0.005,
        "lower_power": float(lower),
        "runtime_seconds": float(runtime),
        "prefix_upper_budget_feasible_rate": 1.0,
        "actor_floor_power_excess_max": 0.0,
        "reconstruction_error_max": 0.0,
        "bound_violation_max": 0.0,
    }


def _matrix_rows():
    rows = []
    path_index = 0
    for environment in spec.ENVIRONMENTS:
        for mode in spec.DISTURBANCE_MODES:
            for seed in spec.EVALUATION_SEEDS:
                if environment == "HalfCheetah-v5":
                    baseline, oracle = False, True
                elif environment == "Hopper-v5":
                    baseline = False
                    oracle = path_index % 40 < 33
                else:
                    walker_index = path_index % 40
                    baseline, oracle = walker_index < 32, True
                candidates = {
                    candidate_id: _candidate(
                        joint=(oracle if candidate_id == "hold_h16" else False),
                        lower=(0.001 if candidate_id == "hold_h16" else 0.004),
                        runtime=(1.0 if candidate_id == "hold_h16" else 2.0),
                    )
                    for candidate_id in spec.CANDIDATES
                }
                for candidate in candidates.values():
                    candidate["recovers_oracle_recoverable_failure"] = bool(
                        not baseline and oracle and candidate["joint_budget_pass"]
                    )
                    candidate["preserves_baseline_feasible_path"] = bool(
                        baseline and candidate["joint_budget_pass"]
                    )
                rows.append({
                    "status": "causal_mpc_path_complete",
                    "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
                    "frozen_core_revision": spec.FROZEN_CORE_REVISION,
                    "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
                    "source_identity": {"source_identity_status": "verified"},
                    "legacy_replay_audit": {"exact": True},
                    "environment": environment,
                    "disturbance_mode": mode,
                    "evaluation_seed": int(seed),
                    "baseline": {
                        "joint_feasible": baseline,
                        "lower_power": 0.01,
                    },
                    "oracle": {"joint_feasible": oracle},
                    "candidates": candidates,
                })
                path_index += 1
    return rows


def test_synthetic_candidate_is_causal_bounded_and_upper_feasible():
    time = np.arange(24, dtype=np.float64)
    total = np.stack(
        (0.45 * np.sin(0.31 * time), 0.35 * np.cos(0.27 * time)),
        axis=1,
    )
    row = evaluate_candidate(total, candidate_id="hold_h16")
    assert row["upper_budget_pass"]
    assert row["reconstruction_error_max"] <= spec.RECONSTRUCTION_TOLERANCE
    assert row["bound_violation_max"] <= spec.BOUND_TOLERANCE
    assert row["prefix_upper_budget_feasible_rate"] == 1.0


def test_analyzer_applies_frozen_recovery_gate_and_selection_order():
    summary = analyze(_matrix_rows())
    assert summary["selected_candidate_id"] == "hold_h16"
    assert summary["status"] == "causal_mpc_advances_to_fresh_training"
    assert summary["oracle_recoverable_path_count"] == 81
    assert all(summary["advancement_gate"].values())


def test_full_design_has_120_data_local_paths_and_oracle_dependencies():
    args = SimpleNamespace(
        environments=list(spec.ENVIRONMENTS),
        disturbance_modes=list(spec.DISTURBANCE_MODES),
        evaluation_seeds=list(spec.EVALUATION_SEEDS),
        python_executable="python3",
        run_name="causal_mpc_test",
        priority="normal",
    )
    paths = selected_paths(args)
    assert len(paths) == spec.EXPECTED_PATH_COUNT == 120
    task = build_scheduler_spec(args, *paths[0])
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["cpu"] == 1
    assert task["ram_mb"] == 1024
    assert task["ckpt_dir"] is None
    assert ".server_artifacts" in task["stage_excludes"]
    dependency = oracle_relative_path(*paths[0])
    assert spec.ORACLE_RUN_NAME in dependency.parts
    assert dependency.name == "oracle_path.json"
