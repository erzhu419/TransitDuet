from types import SimpleNamespace

from scripts import mujoco_v17_6_full_horizon_oracle_spec as spec
from scripts.analyze_mujoco_v17_6_full_horizon_oracle import summarize_paths
from scripts.run_mujoco_v17_6_full_horizon_oracle_path import (
    legacy_replay_audit,
)
from scripts.submit_mujoco_v17_6_full_horizon_oracle_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
    selected_paths,
)


def _row(*, baseline: bool, oracle: bool, upper: bool = True):
    return {
        "baseline": {
            "joint_feasible": baseline,
            "upper_power": 0.01,
            "lower_power": 0.02,
        },
        "oracle": {
            "joint_feasible": oracle,
            "upper_constraint_feasible": upper,
            "upper_power": 0.001,
            "lower_power": 0.002,
            "solver_optimality_max": 1e-10,
            "kkt_residual_inf": 1e-10,
            "bound_violation_max": 0.0,
            "reconstruction_error_max": 0.0,
        },
        "recoverable_by_responsibility_split": oracle and not baseline,
    }


def test_summary_distinguishes_router_and_total_action_limits():
    summary = summarize_paths([
        _row(baseline=False, oracle=True),
        _row(baseline=False, oracle=False),
        _row(baseline=True, oracle=True),
    ])
    assert summary["status"] == "mixed_online_router_and_total_action_limits"
    assert summary["recoverable_path_count"] == 1
    assert summary["oracle_infeasible_path_count"] == 1


def test_legacy_audit_detects_numeric_and_trace_changes():
    common = {
        "RewardTraceSHA256": "r",
        "ExecutedActionTraceSHA256": "a",
        "LatentPolicyTraceSHA256": "l",
        "episode_return": 1.0,
        "UpperHFPowerAbs": 0.1,
        "LowerLFDriftAbs": 0.2,
        "LatentUpperHFPowerAbs": 0.3,
        "LatentLowerLFDriftAbs": 0.4,
        "LowerRouterActionReconstructionRMS": 0.0,
        "ResponsibilityReconstructionRMS": 0.0,
    }
    assert legacy_replay_audit(common, dict(common))["exact"]
    changed = dict(common, LowerLFDriftAbs=0.21)
    assert not legacy_replay_audit(common, changed)["exact"]


def test_full_design_has_120_data_local_paths():
    args = SimpleNamespace(
        environments=list(spec.ENVIRONMENTS),
        disturbance_modes=list(spec.DISTURBANCE_MODES),
        evaluation_seeds=list(spec.EVALUATION_SEEDS),
        python_executable="python3",
        run_name="oracle_test",
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
