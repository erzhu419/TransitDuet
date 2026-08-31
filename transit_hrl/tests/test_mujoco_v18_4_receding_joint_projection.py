from types import SimpleNamespace

from scripts import mujoco_v18_4_receding_joint_projection_spec as spec
from scripts.run_mujoco_v18_4_receding_joint_projection import (
    advancement_gate,
    selection_key,
)
from scripts.submit_mujoco_v18_4_receding_joint_projection_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)


def _complete_summary(candidate_id="joint_mpc_h16_hold"):
    return {
        "candidate_id": candidate_id,
        "valid_path_count": 120,
        "direct_joint_feasible_path_count": 120,
        "direct_reference_feasible_preserved_path_count": 113,
        "direct_actor_floor_recovered_path_count": 7,
        "actor_floor_by_seed": {
            "2802248628": {
                "path_count": 2,
                "direct_recovered_path_count": 2,
                "exact_recovered_path_count": 2,
            },
            "294864529": {
                "path_count": 5,
                "direct_recovered_path_count": 5,
                "exact_recovered_path_count": 5,
            },
        },
        "actor_floor_executed_nonzero_path_count": 7,
        "exact_oracle_audited_path_count": 120,
        "exact_oracle_joint_feasible_path_count": 120,
        "correction_abs_maximum": 0.04,
        "reference_feasible_correction_rms_maximum": 0.009,
        "actor_floor_correction_rms_maximum": 0.014,
    }


def test_v18_4_candidate_set_is_only_horizon_by_causal_forecast():
    assert len(spec.CANDIDATES) == 4
    assert {
        row["planning_horizon"] for row in spec.CANDIDATES.values()
    } == {16, 32}
    assert {
        row["forecast_mode"] for row in spec.CANDIDATES.values()
    } == {"hold", "damped_velocity"}
    assert all(
        set(row) == {"planning_horizon", "forecast_mode"}
        for row in spec.CANDIDATES.values()
    )


def test_v18_4_gate_requires_direct_exact_recovery_and_trust():
    assert all(advancement_gate(_complete_summary()).values())
    failed = _complete_summary()
    failed["direct_joint_feasible_path_count"] = 119
    assert not advancement_gate(failed)[
        "all_paths_directly_joint_feasible"
    ]
    failed = _complete_summary()
    failed["exact_oracle_audited_path_count"] = 119
    assert not advancement_gate(failed)[
        "selected_candidate_exactly_audited_on_all_paths"
    ]
    failed = _complete_summary()
    failed["actor_floor_by_seed"]["2802248628"][
        "exact_recovered_path_count"
    ] = 1
    assert not advancement_gate(failed)[
        "all_actor_floor_seed_groups_recovered"
    ]
    failed = _complete_summary()
    failed["correction_abs_maximum"] = 0.051
    assert not advancement_gate(failed)["global_correction_abs_gate"]


def test_v18_4_selection_uses_direct_audit_not_unaudited_exact_count():
    complete = _complete_summary("complete")
    complete["exact_oracle_audited_path_count"] = 0
    complete["exact_oracle_joint_feasible_path_count"] = 0
    incomplete = _complete_summary("lower_correction")
    incomplete["direct_joint_feasible_path_count"] = 119
    incomplete["reference_feasible_correction_rms_maximum"] = 0.0
    assert selection_key(complete) < selection_key(incomplete)


def test_v18_4_scheduler_is_data_local_and_does_not_read_targets():
    args = SimpleNamespace(
        run_name="v18_4_test",
        python_executable="python3",
        cpu=16,
        workers=16,
        ram_mb=16384,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE == "node003"
    assert task["allowed_nodes"] == ["node003"]
    assert task["cpu"] == 16
    assert task["ram_mb"] == 16384
    assert task["allow_cpu_training"]
    assert spec.REFERENCE_DATASET_RUN in task["cmd"]
    assert "v17_12" not in task["cmd"]
    assert "target" not in task["cmd"]
    assert ".server_artifacts" in task["stage_excludes"]
