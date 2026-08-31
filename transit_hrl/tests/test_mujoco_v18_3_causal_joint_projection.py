from types import SimpleNamespace

from scripts import mujoco_v18_3_causal_joint_projection_spec as spec
from scripts.run_mujoco_v18_3_causal_joint_projection import (
    advancement_gate,
    selection_key,
)
from scripts.submit_mujoco_v18_3_causal_joint_projection_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)


def _complete_summary(candidate_id="joint_projection_instantaneous"):
    return {
        "candidate_id": candidate_id,
        "valid_path_count": 120,
        "direct_joint_feasible_path_count": 120,
        "exact_oracle_joint_feasible_path_count": 120,
        "reference_feasible_preserved_path_count": 113,
        "actor_floor_recovered_path_count": 7,
        "actor_floor_by_seed": {
            "2802248628": {"path_count": 2, "recovered_path_count": 2},
            "294864529": {"path_count": 5, "recovered_path_count": 5},
        },
        "actor_floor_executed_nonzero_path_count": 7,
        "correction_abs_maximum": 0.04,
        "reference_feasible_correction_rms_maximum": 0.009,
        "actor_floor_correction_rms_maximum": 0.014,
    }


def test_v18_3_candidate_set_contains_only_budget_semantics():
    assert set(spec.CANDIDATES) == {
        "joint_projection_instantaneous",
        "joint_projection_prefix_ledger",
    }
    assert {
        row["budget_mode"] for row in spec.CANDIDATES.values()
    } == {"instantaneous", "prefix_ledger"}


def test_v18_3_gate_requires_feasibility_recovery_and_trust():
    assert all(advancement_gate(_complete_summary()).values())
    failed = _complete_summary()
    failed["direct_joint_feasible_path_count"] = 119
    assert not advancement_gate(failed)[
        "all_paths_directly_joint_feasible"
    ]
    failed = _complete_summary()
    failed["correction_abs_maximum"] = 0.051
    assert not advancement_gate(failed)["global_correction_abs_gate"]
    failed = _complete_summary()
    failed["actor_floor_by_seed"]["2802248628"][
        "recovered_path_count"
    ] = 1
    assert not advancement_gate(failed)[
        "all_actor_floor_seed_groups_recovered"
    ]


def test_v18_3_selection_prioritizes_direct_feasibility_before_trust():
    complete = _complete_summary("complete")
    incomplete = _complete_summary("lower_correction")
    incomplete["direct_joint_feasible_path_count"] = 119
    incomplete["reference_feasible_correction_rms_maximum"] = 0.0
    assert selection_key(complete) < selection_key(incomplete)


def test_v18_3_scheduler_is_data_local_and_does_not_read_targets():
    args = SimpleNamespace(
        run_name="v18_3_test",
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
