from types import SimpleNamespace

from scripts import mujoco_v17_12_nearest_feasible_action_oracle_spec as spec
from scripts.run_mujoco_v17_12_nearest_feasible_action_oracle import (
    summarize_rows,
)
from scripts.submit_mujoco_v17_12_nearest_feasible_action_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)


def _summary(feasible: bool, correction: float) -> dict[str, float | bool]:
    return {
        "feasible": feasible,
        "component_correction_rms": correction,
        "total_action_correction_rms": correction,
    }


def test_v17_12_frozen_gate_authorizes_only_complete_small_targets():
    rows = []
    floor_remaining = spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
    for index in range(spec.EXPECTED_PATH_COUNT):
        is_floor = floor_remaining > 0
        if is_floor:
            floor_remaining -= 1
        rows.append({
            "environment": spec.ENVIRONMENTS[
                index % len(spec.ENVIRONMENTS)
            ],
            "reference_joint_feasible": not is_floor,
            "frequency_only": _summary(True, 0.01 if is_floor else 0.0),
            "frequency_total_correction_abs_max": 0.04 if is_floor else 0.0,
            "deployment_aligned": _summary(True, 0.02) if is_floor else None,
            "target_written": is_floor,
        })
    result = summarize_rows(rows)
    assert result["causal_actor_adapter_authorized"]
    assert all(result["advancement_gate"].values())


def test_v17_12_frozen_gate_rejects_large_actor_correction():
    rows = []
    floor_remaining = spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT
    for index in range(spec.EXPECTED_PATH_COUNT):
        is_floor = floor_remaining > 0
        if is_floor:
            floor_remaining -= 1
        correction = 0.06 if is_floor and floor_remaining == 0 else 0.01
        rows.append({
            "environment": spec.ENVIRONMENTS[
                index % len(spec.ENVIRONMENTS)
            ],
            "reference_joint_feasible": not is_floor,
            "frequency_only": _summary(
                True, correction if is_floor else 0.0
            ),
            "frequency_total_correction_abs_max": 0.08 if is_floor else 0.0,
            "deployment_aligned": _summary(True, 0.02) if is_floor else None,
            "target_written": is_floor,
        })
    result = summarize_rows(rows)
    assert not result["causal_actor_adapter_authorized"]
    assert not result["advancement_gate"][
        "actor_floor_total_correction_rms_gate"
    ]


def test_v17_12_scheduler_is_data_local_and_excludes_server_targets():
    args = SimpleNamespace(
        run_name="v17_12_test",
        python_executable="python3",
        cpu=4,
        ram_mb=4096,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert not task["allow_cpu_training"]
    assert ".server_artifacts" in task["stage_excludes"]
