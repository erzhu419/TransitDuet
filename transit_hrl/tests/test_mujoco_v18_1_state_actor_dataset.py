from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mujoco_v18_1_state_actor_dataset_spec as spec
from scripts.export_mujoco_v18_1_state_actor_dataset_path import (
    validate_state_trace,
)
from scripts.submit_mujoco_v18_1_state_actor_dataset_scheduleurm import (
    DATA_LOCAL_NODE,
    artifact_relative_path,
    build_scheduler_spec,
    selected_paths,
)


def synthetic_traces(environment: str, length: int = 5):
    action_dim = spec.EXPECTED_ACTION_DIMENSION[environment]
    observation_dim = spec.EXPECTED_OBSERVATION_DIMENSION[environment]
    state_dim = spec.EXPECTED_LOWER_STATE_DIMENSION[environment]
    upper = np.full((length, action_dim), 0.25, dtype=np.float32)
    lower = np.full((length, action_dim), -0.10, dtype=np.float32)
    responsibility = {
        "total_action": upper + lower,
        "upper_action": upper,
        "lower_action": lower,
        "executed_action": upper + lower,
    }
    actor = {
        "observation": np.zeros((length, observation_dim), dtype=np.float32),
        "lower_policy_state": np.zeros((length, state_dim), dtype=np.float32),
        "disturbance": np.zeros((length, action_dim), dtype=np.float32),
        "upper_policy_action": upper.copy(),
        "latent_lower_action": lower.copy(),
        "upper_decision": np.array(
            [True, False, False, False, True], dtype=np.bool_
        ),
        "episode_step": np.arange(length, dtype=np.int64),
    }
    return responsibility, actor


def test_state_trace_contract_validates_causal_aligned_arrays():
    responsibility, actor = synthetic_traces("Hopper-v5")
    summary = validate_state_trace("Hopper-v5", responsibility, actor)
    assert summary == {
        "trajectory_length": 5,
        "action_dimension": 3,
        "observation_dimension": 11,
        "lower_state_dimension": 136,
        "upper_decision_count": 2,
    }


def test_state_trace_contract_rejects_misalignment_and_reconstruction_drift():
    responsibility, actor = synthetic_traces("Hopper-v5")
    actor["observation"] = actor["observation"][:-1]
    with pytest.raises(ValueError, match="observation trace is invalid"):
        validate_state_trace("Hopper-v5", responsibility, actor)
    responsibility, actor = synthetic_traces("Hopper-v5")
    responsibility["total_action"][2, 1] += 0.01
    with pytest.raises(ValueError, match="does not reconstruct"):
        validate_state_trace("Hopper-v5", responsibility, actor)


def test_dataset_tasks_keep_state_arrays_server_only_on_checkpoint_node():
    args = SimpleNamespace(
        environments=list(spec.ENVIRONMENTS),
        disturbance_modes=list(spec.DISTURBANCE_MODES),
        evaluation_seeds=list(spec.REUSED_SELECTION_SEEDS),
        python_executable="python3",
        run_name="v18_1_dataset_test",
        priority="normal",
    )
    paths = selected_paths(args)
    assert len(paths) == spec.EXPECTED_PATH_COUNT == 120
    task = build_scheduler_spec(args, *paths[0])
    artifact = artifact_relative_path(args.run_name, *paths[0])
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["cpu"] == 1
    assert task["ram_mb"] == 1536
    assert ".server_artifacts" in artifact.parts
    assert ".server_artifacts" not in task["result_dir"]
    assert ".server_artifacts" in task["stage_excludes"]
    assert "nearest_feasible_target" not in task["cmd"]


def test_state_dataset_is_reused_only_and_has_no_target_labels():
    assert spec.REUSED_SELECTION_SEEDS == spec.v17_4.EVALUATION_SEEDS
    assert spec.EXPECTED_PATH_COUNT == 120
    assert "target" not in spec.TRACE_KEYS
    assert "label" not in spec.TRACE_KEYS
    assert "no v17.12 target" in spec.DATASET_CONTRACT["labels"]
