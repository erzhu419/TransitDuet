import json
from types import SimpleNamespace

import numpy as np

from freq_hrl.experiments.mujoco.state_conditioned_actor import (
    apply_state_conditioned_actor,
)
from scripts import mujoco_v18_2_state_conditioned_actor_spec as spec
from scripts.submit_mujoco_v18_2_state_conditioned_actor_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v18_2_state_conditioned_actor import (
    advancement_gate,
    candidate_configs,
    grouped_fold_rows,
    model_to_json,
    selection_key,
    zero_state_actor_model,
)


def test_v18_2_candidate_grid_is_complete_and_unique():
    candidates = candidate_configs()
    assert len(candidates) == spec.EXPECTED_CANDIDATE_COUNT == 16
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    assert {row["proposal_window"] for row in candidates} == {1, 8}
    assert {row["hidden_dim"] for row in candidates} == {32, 64}
    assert {row["actor_floor_path_weight"] for row in candidates} == {
        64.0,
        256.0,
    }
    assert {row["correction_abs_limit"] for row in candidates} == {
        0.010,
        0.025,
    }


def test_v18_2_grouped_fold_holds_every_mode_of_one_seed_out():
    rows = [
        {
            "evaluation_seed": seed,
            "disturbance_mode": mode,
        }
        for seed in spec.REUSED_SELECTION_SEEDS
        for mode in spec.DISTURBANCE_MODES
    ]
    held_seed = spec.REUSED_SELECTION_SEEDS[3]
    fit, held = grouped_fold_rows(rows, held_seed)
    assert len(fit) == 35
    assert len(held) == 5
    assert {row["disturbance_mode"] for row in held} == set(
        spec.DISTURBANCE_MODES
    )
    assert all(row["evaluation_seed"] != held_seed for row in fit)


def test_zero_state_actor_preserves_nonfloor_environment_exactly():
    row = {
        "lower_policy_state": np.zeros((6, 9), dtype=np.float64),
        "total_action": np.full((6, 2), 0.3, dtype=np.float64),
    }
    config = candidate_configs()[0]
    model = zero_state_actor_model(row, config)
    model = json.loads(json.dumps(model_to_json(model)))
    result = apply_state_conditioned_actor(
        row["lower_policy_state"],
        row["total_action"],
        0.4 * row["total_action"],
        model,
    )
    assert np.array_equal(result["corrected_total"], row["total_action"])
    assert np.count_nonzero(result["correction"]) == 0


def complete_summary():
    return {
        "path_count": 120,
        "reference_feasible_path_count": 113,
        "actor_floor_path_count": 7,
        "valid_path_count": 120,
        "reference_feasible_preserved_path_count": 113,
        "actor_floor_recovered_path_count": 7,
        "actor_floor_by_seed": {
            "2802248628": {"path_count": 2, "recovered_path_count": 2},
            "294864529": {"path_count": 5, "recovered_path_count": 5},
        },
        "actor_floor_target_normalized_mse": 0.5,
        "reference_feasible_correction_rms_maximum": 0.004,
        "actor_floor_executed_nonzero_path_count": 7,
    }


def test_v18_2_gate_requires_every_floor_path_and_reference_path():
    summary = complete_summary()
    assert all(advancement_gate(summary).values())
    summary["actor_floor_recovered_path_count"] = 6
    assert not advancement_gate(summary)["all_actor_floor_paths_recovered"]
    summary = complete_summary()
    summary["reference_feasible_preserved_path_count"] = 112
    assert not advancement_gate(summary)[
        "all_reference_feasible_paths_preserved"
    ]


def test_selection_order_prefers_feasibility_before_target_mse():
    base = {
        "corrected_joint_feasible_path_count": 119,
        "actor_floor_recovered_path_count": 6,
        "reference_feasible_preserved_path_count": 113,
        "actor_floor_by_seed": {
            "a": {"path_count": 1, "recovered_path_count": 1},
        },
        "actor_floor_target_normalized_mse": 0.1,
        "reference_feasible_correction_rms_mean": 0.001,
        "correction_abs_maximum": 0.01,
        "candidate_id": "lower_feasibility",
    }
    better = dict(base)
    better.update({
        "corrected_joint_feasible_path_count": 120,
        "actor_floor_target_normalized_mse": 0.7,
        "candidate_id": "better_feasibility",
    })
    assert selection_key(better) < selection_key(base)


def test_v18_2_scheduler_is_data_local_and_cpu_bounded():
    args = SimpleNamespace(
        run_name="v18_2_test",
        python_executable="python3",
        cpu=16,
        workers=16,
        ram_mb=32768,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE == "node003"
    assert task["allowed_nodes"] == ["node003"]
    assert task["cpu"] == 16
    assert task["ram_mb"] == 32768
    assert task["allow_cpu_training"]
    assert ".server_artifacts" in task["cmd"]
    assert ".server_artifacts" in task["stage_excludes"]
    assert "--workers 16" in task["cmd"]
