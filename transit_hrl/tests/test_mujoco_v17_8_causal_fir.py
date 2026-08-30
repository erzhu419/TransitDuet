from types import SimpleNamespace

import numpy as np

from scripts import mujoco_v17_4_streaming_audit_projection_preflight_spec as v17_4
from scripts import mujoco_v17_8_causal_fir_distillation_spec as spec
from scripts.mujoco_v17_8_causal_fir import (
    apply_causal_fir,
    causal_fir_features,
    fit_causal_fir,
)
from scripts.submit_mujoco_v17_8_dataset_scheduleurm import (
    DATA_LOCAL_NODE,
    artifact_relative_path,
    build_scheduler_spec,
    selected_paths,
)
from scripts.submit_mujoco_v17_8_selection_scheduleurm import (
    build_scheduler_spec as build_selection_scheduler_spec,
)
from scripts.train_mujoco_v17_8_causal_fir import (
    candidate_configs,
    reused_advancement_gate,
)


def test_causal_features_use_current_and_past_but_never_future_actions():
    total = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    features = causal_fir_features(total, window=2)
    assert np.array_equal(features, np.array([
        [1.0, 10.0, 0.0, 0.0],
        [2.0, 20.0, 1.0, 10.0],
        [3.0, 30.0, 2.0, 20.0],
    ]))
    changed = total.copy()
    changed[2] = 999.0
    assert np.array_equal(
        causal_fir_features(changed, window=2)[:2], features[:2]
    )


def test_multivariate_ridge_fit_recovers_cross_action_fir_mapping():
    generator = np.random.default_rng(17801)
    coefficient = np.array([
        [[0.4, -0.2], [0.3, 0.1]],
        [[-0.1, 0.25], [0.2, -0.3]],
        [[0.05, 0.02], [-0.04, 0.07]],
    ])
    totals = [generator.normal(size=(80, 2)) for _ in range(6)]
    targets = [
        causal_fir_features(total, window=3)
        @ coefficient.reshape(6, 2)
        for total in totals
    ]
    model = fit_causal_fir(
        totals,
        targets,
        window=3,
        ridge_penalty=1e-10,
        feature_scale_floor=1e-12,
    )
    assert np.allclose(model["coefficients"], coefficient, atol=1e-8)


def test_physical_projection_preserves_total_and_both_component_boxes():
    total = np.array([[1.8, -1.8], [0.5, -0.25]], dtype=np.float64)
    model = {
        "window": 1,
        "action_dimension": 2,
        "coefficients": np.array([[[5.0, 0.0], [0.0, 5.0]]]),
    }
    split = apply_causal_fir(
        total,
        model,
        output_gain=1.0,
        upper_action_limit=1.0,
        lower_action_limit=1.0,
    )
    assert np.max(np.abs(split["upper"])) <= 1.0
    assert np.max(np.abs(split["lower"])) <= 1.0
    assert np.array_equal(split["upper"] + split["lower"], total)


def test_frozen_grid_and_fresh_seed_roles_are_disjoint():
    assert len(candidate_configs()) == 80
    known = set(v17_4.OPTIMIZER_SEEDS)
    known.update(v17_4.TRAIN_SEEDS)
    known.update(v17_4.SELECTION_SEEDS)
    known.update(v17_4.EVALUATION_SEEDS)
    assert len(set(spec.FRESH_VALIDATION_SEEDS)) == 8
    assert not known.intersection(spec.FRESH_VALIDATION_SEEDS)


def test_reused_gate_requires_all_registered_conditions():
    by_environment = {
        "HalfCheetah-v5": {
            "recovered_failure_count": 35,
            "baseline_feasible_path_count": 0,
            "preserved_baseline_feasible_path_count": 0,
            "mean_lower_power": 0.001,
            "baseline_mean_lower_power": 0.002,
        },
        "Hopper-v5": {
            "recovered_failure_count": 24,
            "baseline_feasible_path_count": 0,
            "preserved_baseline_feasible_path_count": 0,
            "mean_lower_power": 0.001,
            "baseline_mean_lower_power": 0.002,
        },
        "Walker2d-v5": {
            "recovered_failure_count": 6,
            "baseline_feasible_path_count": 32,
            "preserved_baseline_feasible_path_count": 30,
            "mean_lower_power": 0.001,
            "baseline_mean_lower_power": 0.002,
        },
    }
    summary = {
        "oracle_recoverable_failure_count": 81,
        "valid_path_count": 120,
        "upper_budget_path_count": 120,
        "recovered_failure_count": 65,
        "by_environment": by_environment,
    }
    assert all(reused_advancement_gate(summary).values())
    summary["upper_budget_path_count"] = 119
    assert not reused_advancement_gate(summary)[
        "all_paths_meet_endpoint_upper_budget"
    ]


def test_dataset_tasks_keep_arrays_server_only_on_checkpoint_node():
    args = SimpleNamespace(
        environments=list(spec.ENVIRONMENTS),
        disturbance_modes=list(spec.DISTURBANCE_MODES),
        evaluation_seeds=list(spec.REUSED_SELECTION_SEEDS),
        python_executable="python3",
        run_name="v17_8_dataset_test",
        priority="normal",
    )
    paths = selected_paths(args)
    assert len(paths) == spec.REUSED_EXPECTED_PATH_COUNT == 120
    task = build_scheduler_spec(args, *paths[0])
    artifact = artifact_relative_path(args.run_name, *paths[0])
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["cpu"] == 1
    assert task["ram_mb"] == 1024
    assert ".server_artifacts" in artifact.parts
    assert ".server_artifacts" not in task["result_dir"]
    assert ".server_artifacts" in task["stage_excludes"]


def test_selection_task_explicitly_registers_cpu_only_ridge_workload():
    args = SimpleNamespace(
        dataset_run_name="v17_8_dataset_test",
        run_name="v17_8_selection_test",
        python_executable="python3",
        cpu=8,
        ram_mb=8192,
        priority="normal",
    )
    task = build_selection_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["vram"] == 0
    assert task["cpu"] == 8
    assert task["allow_cpu_training"]
    assert "NumPy ridge" in task["cpu_training_justification"]
