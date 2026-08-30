from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scripts import mujoco_v17_13_causal_actor_adapter_spec as spec
from scripts.submit_mujoco_v17_13_causal_actor_adapter_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v17_13_causal_actor_adapter import (
    _fit_model,
    _solve_corrected_oracle,
    attach_actor_targets,
    build_statistics_cache,
    candidate_configs,
    reused_advancement_gate,
    select_prefilter_candidate_ids,
    target_artifact_path,
)


def _panel():
    floor_ids = {
        ("standard", 294864529),
        ("low_frequency", 294864529),
        ("high_frequency", 294864529),
        ("mixed", 2802248628),
        ("mixed", 294864529),
        ("ood_chirp", 2802248628),
        ("ood_chirp", 294864529),
    }
    rows = []
    index = 0
    for environment in spec.ENVIRONMENTS:
        for mode in spec.DISTURBANCE_MODES:
            for seed in spec.REUSED_SELECTION_SEEDS:
                total = np.full((6, 2), 0.2 + 0.001 * index)
                rows.append({
                    "environment": environment,
                    "disturbance_mode": mode,
                    "evaluation_seed": seed,
                    "total_action": total,
                    "baseline_upper_action": 0.4 * total,
                    "oracle_joint_feasible": not (
                        environment == "Hopper-v5" and (mode, seed) in floor_ids
                    ),
                })
                index += 1
    return rows


def test_actor_targets_are_complete_aligned_and_change_executed_action(tmp_path):
    panel = _panel()
    for row in panel:
        if row["oracle_joint_feasible"]:
            continue
        path = target_artifact_path(tmp_path, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = row["total_action"]
        correction = np.full_like(reference, 0.01)
        target_total = reference + correction
        target_upper = 0.5 * target_total
        np.savez_compressed(
            path,
            reference_total_action=reference,
            target_upper_action=target_upper,
            target_lower_action=target_total - target_upper,
            target_total_action=target_total,
            total_action_correction=correction,
        )
    attached = attach_actor_targets(panel, tmp_path)
    floor = [row for row in attached if row["actor_floor"]]
    assert len(attached) == spec.EXPECTED_PATH_COUNT == 120
    assert len(floor) == spec.EXPECTED_ACTOR_FLOOR_PATH_COUNT == 7
    assert all(row["target_executed_correction_rms"] > 0.0 for row in floor)
    assert all(
        np.array_equal(row["target_total_correction"], np.zeros((6, 2)))
        for row in attached if not row["actor_floor"]
    )


def test_frozen_candidate_grid_and_prefilter_union_are_deterministic():
    configs = candidate_configs()
    assert len(configs) == 900
    assert len({row["candidate_id"] for row in configs}) == len(configs)
    summaries = []
    for index in range(40):
        summaries.append({
            "candidate_id": f"candidate_{index:02d}",
            "actor_floor_target_normalized_mse": 0.2 + index / 100.0,
            "reference_feasible_correction_rms_mean": (39 - index) / 10000.0,
            "actor_floor_target_correction_rms_mean": 0.002,
        })
    first = select_prefilter_candidate_ids(summaries)
    second = select_prefilter_candidate_ids(list(reversed(summaries)))
    assert first == second
    assert 16 <= len(first) <= 3 * spec.PREFILTER_TOP_PER_RANKING


def test_cached_sufficient_statistics_match_direct_weighted_ridge():
    generator = np.random.default_rng(171314)
    rows = []
    for index, floor in enumerate((False, True, False)):
        total = generator.normal(scale=0.2, size=(15 + index, 2))
        upper = generator.normal(scale=0.1, size=total.shape)
        target = 0.03 * (total - upper) if floor else np.zeros_like(total)
        rows.append({
            "environment": "Hopper-v5",
            "disturbance_mode": f"mode_{index}",
            "evaluation_seed": index,
            "total_action": total,
            "baseline_upper_action": upper,
            "target_total_correction": target,
            "actor_floor": floor,
        })
    direct = _fit_model(
        rows,
        window=4,
        ridge_penalty=1e-4,
        actor_floor_path_weight=16.0,
    )
    cached = _fit_model(
        rows,
        window=4,
        ridge_penalty=1e-4,
        actor_floor_path_weight=16.0,
        statistics_cache=build_statistics_cache(rows),
    )
    assert np.allclose(
        direct["coefficients"], cached["coefficients"], atol=1e-12
    )


def test_complete_120_path_gate_requires_recovery_preservation_and_execution():
    summary = {
        "path_count": 120,
        "valid_path_count": 120,
        "reference_feasible_path_count": 113,
        "reference_feasible_preserved_path_count": 113,
        "actor_floor_path_count": 7,
        "actor_floor_recovered_path_count": 7,
        "actor_floor_executed_nonzero_path_count": 7,
        "actor_floor_target_normalized_mse": 0.5,
        "reference_feasible_correction_rms_maximum": 0.005,
        "actor_floor_by_seed": {
            "1": {"path_count": 5, "recovered_path_count": 5},
            "2": {"path_count": 2, "recovered_path_count": 2},
        },
    }
    assert all(reused_advancement_gate(summary).values())
    summary["actor_floor_executed_nonzero_path_count"] = 6
    assert not reused_advancement_gate(summary)[
        "all_actor_floor_paths_change_executed_action"
    ]


def test_v17_13_scheduler_is_data_local_and_keeps_targets_server_only():
    args = SimpleNamespace(
        run_name="v17_13_test",
        python_executable="python3",
        cpu=8,
        workers=8,
        ram_mb=8192,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["allow_cpu_training"]
    assert "weighted ridge" in task["cpu_training_justification"]
    assert ".server_artifacts" in task["stage_excludes"]
    assert ".server_artifacts" not in task["result_dir"]


def test_exact_oracle_worker_is_process_pool_safe():
    with ProcessPoolExecutor(max_workers=2) as executor:
        result = executor.submit(
            _solve_corrected_oracle, np.zeros((12, 2), dtype=np.float64)
        ).result(timeout=20)
    assert result["joint_feasible"]
    assert result["upper_power"] == 0.0
    assert result["lower_power"] == 0.0
