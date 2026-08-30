import json
from types import SimpleNamespace

from scripts import mujoco_v17_14_exhaustive_actor_oracle_spec as spec
from scripts.run_mujoco_v17_14_exhaustive_actor_oracle import (
    load_v17_13_summary,
    remainder_candidate_configs,
)
from scripts.submit_mujoco_v17_14_exhaustive_actor_oracle_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v17_13_causal_actor_adapter import candidate_configs


def _exact_summary(config, *, best=False):
    return {
        **config,
        "corrected_joint_feasible_path_count": 116 if best else 115,
        "actor_floor_recovered_path_count": 3 if best else 2,
        "reference_feasible_preserved_path_count": 113,
        "actor_floor_by_seed": {
            "1": {"path_count": 2, "recovered_path_count": 1},
            "2": {"path_count": 5, "recovered_path_count": 2 if best else 1},
        },
        "actor_floor_target_normalized_mse": 0.6 if best else 0.8,
        "reference_feasible_correction_rms_mean": 0.001,
        "correction_abs_maximum": 0.01,
    }


def _source_summary():
    configs = candidate_configs()
    mild = [row for row in configs if row["output_gain"] in (0.5, 1.0)]
    half_gain = [row for row in mild if row["output_gain"] == 0.5]
    exact_configs = [*half_gain[:47], next(
        row for row in mild if row["output_gain"] == 1.0
    )]
    exact = [
        _exact_summary(row, best=index == 0)
        for index, row in enumerate(exact_configs)
    ]
    return {
        "status": "causal_actor_adapter_stops_before_fresh_path_access",
        "development_protocol_version": "mujoco_v17_13_causal_actor_adapter_v1",
        "candidate_count": 900,
        "full_oracle_candidate_count": 48,
        "prefilter_candidate_summaries": [
            {"candidate_id": row["candidate_id"]} for row in configs
        ],
        "full_oracle_candidate_summaries": exact,
        "selected_candidate_id": exact[0]["candidate_id"],
        "selected_out_of_fold_rows": [{} for _ in range(120)],
        "fresh_validation_paths_accessed": False,
        "fresh_path_access_allowed": False,
        "source_identity": {
            "source_identity_status": "verified",
            "code_revision": spec.v17_13.FROZEN_CORE_REVISION,
            "source_manifest_sha256": (
                spec.v17_13.FROZEN_SOURCE_MANIFEST_SHA256
            ),
        },
    }


def test_v17_14_partition_covers_every_unexamined_frozen_candidate(tmp_path):
    path = tmp_path / "selection_summary.json"
    path.write_text(json.dumps(_source_summary()), encoding="utf-8")
    source = load_v17_13_summary(path)
    remainder = remainder_candidate_configs(source)
    examined = {
        row["candidate_id"]
        for row in source["full_oracle_candidate_summaries"]
    }
    remainder_ids = {row["candidate_id"] for row in remainder}
    assert len(examined) == 48
    assert len(remainder_ids) == spec.EXPECTED_REMAINDER_CANDIDATE_COUNT == 852
    assert not examined.intersection(remainder_ids)
    assert len(examined | remainder_ids) == 900


def test_v17_14_scheduler_is_data_local_and_uses_bounded_workers():
    args = SimpleNamespace(
        run_name="v17_14_test",
        python_executable="python3",
        cpu=32,
        workers=32,
        ram_mb=32768,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["cpu"] == 32
    assert task["allow_cpu_training"]
    assert "exact SciPy" in task["cpu_training_justification"]
    assert ".server_artifacts" in task["stage_excludes"]
    assert "--oracle-workers 32" in task["cmd"]


def test_v17_14_frozen_counts_cover_the_full_v17_13_grid():
    assert spec.EXPECTED_FULL_GRID_CANDIDATE_COUNT == 900
    assert spec.EXPECTED_V17_13_ORACLE_CANDIDATE_COUNT == 48
    assert spec.EXPECTED_REMAINDER_CANDIDATE_COUNT == 852
