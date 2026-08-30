from types import SimpleNamespace

from scripts import mujoco_v17_9_prefix_hpf_fir_spec as spec
from scripts.submit_mujoco_v17_9_selection_scheduleurm import (
    CPU_JUSTIFICATION,
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)
from scripts.train_mujoco_v17_9_prefix_hpf_fir import candidate_configs


def test_v17_9_grid_is_gain_one_and_shared_across_environments():
    candidates = candidate_configs()
    assert len(candidates) == 8
    assert {row["output_gain"] for row in candidates} == {1.0}
    assert {row["router_mode"] for row in candidates} == {
        "prefix_hpf_innovation_projection"
    }
    assert set(spec.FRESH_VALIDATION_SEEDS).isdisjoint(
        spec.REUSED_SELECTION_SEEDS
    )


def test_v17_9_selection_is_explicit_cpu_work_on_dataset_node():
    args = SimpleNamespace(
        dataset_run_name="v17_8_dataset_test",
        run_name="v17_9_selection_test",
        python_executable="python3",
        cpu=8,
        ram_mb=8192,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE
    assert task["allowed_nodes"] == [DATA_LOCAL_NODE]
    assert task["vram"] == 0
    assert task["cpu"] == 8
    assert task["allow_cpu_training"]
    assert task["cpu_training_justification"] == CPU_JUSTIFICATION
    assert ".server_artifacts" in task["stage_excludes"]
