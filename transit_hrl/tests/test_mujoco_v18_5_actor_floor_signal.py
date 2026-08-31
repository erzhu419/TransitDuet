from types import SimpleNamespace

from scripts import mujoco_v18_5_actor_floor_signal_spec as spec
from scripts.run_mujoco_v18_5_actor_floor_signal import (
    SCORE_FIELDS,
    assess_score,
)
from scripts.submit_mujoco_v18_5_actor_floor_signal_scheduleurm import (
    DATA_LOCAL_NODE,
    build_scheduler_spec,
)


def _assessment_rows(*, separate_hopper=True):
    rows = []
    floor_modes = (
        "high_frequency",
        "low_frequency",
        "mixed",
        "mixed",
        "ood_chirp",
        "ood_chirp",
        "standard",
    )
    floor_seeds = (
        294864529,
        294864529,
        294864529,
        2802248628,
        294864529,
        2802248628,
        294864529,
    )
    for index, (mode, seed) in enumerate(
        zip(floor_modes, floor_seeds, strict=True)
    ):
        rows.append({
            "candidate_id": "actor_floor_h16_hold",
            "environment": "Hopper-v5",
            "disturbance_mode": mode,
            "evaluation_seed": seed,
            "actor_floor": True,
            "reference_feasible": False,
            "floor_ratio_mean": 2.0 - 0.01 * index,
        })
    environments = (
        ["Hopper-v5"] * 33
        + ["HalfCheetah-v5"] * 40
        + ["Walker2d-v5"] * 40
    )
    for index, environment in enumerate(environments):
        score = 0.1 + index * 1e-5
        if environment == "Hopper-v5" and not separate_hopper:
            score = 3.0
        rows.append({
            "candidate_id": "actor_floor_h16_hold",
            "environment": environment,
            "disturbance_mode": f"reference_{index}",
            "evaluation_seed": 1000 + index,
            "actor_floor": False,
            "reference_feasible": True,
            "floor_ratio_mean": score,
        })
    return rows


def test_v18_5_candidate_set_has_only_two_h16_causal_forecasts():
    assert set(spec.CANDIDATES) == {
        "actor_floor_h16_hold",
        "actor_floor_h16_damped_velocity",
    }
    assert all(
        row["planning_horizon"] == 16
        for row in spec.CANDIDATES.values()
    )
    assert len(SCORE_FIELDS) == 6


def test_v18_5_signal_gate_requires_within_hopper_separation():
    supported = assess_score(
        "actor_floor_h16_hold",
        "floor_ratio_mean",
        _assessment_rows(),
    )
    assert supported["global_rank_auc"] == 1.0
    assert supported["actor_floor_environment_rank_auc"] == 1.0
    assert supported["top_k_actor_floor_count"]["14"] == 7
    assert supported["unresolved_v17_14_path_rank"] <= 7
    assert supported["feedback_screen_eligible"]

    confounded = assess_score(
        "actor_floor_h16_hold",
        "floor_ratio_mean",
        _assessment_rows(separate_hopper=False),
    )
    assert confounded["actor_floor_environment_rank_auc"] == 0.0
    assert not confounded["feedback_screen_eligible"]


def test_v18_5_scheduler_is_data_local_and_target_free():
    args = SimpleNamespace(
        run_name="v18_5_test",
        python_executable="python3",
        cpu=16,
        workers=16,
        ram_mb=8192,
        priority="normal",
    )
    task = build_scheduler_spec(args)
    assert task["require_node"] == DATA_LOCAL_NODE == "node003"
    assert task["allowed_nodes"] == ["node003"]
    assert task["cpu"] == 16
    assert task["ram_mb"] == 8192
    assert spec.REFERENCE_DATASET_RUN in task["cmd"]
    assert "v17_12" not in task["cmd"]
    assert "target" not in task["cmd"]
    assert ".server_artifacts" in task["stage_excludes"]
