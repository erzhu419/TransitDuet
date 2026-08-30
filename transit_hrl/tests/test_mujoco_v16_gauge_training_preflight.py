from argparse import Namespace
from copy import deepcopy

from freq_hrl.domains.mujoco import lower_action_router_contract
from scripts import mujoco_v15_2_multisource_distillation_preflight_spec as v15_2
from scripts import mujoco_v16_gauge_training_preflight_spec as spec
from scripts.analyze_mujoco_v16_gauge_training_preflight import analyze_cells
from scripts.submit_mujoco_v16_gauge_training_preflight_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    selected_cells,
)


def _args() -> Namespace:
    return Namespace(
        run_name="mujoco_v16_unit",
        arms=list(spec.ARMS),
        nodes=[f"node00{index}" for index in range(1, 7)],
        environments=list(spec.ENVIRONMENTS),
        optimizer_seeds=list(spec.OPTIMIZER_SEEDS),
        python_executable="python3",
        priority="normal",
    )


def _cell(environment: str, arm: str, seed: int):
    arm_spec = spec.ARMS[arm]
    summary = {
        "environment": environment,
        "optimizer_seed": seed,
        "method": "freq_hrl",
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "lower_action_router_mode": arm_spec["lower_action_router_mode"],
        "lower_action_router_strength": arm_spec["lower_action_router_strength"],
        "lower_action_router_contract": lower_action_router_contract(
            arm_spec["lower_action_router_mode"]
        ),
        "leakage_constraint_scope": arm_spec["leakage_constraint_scope"],
        "leakage_constraint_cost_mode": arm_spec["leakage_cost_mode"],
        "upper_constraint_mode": "primal_dual",
        "upper_dual_lr": arm_spec["upper_dual_lr"],
        "lower_dual_lr": arm_spec["lower_dual_lr"],
    }
    if arm == spec.JOINT_PD_CONTROL:
        reward, lower, upper, latent_lower, latent_upper = (
            100.0, 0.0020, 0.0040, 0.0010, 0.0020
        )
    elif arm == spec.GAUGE_REWARD_CONTROL:
        reward, lower, upper, latent_lower, latent_upper = (
            100.0, 0.0006, 0.0008, 0.0010, 0.0020
        )
    else:
        reward, lower, upper, latent_lower, latent_upper = (
            99.0, 0.0003, 0.0004, 0.0008, 0.0015
        )
    rows = []
    for mode in spec.EVALUATION_DISTURBANCE_MODES:
        for evaluation_seed in spec.EVALUATION_SEEDS:
            rows.append({
                "seed": evaluation_seed,
                "environment": environment,
                "disturbance_mode": mode,
                "episode_return": reward,
                "protocol_valid": 1.0,
                "LowerActionRouterMode": arm_spec["lower_action_router_mode"],
                "LowerLFDriftAbs": lower,
                "UpperHFPowerAbs": upper,
                "LatentLowerLFDriftAbs": latent_lower,
                "LatentUpperHFPowerAbs": latent_upper,
                "ResponsibilityReconstructionRMS": 0.0,
                "LowerRouterActionReconstructionRMS": 0.0,
            })
    return environment, arm, seed, summary, rows


def _supported_cells():
    return [
        _cell(environment, arm, seed)
        for environment in spec.ENVIRONMENTS
        for arm in spec.ARMS
        for seed in spec.OPTIMIZER_SEEDS
    ]


def test_v16_design_uses_fresh_paired_roles_and_capacity_matched_arms():
    current = set(
        spec.OPTIMIZER_SEEDS
        + spec.TRAIN_SEEDS
        + spec.SELECTION_SEEDS
        + spec.EVALUATION_SEEDS
    )
    previous = set(v15_2.DISTILL_ROOTS + v15_2.DESIGN_ROOTS + v15_2.VALIDATION_ROOTS)

    assert len(current) == 19
    assert not current & previous
    assert spec.EXPECTED_CELL_COUNT == 27
    assert all(arm["upper_constraint_mode"] == "primal_dual" for arm in spec.ARMS.values())
    assert spec.ARMS[spec.GAUGE_REWARD_CONTROL]["upper_dual_lr"] == 0.0
    assert spec.ARMS[spec.GAUGE_PD_CANDIDATE]["upper_dual_lr"] > 0.0


def test_v16_launcher_emits_dynamic_single_core_paired_cells():
    args = _args()
    assert len(selected_cells(args)) == spec.EXPECTED_CELL_COUNT
    scheduler = build_scheduler_spec(
        args,
        spec.ENVIRONMENTS[0],
        spec.GAUGE_PD_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    command = build_training_command(
        args,
        spec.ENVIRONMENTS[0],
        spec.GAUGE_PD_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )

    assert scheduler["require_node"] is None
    assert scheduler["cpu"] == 1
    assert set(scheduler["allowed_nodes"]) == set(args.nodes)
    assert "--lower-action-router-mode causal_total_action_gauge" in command
    assert "--lower-action-router-strength 1.0" in command
    assert "--upper-dual-lr 0.03" in command
    assert f"--code-revision {spec.FROZEN_ALGORITHM_REVISION}" in command
    assert command.endswith("&& echo DONE")


def test_v16_analysis_requires_reward_reconstruction_and_learned_latent_gain():
    result = analyze_cells(_supported_cells())

    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["support_gate"] is True
    assert result["cell_count"] == 9
    assert result["latent_improvement_count"] == 9

    failed = deepcopy(_supported_cells())
    for environment, arm, seed, _, rows in failed:
        if arm == spec.GAUGE_PD_CANDIDATE:
            for row in rows:
                row["LatentLowerLFDriftAbs"] = 0.0010
                row["LatentUpperHFPowerAbs"] = 0.0020
    rejected = analyze_cells(failed)
    assert rejected["status"] == spec.NOT_SUPPORTED_STATUS
    assert rejected["latent_improvement_count"] == 0


def test_v16_analysis_rejects_incomplete_path_registry():
    cells = _supported_cells()
    cells[0][4].pop()

    try:
        analyze_cells(cells)
    except ValueError as exc:
        assert "invalid or incomplete" in str(exc)
    else:
        raise AssertionError("incomplete v16 cells must be rejected")
