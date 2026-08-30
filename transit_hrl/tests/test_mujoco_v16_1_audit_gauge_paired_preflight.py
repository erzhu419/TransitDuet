from argparse import Namespace
from copy import deepcopy

from freq_hrl.domains.mujoco import lower_action_router_contract
from scripts import mujoco_v16_1_audit_gauge_paired_preflight_spec as spec
from scripts import mujoco_v16_gauge_training_preflight_spec as v16
from scripts.analyze_mujoco_v16_1_audit_gauge_paired_preflight import (
    analyze_cells,
)
from scripts.submit_mujoco_v16_1_audit_gauge_paired_preflight_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    selected_cells,
)


def _args() -> Namespace:
    return Namespace(
        run_name="mujoco_v16_1_unit",
        phases=["anchor", "continuation"],
        arms=list(spec.ARMS),
        nodes=[f"node00{index}" for index in range(1, 7)],
        environments=list(spec.ENVIRONMENTS),
        optimizer_seeds=list(spec.OPTIMIZER_SEEDS),
        python_executable="python3",
        priority="normal",
    )


def _cell(phase: str, environment: str, arm: str, seed: int):
    is_anchor = phase == "anchor"
    arm_spec = spec.ANCHOR_SPEC if is_anchor else spec.ARMS[arm]
    if is_anchor or arm == spec.REWARD_CONTINUATION_CONTROL:
        reward, lower, upper, latent_lower, latent_upper = (
            100.0,
            0.0020,
            0.0040,
            0.0010,
            0.0020,
        )
    else:
        reward, lower, upper, latent_lower, latent_upper = (
            99.0,
            0.0010,
            0.0020,
            0.0009,
            0.0018,
        )
    summary = {
        "environment": environment,
        "optimizer_seed": seed,
        "method": "freq_hrl",
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "lower_action_router_mode": arm_spec["lower_action_router_mode"],
        "lower_action_router_alpha": arm_spec["lower_action_router_alpha"],
        "lower_action_router_strength": arm_spec["lower_action_router_strength"],
        "lower_action_router_contract": lower_action_router_contract(
            arm_spec["lower_action_router_mode"]
        ),
        "leakage_constraint_scope": arm_spec["leakage_constraint_scope"],
        "leakage_constraint_cost_mode": arm_spec["leakage_cost_mode"],
        "upper_constraint_mode": "primal_dual",
        "upper_dual_lr": arm_spec["upper_dual_lr"],
        "lower_dual_lr": arm_spec["lower_dual_lr"],
        "checkpoint_score_mode": arm_spec["checkpoint_score_mode"],
        "selected_checkpoint_iteration": 20 if is_anchor else 3,
    }
    if not is_anchor:
        summary["paired_checkpoint_continuation"] = {
            "enabled": True,
            "checkpoint_environment": environment,
            "checkpoint_optimizer_seed": seed,
            "checkpoint_router_mode": spec.ANCHOR_SPEC[
                "lower_action_router_mode"
            ],
        }
        summary["selected_checkpoint_diagnostics"] = {
            "constraints": [
                {"normalized_violation": 0.0}
                for _ in range(len(spec.TRAINING_DISTURBANCE_MODES) * 6)
            ]
        }
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
                "LowerRouterAuditAlphaMean": 0.23,
                "LowerRouterAuditAlphaFinal": 0.25,
                "ResponsibilityReconstructionRMS": 0.0,
                "LowerRouterActionReconstructionRMS": 0.0,
            })
    return phase, environment, arm, seed, summary, rows


def _supported_cells():
    cells = []
    for environment in spec.ENVIRONMENTS:
        for seed in spec.OPTIMIZER_SEEDS:
            cells.append(_cell("anchor", environment, spec.ANCHOR_ARM, seed))
            cells.extend(
                _cell("continuation", environment, arm, seed)
                for arm in spec.ARMS
            )
    return cells


def test_v16_1_design_uses_new_disjoint_roles_and_matched_capacity():
    current = set(
        spec.OPTIMIZER_SEEDS
        + spec.PRETRAIN_SEEDS
        + spec.PRETRAIN_SELECTION_SEEDS
        + spec.CONTINUATION_TRAIN_SEEDS
        + spec.CONTINUATION_SELECTION_SEEDS
        + spec.EVALUATION_SEEDS
    )
    previous = set(
        v16.OPTIMIZER_SEEDS
        + v16.TRAIN_SEEDS
        + v16.SELECTION_SEEDS
        + v16.EVALUATION_SEEDS
    )

    assert len(current) == 27
    assert not current & previous
    assert spec.EXPECTED_ANCHOR_CELL_COUNT == 9
    assert spec.EXPECTED_CONTINUATION_CELL_COUNT == 18
    assert spec.ANCHOR_SPEC["upper_constraint_mode"] == "primal_dual"
    assert spec.ANCHOR_SPEC["upper_dual_lr"] == 0.0
    assert spec.ARMS[spec.PRIMAL_DUAL_CANDIDATE]["upper_dual_lr"] > 0.0


def test_v16_1_launcher_emits_dependency_gated_dynamic_cells():
    args = _args()
    cells = selected_cells(args)
    assert len(cells) == 27
    continuation = next(cell for cell in cells if cell[0] == "continuation")
    scheduler = build_scheduler_spec(args, *continuation)
    command = build_training_command(args, *continuation)

    assert scheduler["require_node"] is None
    assert scheduler["cpu"] == 1
    assert len(scheduler["wait_for_files"]) == 2
    assert set(scheduler["allowed_nodes"]) == set(args.nodes)
    assert "--lower-action-router-mode causal_audit_aligned_gauge" in command
    assert "--lower-action-router-alpha 0.2" in command
    assert "--checkpoint-score-mode paired_relative_frequency_feasibility_first" in command
    assert "--initial-checkpoint-path" in command
    assert f"--code-revision {spec.FROZEN_ALGORITHM_REVISION}" in command
    assert command.endswith("&& echo DONE")


def test_v16_1_analysis_requires_paired_reward_and_frequency_gates():
    result = analyze_cells(_supported_cells())

    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["support_gate"] is True
    assert result["cell_count"] == 9
    assert result["canonical_improvement_count"] == 9

    failed = deepcopy(_supported_cells())
    for phase, _, arm, _, _, rows in failed:
        if phase == "continuation" and arm == spec.PRIMAL_DUAL_CANDIDATE:
            for row in rows:
                row["episode_return"] = 80.0
    rejected = analyze_cells(failed)
    assert rejected["status"] == spec.NOT_SUPPORTED_STATUS
    assert not all(row["reward_noninferior"] for row in rejected["cells"])


def test_v16_1_analysis_rejects_an_unpaired_path_registry():
    cells = _supported_cells()
    cells[0][5].pop()

    try:
        analyze_cells(cells)
    except ValueError as exc:
        assert "invalid or incomplete" in str(exc)
    else:
        raise AssertionError("incomplete v16.1 cells must be rejected")
