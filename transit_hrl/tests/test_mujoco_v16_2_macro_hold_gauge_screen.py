import argparse

from scripts import mujoco_v16_2_macro_hold_gauge_screen_spec as spec
from scripts import analyze_mujoco_v16_2_macro_hold_gauge_screen as analyzer
from scripts.submit_mujoco_v16_2_macro_hold_gauge_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    selected_cells,
)


def _args():
    return argparse.Namespace(
        run_name="v16_2_test",
        arms=list(spec.ARMS),
        nodes=[f"node{index:03d}" for index in range(1, 7)],
        environments=list(spec.ENVIRONMENTS),
        optimizer_seeds=list(spec.OPTIMIZER_SEEDS),
        python_executable="python3",
        priority="normal",
    )


def test_frozen_v16_2_matrix_and_scheduler_contract():
    spec.validate()
    args = _args()
    assert len(selected_cells(args)) == 27
    command = build_training_command(
        args,
        "Walker2d-v5",
        spec.MACRO_HOLD_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--lower-action-router-mode causal_macro_hold_audit_gauge" in command
    assert f"--control-protocol-version {spec.FROZEN_CORE_PROTOCOL_VERSION}" in command
    scheduler = build_scheduler_spec(
        args,
        "Walker2d-v5",
        spec.MACRO_HOLD_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler["require_node"] is None
    assert scheduler["allowed_nodes"] == args.nodes
    assert scheduler["cpu"] == 1


def _fake_cell(_run_name, environment, arm, optimizer_seed):
    arm_spec = spec.ARMS[arm]
    summary = {
        "protocol_version": spec.FROZEN_CORE_PROTOCOL_VERSION,
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "environment": environment,
        "optimizer_seed": optimizer_seed,
        "method": arm_spec["method"],
        "responsibility_mode": arm_spec["responsibility_mode"],
        "lower_action_router_mode": arm_spec["lower_action_router_mode"],
        "lower_action_router_alpha": arm_spec["lower_action_router_alpha"],
        "lower_action_router_strength": arm_spec["lower_action_router_strength"],
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": arm_spec["leakage_constraint_scope"],
        "leakage_constraint_cost_mode": arm_spec["leakage_cost_mode"],
        "checkpoint_score_mode": spec.CHECKPOINT_SCORE_MODE,
        "selected_checkpoint_iteration": 20,
        "parameter_count": 1234,
    }
    if arm == spec.PRIMITIVE_GAUGE_CONTROL:
        upper, lower, latent_upper, latent_lower = 0.006, 0.002, 0.004, 0.004
    elif arm == spec.MACRO_HOLD_CANDIDATE:
        upper, lower, latent_upper, latent_lower = 0.003, 0.002, 0.004, 0.004
    else:
        upper, lower, latent_upper, latent_lower = 0.004, 0.004, 0.004, 0.004
    rows = []
    for mode in spec.EVALUATION_DISTURBANCE_MODES:
        for seed in spec.EVALUATION_SEEDS:
            rows.append({
                "disturbance_mode": mode,
                "seed": seed,
                "protocol_valid": 1,
                "episode_return": 100.0,
                "UpperHFPowerAbs": upper,
                "LowerLFDriftAbs": lower,
                "LatentUpperHFPowerAbs": latent_upper,
                "LatentLowerLFDriftAbs": latent_lower,
                "LowerRouterClipRate": 0.0,
                "LowerRouterActionReconstructionRMS": 0.0,
                "ResponsibilityReconstructionRMS": 0.0,
                "LowerRouterAuditAlphaMean": 0.20,
                "LowerRouterAuditAlphaFinal": 0.18,
            })
    return summary, rows


def test_analyzer_requires_all_component_gates(monkeypatch):
    monkeypatch.setattr(analyzer, "_load_cell", _fake_cell)
    result = analyzer.analyze("synthetic")
    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["supported_cell_count"] == 9
    assert all(row["environment_gate"] for row in result["environment_results"])
    for row in result["cells"]:
        assert row["supported"]
        assert all(row["gates"].values())
