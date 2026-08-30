import argparse
import json

from scripts import analyze_mujoco_v17_zero_dc_plan_screen as analyzer
from scripts import mujoco_v17_zero_dc_plan_screen_spec as spec
from scripts.run_mujoco_cell_small_export import export_small_cell
from scripts.submit_mujoco_v17_zero_dc_plan_screen_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    selected_cells,
)


def _args():
    return argparse.Namespace(
        run_name="v17_test",
        arms=list(spec.ARMS),
        nodes=[f"node{index:03d}" for index in range(1, 7)],
        environments=list(spec.ENVIRONMENTS),
        optimizer_seeds=list(spec.OPTIMIZER_SEEDS),
        python_executable="python3",
        priority="normal",
    )


def test_frozen_v17_matrix_scheduler_and_small_result_contract():
    spec.validate()
    args = _args()
    assert len(selected_cells(args)) == 27
    command = build_training_command(
        args,
        "Walker2d-v5",
        spec.ZERO_DC_PLAN_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--upper-action-decoder-mode causal_smoothstep_plan" in command
    assert "--lower-action-router-mode causal_macro_zero_dc" in command
    assert f"--control-protocol-version {spec.FROZEN_CORE_PROTOCOL_VERSION}" in command
    assert "--iterations 128" in command
    assert ".server_artifacts/v17_test/" in command
    scheduler = build_scheduler_spec(
        args,
        "Walker2d-v5",
        spec.ZERO_DC_PLAN_CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler["require_node"] is None
    assert scheduler["allowed_nodes"] == args.nodes
    assert scheduler["cpu"] == 1
    assert ".server_artifacts" in scheduler["ckpt_dir"]
    assert ".server_artifacts" not in scheduler["result_dir"]
    assert ".server_artifacts" in scheduler["stage_excludes"]


def test_small_export_omits_checkpoint_and_training_history(tmp_path):
    source = tmp_path / "full"
    target = tmp_path / "small"
    source.mkdir()
    (source / "cell_summary.json").write_text("{}\n", encoding="utf-8")
    (source / "evaluation_rows.csv").write_text("metric\n", encoding="utf-8")
    (source / "checkpoint.pt").write_bytes(b"checkpoint")
    (source / "training_history.json").write_text("[]\n", encoding="utf-8")
    export_small_cell(
        source,
        target,
        server_full_output_dir=".server_artifacts/run/cell",
    )
    assert sorted(path.name for path in target.iterdir()) == [
        "cell_summary.json",
        "evaluation_rows.csv",
        "server_artifact_location.json",
    ]
    location = json.loads(
        (target / "server_artifact_location.json").read_text(encoding="utf-8")
    )
    assert location["server_only_files"] == [
        "checkpoint.pt",
        "training_history.json",
    ]


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
        "upper_action_decoder_mode": arm_spec["upper_action_decoder_mode"],
        "upper_action_decoder_contract": (
            spec.SMOOTH_PLAN_CONTRACT
            if arm != spec.HOLD_DIRECT_CONTROL
            else "macro_target_zero_order_hold_v1"
        ),
        "lower_action_router_mode": arm_spec["lower_action_router_mode"],
        "lower_action_router_contract": (
            spec.ZERO_DC_ROUTER_CONTRACT
            if arm == spec.ZERO_DC_PLAN_CANDIDATE
            else "direct_lower_action_with_no_frequency_projection_v1"
        ),
        "lower_action_router_alpha": arm_spec["lower_action_router_alpha"],
        "lower_action_router_strength": (
            0.0
            if arm_spec["lower_action_router_mode"] == "direct"
            else arm_spec["lower_action_router_strength"]
        ),
        "lower_action_router_observe_strength": False,
        "leakage_constraint_scope": arm_spec["leakage_constraint_scope"],
        "leakage_constraint_cost_mode": arm_spec["leakage_cost_mode"],
        "checkpoint_score_mode": spec.CHECKPOINT_SCORE_MODE,
        "selected_checkpoint_iteration": 40,
        "capacity_actual_parameter_count": 1234,
    }
    if arm == spec.HOLD_DIRECT_CONTROL:
        reward, upper, raw_lower, latent_lower = 100.0, 0.004, 0.004, 0.004
        projection = 0.0
    elif arm == spec.SMOOTH_DIRECT_CONTROL:
        reward, upper, raw_lower, latent_lower = 100.0, 0.0025, 0.004, 0.004
        projection = 0.0
    else:
        reward, upper, raw_lower, latent_lower = 98.0, 0.002, 0.002, 0.004
        projection = 0.25
    rows = []
    for mode in spec.EVALUATION_DISTURBANCE_MODES:
        for seed in spec.EVALUATION_SEEDS:
            rows.append({
                "disturbance_mode": mode,
                "seed": seed,
                "protocol_valid": 1,
                "episode_return": reward,
                "UpperHFPowerAbs": upper,
                "RawLowerLFDriftAbs": raw_lower,
                "LatentLowerLFDriftAbs": latent_lower,
                "LowerRouterMacroProjectionRate": projection,
                "LowerRouterMacroDebtRMSMean": 0.01 if projection else 0.0,
                "LowerRouterMacroCompletionErrorMax": 0.0,
                "ResponsibilityReconstructionRMS": 0.0,
                "LowerRouterClipRate": projection,
            })
    return summary, rows


def test_analyzer_requires_physical_architecture_gates(monkeypatch):
    monkeypatch.setattr(analyzer, "_load_cell", _fake_cell)
    result = analyzer.analyze("synthetic")
    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["supported_cell_count"] == 9
    assert result["gate_counts"] == {
        "trained_checkpoint": 9,
        "reward_noninferior": 9,
        "smooth_upper_ablation": 9,
        "candidate_upper_hf_reduction": 9,
        "candidate_upper_hf_budget": 9,
        "raw_lower_lf_reduction_vs_smooth": 9,
        "raw_lower_lf_reduction_vs_latent": 9,
        "raw_joint_merit_reduction": 9,
        "complete_macro_zero_sum": 9,
        "projection_active": 9,
        "responsibility_reconstruction_exact": 9,
    }
    assert all(row["environment_gate"] for row in result["environment_results"])
    for row in result["cells"]:
        assert row["supported"]
        assert all(row["gates"].values())
