import argparse

from scripts import analyze_mujoco_v17_1_headroom_homotopy_preflight as analyzer
from scripts import mujoco_v17_1_headroom_homotopy_preflight_spec as spec
from scripts.submit_mujoco_v17_1_headroom_homotopy_preflight_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    selected_cells,
)


def _args():
    return argparse.Namespace(
        run_name="v17_1_test",
        arms=list(spec.ARMS),
        nodes=[f"node{index:03d}" for index in range(1, 7)],
        environments=list(spec.ENVIRONMENTS),
        optimizer_seeds=list(spec.OPTIMIZER_SEEDS),
        python_executable="python3",
        priority="normal",
        recovery_only=False,
    )


def test_frozen_v171_matrix_is_dynamic_and_small_result_only():
    spec.validate()
    args = _args()
    assert len(selected_cells(args)) == 15
    candidate = build_training_command(
        args,
        "Walker2d-v5",
        spec.HEADROOM_HOMOTOPY_PROMOTION_05,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--upper-action-decoder-mode causal_smoothstep_plan" in candidate
    assert "--upper-promotion-gain 0.5" in candidate
    assert "--lower-action-router-mode causal_macro_zero_dc_headroom" in candidate
    assert "--lower-action-router-training-schedule delayed_cosine" in candidate
    assert "--lower-action-router-observe-strength" in candidate
    assert (
        f"--control-protocol-version {spec.CANDIDATE_CORE_PROTOCOL_VERSION}"
        in candidate
    )
    assert "--safety-selection-seeds" in candidate
    assert ".server_artifacts/v17_1_test/" in candidate
    control = build_training_command(
        args,
        "Walker2d-v5",
        spec.SMOOTH_DIRECT_CONTROL,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert f"--control-protocol-version {spec.DIRECT_CORE_PROTOCOL_VERSION}" in control
    scheduler = build_scheduler_spec(
        args,
        "Walker2d-v5",
        spec.HEADROOM_HOMOTOPY_PROMOTION_05,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler["require_node"] is None
    assert scheduler["allowed_nodes"] == args.nodes
    assert scheduler["cpu"] == 1
    assert scheduler["ram_mb"] == 1024
    assert ".server_artifacts" in scheduler["ckpt_dir"]
    assert ".server_artifacts" not in scheduler["result_dir"]
    assert ".server_artifacts" in scheduler["stage_excludes"]


def _fake_cell(_run_name, environment, arm, optimizer_seed):
    arm_spec = spec.ARMS[arm]
    candidate = arm in spec.CANDIDATE_ARMS
    summary = {
        "protocol_version": arm_spec["control_protocol_version"],
        "code_revision": spec.FROZEN_ALGORITHM_REVISION,
        "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "environment": environment,
        "optimizer_seed": optimizer_seed,
        "method": arm_spec["method"],
        "responsibility_mode": arm_spec["responsibility_mode"],
        "upper_action_decoder_mode": arm_spec["upper_action_decoder_mode"],
        "upper_action_decoder_contract": spec.SMOOTH_PLAN_CONTRACT,
        "upper_promotion_gain": arm_spec["upper_promotion_gain"],
        "lower_action_router_mode": arm_spec["lower_action_router_mode"],
        "lower_action_router_contract": (
            spec.HEADROOM_ROUTER_CONTRACT
            if candidate
            else "direct_lower_action_with_no_frequency_projection_v1"
        ),
        "lower_action_headroom_contract": (
            spec.HEADROOM_ACTION_CONTRACT if candidate else "disabled"
        ),
        "lower_action_router_alpha": arm_spec["lower_action_router_alpha"],
        "lower_action_router_strength": arm_spec["lower_action_router_strength"],
        "lower_action_router_training_schedule": arm_spec[
            "lower_action_router_training_schedule"
        ],
        "lower_action_router_warmup_fraction": arm_spec[
            "lower_action_router_warmup_fraction"
        ],
        "lower_action_router_ramp_fraction": arm_spec[
            "lower_action_router_ramp_fraction"
        ],
        "lower_action_router_observe_strength": True,
        "lower_action_router_training_strengths_by_iteration": (
            analyzer._expected_training_strengths(arm_spec)
        ),
        "leakage_constraint_scope": arm_spec["leakage_constraint_scope"],
        "leakage_constraint_cost_mode": arm_spec["leakage_cost_mode"],
        "checkpoint_score_mode": spec.CHECKPOINT_SCORE_MODE,
        "selected_checkpoint_iteration": 40,
        "capacity_actual_parameter_count": 1234,
    }
    if arm == spec.SMOOTH_DIRECT_CONTROL:
        reward, upper, raw_lower, latent_lower = 100.0, 0.004, 0.004, 0.004
    elif arm == spec.HEADROOM_EXACT:
        reward, upper, raw_lower, latent_lower = 80.0, 0.002, 0.002, 0.004
    elif arm == spec.HEADROOM_HOMOTOPY:
        reward, upper, raw_lower, latent_lower = 98.0, 0.0035, 0.0038, 0.004
    elif arm == spec.HEADROOM_HOMOTOPY_PROMOTION_05:
        reward, upper, raw_lower, latent_lower = 101.0, 0.0025, 0.002, 0.004
    else:
        reward, upper, raw_lower, latent_lower = 94.0, 0.0025, 0.002, 0.004
    projection = 0.25 if candidate else 0.0
    promotion = 0.02 if float(arm_spec["upper_promotion_gain"]) > 0.0 else 0.0
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
                "LowerRouterMacroCompletionErrorMax": 0.0,
                "ResponsibilityReconstructionRMS": 0.0,
                "AdditiveActionClipRate": 0.0,
                "LowerRouterHeadroomClipRate": 0.05 if candidate else 0.0,
                "UpperPromotionRMS": promotion,
                "UpperPromotionActivationRate": 0.5 if promotion else 0.0,
                "UpperPromotionGain": arm_spec["upper_promotion_gain"],
                "LowerActionRouterStrength": arm_spec[
                    "lower_action_router_strength"
                ],
            })
    return summary, rows


def test_analyzer_selects_one_global_arm_only_after_all_environment_gates(
    monkeypatch,
):
    monkeypatch.setattr(analyzer, "_load_cell", _fake_cell)
    result = analyzer.analyze("synthetic")
    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["cell_count"] == 12
    assert (
        result["selected_arm_for_fresh_multiseed"]
        == spec.HEADROOM_HOMOTOPY_PROMOTION_05
    )
    eligibility = {
        row["arm"]: row["eligible_for_fresh_multiseed"]
        for row in result["candidate_results"]
    }
    assert eligibility == {
        spec.HEADROOM_EXACT: False,
        spec.HEADROOM_HOMOTOPY: False,
        spec.HEADROOM_HOMOTOPY_PROMOTION_05: True,
        spec.HEADROOM_HOMOTOPY_PROMOTION_10: False,
    }
