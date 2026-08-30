import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from freq_hrl.experiments.mujoco import control_validation
from scripts import (
    mujoco_v17_2_smooth_macro_gauge_preflight_spec as v17_2,
)
from scripts import (
    mujoco_v17_3_audit_optimal_macro_gauge_preflight_spec as spec,
)
from scripts.analyze_mujoco_v17_3_audit_optimal_macro_gauge_preflight import (
    analyze,
)
from scripts.run_mujoco_v17_3_paired_gauge_cell import (
    paired_intervention_audit,
    run_cell,
)
from scripts.submit_mujoco_v17_3_audit_optimal_macro_gauge_preflight_scheduleurm import (
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
)


def _paired_rows():
    control = []
    candidate = []
    for mode in spec.EVALUATION_DISTURBANCE_MODES:
        for seed in spec.EVALUATION_SEEDS:
            identity = f"{mode}-{seed}"
            common = {
                "disturbance_mode": mode,
                "seed": seed,
                "episode_return": 100.0,
                "RewardTraceSHA256": f"reward-{identity}",
                "ExecutedActionTraceSHA256": f"action-{identity}",
                "LatentPolicyTraceSHA256": f"latent-{identity}",
                "LatentUpperHFPowerAbs": 0.5,
                "LatentLowerLFDriftAbs": 0.5,
                "AdditiveActionClipExcessMax": 0.0,
                "AdditiveActionClipExcessRMS": 0.0,
                "LowerRouterActionReconstructionRMS": 0.0,
                "ResponsibilityReconstructionRMS": 0.0,
                "LowerRouterHeadroomClipRate": 0.0,
                "protocol_valid": 1.0,
                "parameter_count": 100,
                "episode_length": 512,
                "upper_decision_count": 32,
                "upper_transition_count": 32,
                "lower_transition_count": 512,
            }
            control.append({
                **common,
                "UpperHFPowerAbs": 1.0,
                "LowerLFDriftAbs": 1.0,
                "LowerActionRouterStrength": spec.CONTROL_STRENGTH,
                "paired_intervention": spec.CONTROL_INTERVENTION,
            })
            candidate.append({
                **common,
                "UpperHFPowerAbs": 0.8,
                "LowerLFDriftAbs": 0.8,
                "LowerActionRouterStrength": spec.CANDIDATE_STRENGTH,
                "paired_intervention": spec.CANDIDATE_INTERVENTION,
            })
    return control, candidate


def _write_csv(path: Path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_run(root: Path, run_name: str):
    run_root = root / "results" / run_name
    run_root.mkdir(parents=True)
    (run_root / "preregistration.json").write_text(json.dumps({
        "status": spec.PREREGISTRATION_STATUS,
        "evidence_role": spec.EVIDENCE_ROLE,
        "frozen_algorithm_revision": spec.FROZEN_ALGORITHM_REVISION,
        "frozen_source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
        "environments": list(spec.ENVIRONMENTS),
        "router_alpha": spec.ROUTER_ALPHA,
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "train_seeds": list(spec.TRAIN_SEEDS),
        "selection_seeds": list(spec.SELECTION_SEEDS),
        "evaluation_seeds": list(spec.EVALUATION_SEEDS),
        "selection_contract": spec.SELECTION_CONTRACT,
    }), encoding="utf-8")
    (run_root / "run_scoped_result_sync.json").write_text(json.dumps({
        "cell_count": spec.EXPECTED_CELL_COUNT,
        "artifact_contract": "small_results_only_v1",
        "tasks": {},
        "nodes": {},
    }), encoding="utf-8")
    for environment in spec.ENVIRONMENTS:
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            control, candidate = _paired_rows()
            audit = paired_intervention_audit(control, candidate)
            directory = root / cell_relative_dir(
                run_name, environment, optimizer_seed
            )
            directory.mkdir(parents=True)
            summary = {
                "development_protocol_version": spec.DEVELOPMENT_PROTOCOL_VERSION,
                "evidence_role": spec.EVIDENCE_ROLE,
                "protocol_version": spec.FROZEN_CORE_PROTOCOL_VERSION,
                "code_revision": spec.FROZEN_ALGORITHM_REVISION,
                "source_manifest_sha256": spec.FROZEN_SOURCE_MANIFEST_SHA256,
                "environment": environment,
                "optimizer_seed": optimizer_seed,
                "method": "freq_hrl",
                "responsibility_mode": "additive",
                "upper_action_decoder_mode": "causal_smoothstep_plan",
                "upper_action_decoder_contract": spec.SMOOTH_PLAN_CONTRACT,
                "lower_action_router_mode": (
                    "causal_audit_optimal_macro_gauge"
                ),
                "lower_action_router_alpha": spec.ROUTER_ALPHA,
                "lower_action_router_strength": spec.CONTROL_STRENGTH,
                "lower_action_router_observe_strength": False,
                "lower_action_router_contract": spec.ROUTER_CONTRACT,
                "policy_filter_state_contract": spec.POLICY_STATE_CONTRACT,
                "lower_action_router_training_schedule": "constant",
                "checkpoint_score_mode": spec.CHECKPOINT_SCORE_MODE,
                "selected_checkpoint_iteration": (
                    spec.CHECKPOINT_MINIMUM_ITERATION
                ),
                "heldout_evaluation_pass_count": 2,
                "paired_intervention_audit": audit,
            }
            (directory / "cell_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            _write_csv(
                directory / "evaluation_rows.csv", [*control, *candidate]
            )


def test_v173_design_is_fresh_paired_and_dynamic():
    old_seeds = set(
        v17_2.OPTIMIZER_SEEDS
        + v17_2.TRAIN_SEEDS
        + v17_2.SELECTION_SEEDS
        + v17_2.EVALUATION_SEEDS
    )
    new_seeds = set(
        spec.OPTIMIZER_SEEDS
        + spec.TRAIN_SEEDS
        + spec.SELECTION_SEEDS
        + spec.EVALUATION_SEEDS
    )
    assert not old_seeds & new_seeds
    assert len(new_seeds) == 17
    assert spec.EXPECTED_CELL_COUNT == 3
    assert spec.EXPECTED_PATHS_PER_INTERVENTION == 40
    args = SimpleNamespace(
        run_name="v173_test",
        python_executable="python3",
        nodes=[f"node00{index}" for index in range(1, 7)],
        priority="normal",
    )
    command = build_training_command(
        args, "HalfCheetah-v5", spec.OPTIMIZER_SEEDS[0]
    )
    scheduler_spec = build_scheduler_spec(
        args, "HalfCheetah-v5", spec.OPTIMIZER_SEEDS[0]
    )
    assert "run_mujoco_v17_3_paired_gauge_cell.py" in command
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["ram_mb"] == 1024


def test_paired_audit_detects_trace_mismatch():
    control, candidate = _paired_rows()
    audit = paired_intervention_audit(control, candidate)
    assert audit["all_trace_hashes_match"]
    assert audit["path_count"] == spec.EXPECTED_PATHS_PER_INTERVENTION
    candidate[0]["ExecutedActionTraceSHA256"] = "changed"
    mismatch = paired_intervention_audit(control, candidate)
    assert mismatch["trace_mismatches"]["ExecutedActionTraceSHA256"] == 1
    assert not mismatch["all_trace_hashes_match"]


def test_analysis_supports_only_complete_strict_paired_improvement(tmp_path):
    run_name = "synthetic_v173"
    _write_synthetic_run(tmp_path, run_name)
    result = analyze(run_name, root=tmp_path)
    assert result["status"] == spec.SUPPORTED_STATUS
    assert result["cell_count"] == spec.EXPECTED_CELL_COUNT
    assert result["paired_path_count"] == 120
    assert result["evaluation_row_count"] == 240
    assert result["eligible_for_leakage_active_multiseed"]
    assert all(row["supported"] for row in result["cells"])


def test_worker_runs_one_real_paired_smoke(monkeypatch):
    pytest.importorskip("gymnasium")
    monkeypatch.setattr(
        control_validation,
        "verify_current_freq_hrl_source_identity",
        lambda **kwargs: {
            "code_revision": str(kwargs["code_revision"]),
            "source_manifest_sha256": str(
                kwargs["expected_source_manifest_sha256"]
            ),
            "source_identity_status": "verified",
        },
    )
    monkeypatch.setattr(spec, "STEPS", 16)
    monkeypatch.setattr(spec, "EPISODE_HORIZON", 16)
    monkeypatch.setattr(spec, "ITERATIONS", 1)
    monkeypatch.setattr(spec, "UPPER_PERIOD", 4)
    monkeypatch.setattr(spec, "HIDDEN_DIM", 8)
    monkeypatch.setattr(spec, "TRAIN_SEEDS", spec.TRAIN_SEEDS[:1])
    monkeypatch.setattr(spec, "SELECTION_SEEDS", spec.SELECTION_SEEDS[:1])
    monkeypatch.setattr(spec, "EVALUATION_SEEDS", spec.EVALUATION_SEEDS[:1])
    monkeypatch.setattr(spec, "TRAINING_DISTURBANCE_MODES", ("standard",))
    monkeypatch.setattr(spec, "EVALUATION_DISTURBANCE_MODES", ("standard",))
    monkeypatch.setattr(spec, "CHECKPOINT_MINIMUM_ITERATION", 0)
    monkeypatch.setattr(spec, "CHECKPOINT_EVALUATION_INTERVAL", 1)
    payload, rows, _ = run_cell(
        env_id="HalfCheetah-v5",
        optimizer_seed=spec.OPTIMIZER_SEEDS[0],
    )
    assert len(rows) == 2
    assert payload["paired_intervention_audit"]["all_trace_hashes_match"]
    assert rows[0]["RewardTraceSHA256"] == rows[1]["RewardTraceSHA256"]
    assert rows[0]["ExecutedActionTraceSHA256"] == rows[1][
        "ExecutedActionTraceSHA256"
    ]
