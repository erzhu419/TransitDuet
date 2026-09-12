import argparse
import ast
import json
from pathlib import Path
import shlex
from unittest import mock

from freq_hrl.experiments.mujoco import control_validation
from scripts import analyze_mujoco_v24_policy_mean_upper_projection_target_development as analysis
from scripts import mujoco_v24_policy_mean_upper_projection_target_development_spec as spec
from scripts import submit_mujoco_v24_policy_mean_upper_projection_target_development_scheduleurm as submit


def _args(**overrides):
    values = {
        "run_name": "v24_test",
        "arms": list(spec.ARMS),
        "nodes": [
            "node001",
            "node002",
            "node003",
            "node004",
            "node005",
            "node006",
        ],
        "environments": list(spec.ENVIRONMENTS),
        "optimizer_seeds": list(spec.OPTIMIZER_SEEDS),
        "python_executable": "/opt/freqhrl/bin/python",
        "priority": "normal",
        "dispatch": False,
        "dry_run": True,
        "sync_only": False,
        "sync_workers": 6,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_frozen_roots_are_disjoint_from_earlier_mujoco_literals():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    historical_integers = set()
    for path in scripts.glob("*mujoco*.py"):
        if path.name == (
            "mujoco_v24_policy_mean_upper_projection_target_"
            "development_spec.py"
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        historical_integers.update(
            int(node.value)
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, int)
                and not isinstance(node.value, bool)
            )
        )
    fresh = set(
        spec.OPTIMIZER_SEEDS
        + spec.TRAIN_SEEDS
        + spec.SELECTION_SEEDS
        + spec.EVALUATION_SEEDS
    )
    assert len(fresh) == 20
    assert not fresh & historical_integers
    assert spec.EXPECTED_CELL_COUNT == 48


def test_policy_mean_candidate_isolates_only_the_target_estimator():
    causal = spec.ARMS[spec.DECISION_TIME_UNIFORM_010]
    candidate = spec.ARMS[spec.POLICY_MEAN_UNIFORM_010]
    differing = {key for key in causal if causal[key] != candidate[key]}
    assert differing == {"arm_role", "upper_projection_target_aggregation"}
    assert causal["upper_projection_target_aggregation"] == "decision_time"
    assert (
        candidate["upper_projection_target_aggregation"]
        == "decision_policy_mean"
    )


def test_submitter_threads_policy_mean_mode_and_keeps_scheduler_unpinned():
    args = _args()
    causal = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.DECISION_TIME_UNIFORM_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    candidate = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.POLICY_MEAN_UNIFORM_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--upper-projection-target-aggregation decision_time" in causal
    assert (
        "--upper-projection-target-aggregation decision_policy_mean"
        in candidate
    )
    assert "--upper-projection-consistency-coef 0.1" in candidate
    assert "--lower-projection-consistency-coef 0.1" in candidate
    assert "--iterations 512" in candidate
    assert "--checkpoint-minimum-iteration 383" in candidate
    assert spec.FROZEN_ALGORITHM_REVISION in candidate

    tokens = shlex.split(candidate)
    separator = tokens.index("--")
    parsed = control_validation.build_parser().parse_args([
        *tokens[separator + 1 :],
        "--output-dir",
        "/tmp/v24_cli_contract",
    ])
    assert parsed.upper_projection_target_aggregation == "decision_policy_mean"
    assert parsed.terminal_reserve_context
    assert parsed.terminal_reserve_projection

    scheduler_spec = submit.build_scheduler_spec(
        args,
        "HalfCheetah-v5",
        spec.POLICY_MEAN_UNIFORM_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["ram_mb"] == 1536
    assert scheduler_spec["allow_duplicate"] is False
    assert ".server_artifacts" in scheduler_spec["stage_excludes"]


def test_explicit_v24_protocol_accepts_all_frozen_arm_shapes():
    for index, (arm_name, arm) in enumerate(spec.ARMS.items()):
        payload, rows, model = control_validation.train_mujoco_method(
            method="freq_hrl",
            env_id="HalfCheetah-v5",
            disturbance_mode="standard",
            train_seeds=[2401],
            selection_seeds=[2403],
            eval_seeds=[2405],
            steps=8,
            episode_horizon=8,
            iterations=1,
            optimizer_seed=2411 + index,
            upper_period=4,
            hidden_dim=8,
            ppo_clip_ratio=0.1,
            upper_projection_consistency_coef=(
                arm["upper_projection_consistency_coef"]
            ),
            lower_projection_consistency_coef=(
                arm["lower_projection_consistency_coef"]
            ),
            upper_projection_target_aggregation=(
                arm["upper_projection_target_aggregation"]
            ),
            projection_consistency_training_schedule=(
                arm["projection_consistency_training_schedule"]
            ),
            projection_consistency_warmup_fraction=(
                arm["projection_consistency_warmup_fraction"]
            ),
            projection_consistency_ramp_fraction=(
                arm["projection_consistency_ramp_fraction"]
            ),
            terminal_reserve_context=True,
            terminal_reserve_projection=True,
            lower_lf_rms_budget=spec.LOWER_LF_RMS_BUDGET,
            upper_hf_rms_budget=spec.UPPER_HF_RMS_BUDGET,
            checkpoint_minimum_iteration=0,
            checkpoint_smoothing_window=1,
            checkpoint_min_delta=0.0,
            checkpoint_evaluation_interval=1,
            training_disturbance_modes=["standard"],
            evaluation_disturbance_modes=["standard"],
            control_protocol_version=spec.FROZEN_CORE_PROTOCOL_VERSION,
        )
        assert payload["protocol_version"] == spec.FROZEN_CORE_PROTOCOL_VERSION
        assert payload["upper_projection_target_aggregation"] == (
            arm["upper_projection_target_aggregation"]
        )
        target = payload["upper_policy_mean_projection_target_training"]
        if arm_name == spec.POLICY_MEAN_UNIFORM_010:
            assert target["target_count"] > 0.0
            assert target["sampled_target_delta_rms_mean"] > 0.0
        else:
            assert target["target_count"] == 0.0
        assert model.config.upper_projection_target_aggregation == (
            arm["upper_projection_target_aggregation"]
        )
        assert len(rows) == 1


def test_cells_are_interleaved_and_scheduler_lookup_accepts_warning():
    args = _args()
    cells = submit.selected_cells(args)
    assert len(cells) == 48
    assert cells[:4] == [
        ("HalfCheetah-v5", arm, spec.OPTIMIZER_SEEDS[0])
        for arm in spec.ARMS
    ]
    signature = submit.task_signature(
        args.run_name,
        "HalfCheetah-v5",
        spec.POLICY_MEAN_UNIFORM_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    payload = json.dumps({
        "results": [{
            "source": "archive",
            "id": "t1",
            "status": "done",
            "signature": signature,
            "node": "node001",
        }]
    })
    completed = mock.Mock(stdout="Warning: compacted archive\n" + payload)
    with mock.patch.object(submit.subprocess, "run", return_value=completed):
        tasks = submit._scheduler_tasks(args.run_name)
    assert tasks[signature]["id"] == "t1"


def _weight_training(arm: str):
    result = {}
    for level in ("upper", "lower"):
        active = float(
            spec.ARMS[arm][f"{level}_projection_consistency_coef"]
        ) > 0.0
        mse = 0.0
        if active:
            if level == "upper" and arm == spec.DECISION_TIME_UNIFORM_010:
                mse = 0.50
            elif level == "upper" and arm == spec.POLICY_MEAN_UNIFORM_010:
                mse = 0.40
            else:
                mse = 0.45
        result[level] = {
            "active_iteration_count": 256.0 if active else 0.0,
            "unweighted_mse_mean": mse,
            "weighted_mse_mean": mse,
            "weight_mean": 1.0 if active else 0.0,
            "weight_max": 1.0 if active else 0.0,
        }
    return result


def _synthetic_load(passes: bool):
    def load(run_name, environment, arm, optimizer_seed):
        del run_name, environment, optimizer_seed
        summary = {
            "selected_checkpoint_iteration": 400,
            "capacity_actual_parameter_count": 12345,
            "projection_consistency_weight_training": _weight_training(arm),
            "upper_policy_mean_projection_target_training": {
                "active_iteration_count": (
                    float(spec.ITERATIONS)
                    if arm == spec.POLICY_MEAN_UNIFORM_010 else 0.0
                ),
                "target_count": (
                    100.0 if arm == spec.POLICY_MEAN_UNIFORM_010 else 0.0
                ),
                "sampled_target_delta_rms_mean": (
                    0.20 if arm == spec.POLICY_MEAN_UNIFORM_010 else 0.0
                ),
            },
        }
        if arm == spec.TERMINAL_RESERVE_ZERO:
            reward, component, total = 100.0, 1.00, 0.20
        elif arm == spec.MACRO_MEAN_UNIFORM_010:
            reward, component, total = 102.0, 0.85, 0.17
        elif arm == spec.DECISION_TIME_UNIFORM_010:
            reward, component, total = 100.0, 0.90, 0.18
        elif passes:
            reward, component, total = 104.0, 0.75, 0.15
        else:
            reward, component, total = 80.0, 1.20, 0.30
        policy_mean = arm == spec.POLICY_MEAN_UNIFORM_010
        rows = [{
            "disturbance_mode": "standard",
            "seed": spec.EVALUATION_SEEDS[0],
            "episode_return": reward,
            "terminal_reserve_certificate_violation_count": 0.0,
            "terminal_reserve_component_correction_rms_mean": component,
            "terminal_reserve_correction_rms_mean": total,
            "terminal_reserve_total_action_change_rate": 0.4,
            "terminal_reserve_projection_converged_rate": 0.25,
            "terminal_reserve_recursive_fallback_rate": 0.0,
            "terminal_reserve_upper_prefix_power_max": (
                spec.UPPER_HF_RMS_BUDGET**2
            ),
            "terminal_reserve_lower_prefix_power_max": (
                spec.LOWER_LF_RMS_BUDGET**2
            ),
            "terminal_reserve_upper_policy_mean_target_count": (
                2.0 if policy_mean else 0.0
            ),
            "terminal_reserve_upper_policy_mean_target_delta_rms_mean": 0.0,
        }]
        return summary, rows

    return load


def test_analysis_advances_only_the_policy_mean_candidate_when_all_gates_pass():
    with mock.patch.object(analysis, "_validate_cell", return_value=None):
        with mock.patch.object(
            analysis, "_load_cell", side_effect=_synthetic_load(True)
        ):
            result = analysis.analyze("synthetic")
    assert result["status"] == spec.ADVANCES_STATUS
    assert result["selected_candidate"] == spec.POLICY_MEAN_UNIFORM_010
    assert result["candidate_results"][
        spec.POLICY_MEAN_UNIFORM_010
    ]["supported"]
    assert all(
        item["supported"] for item in result["validity"].values()
    )
    assert min(
        item["minimum_projection_converged_rate"]
        for item in result["validity"].values()
    ) == 0.25


def test_analysis_stops_when_policy_mean_candidate_fails():
    with mock.patch.object(analysis, "_validate_cell", return_value=None):
        with mock.patch.object(
            analysis, "_load_cell", side_effect=_synthetic_load(False)
        ):
            result = analysis.analyze("synthetic")
    assert result["status"] == spec.STOPS_STATUS
    assert result["selected_candidate"] is None
