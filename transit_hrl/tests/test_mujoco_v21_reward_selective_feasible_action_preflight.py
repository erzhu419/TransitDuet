import argparse
import ast
import json
from pathlib import Path
import shlex
from unittest import mock

from freq_hrl.experiments.mujoco import control_validation
from scripts import (
    analyze_mujoco_v21_reward_selective_feasible_action_preflight as analysis,
)
from scripts import (
    mujoco_v21_reward_selective_feasible_action_preflight_spec as spec,
)
from scripts import (
    submit_mujoco_v21_reward_selective_feasible_action_preflight_scheduleurm
    as submit,
)


def _args(**overrides):
    values = {
        "run_name": "v21_test",
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


def test_frozen_seed_roles_are_disjoint_from_all_earlier_mujoco_literals():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    historical_integers = set()
    for path in scripts.glob("*mujoco*.py"):
        if "v21_reward_selective_feasible_action" in path.name:
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


def test_submitter_builds_frozen_reward_selective_command():
    args = _args()
    raw = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.RAW_CONTEXT,
        spec.OPTIMIZER_SEEDS[0],
    )
    uniform = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.DELAYED_UNIFORM_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    selective = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.DELAYED_REWARD_SELECTIVE_010,
        spec.OPTIMIZER_SEEDS[0],
    )

    assert "--terminal-reserve-context" in raw
    assert "--terminal-reserve-projection" not in raw
    assert "--terminal-reserve-projection" in selective
    assert "--iterations 512" in selective
    assert "--checkpoint-minimum-iteration 383" in selective
    assert "--upper-projection-consistency-coef 0.1" in selective
    assert "--projection-consistency-weighting uniform" in uniform
    assert (
        "--projection-consistency-weighting exp_reward_advantage"
        in selective
    )
    assert (
        "--projection-consistency-advantage-temperature 1.0"
        in selective
    )
    assert (
        "--projection-consistency-advantage-weight-clip 5.0"
        in selective
    )
    assert (
        "--projection-consistency-training-schedule delayed_linear"
        in selective
    )
    assert spec.FROZEN_ALGORITHM_REVISION in selective
    assert "--source-manifest-sha256" not in selective

    tokens = shlex.split(selective)
    separator = tokens.index("--")
    parsed = control_validation.build_parser().parse_args([
        *tokens[separator + 1:],
        "--output-dir",
        "/tmp/v21_cli_contract",
    ])
    assert parsed.terminal_reserve_context
    assert parsed.terminal_reserve_projection
    assert parsed.iterations == spec.ITERATIONS
    assert parsed.projection_consistency_weighting == (
        "exp_reward_advantage"
    )

    scheduler_spec = submit.build_scheduler_spec(
        args,
        "HalfCheetah-v5",
        spec.DELAYED_REWARD_SELECTIVE_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["ram_mb"] == 1536
    assert scheduler_spec["allow_duplicate"] is False
    assert ".server_artifacts" in scheduler_spec["stage_excludes"]
    assert submit.SMALL_RESULT_FILES == (
        "cell_summary.json",
        "evaluation_rows.csv",
        "server_artifact_location.json",
    )


def test_scheduler_lookup_accepts_warning_before_archive_json():
    signature = submit.task_signature(
        "v21_test",
        "HalfCheetah-v5",
        spec.RAW_CONTEXT,
        spec.OPTIMIZER_SEEDS[0],
    )
    payload = json.dumps({
        "results": [{
            "source": "archive",
            "id": "t1",
            "status": "done",
            "signature": signature,
            "node": "node001",
        }],
    })
    completed = mock.Mock(stdout="Warning: compacted archive\n" + payload)
    with mock.patch.object(
        submit.subprocess, "run", return_value=completed
    ) as run:
        tasks = submit._scheduler_tasks("v21_test")

    assert tasks[signature]["id"] == "t1"
    command = run.call_args.args[0]
    assert "results" in command
    assert "--include-empty" in command
    assert "--no-log-scan" in command


def _weight_level(kind: str, *, valid: bool):
    if kind == "selective":
        return {
            "active_iteration_count": 128.0,
            "unweighted_mse_mean": 0.5,
            "weighted_mse_mean": 0.4,
            "weight_mean": 1.0,
            "weight_max": 3.0 if valid else 1.0,
        }
    if kind == "uniform":
        return {
            "active_iteration_count": 128.0,
            "unweighted_mse_mean": 0.5,
            "weighted_mse_mean": 0.5,
            "weight_mean": 1.0,
            "weight_max": 1.0,
        }
    return {
        "active_iteration_count": 0.0,
        "unweighted_mse_mean": 0.0,
        "weighted_mse_mean": 0.0,
        "weight_mean": 0.0,
        "weight_max": 0.0,
    }


def _synthetic_cell(
    arm: str,
    *,
    reward_supported: bool,
    weight_valid: bool,
):
    if arm == spec.RAW_CONTEXT:
        kind = "inactive"
        return {
            "reward": 100.0,
            "parameter_count": 12345,
            "selected_checkpoint_iteration": 400,
            "weight_training": {
                level: _weight_level(kind, valid=True)
                for level in ("upper", "lower")
            },
            "raw_prefix_budget_violation_count": 10.0,
            "raw_upper_prefix_power_max": 0.10,
            "raw_lower_prefix_power_max": 0.10,
        }
    if arm == spec.TERMINAL_RESERVE_ZERO:
        kind = "inactive"
        reward, component, total = 90.0, 1.0, 0.20
    elif arm == spec.DELAYED_UNIFORM_010:
        kind = "uniform"
        reward, component, total = 95.0, 0.80, 0.16
    else:
        kind = "selective"
        reward = 98.0 if reward_supported else 80.0
        component, total = 0.78, 0.155
    return {
        "reward": reward,
        "parameter_count": 12345,
        "selected_checkpoint_iteration": 400,
        "weight_training": {
            level: _weight_level(kind, valid=weight_valid)
            for level in ("upper", "lower")
        },
        "certificate_violation_count": 0.0,
        "component_correction_rms": component,
        "total_correction_rms": total,
        "total_action_change_rate": 0.40,
        "fixed_total_rate": 0.60,
        "projection_converged_rate": 1.0,
        "recursive_fallback_rate": 0.0,
        "upper_prefix_power_max": spec.UPPER_HF_RMS_BUDGET ** 2,
        "lower_prefix_power_max": spec.LOWER_LF_RMS_BUDGET ** 2,
    }


def _run_synthetic_analysis(
    *,
    reward_supported: bool,
    weight_valid: bool = True,
):
    optimizer_seeds = (101, 103, 107, 109)

    def load_cell(run_name, environment, arm, optimizer_seed):
        del run_name, environment, optimizer_seed
        return {
            "cell": _synthetic_cell(
                arm,
                reward_supported=reward_supported,
                weight_valid=weight_valid,
            )
        }, [{}]

    with (
        mock.patch.object(spec, "OPTIMIZER_SEEDS", optimizer_seeds),
        mock.patch.object(analysis, "_load_cell", side_effect=load_cell),
        mock.patch.object(analysis, "_validate_cell"),
        mock.patch.object(
            analysis,
            "_summarize_cell",
            side_effect=lambda summary, rows, projected: summary["cell"],
        ),
        mock.patch.object(
            analysis, "_path_registry", return_value={("standard", 1)}
        ),
    ):
        return analysis.analyze("synthetic")


def test_analysis_advances_only_when_reward_and_correction_gates_pass():
    result = _run_synthetic_analysis(reward_supported=True)
    assert result["support_gate"]
    assert result["selected_candidate"] == (
        spec.DELAYED_REWARD_SELECTIVE_010
    )
    candidate = result["candidate_results"][0]
    assert candidate["eligible"]
    assert candidate["reward_improved_environment_count"] == 3
    assert candidate["component_improved_environment_count"] == 3
    assert candidate["total_improved_environment_count"] == 3
    assert candidate["weight_audit"]["supported"]


def test_analysis_stops_when_reward_selective_policy_loses_reward():
    result = _run_synthetic_analysis(reward_supported=False)
    assert not result["support_gate"]
    assert result["selected_candidate"] is None
    gates = result["candidate_results"][0]["gates"]
    assert not gates["reward_improves_in_two_environments"]
    assert not gates["reward_floor_in_every_environment"]


def test_analysis_stops_when_weighting_did_not_become_selective():
    result = _run_synthetic_analysis(
        reward_supported=True,
        weight_valid=False,
    )
    assert not result["support_gate"]
    assert result["selected_candidate"] is None
    candidate = result["candidate_results"][0]
    assert not candidate["weight_audit"]["supported"]
    assert not candidate["gates"]["weight_audit_supported"]


def test_spec_freezes_small_preflight_before_scaleup():
    assert spec.EXPECTED_CELL_COUNT == 48
    assert len(spec.OPTIMIZER_SEEDS) == 4
    assert spec.ITERATIONS == 512
    assert spec.CHECKPOINT_MINIMUM_ITERATION == 383
    assert spec.CANDIDATES == (spec.DELAYED_REWARD_SELECTIVE_010,)
