import argparse
import ast
import json
from pathlib import Path
import shlex
from unittest import mock

import numpy as np

from freq_hrl.experiments.mujoco import control_validation
from scripts import analyze_mujoco_v22_uniform_terminal_reserve_confirmation as analysis
from scripts import mujoco_v22_uniform_terminal_reserve_confirmation_spec as spec
from scripts import submit_mujoco_v22_uniform_terminal_reserve_confirmation_scheduleurm as submit


def _args(**overrides):
    values = {
        "run_name": "v22_test",
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


def test_frozen_seed_roles_are_disjoint_from_earlier_mujoco_literals():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    historical_integers = set()
    for path in scripts.glob("*mujoco*.py"):
        if "v22_uniform_terminal_reserve_confirmation" in path.name:
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
    assert len(fresh) == 48
    assert not fresh & historical_integers
    assert spec.EXPECTED_CELL_COUNT == 288


def test_submitter_preserves_the_unchanged_uniform_v21_mechanism():
    args = _args()
    raw = submit.build_training_command(
        args, "HalfCheetah-v5", spec.RAW_CONTEXT, spec.OPTIMIZER_SEEDS[0]
    )
    candidate = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--terminal-reserve-context" in raw
    assert "--terminal-reserve-projection" not in raw
    assert "--terminal-reserve-projection" in candidate
    assert "--iterations 512" in candidate
    assert "--checkpoint-minimum-iteration 383" in candidate
    assert "--upper-projection-consistency-coef 0.1" in candidate
    assert "--lower-projection-consistency-coef 0.1" in candidate
    assert "--projection-consistency-weighting uniform" in candidate
    assert "--projection-consistency-training-schedule delayed_linear" in candidate
    assert spec.FROZEN_ALGORITHM_REVISION in candidate

    tokens = shlex.split(candidate)
    separator = tokens.index("--")
    parsed = control_validation.build_parser().parse_args([
        *tokens[separator + 1 :],
        "--output-dir",
        "/tmp/v22_cli_contract",
    ])
    assert parsed.terminal_reserve_context
    assert parsed.terminal_reserve_projection
    assert parsed.iterations == spec.ITERATIONS
    assert parsed.projection_consistency_weighting == "uniform"

    scheduler_spec = submit.build_scheduler_spec(
        args,
        "HalfCheetah-v5",
        spec.CANDIDATE,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["ram_mb"] == 1536
    assert scheduler_spec["allow_duplicate"] is False
    assert ".server_artifacts" in scheduler_spec["stage_excludes"]


def test_scheduler_lookup_accepts_warning_before_json():
    signature = submit.task_signature(
        "v22_test",
        "HalfCheetah-v5",
        spec.CANDIDATE,
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
        tasks = submit._scheduler_tasks("v22_test")
    assert tasks[signature]["id"] == "t1"


def test_reward_statistic_uses_paired_ratio_of_means_not_mean_of_ratios():
    uniform = np.asarray([280.068, 97.243, 279.455, 131.224])
    reserve = np.asarray([102.495, 322.626, 301.697, 302.578])
    old_mean_of_ratios = np.mean((uniform - reserve) / np.abs(reserve))
    assert old_mean_of_ratios > 0.0
    assert analysis._reward_delta(uniform, reserve) < 0.0


def _weight_training(active: bool):
    row = {
        "active_iteration_count": 128.0 if active else 0.0,
        "unweighted_mse_mean": 0.5 if active else 0.0,
        "weighted_mse_mean": 0.5 if active else 0.0,
        "weight_mean": 1.0 if active else 0.0,
        "weight_max": 1.0 if active else 0.0,
    }
    return {level: dict(row) for level in ("upper", "lower")}


def _synthetic_load(*, supported: bool):
    def load(run_name, environment, arm, optimizer_seed):
        del run_name, environment, optimizer_seed
        projected = arm != spec.RAW_CONTEXT
        active = arm == spec.CANDIDATE
        summary = {
            "selected_checkpoint_iteration": 400,
            "capacity_actual_parameter_count": 12345,
            "projection_consistency_weight_training": _weight_training(active),
        }
        if arm == spec.RAW_CONTEXT:
            reward, component, total = 110.0, 0.0, 0.0
        elif arm == spec.PRIMARY_PROJECTED_BASELINE:
            reward, component, total = 100.0, 1.0, 0.20
        elif supported:
            reward, component, total = 104.0, 0.80, 0.16
        else:
            reward, component, total = 80.0, 1.20, 0.30
        rows = []
        for mode in spec.EVALUATION_DISTURBANCE_MODES:
            for seed in spec.EVALUATION_SEEDS:
                row = {
                    "disturbance_mode": mode,
                    "seed": seed,
                    "episode_return": reward,
                }
                if projected:
                    row.update({
                        "terminal_reserve_certificate_violation_count": 0.0,
                        "terminal_reserve_component_correction_rms_mean": component,
                        "terminal_reserve_correction_rms_mean": total,
                        "terminal_reserve_total_action_change_rate": 0.4,
                        "terminal_reserve_fixed_total_rate": 0.6,
                        "terminal_reserve_projection_converged_rate": 1.0,
                        "terminal_reserve_recursive_fallback_rate": 0.0,
                        "terminal_reserve_upper_prefix_power_max": (
                            spec.UPPER_HF_RMS_BUDGET**2
                        ),
                        "terminal_reserve_lower_prefix_power_max": (
                            spec.LOWER_LF_RMS_BUDGET**2
                        ),
                    })
                else:
                    row.update({
                        "terminal_reserve_raw_prefix_budget_violation_count": 1.0,
                        "terminal_reserve_raw_upper_prefix_power_max": 0.1,
                        "terminal_reserve_raw_lower_prefix_power_max": 0.1,
                    })
                rows.append(row)
        return summary, rows

    return load


def test_synthetic_confirmation_requires_joint_reward_and_correction_gate():
    with mock.patch.object(analysis, "_validate_cell", return_value=None):
        with mock.patch.object(
            analysis, "_load_cell", side_effect=_synthetic_load(supported=True)
        ):
            supported = analysis.analyze("synthetic")
        with mock.patch.object(
            analysis, "_load_cell", side_effect=_synthetic_load(supported=False)
        ):
            rejected = analysis.analyze("synthetic")
    assert supported["status"] == spec.SUPPORTED_STATUS
    assert supported["selected_candidate"] == spec.CANDIDATE
    assert rejected["status"] == spec.NOT_SUPPORTED_STATUS
    assert rejected["selected_candidate"] is None
