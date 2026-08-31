import argparse
import ast
import json
from pathlib import Path
import shlex
from unittest import mock

from freq_hrl.experiments.mujoco import control_validation
from scripts import analyze_mujoco_v20_reward_guarded_reserve_training as analysis
from scripts import mujoco_v20_reward_guarded_reserve_training_spec as spec
from scripts import (
    submit_mujoco_v20_reward_guarded_reserve_training_scheduleurm as submit,
)


def _args(**overrides):
    values = {
        "run_name": "v20_test",
        "arms": list(spec.ARMS),
        "nodes": [
            "node001", "node002", "node003",
            "node004", "node005", "node006",
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
        if "v20_reward_guarded_reserve" in path.name:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        historical_integers.update(
            int(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        )
    fresh = set(
        spec.OPTIMIZER_SEEDS
        + spec.TRAIN_SEEDS
        + spec.SELECTION_SEEDS
        + spec.EVALUATION_SEEDS
    )
    assert not fresh & historical_integers


def test_submitter_builds_frozen_long_horizon_guarded_command():
    args = _args()
    raw = submit.build_training_command(
        args, "HalfCheetah-v5", spec.RAW_CONTEXT_LONG,
        spec.OPTIMIZER_SEEDS[0],
    )
    guarded = submit.build_training_command(
        args, "HalfCheetah-v5", spec.DELAYED_REWARD_GUARDED_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--terminal-reserve-context" in raw
    assert "--terminal-reserve-projection" not in raw
    assert "--terminal-reserve-context" in guarded
    assert "--terminal-reserve-projection" in guarded
    assert "--iterations 384" in guarded
    assert "--checkpoint-minimum-iteration 287" in guarded
    assert "--upper-projection-consistency-coef 0.1" in guarded
    assert (
        "--projection-consistency-update-mode reward_guarded_projection"
        in guarded
    )
    assert (
        "--projection-consistency-training-schedule delayed_linear" in guarded
    )
    assert "--projection-consistency-warmup-fraction 0.5" in guarded
    assert "--projection-consistency-ramp-fraction 0.25" in guarded
    assert spec.FROZEN_ALGORITHM_REVISION in guarded
    assert "--source-manifest-sha256" not in guarded
    tokens = shlex.split(guarded)
    separator = tokens.index("--")
    parsed = control_validation.build_parser().parse_args([
        *tokens[separator + 1:],
        "--output-dir", "/tmp/v20_cli_contract",
    ])
    assert parsed.terminal_reserve_context
    assert parsed.terminal_reserve_projection
    assert parsed.iterations == spec.ITERATIONS
    assert parsed.projection_consistency_update_mode == (
        "reward_guarded_projection"
    )
    scheduler_spec = submit.build_scheduler_spec(
        args, "HalfCheetah-v5", spec.DELAYED_REWARD_GUARDED_010,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["ram_mb"] == 1536
    assert scheduler_spec["allow_duplicate"] is False


def test_scheduler_lookup_reads_compacted_successes_from_results_archive():
    signature = submit.task_signature(
        "v20_test", "HalfCheetah-v5", spec.RAW_CONTEXT_LONG,
        spec.OPTIMIZER_SEEDS[0],
    )
    completed = mock.Mock(stdout=json.dumps({
        "results": [{
            "source": "archive",
            "id": "t1",
            "status": "done",
            "signature": signature,
            "node": "node001",
        }],
    }))
    with mock.patch.object(submit.subprocess, "run", return_value=completed) as run:
        tasks = submit._scheduler_tasks("v20_test")

    assert tasks[signature]["id"] == "t1"
    command = run.call_args.args[0]
    assert "results" in command
    assert "--include-empty" in command
    assert "--no-log-scan" in command


def _guard_level(*, active: bool, safe: bool = True):
    return {
        "active_iteration_count": 192.0 if active else 0.0,
        "attempted_mass": 192.0 if active else 0.0,
        "accepted_mass": 96.0 if active else 0.0,
        "acceptance_rate": 0.5 if active else 0.0,
        "reward_loss_delta_max": (
            -1e-6 if active and safe else (1e-3 if active else 0.0)
        ),
        "native_constraint_loss_delta_max": 0.0,
        "consistency_loss_delta_mean": -0.01 if active else 0.0,
        "gradient_conflict_rate": 0.25 if active else 0.0,
    }


def _synthetic_cell(arm: str, *, supported: bool, guard_safe: bool):
    guarded = arm == spec.DELAYED_REWARD_GUARDED_010
    base = {
        "reward": 95.0,
        "parameter_count": 12345,
        "selected_checkpoint_iteration": 320,
        "projection_guard_training": {
            "upper": _guard_level(active=guarded, safe=guard_safe),
            "lower": _guard_level(active=guarded, safe=guard_safe),
        },
    }
    if arm == spec.PRIMARY_RAW_BASELINE:
        return {
            **base,
            "reward": 100.0,
            "raw_prefix_budget_violation_count": 10.0,
            "raw_upper_prefix_power_max": 0.1,
            "raw_lower_prefix_power_max": 0.1,
        }
    if arm == spec.PRIMARY_MECHANISM_BASELINE:
        component, total, reward = 1.0, 0.20, 95.0
    elif supported and guarded:
        component, total, reward = 0.70, 0.14, 98.0
    elif supported:
        component, total, reward = 0.80, 0.16, 97.0
    else:
        component, total, reward = 1.0, 0.20, 70.0
    return {
        **base,
        "reward": reward,
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


def _run_synthetic_analysis(*, supported: bool, guard_safe: bool = True):
    optimizer_seeds = (101, 103, 107, 109)

    def load_cell(run_name, environment, arm, optimizer_seed):
        del run_name, environment, optimizer_seed
        return {
            "cell": _synthetic_cell(
                arm, supported=supported, guard_safe=guard_safe
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


def test_analysis_selects_only_a_candidate_that_passes_every_frozen_gate():
    result = _run_synthetic_analysis(supported=True)
    assert result["support_gate"]
    assert result["selected_candidate"] == spec.DELAYED_REWARD_GUARDED_010
    selected = next(
        row for row in result["candidate_results"]
        if row["candidate"] == result["selected_candidate"]
    )
    assert selected["eligible"]
    assert selected["component_correction_supported_environment_count"] == 3
    assert selected["guard_audit"]["supported"]


def test_analysis_rejects_a_guard_that_worsens_training_reward_surrogate():
    result = _run_synthetic_analysis(supported=True, guard_safe=False)
    guarded = next(
        row for row in result["candidate_results"]
        if row["candidate"] == spec.DELAYED_REWARD_GUARDED_010
    )
    assert not guarded["guard_audit"]["supported"]
    assert not guarded["eligible"]
    assert result["selected_candidate"] == spec.DELAYED_SCALARIZED_010


def test_analysis_stops_without_diagnostic_best_arm_when_no_gate_passes():
    result = _run_synthetic_analysis(supported=False)
    assert not result["support_gate"]
    assert result["selected_candidate"] is None
    assert result["selected_consistency_coef"] is None
    assert result["status"] == spec.NOT_SUPPORTED_STATUS
