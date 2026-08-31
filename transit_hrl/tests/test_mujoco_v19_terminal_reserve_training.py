import argparse
import ast
import json
from pathlib import Path
import shlex
from unittest import mock

from freq_hrl.experiments.mujoco import control_validation
from scripts import analyze_mujoco_v19_terminal_reserve_training as analysis
from scripts import mujoco_v19_terminal_reserve_training_spec as spec
from scripts import submit_mujoco_v19_terminal_reserve_training_scheduleurm as submit


def _args(**overrides):
    values = {
        "run_name": "v19_test",
        "arms": list(spec.ARMS),
        "nodes": ["node001", "node002", "node003", "node004", "node005", "node006"],
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
        if "v19_terminal_reserve" in path.name:
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


def test_submitter_builds_capacity_matched_raw_and_projected_commands():
    args = _args()
    raw = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.RAW_CONTEXT_BASELINE,
        spec.OPTIMIZER_SEEDS[0],
    )
    candidate = submit.build_training_command(
        args,
        "HalfCheetah-v5",
        spec.TERMINAL_RESERVE_CONSISTENCY_001,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert "--terminal-reserve-context" in raw
    assert "--terminal-reserve-projection" not in raw
    assert "--terminal-reserve-context" in candidate
    assert "--terminal-reserve-projection" in candidate
    assert "--ppo-clip-ratio 0.1" in candidate
    assert "--upper-projection-consistency-coef 0.01" in candidate
    assert spec.FROZEN_ALGORITHM_REVISION in candidate
    assert "--source-manifest-sha256" not in candidate
    tokens = shlex.split(candidate)
    separator = tokens.index("--")
    parsed = control_validation.build_parser().parse_args([
        *tokens[separator + 1:],
        "--output-dir",
        "/tmp/v19_cli_contract",
    ])
    assert parsed.terminal_reserve_context
    assert parsed.terminal_reserve_projection
    assert parsed.ppo_clip_ratio == spec.PPO_CLIP_RATIO
    scheduler_spec = submit.build_scheduler_spec(
        args,
        "HalfCheetah-v5",
        spec.TERMINAL_RESERVE_CONSISTENCY_001,
        spec.OPTIMIZER_SEEDS[0],
    )
    assert scheduler_spec["require_node"] is None
    assert scheduler_spec["allowed_nodes"] == args.nodes
    assert scheduler_spec["cpu"] == 1
    assert scheduler_spec["allow_duplicate"] is False


def test_scheduler_lookup_reads_compacted_successes_from_results_archive():
    signature = submit.task_signature(
        "v19_test",
        "HalfCheetah-v5",
        spec.RAW_CONTEXT_BASELINE,
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
        tasks = submit._scheduler_tasks("v19_test")

    assert tasks[signature]["id"] == "t1"
    command = run.call_args.args[0]
    assert "results" in command
    assert "--include-empty" in command
    assert "--no-log-scan" in command


def _synthetic_cell(arm: str, *, supported: bool):
    base = {
        "reward": 95.0,
        "parameter_count": 12345,
        "selected_checkpoint_iteration": 64,
    }
    if arm == spec.RAW_CONTEXT_BASELINE:
        return {
            **base,
            "reward": 100.0,
            "raw_prefix_budget_violation_count": 10.0,
            "raw_upper_prefix_power_max": 0.1,
            "raw_lower_prefix_power_max": 0.1,
        }
    correction = {
        spec.PRIMARY_MECHANISM_BASELINE: 1.0,
        spec.TERMINAL_RESERVE_CONSISTENCY_001: 0.8 if supported else 1.0,
        spec.TERMINAL_RESERVE_CONSISTENCY_003: 0.98,
        spec.TERMINAL_RESERVE_CONSISTENCY_010: 1.1,
    }[arm]
    reward = 70.0 if arm == spec.TERMINAL_RESERVE_CONSISTENCY_010 else 95.0
    return {
        **base,
        "reward": reward,
        "certificate_violation_count": 0.0,
        "component_correction_rms": correction,
        "total_correction_rms": correction,
        "total_action_change_rate": 0.1,
        "fixed_total_rate": 0.9,
        "projection_converged_rate": 1.0,
        "recursive_fallback_rate": 0.0,
        "upper_prefix_power_max": spec.UPPER_HF_RMS_BUDGET ** 2,
        "lower_prefix_power_max": spec.LOWER_LF_RMS_BUDGET ** 2,
    }


def _run_synthetic_analysis(*, supported: bool):
    optimizer_seeds = (101, 103, 107, 109)

    def load_cell(run_name, environment, arm, optimizer_seed):
        del run_name, environment, optimizer_seed
        return {"cell": _synthetic_cell(arm, supported=supported)}, [{}]

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
            analysis,
            "_path_registry",
            return_value={("standard", 1)},
        ),
    ):
        return analysis.analyze("synthetic")


def test_analysis_selects_only_a_candidate_that_passes_every_frozen_gate():
    result = _run_synthetic_analysis(supported=True)
    assert result["support_gate"]
    assert (
        result["selected_candidate"]
        == spec.TERMINAL_RESERVE_CONSISTENCY_001
    )
    selected = next(
        row
        for row in result["candidate_results"]
        if row["candidate"] == result["selected_candidate"]
    )
    assert selected["eligible"]
    assert selected["component_correction_supported_environment_count"] == 3


def test_analysis_stops_without_diagnostic_best_arm_when_no_gate_passes():
    result = _run_synthetic_analysis(supported=False)
    assert not result["support_gate"]
    assert result["selected_candidate"] is None
    assert result["selected_consistency_coef"] is None
    assert result["status"] == spec.NOT_SUPPORTED_STATUS
