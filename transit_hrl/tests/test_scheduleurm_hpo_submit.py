import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.submit_hyperparameter_pilot_scheduleurm import (
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    build_parser,
    build_scheduler_command,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    cells_without_local_summary,
    execute_bulk,
    experiment_cells,
    normalize_args,
    source_identity,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit_hpo",
        train_seeds=[42],
        checkpoint_validation_seeds=[57721],
        tuning_validation_seeds=[68207],
        steps=24,
        assets=2,
        iterations=1,
        nodes=["jtl110cpu", "jtl110cpu2"],
        python_executable="python3",
        launch_subdir=".",
        project="Freq-HRL",
        ram_mb=2048,
        cpu=1,
        priority="normal",
        skip_launch_staging=False,
        allow_duplicate=False,
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
    )


class ScheduleurmHpoSubmitTest(unittest.TestCase):
    def test_grid_skips_candidates_from_the_wrong_family(self):
        cells = experiment_cells(
            ["freq_hrl", "flat_sac"],
            ["ppo_lr1e4_std15", "off_lr1e4_w2048_b64"],
            ["persistent_shift"],
            [7],
        )
        self.assertEqual(cells, [
            ("freq_hrl", "ppo_lr1e4_std15", "persistent_shift", 7),
            ("flat_sac", "off_lr1e4_w2048_b64", "persistent_shift", 7),
        ])

    def test_training_command_contains_only_nested_validation_roles(self):
        args = args_fixture()
        command = build_training_command(
            args,
            policy_mode="freq_hrl",
            candidate_id="ppo_lr1e4_std15",
            scenario="persistent_shift",
            replicate_seed=7,
            output_dir=cell_relative_dir(
                "unit_hpo", "freq_hrl", "ppo_lr1e4_std15", "persistent_shift", 7
            ),
        )
        self.assertIn("--checkpoint-validation-seeds 57721", command)
        self.assertIn("--tuning-validation-seeds 68207", command)
        self.assertNotIn("--eval-seeds", command)
        self.assertNotIn("31415", command)
        self.assertIn("--code-revision " + "a" * 40, command)
        self.assertIn("--source-manifest-sha256 " + "b" * 64, command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_scheduler_uses_dynamic_cpu_nodes_without_hard_pin(self):
        args = args_fixture()
        command = build_scheduler_command(
            args,
            policy_mode="freq_hrl",
            candidate_id="ppo_lr1e4_std15",
            scenario="persistent_shift",
            replicate_seed=7,
        )
        self.assertNotIn("--require-node", command)
        allowed = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--allowed-node"
        ]
        self.assertEqual(allowed, ["jtl110cpu", "jtl110cpu2"])
        self.assertEqual(command[command.index("--cpu") + 1], "1")
        spec = build_scheduler_spec(
            args,
            policy_mode="freq_hrl",
            candidate_id="ppo_lr1e4_std15",
            scenario="persistent_shift",
            replicate_seed=7,
        )
        self.assertEqual(spec["allowed_nodes"], ["jtl110cpu", "jtl110cpu2"])
        self.assertIsNone(spec.get("require_node"))
        self.assertEqual(spec["cpu"], 1)
        self.assertTrue(spec["allow_cpu_training"])

    def test_linux_pool_uses_shared_interpreter_and_dynamic_six_node_placement(self):
        args = normalize_args(build_parser().parse_args(["--run-name", "unit_linux"]))
        self.assertEqual(args.nodes, list(LINUX_CPU_NODES))
        self.assertEqual(args.python_executable, DEFAULT_LINUX_PYTHON)
        args.launch_subdir = "scripts"
        args.project = "Freq-HRL-Linux"
        args.code_revision = "a" * 40
        args.source_manifest_sha256 = "b" * 64
        spec = build_scheduler_spec(
            args,
            policy_mode="freq_hrl",
            candidate_id="ppo_lr1e4_std15",
            scenario="persistent_shift",
            replicate_seed=7,
        )
        self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertTrue(str(spec["cwd"]).endswith("/scripts"))
        self.assertTrue(str(spec["cmd"]).startswith("cd .. && "))
        self.assertIn(DEFAULT_LINUX_PYTHON + " -u -m", str(spec["cmd"]))
        self.assertNotIn("require_node", spec)

    def test_source_revision_override_requires_the_same_source_manifest(self):
        frozen_revision = "8" * 40
        with patch(
            "scripts.submit_hyperparameter_pilot_scheduleurm."
            "registered_git_source_identity",
            return_value=("9" * 40, "a" * 64),
        ), patch(
            "scripts.submit_hyperparameter_pilot_scheduleurm."
            "git_source_manifest_sha256",
            return_value="a" * 64,
        ) as committed_manifest:
            self.assertEqual(source_identity(frozen_revision), (frozen_revision, "a" * 64))
        committed_manifest.assert_called_once()

    def test_missing_cell_filter_skips_only_materialized_summaries(self):
        cells = [
            ("freq_hrl", "ppo_lr1e4_std15", "persistent_shift", 7),
            ("freq_hrl", "ppo_lr1e4_std15", "persistent_shift", 8),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / cell_relative_dir("unit_hpo", *cells[0])
            completed.mkdir(parents=True)
            (completed / "cell_summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                cells_without_local_summary(
                    cells,
                    run_name="unit_hpo",
                    root=root,
                ),
                [cells[1]],
            )

    def test_bulk_submission_uses_one_atomic_scheduler_call(self):
        specs = [{"signature": "unit/1"}, {"signature": "unit/2"}]
        completed = SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "count": 2,
                "submitted": [{"id": "t1"}, {"id": "t2"}],
            }),
            stderr="",
        )
        with patch(
            "scripts.submit_hyperparameter_pilot_scheduleurm.subprocess.run",
            return_value=completed,
        ) as run:
            execute_bulk(specs, dry_run=False, intent_label="unit")
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertIn("submit-jsonl", command)
        self.assertIn("--trusted", command)
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), specs)


if __name__ == "__main__":
    unittest.main()
