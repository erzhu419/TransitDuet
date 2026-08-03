import argparse
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.submit_strong_learned_baselines_scheduleurm import (
    DEFAULT_LINUX_PYTHON,
    LINUX_CPU_NODES,
    build_parser,
    build_scheduler_command,
    build_scheduler_spec,
    build_training_command,
    cell_relative_dir,
    cells_without_local_results,
    execute_bulk,
    experiment_cells,
    normalize_args,
    resolved_hyperparameters,
)


def args_fixture() -> argparse.Namespace:
    return argparse.Namespace(
        run_name="unit",
        train_seeds=[42],
        validation_seeds=[57721],
        eval_seeds=[31415],
        steps=24,
        assets=2,
        iterations=1,
        min_pairs=1,
        ppo_hidden_dim=32,
        ppo_learning_rate=3e-4,
        ppo_epochs=2,
        ppo_minibatch_size=64,
        ppo_init_log_std=-1.0,
        training_reward_scale=100.0,
        offpolicy_hidden_dim=32,
        offpolicy_learning_rate=3e-4,
        offpolicy_replay_capacity=1000,
        offpolicy_warmup_steps=8,
        offpolicy_batch_size=8,
        offpolicy_updates_per_step=1,
        nodes=["jtl110cpu", "jtl110cpu2"],
        python_executable="python3",
        launch_subdir=".",
        project="Freq-HRL-Confirmatory",
        ram_mb=2048,
        cpu=1,
        priority="normal",
        skip_launch_staging=False,
        stage_input_paths=[],
        allow_duplicate=False,
        code_revision="a" * 40,
        source_manifest_sha256="b" * 64,
    )


class ScheduleurmStrongBaselineSubmitTest(unittest.TestCase):
    def test_cell_grid_expands_training_replicates(self):
        cells = experiment_cells(
            ["persistent_shift"], ["freq_hrl", "flat_ppo"], [7, 11]
        )
        self.assertEqual(len(cells), 4)
        self.assertEqual(cells[-1], ("persistent_shift", "flat_ppo", 11))

    def test_training_command_freezes_seed_roles_and_budget(self):
        args = args_fixture()
        command = build_training_command(
            args,
            scenario="persistent_shift",
            mode="freq_hrl",
            replicate_seed=7,
            output_dir=cell_relative_dir(
                "unit", "persistent_shift", "freq_hrl", 7
            ),
        )
        self.assertIn("--train-seeds 42 --validation-seeds 57721", command)
        self.assertIn("--eval-seeds 31415", command)
        self.assertIn("--optimizer-seeds 7", command)
        self.assertIn("OMP_NUM_THREADS=1", command)
        self.assertIn("--code-revision " + "a" * 40, command)
        self.assertIn("--source-manifest-sha256 " + "b" * 64, command)
        self.assertIn("CUDA_VISIBLE_DEVICES=", command)
        self.assertTrue(command.endswith("&& echo DONE"))

    def test_gru_controls_are_resolved_as_ppo_family(self):
        parameters, candidate_id = resolved_hyperparameters(
            args_fixture(), "generic_hrl_gru_ppo"
        )
        self.assertEqual(candidate_id, "exploratory_manual")
        self.assertIn("epochs", parameters)
        self.assertIn("init_log_std", parameters)
        self.assertNotIn("replay_capacity", parameters)

    def test_confirmatory_command_uses_policy_specific_frozen_parameters(self):
        args = args_fixture()
        args.confirmatory = True
        args.frozen_config_sha256 = "b" * 64
        args.frozen_selected = {
            "freq_hrl": {
                "candidate_id": "ppo_unit",
                "parameters": {
                    "hidden_dim": 64,
                    "learning_rate": 0.0001,
                    "epochs": 4,
                    "minibatch_size": 512,
                    "init_log_std": -1.5,
                    "reward_scale": 100.0,
                },
            }
        }
        command = build_training_command(
            args,
            scenario="persistent_shift",
            mode="freq_hrl",
            replicate_seed=7,
            output_dir=cell_relative_dir(
                "unit", "persistent_shift", "freq_hrl", 7
            ),
        )
        self.assertIn("--ppo-learning-rate 0.0001", command)
        self.assertIn("--ppo-init-log-std -1.5", command)
        self.assertIn("--selected-candidate-id ppo_unit", command)
        self.assertIn("--frozen-config-sha256 " + "b" * 64, command)
        self.assertIn("--confirmatory", command)

    def test_scheduler_uses_dynamic_allowed_nodes_without_hard_pin(self):
        command = build_scheduler_command(
            args_fixture(),
            scenario="persistent_shift",
            mode="freq_hrl",
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

    def test_linux_pool_uses_shared_interpreter_and_dynamic_six_node_placement(self):
        with patch(
            "scripts.submit_strong_learned_baselines_scheduleurm.source_identity",
            return_value=("a" * 40, "b" * 64),
        ):
            args = normalize_args(build_parser().parse_args([
                "--run-name",
                "unit_linux",
                "--smoke",
                "--skip-launch-staging",
            ]))
        self.assertEqual(args.nodes, list(LINUX_CPU_NODES))
        self.assertEqual(args.python_executable, DEFAULT_LINUX_PYTHON)
        args.code_revision = "a" * 40
        args.source_manifest_sha256 = "b" * 64
        spec = build_scheduler_spec(
            args,
            scenario="persistent_shift",
            mode="freq_hrl",
            replicate_seed=7,
        )
        self.assertEqual(spec["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertTrue(str(spec["cwd"]).endswith("/scripts"))
        self.assertTrue(str(spec["cmd"]).startswith("cd .. && "))
        self.assertIn(DEFAULT_LINUX_PYTHON + " -u -m", str(spec["cmd"]))
        self.assertNotIn("require_node", spec)
        self.assertEqual(
            spec["stage_input_paths"],
            [str((Path(__file__).resolve().parents[1] / "freq_hrl").resolve())],
        )

    def test_missing_cell_filter_skips_only_materialized_results(self):
        cells = [
            ("persistent_shift", "freq_hrl", 7),
            ("persistent_shift", "freq_hrl", 8),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            completed = root / cell_relative_dir("unit", *cells[0])
            completed.mkdir(parents=True)
            (completed / "per_seed.csv").write_text("seed\n1\n", encoding="utf-8")
            self.assertEqual(
                cells_without_local_results(cells, run_name="unit", root=root),
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
            "scripts.submit_strong_learned_baselines_scheduleurm.subprocess.run",
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
