import argparse
import unittest

from scripts.submit_hyperparameter_pilot_scheduleurm import (
    build_scheduler_command,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
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

    def test_scheduler_uses_dynamic_cpu_nodes_without_hard_pin(self):
        command = build_scheduler_command(
            args_fixture(),
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


if __name__ == "__main__":
    unittest.main()
