import argparse
import unittest

from scripts.submit_strong_learned_baselines_scheduleurm import (
    build_scheduler_command,
    build_training_command,
    cell_relative_dir,
    experiment_cells,
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
        offpolicy_hidden_dim=32,
        offpolicy_replay_capacity=1000,
        offpolicy_warmup_steps=8,
        offpolicy_batch_size=8,
        offpolicy_updates_per_step=1,
        nodes=["jtl110cpu", "jtl110cpu2"],
        ram_mb=2048,
        cpu=1,
        priority="normal",
        skip_launch_staging=False,
        allow_duplicate=False,
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


if __name__ == "__main__":
    unittest.main()
