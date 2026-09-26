import unittest

from scripts import pointmaze_learned_termination_stage10_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_learned_termination_stage10_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeLearnedTerminationStageTenTest(unittest.TestCase):
    def test_reuses_stage9_paths_without_new_eval_selection(self):
        self.assertEqual(len(spec.cells(preflight=False)), 8)
        for _, root in spec.cells(preflight=False):
            options = spec.cell_options(root, preflight=False)
            self.assertEqual(len(options["branch_fit"]), 8)
            self.assertEqual(len(options["selection"]), 8)
            self.assertEqual(len(options["trigger_eval"]), 16)
            self.assertEqual(options["termination_iterations"], 18)
            # 8 warm-up, 18*8 training, and 5*8 selection episodes.
            self.assertEqual((8 + 18 * 8 + 5 * 8) * options["horizon"], 230400)

    def test_scheduler_keeps_single_core_dynamic_and_no_checkpoint(self):
        root = spec.PREFLIGHT_OPTIMIZER_SEEDS[0]
        command = training_command("unit_stage10", root, preflight=True)
        self.assertIn("--termination-iterations 2", command)
        self.assertIn("--max-offset-steps 25", command)
        self.assertIn("--trigger-eval-seeds", command)
        diagnostic = training_command(
            "unit_stage10", root, preflight=True, stochastic_repetitions=4
        )
        self.assertIn("--termination-stochastic-repetitions 4", diagnostic)
        task = task_specification("unit_stage10", root, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertTrue(task["allow_no_ckpt"])


if __name__ == "__main__":
    unittest.main()
