import unittest

from scripts import pointmaze_exogenous_stage5_spec as stage5
from scripts import pointmaze_exogenous_stage5_stability_spec as spec
from scripts.analyze_pointmaze_exogenous_stage5_stability import analyze_records
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_exogenous_stage5_stability_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeExogenousStabilityScreenTest(unittest.TestCase):
    def test_matrix_and_seed_roles_are_paired_and_fresh(self):
        self.assertEqual(len(spec.cells(preflight=False)), 8)
        self.assertEqual(len(spec.cells(preflight=True)), 4)
        old = {
            seed
            for root in stage5.OPTIMIZER_SEEDS
            for values in stage5.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for root in spec.OPTIMIZER_SEEDS:
            roles = spec.seed_roles(root)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 40)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(old.intersection(values))
            self.assertFalse(observed.intersection(values))
            observed.update(values)

    def test_commands_encode_the_factorial_without_node_binding(self):
        for arm in spec.ARMS:
            cell = (arm, spec.PREFLIGHT_OPTIMIZER_SEED)
            options = spec.cell_options(*cell, preflight=True)
            command = training_command("unit_stability", cell, preflight=True)
            self.assertIn(
                f"--checkpoint-rank-mode {options['checkpoint_rank_mode']}",
                command,
            )
            task = task_specification("unit_stability", cell, preflight=True)
            self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["cpu"], 1)

    def test_registered_selection_rule_can_select_combined_arm(self):
        records = []
        values = {
            "v1_control": ((0.20, 0.30), (100.0, 110.0)),
            "dense_rank": ((0.24, 0.33), (115.0, 120.0)),
            "more_rollouts": ((0.27, 0.36), (98.0, 112.0)),
            "dense_rank_more_rollouts": ((0.45, 0.55), (145.0, 155.0)),
        }
        for arm in spec.ARMS:
            successes, returns = values[arm]
            for index, root in enumerate(spec.OPTIMIZER_SEEDS):
                records.append({
                    "arm": arm,
                    "root": root,
                    "success": successes[index],
                    "return": returns[index],
                    "initial_success": 0.10,
                    "initial_return": 80.0,
                    "selected_checkpoint_iteration": 767,
                    "runtime_versions": {"python": "test"},
                })
        analysis = analyze_records(records)
        self.assertEqual(
            analysis["selected_candidate"], "dense_rank_more_rollouts"
        )
        self.assertEqual(analysis["stage5_v2_freeze_status"], "authorized")


if __name__ == "__main__":
    unittest.main()

