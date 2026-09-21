import unittest

from scripts import pointmaze_exogenous_stage5_spec as v1
from scripts import pointmaze_exogenous_stage5_stability_spec as stability
from scripts import pointmaze_exogenous_stage5_v2_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_exogenous_stage5_scheduleurm import (
    _experiment_protocol,
    _preflight_optimizer_seeds,
    task_specification,
    training_command,
)


class PointMazeExogenousStageFiveV2SchedulerTest(unittest.TestCase):
    def test_fresh_formal_and_preflight_seed_roles(self):
        self.assertEqual(len(spec.cells(preflight=False)), 16)
        self.assertEqual(len(spec.cells(preflight=True)), 2)
        self.assertEqual(_preflight_optimizer_seeds(spec), (174001,))
        old = set()
        for source, roots in (
            (v1, v1.OPTIMIZER_SEEDS),
            (stability, stability.OPTIMIZER_SEEDS),
        ):
            for root in roots:
                for values in source.seed_roles(root).values():
                    old.update(values)
        observed = set()
        for root in (*spec.PREFLIGHT_OPTIMIZER_SEEDS, *spec.OPTIMIZER_SEEDS):
            roles = spec.seed_roles(root)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 40)
            self.assertEqual(len(values), len(set(values)))
            self.assertFalse(old.intersection(values))
            self.assertFalse(observed.intersection(values))
            observed.update(values)
        self.assertFalse(
            set(spec.PREFLIGHT_OPTIMIZER_SEEDS).intersection(spec.OPTIMIZER_SEEDS)
        )

    def test_v2_equalizes_rollout_budget_and_records_protocols(self):
        for method in spec.METHODS:
            cell = (method, spec.OPTIMIZER_SEEDS[0])
            command = training_command(
                "unit_stage5_v2",
                cell,
                preflight=False,
                protocol_spec=spec,
            )
            self.assertIn("--checkpoint-rank-mode success_then_return", command)
            self.assertEqual(command.count(" 1700011"), 1)
            options = spec.cell_options(cell[1], preflight=False)
            self.assertEqual(len(options["train"]), 8)
            task = task_specification(
                "unit_stage5_v2",
                cell,
                preflight=False,
                protocol_spec=spec,
            )
            self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)
            self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
            self.assertIsNone(task["require_node"])
        self.assertEqual(_experiment_protocol(spec), spec.EXPERIMENT_PROTOCOL)
        self.assertEqual(_experiment_protocol(v1), v1.PROTOCOL)


if __name__ == "__main__":
    unittest.main()

