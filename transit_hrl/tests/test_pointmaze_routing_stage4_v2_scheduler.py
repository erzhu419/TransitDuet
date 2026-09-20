import unittest

from freq_hrl.experiments.pointmaze_routing_masked_attribution import (
    POINTMAZE_MASKED_ROUTING_REPRESENTATIONS,
)
from scripts import pointmaze_routing_stage4_spec as v1
from scripts import pointmaze_routing_stage4_v2_spec as spec
from scripts.submit_hyperparameter_pilot_scheduleurm import LINUX_CPU_NODES
from scripts.submit_pointmaze_routing_stage4_v2_scheduleurm import (
    task_specification,
    training_command,
)


class PointMazeMaskedRoutingSchedulerTest(unittest.TestCase):
    def test_matrix_and_all_seed_roles_are_fresh(self):
        self.assertEqual(len(spec.cells(preflight=False)), 120)
        self.assertEqual(len(spec.cells(preflight=True)), 15)
        v1_seeds = {
            seed
            for root in v1.OPTIMIZER_SEEDS
            for values in v1.seed_roles(root).values()
            for seed in values
        }
        observed = set()
        for optimizer_seed in spec.OPTIMIZER_SEEDS:
            roles = spec.seed_roles(optimizer_seed)
            values = [seed for seeds in roles.values() for seed in seeds]
            self.assertEqual(len(values), 36)
            self.assertEqual(len(set(values)), 36)
            self.assertFalse(observed.intersection(values))
            self.assertFalse(v1_seeds.intersection(values))
            observed.update(values)

    def test_tasks_are_dynamic_and_use_the_frozen_v2_runner(self):
        cell = spec.cells(preflight=True)[0]
        task = task_specification("unit_v2_preflight", cell, preflight=True)
        self.assertEqual(task["allowed_nodes"], list(LINUX_CPU_NODES))
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 2560)
        command = training_command("unit_v2_preflight", cell, preflight=True)
        self.assertIn("run_pointmaze_routing_stage4_v2.py", command)
        self.assertIn("--iterations 2", command)
        self.assertIn("--horizon 64", command)

    def test_protocol_freezes_equal_shape_masks(self):
        self.assertEqual(spec.PROTOCOL, "pointmaze_frequency_routing_stage4_v2")
        self.assertEqual(len(spec.ALGORITHM_REVISION), 40)
        int(spec.ALGORITHM_REVISION, 16)
        self.assertEqual(
            POINTMAZE_MASKED_ROUTING_REPRESENTATIONS[
                "hrl_multiscale_routed"
            ],
            "multiscale_routed_masked",
        )
        self.assertEqual(
            POINTMAZE_MASKED_ROUTING_REPRESENTATIONS[
                "hrl_multiscale_swapped"
            ],
            "multiscale_swapped_masked",
        )


if __name__ == "__main__":
    unittest.main()
