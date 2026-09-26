import unittest

from scripts import pointmaze_termination_mc_credit_v3_spec as spec
from scripts.submit_pointmaze_termination_mc_credit_v3_scheduleurm import (
    task_specification,
)


class TerminationMCCreditTest(unittest.TestCase):
    def test_frozen_credit_screen_keeps_budget_and_dynamic_placement(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        for root in spec.roots(preflight=False):
            task = task_specification("unit_mc_credit", root, preflight=False)
            self.assertIn("--termination-gae-lambda 1.0", task["cmd"])
            self.assertIn("--termination-stochastic-repetitions 4", task["cmd"])
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
