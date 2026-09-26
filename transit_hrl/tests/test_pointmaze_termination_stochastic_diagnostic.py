import unittest

from scripts import pointmaze_termination_stochastic_diagnostic_spec as spec
from scripts.submit_pointmaze_termination_stochastic_diagnostic_scheduleurm import (
    task_specification,
)


class StochasticTerminationDiagnosticTest(unittest.TestCase):
    def test_frozen_diagnostic_uses_old_paths_and_four_policy_draws(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        self.assertEqual(spec.STOCHASTIC_REPETITIONS, 4)
        for root in spec.roots(preflight=False):
            task = task_specification("unit_stochastic", root, preflight=False)
            self.assertIn(
                "--termination-stochastic-repetitions 4", task["cmd"]
            )
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)


if __name__ == "__main__":
    unittest.main()
