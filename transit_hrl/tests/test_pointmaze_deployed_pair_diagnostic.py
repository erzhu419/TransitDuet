import unittest

from freq_hrl.experiments.pointmaze_deployed_pair_diagnostic import select_bins
from scripts import pointmaze_deployed_pair_diagnostic_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import (
    task_specification,
)


class DeployedPairDiagnosticTest(unittest.TestCase):
    def test_selection_is_fixed_and_excludes_initial_and_final_bins(self):
        decisions = [0, *[50 * i + (10 if i % 2 else 25)
                          for i in range(1, 24)]]
        selected = select_bins(
            root=209011, seed=2180301, decision_steps=decisions,
            period=50, deadline=25, pairs_per_class=2,
        )
        self.assertEqual(selected, select_bins(
            root=209011, seed=2180301, decision_steps=decisions,
            period=50, deadline=25, pairs_per_class=2,
        ))
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(1 <= index < 23 for index in selected))
        self.assertEqual(sum(decisions[i] - 50 * i < 25 for i in selected), 2)

    def test_scheduler_stages_only_compact_source_result(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_deployed_pair", 209011, preflight=False)
        self.assertEqual(task["cpu"], 1)
        self.assertIsNone(task["require_node"])
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertIn("--pairs-per-class 2", task["cmd"])
        self.assertIn(str(spec.source_result(209011, preflight=False).parent),
                      task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
