import unittest

from freq_hrl.experiments.pointmaze_timing_pair import timing_pair_opportunities
from scripts import pointmaze_timing_pair_stage12_spec as spec
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import (
    task_specification,
)


class TimingPairStage12Test(unittest.TestCase):
    def test_windowed_opportunities_fit_episode_and_budget(self):
        for root in spec.roots(preflight=False):
            options = spec.cell_options(root, preflight=False)
            task = task_specification(
                "unit_windowed_pair", root, preflight=False,
                protocol_spec=spec,
            )
            self.assertIn("--credit-window-steps 50", task["cmd"])
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["project"], spec.EXPERIMENT_PROTOCOL)
            for seed in options["branch_fit"]:
                pairs = timing_pair_opportunities(
                    seed=seed,
                    optimizer_seed=root,
                    horizon=options["horizon"],
                    period_steps=50,
                    max_offset_steps=25,
                    check_stride_steps=5,
                    pairs_per_seed=options["pairs_per_seed"],
                    credit_window_steps=50,
                )
                self.assertEqual(len(pairs), options["pairs_per_seed"])
                self.assertTrue(all(50 * index + offset + 50 <= 1200
                                    for index, offset in pairs))


if __name__ == "__main__":
    unittest.main()
