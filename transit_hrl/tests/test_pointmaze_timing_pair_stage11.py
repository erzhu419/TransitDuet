import unittest

from freq_hrl.experiments.pointmaze_timing_pair import (
    timing_pair_opportunities,
    timing_pair_schedule,
)
from scripts import pointmaze_timing_pair_stage11_spec as spec
from scripts.submit_pointmaze_timing_pair_stage11_scheduleurm import (
    task_specification,
)


class TimingPairStage11Test(unittest.TestCase):
    def test_scheduler_uses_registered_roots_and_one_cpu(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        for root in spec.roots(preflight=False):
            task = task_specification("unit_timing_pair", root, preflight=False)
            self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
            self.assertIn("--pairs-per-seed 12", task["cmd"])
            self.assertEqual(task["cpu"], 1)
            self.assertIsNone(task["require_node"])

    def test_paired_schedules_differ_only_at_one_legal_check(self):
        opportunities = timing_pair_opportunities(
            seed=2180201,
            optimizer_seed=209011,
            horizon=1200,
            period_steps=50,
            max_offset_steps=25,
            check_stride_steps=5,
            pairs_per_seed=12,
        )
        self.assertEqual(len(opportunities), 12)
        self.assertEqual(len({bin_index for bin_index, _ in opportunities}), 12)
        self.assertEqual({offset for _, offset in opportunities}, {0, 5, 10, 15, 20})
        for bin_index, offset in opportunities:
            wait, now = timing_pair_schedule(
                horizon=1200,
                period_steps=50,
                max_offset_steps=25,
                bin_index=bin_index,
                offset=offset,
            )
            self.assertEqual(len(wait), len(now), 24)
            self.assertEqual(
                [index for index, (a, b) in enumerate(zip(wait, now)) if a != b],
                [bin_index],
            )
            self.assertEqual(now[bin_index], bin_index * 50 + offset)
            self.assertEqual(wait[bin_index], bin_index * 50 + 25)


if __name__ == "__main__":
    unittest.main()
