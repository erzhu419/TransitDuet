"""Check matched-label algebra and root-level inference without simulation."""

import unittest

import numpy as np

from scripts.run_pointmaze_review_controls import balanced_arms, single_arm_target, summarize


class PointMazeReviewControlsTests(unittest.TestCase):
    def test_single_arm_target_averages_to_paired_difference(self):
        row = {"feature_names": ("target_error_0", "target_error_1"), "features": [0.6, 0.8]}
        keep = single_arm_target({**row, "branch_ise": 1.2}, False)
        renew = single_arm_target({**row, "branch_ise": 0.9}, True)
        self.assertAlmostEqual((keep + renew) / 2, 0.3)
        self.assertEqual(sum(balanced_arms(np.random.default_rng(1), 24)), 12)

    def test_complete_cohort_uses_eight_root_effects_and_requires_both_halves(self):
        plan = {"calls_per_episode": 24, "cells": []}
        records = []
        for i in range(8):
            path_seeds = [1000 + 2 * i, 1001 + 2 * i]
            plan["cells"].append({"root": i, "evaluation_paths": path_seeds})
            records.append({
                "root": i, "status": "complete", "preflight": False,
                "fit_rows": [192, 384], "supervision_steps_per_rule": 200_000,
                "supervision_planner_calls_per_rule": 3_000,
                "ledger": {"cpu_seconds": 1.0},
                "paths": [{"seed": seed, "donor": path_seeds[j ^ 1], "ridge": 1.0,
                           "yoked": 1.1 + i / 100, "single_arm": 2.0 if j == 0 else 0.9,
                           "yoked_schedule_differs": True}
                          for j, seed in enumerate(path_seeds)],
            })
        report = summarize(plan, records)
        self.assertEqual(report["contrasts"]["yoked"]["paired_gain"]["n"], 8)
        self.assertTrue(report["contrasts"]["yoked"]["supported"])
        self.assertFalse(report["contrasts"]["single_arm"]["supported"])
        with self.assertRaisesRegex(ValueError, "complete frozen cohort"):
            summarize(plan, records[:-1])


if __name__ == "__main__":
    unittest.main()
