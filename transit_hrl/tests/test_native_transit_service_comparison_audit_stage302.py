import unittest

import numpy as np

from scripts import run_native_transit_service_comparison_audit_stage302 as audit
from scripts import submit_native_transit_service_comparison_audit_stage302_scheduleurm as submit


class ServiceComparisonAuditTests(unittest.TestCase):
    def test_joint_minus_service_only_uses_correct_arm_and_sign(self):
        actual = np.zeros((20, 32, 7, 4)); cube = np.arange(20).reshape(10, 2, 1)
        actual[:, :, 3, 0] = -3.; actual[:, :, 4, 0] = -4.
        result = audit.paired_statistics(actual, cube, arms=audit.source.ARMS,
            comparators=audit.COMPARATORS, seed=301)
        for half in ("all32", "first16", "last16"):
            comparison = result[half]["joint_budgeted_vs_service_only"]
            self.assertEqual(comparison["mean"]["cost"], -1.)
            np.testing.assert_allclose(comparison["prefix_CI95"]["cost"], [-1., -1.], rtol=0., atol=1e-12)
            np.testing.assert_allclose(comparison["conditional_MC_CI95"]["cost"], [-1., -1.], rtol=0., atol=1e-12)

    def test_service_retention_uses_shared_noise_and_named_arms(self):
        arms = ("service_only", "raw", "zero_abstain", "joint_budgeted")
        actual = np.zeros((20, 32, 4, 4)); cube = np.arange(20).reshape(10, 2, 1)
        actual[:, :, 2, 3] = np.random.default_rng(5).normal(-2., 3., (20, 32))
        actual[:, :, 0, 3] = .5*actual[:, :, 2, 3]
        result = audit.paired_statistics(actual, cube, arms=arms, primary="service_only",
            comparators=("raw",), seed=301)
        for half in ("all32", "first16", "last16"):
            margin = result[half]["half_base_wait_margin"]
            self.assertEqual(margin["mean"]["wait_min"], 0.)
            self.assertEqual(margin["conditional_MC_CI95"]["wait_min"], [0., 0.])
            self.assertFalse(result["retention_margin_both_CIs_negative"][half])

    def test_half_regression_is_not_hidden_by_pooled_gain(self):
        actual = np.zeros((20, 32, 7, 4)); cube = np.arange(20).reshape(10, 2, 1)
        actual[:, :16, 4, 0] = -4.; actual[:, 16:, 4, 0] = 1.
        result = audit.paired_statistics(actual, cube, arms=audit.source.ARMS,
            comparators=("service_only",), seed=301)
        self.assertLess(result["all32"]["joint_budgeted_vs_service_only"]["mean"]["cost"], 0.)
        self.assertGreater(result["last16"]["joint_budgeted_vs_service_only"]["prefix_CI95"]["cost"][0], 0.)

    def test_scheduler_keeps_audit_offline_and_unpinned(self):
        job = submit.task("unit_stage302")
        self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 3072, 0))
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn("unit_stage302/inputs", job["stage_input_paths"][-1])
        self.assertEqual(audit.contract()["new_native_episodes"], 0)


if __name__ == "__main__": unittest.main()
