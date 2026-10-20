import unittest

import numpy as np

from scripts import run_native_transit_service_audit_stage299 as audit
from scripts import submit_native_transit_service_audit_stage299_scheduleurm as submit


class FrozenPairedAuditTests(unittest.TestCase):
    def fixture(self):
        return np.zeros((20, 32, 7, 4)), np.arange(20).reshape(10, 2, 1)

    def test_retention_uses_shared_noise_not_independent_arm_errors(self):
        actual, cube = self.fixture()
        rng = np.random.default_rng(4)
        actual[:, :, 2, 3] = rng.normal(-2., 3., size=(20, 32))
        actual[:, :, 4, 3] = .5*actual[:, :, 2, 3]
        result = audit.paired_statistics(actual, cube)
        for half in ("all32", "first16", "last16"):
            margin = result[half]["half_base_wait_margin"]
            self.assertEqual(margin["mean"]["wait_min"], 0.)
            self.assertEqual(margin["prefix_CI95"]["wait_min"], [0., 0.])
            self.assertEqual(margin["conditional_MC_CI95"]["wait_min"], [0., 0.])
            self.assertFalse(result["retention_margin_both_CIs_negative"][half])

    def test_half_failure_is_not_hidden_by_pooled_retention(self):
        actual, cube = self.fixture()
        actual[:, :, 2, 3] = -2.
        actual[:, :16, 4, 3] = -2.
        actual[:, 16:, 4, 3] = -.5
        result = audit.paired_statistics(actual, cube)
        self.assertTrue(result["retention_margin_both_CIs_negative"]["all32"])
        self.assertTrue(result["retention_margin_both_CIs_negative"]["first16"])
        self.assertFalse(result["retention_margin_both_CIs_negative"]["last16"])

    def test_comparisons_and_fixed_stratum_leave_one_prefix(self):
        actual, cube = self.fixture()
        actual[:, :, 4, 0] = np.arange(20)[:, None]
        actual[:, :, 5, 0] = 3.
        result = audit.paired_statistics(actual, cube)
        self.assertAlmostEqual(result["all32"]["joint_budgeted_vs_service_rank_matched"]["mean"]["cost"], 6.5)
        low, high = result["leave_one_prefix_mean_range"]["joint_budgeted_vs_raw"]["cost"]
        self.assertAlmostEqual(low, 9.45)
        self.assertAlmostEqual(high, 9.55)
        self.assertEqual(len(result["descriptive_groups"]), 10)

    def test_scheduler_is_offline_and_has_own_staging_inputs(self):
        job = submit.task("unit_stage299")
        self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 3072, 0))
        self.assertIsNone(job["require_node"])
        self.assertEqual(job["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIn("unit_stage299/inputs", job["stage_input_paths"][-1])
        self.assertIn("service_audit_stage299.py", job["cmd"])
        self.assertEqual(audit.contract()["new_native_episodes"], 0)


if __name__ == "__main__": unittest.main()
