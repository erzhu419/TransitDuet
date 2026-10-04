import unittest

from scripts import pointmaze_option_residual_diagnostic_spec as spec


class OptionResidualDiagnosticTest(unittest.TestCase):
    def test_frozen_roster_and_budget(self):
        self.assertEqual(spec.roots(), (410011,))
        self.assertEqual(spec.EPISODES_PER_PERIOD, 32)
        self.assertEqual(spec.budget()["native_episodes"], 128)
        self.assertEqual(spec.contract()["selection"], "single_root_410011_fixed_before_reading_diagnostic_results")

    def test_checkpoint_roster_is_final_branch_only(self):
        for period in spec.PERIODS:
            for arm in ("learned", "forecast"):
                self.assertIn(f"period_{period}_{arm}.pt", str(spec.checkpoint(410011, period, arm)))


if __name__ == "__main__":
    unittest.main()
