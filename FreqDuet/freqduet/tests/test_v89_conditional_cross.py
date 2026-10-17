from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v89_conditional_cross import DAYS, ITEMS, MODES, RISK_KEYS, audit, effects, use_early


class ConditionalCrossTest(unittest.TestCase):
    def test_new_days_do_not_reuse_v87_development_or_smoke_days(self):
        self.assertFalse(set(DAYS) & (set(audit.experiment.CAL_DAYS) | set(audit.experiment.EVAL_DAYS) | {1879901}))
        self.assertEqual(len(ITEMS) * len(MODES), 480)

    def test_native_endpoints_and_each_conditional_corner(self):
        for group in audit.GROUPS:
            self.assertFalse(use_early("late", group))
            self.assertTrue(use_early("early", group))
            self.assertEqual(use_early("no_gain_early", group), group == "more_hold_gain_not_up")
            self.assertEqual(use_early("gain_early", group), group == "more_hold_gain_up")
            self.assertEqual(use_early("both_early", group), group in ("more_hold_gain_up", "more_hold_gain_not_up"))

    def test_interaction_is_not_assumed_additive(self):
        e = effects(dict(late=10., no_gain_early=8., gain_early=12., both_early=9., early=7.))
        self.assertEqual(e["no_gain_vs_late"], -2.)
        self.assertEqual(e["gain_vs_late"], 2.)
        self.assertEqual(e["interaction"], -1.)
        self.assertEqual(e["both_vs_late"], e["no_gain_vs_late"] + e["gain_vs_late"] + e["interaction"])

    def test_joint_risk_bins_keep_zero_load_pressure_and_command_history_distinct(self):
        self.assertEqual(len(set(RISK_KEYS)), 4 * 2 * 2 * 2)
        self.assertIn(("more_hold_gain_not_up", True, True, False), RISK_KEYS)
        self.assertIn(("more_hold_gain_not_up", False, False, True), RISK_KEYS)


if __name__ == "__main__":
    unittest.main()
