from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_freqduet_v69_signal as audit


def contexts(signals, noises):
    return [dict(scenario_seed=day, signal_noise=dict(event=dict(metric=dict(
        debiased_signal_squared=s, single_future_noise_variance=n))))
        for day, s, n in zip(audit.pilot.labels.DEMAND_SEEDS, signals, noises) for _ in range(4)]


class SignalAuditTest(unittest.TestCase):
    def test_joint_day_resampling_and_eight_future_noise_scaling(self):
        result = audit.signal_ratio(contexts(range(1, 9), range(4, 33, 4)), "metric")
        self.assertEqual(result["single_future_signal_to_noise"], .25)
        self.assertEqual(result["single_future_ci95"], [.25, .25])
        self.assertEqual(result["eight_future_mean_signal_to_noise"], 2.)
        self.assertEqual(result["eight_future_mean_ci95"], [2., 2.])
        self.assertEqual(result["positive_signal_days"], 8)

    def test_negative_corrected_signal_is_not_clipped_and_concentration_uses_days(self):
        result = audit.signal_ratio(contexts([1.] * 7 + [-10.], [4.] * 8), "metric")
        self.assertEqual(result["single_future_signal_to_noise"], -.09375)
        self.assertEqual(result["positive_signal_days"], 7)
        self.assertAlmostEqual(result["max_day_share_of_positive_signal"], 1. / 7.)
        self.assertEqual(result["leave_one_day_out_signal_range"], [-4. / 7., 1.])


if __name__ == "__main__":
    unittest.main()
