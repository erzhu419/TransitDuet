import unittest

from freq_hrl.core import CausalPromotionGate


class PromotionGateTest(unittest.TestCase):
    def test_isolated_spike_does_not_promote(self):
        gate = CausalPromotionGate(
            update_interval_s=60,
            window_s=300,
            residual_threshold=1.0,
            persistence_ratio=0.6,
            cooldown_s=0,
        )
        promoted = []
        for residual in [0.0, 0.0, 4.0, 0.0, 0.0, 0.0]:
            signal = gate.update({"x_high": [residual], "x_low": [1.0]})
            promoted.append(signal["promote"])
        self.assertFalse(any(promoted))

    def test_persistent_spike_promotes(self):
        gate = CausalPromotionGate(
            update_interval_s=60,
            window_s=300,
            residual_threshold=1.0,
            persistence_ratio=0.6,
            cooldown_s=0,
        )
        promoted = []
        for residual in [0.0, 3.0, 3.0, 3.0, 3.0, 3.0]:
            signal = gate.update({"x_high": [residual], "x_low": [1.0]})
            promoted.append(signal["promote"])
        self.assertTrue(any(promoted))

    def test_min_age_blocks_startup_candidate(self):
        gate = CausalPromotionGate(
            update_interval_s=60,
            window_s=300,
            residual_threshold=1.0,
            persistence_ratio=0.6,
            cooldown_s=0,
            min_age_s=600,
        )
        early = []
        for residual in [3.0, 3.0, 3.0, 3.0, 3.0]:
            signal = gate.update({"x_high": [residual], "x_low": [1.0]})
            early.append(signal)
        self.assertFalse(any(signal["promote"] for signal in early))
        self.assertEqual(early[-1]["reason"], "candidate_before_min_age")

        later = []
        for residual in [3.0, 3.0, 3.0, 3.0, 3.0]:
            later.append(gate.update({"x_high": [residual], "x_low": [1.0]}))
        self.assertTrue(any(signal["promote"] for signal in later))

    def test_activation_strength_blocks_marginal_candidate(self):
        gate = CausalPromotionGate(
            update_interval_s=60,
            window_s=300,
            residual_threshold=1.0,
            persistence_ratio=0.6,
            cooldown_s=0,
            activation_strength_threshold=0.5,
        )
        marginal = []
        for residual in [0.0, 1.05, 1.05, 1.05, 0.0]:
            marginal.append(gate.update({"x_high": [residual], "x_low": [0.0]}))
        self.assertFalse(any(signal["promote"] for signal in marginal))
        self.assertEqual(marginal[-1]["reason"], "candidate_below_activation_strength")

        strong = gate.update({"x_high": [2.0], "x_low": [0.0]})
        self.assertTrue(strong["promote"])

    def test_startup_strength_threshold_relaxes_after_age(self):
        gate = CausalPromotionGate(
            update_interval_s=60,
            window_s=300,
            residual_threshold=1.0,
            persistence_ratio=0.6,
            cooldown_s=0,
            startup_strength_age_s=360,
            startup_strength_threshold=0.5,
        )
        startup = []
        for residual in [0.0, 1.05, 1.05, 1.05, 0.0]:
            startup.append(gate.update({"x_high": [residual], "x_low": [0.0]}))
        self.assertFalse(any(signal["promote"] for signal in startup))

        mature = gate.update({"x_high": [1.05], "x_low": [0.0]})
        self.assertTrue(mature["promote"])


if __name__ == "__main__":
    unittest.main()
