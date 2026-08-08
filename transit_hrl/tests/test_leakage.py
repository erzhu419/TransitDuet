import unittest

import numpy as np

from freq_hrl.core import (
    CausalLeakageRewardShaper,
    CausalLowFrequencyEffectProjector,
    CausalRollingBandTracker,
    CumulativeActionEffectOperator,
    LeakageRegularizer,
    evaluate_rms_leakage_budget,
)


class LeakageTest(unittest.TestCase):
    def test_online_rolling_bands_match_batch_diagnostics(self):
        values = np.asarray([
            [1.0, -0.5],
            [0.2, 0.7],
            [-0.4, 0.3],
            [0.9, -0.8],
            [0.1, 0.2],
        ])
        tracker = CausalRollingBandTracker(window=3)
        tracker.reset(2)
        online = [tracker.update(row) for row in values]
        regularizer = LeakageRegularizer(
            upper_hf_window=3, lower_lf_window=3
        )
        metrics = regularizer.compute(values, values)
        online_low = np.asarray([row["low"] for row in online])
        online_high = np.asarray([row["high"] for row in online])
        self.assertAlmostEqual(
            float(np.mean(np.square(online_low))),
            float(metrics["LowerLFDriftAbs"]),
        )
        self.assertAlmostEqual(
            float(np.mean(np.square(online_high))),
            float(metrics["UpperHFPowerAbs"]),
        )

    def test_lower_drift_penalty_increases_with_cumulative_bias(self):
        op = CumulativeActionEffectOperator()
        reg = LeakageRegularizer(lower_lf_window=8)
        zero_mean_actions = np.array([1.0, -1.0] * 20)
        biased_actions = np.ones(40)
        zero_metrics = reg.compute(
            upper_effect=np.zeros(40),
            lower_effect=op.lower_effect(zero_mean_actions),
        )
        biased_metrics = reg.compute(
            upper_effect=np.zeros(40),
            lower_effect=op.lower_effect(biased_actions),
        )
        self.assertGreater(
            biased_metrics["lower_lf_penalty"],
            zero_metrics["lower_lf_penalty"],
        )

    def test_upper_hf_penalty_increases_with_oscillation(self):
        reg = LeakageRegularizer(upper_hf_window=4)
        smooth = np.linspace(0.0, 1.0, 40)
        oscillating = np.array([1.0, -1.0] * 20)
        smooth_metrics = reg.compute(smooth, np.zeros(40))
        oscillating_metrics = reg.compute(oscillating, np.zeros(40))
        self.assertGreater(
            oscillating_metrics["upper_hf_penalty"],
            smooth_metrics["upper_hf_penalty"],
        )

    def test_causal_reward_shaper_subtracts_online_penalty(self):
        shaper = CausalLeakageRewardShaper(
            regularizer=LeakageRegularizer(upper_hf_window=2, lower_lf_window=2),
            reward_penalty_scale=0.1,
            enabled=True,
        )
        info = None
        for value in [1.0, -1.0, 1.0, -1.0]:
            info = shaper.update(
                upper_effect=np.asarray([value]),
                lower_effect=np.asarray([0.5]),
                reward=1.0,
            )
        self.assertIsNotNone(info)
        self.assertGreater(info["leakage_reward_penalty"], 0.0)
        self.assertLess(info["shaped_reward"], info["raw_reward"])

    def test_disabled_reward_shaper_reports_zero_penalty(self):
        shaper = CausalLeakageRewardShaper(
            reward_penalty_scale=1.0,
            enabled=False,
        )
        info = shaper.update(upper_effect=[1.0], lower_effect=[1.0], reward=1.0)
        self.assertEqual(info["leakage_reward_penalty"], 0.0)
        self.assertEqual(info["shaped_reward"], 1.0)

    def test_causal_effect_projector_removes_slow_baseline(self):
        reg = LeakageRegularizer(lower_lf_window=8)
        raw = np.ones(40)
        projector = CausalLowFrequencyEffectProjector(window=8, gain=1.0)
        projected = projector.transform_sequence(raw)
        raw_metrics = reg.compute(upper_effect=np.zeros(40), lower_effect=raw)
        projected_metrics = reg.compute(upper_effect=np.zeros(40), lower_effect=projected)
        self.assertLess(projected_metrics["LowerLFDrift"], raw_metrics["LowerLFDrift"])
        self.assertAlmostEqual(float(np.max(np.abs(projected))), 0.0)

    def test_causal_effect_projector_keeps_fast_residuals(self):
        projector = CausalLowFrequencyEffectProjector(window=4, gain=1.0)
        projected = projector.transform_sequence(np.array([1.0, -1.0] * 8))
        self.assertGreater(float(np.mean(np.abs(projected))), 0.4)

    def test_fixed_rms_budget_preserves_action_effect_scale(self):
        large = evaluate_rms_leakage_budget(4e-6, rms_budget=1e-3)
        small = evaluate_rms_leakage_budget(4e-10, rms_budget=1e-3)
        self.assertAlmostEqual(float(large["budget_ratio"]), 2.0)
        self.assertAlmostEqual(float(small["budget_ratio"]), 0.02)
        self.assertTrue(bool(large["budget_violated"]))
        self.assertFalse(bool(small["budget_violated"]))

    def test_fixed_rms_budget_rejects_data_dependent_zero_budget(self):
        with self.assertRaises(ValueError):
            evaluate_rms_leakage_budget(0.0, rms_budget=0.0)


if __name__ == "__main__":
    unittest.main()
