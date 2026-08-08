import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.theory_appendix import (
    build_theory_payload,
    conditional_no_tradeoff_margin,
    finite_sample_mean_ci_radius,
    hierarchical_credit_residual_bound,
    promotion_detection_delay_bound,
    promotion_false_positive_bound,
    responsibility_reconstruction_error,
    shaped_return_deviation_bound,
    stress_claim_coverage_fraction,
    write_outputs,
)


class TheoryAppendixTest(unittest.TestCase):
    def test_bounds_are_directional(self):
        self.assertAlmostEqual(
            shaped_return_deviation_bound(0.5, [1.0, -2.0, 3.0]),
            2.0,
        )
        loose = promotion_false_positive_bound(
            window_bins=10,
            persistence_ratio=0.35,
            event_probability=0.25,
        )
        tight = promotion_false_positive_bound(
            window_bins=10,
            persistence_ratio=0.35,
            event_probability=0.05,
        )
        self.assertLess(tight, loose)
        self.assertEqual(
            promotion_detection_delay_bound(
                update_interval_s=60.0,
                window_bins=10,
                persistence_ratio=0.35,
            ),
            600.0,
        )
        self.assertLess(
            finite_sample_mean_ci_radius(sample_std=1.0, n=16),
            finite_sample_mean_ci_radius(sample_std=1.0, n=4),
        )
        self.assertAlmostEqual(
            hierarchical_credit_residual_bound(
                total_credit=[1.0, 2.0],
                upper_credit=[0.25, 1.0],
                lower_credit=[0.75, 0.5],
            ),
            0.5,
        )
        self.assertGreater(
            conditional_no_tradeoff_margin(
                baseline_advantage=0.20,
                leakage_penalty_budget=0.05,
                constraint_slack=0.03,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            stress_claim_coverage_fraction(supported_regimes=4, required_regimes=5),
            0.8,
        )
        self.assertLessEqual(
            responsibility_reconstruction_error(
                upper_policy=[0.2, -0.7],
                raw_lower=[0.5, -0.1],
                transferred_lf=[0.12, -0.08],
            ),
            1e-15,
        )

    def test_theory_appendix_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_theory_payload(root / "results")
            write_outputs(root / "out", payload)
            self.assertTrue((root / "out" / "summary.json").exists())
            report = (root / "out" / "report.md").read_text()
            self.assertGreaterEqual(len(payload["theorems"]), 9)
            self.assertIn("Theorem 1", report)
            self.assertIn("Theorem 5", report)
            self.assertIn("Proposition 8", report)
            self.assertIn("Proposition 9", report)
            self.assertIn("Proposition 10", report)
            self.assertIn("Proposition 11", report)
            self.assertIn("Proof:", report)
            self.assertIn("Limitation:", report)


if __name__ == "__main__":
    unittest.main()
