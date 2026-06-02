import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.theory_appendix import (
    build_theory_payload,
    finite_sample_mean_ci_radius,
    hierarchical_credit_residual_bound,
    promotion_detection_delay_bound,
    promotion_false_positive_bound,
    shaped_return_deviation_bound,
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

    def test_theory_appendix_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = build_theory_payload(root / "results")
            write_outputs(root / "out", payload)
            self.assertTrue((root / "out" / "summary.json").exists())
            self.assertIn("Theorem 1", (root / "out" / "report.md").read_text())
            self.assertIn("Theorem 5", (root / "out" / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
