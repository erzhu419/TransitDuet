import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.theory_appendix import (
    build_theory_payload,
    conditional_no_tradeoff_margin,
    finite_sample_mean_ci_radius,
    hierarchical_credit_residual_bound,
    ideal_transfer_relative_leakage_reduction,
    lower_router_constant_transient,
    lower_router_frequency_response_power,
    physical_power_excess_upper_bound,
    promotion_detection_delay_bound,
    promotion_false_positive_bound,
    promotion_warm_window_delay_bound,
    projected_dual_regret_term,
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
        self.assertEqual(
            promotion_warm_window_delay_bound(
                update_interval_s=60.0,
                window_bins=10,
                persistence_ratio=0.35,
            ),
            240.0,
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
        self.assertAlmostEqual(
            ideal_transfer_relative_leakage_reduction(
                lower_lf_norm=0.5,
                transfer_error_norm=0.2,
            ),
            0.84,
        )
        self.assertEqual(
            lower_router_frequency_response_power(
                alpha=0.1,
                angular_frequency=0.0,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            lower_router_frequency_response_power(
                alpha=0.1,
                angular_frequency=3.141592653589793,
            ),
            4.0 / (2.0 - 0.1) ** 2,
        )
        self.assertAlmostEqual(
            lower_router_frequency_response_power(
                alpha=0.1,
                angular_frequency=0.0,
                strength=0.1,
            ),
            0.81,
        )
        self.assertLess(
            lower_router_constant_transient(
                latent_magnitude=1.0,
                alpha=0.1,
                step=32,
            ),
            0.04,
        )
        self.assertAlmostEqual(
            lower_router_constant_transient(
                latent_magnitude=1.0,
                alpha=0.1,
                step=1000,
                strength=0.1,
            ),
            0.9,
        )
        self.assertAlmostEqual(
            physical_power_excess_upper_bound(
                action_limit=1.0,
                rms_budget=0.05,
            ),
            0.9975,
        )
        self.assertGreater(
            projected_dual_regret_term(
                dual_radius=2.0,
                step_size=0.05,
                horizon=400,
                gradient_bound=1.0,
            ),
            0.0,
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
            self.assertGreaterEqual(len(payload["formal_statements"]), 14)
            router_markov = next(
                row for row in payload["formal_statements"]
                if row["id"] == "F12"
            )
            self.assertIn("(s_t,b_t,beta)", router_markov["statement"])
            self.assertIn(
                "nonstationary optimization",
                router_markov["limitation"],
            )
            self.assertIn("F1 (lemma)", report)
            self.assertIn("F3 (proposition)", report)
            self.assertIn("F9 (lemma)", report)
            self.assertIn("F11 (proposition)", report)
            self.assertIn("F14 (proposition)", report)
            self.assertIn("R1 (reporting_approximation)", report)
            self.assertIn("Proof:", report)
            self.assertIn("Limitation:", report)
            self.assertFalse(payload["independent_proof_verification"])


if __name__ == "__main__":
    unittest.main()
