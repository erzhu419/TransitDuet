import unittest

import pandas as pd

from scripts.decide_freqduet_protocol_v4_selection import (
    NO_HARM_LIMITS,
    PRIMARY,
    decide,
)


REFERENCE = "main"
SIMPLE = "csac"


def candidate_row(name, mean=0.0, primary_high=0.0):
    row = {
        "candidate": name,
        "reference": REFERENCE,
        f"delta_{PRIMARY}_mean": mean,
        f"delta_{PRIMARY}_ci_high": primary_high,
    }
    for metric, (side, limit) in NO_HARM_LIMITS.items():
        row[f"delta_{metric}_ci_{side}"] = limit
    return row


class ProtocolV4SelectionDecisionTest(unittest.TestCase):
    def test_retains_reference_without_locked_superiority(self):
        frame = pd.DataFrame([
            candidate_row("nopromotion", mean=-0.005, primary_high=0.002),
        ])

        result = decide(
            frame,
            reference=REFERENCE,
            simple_config=SIMPLE,
            frequency_failure_configs={"nofreq", "rawhistory"},
        )

        self.assertEqual(result["status"], "reference_retained")
        self.assertEqual(result["selected_config"], REFERENCE)

    def test_selects_superior_single_axis_candidate(self):
        frame = pd.DataFrame([
            candidate_row("nopromotion", mean=-0.02, primary_high=-0.005),
        ])

        result = decide(
            frame,
            reference=REFERENCE,
            simple_config=SIMPLE,
            frequency_failure_configs={"nofreq", "rawhistory"},
        )

        self.assertEqual(result["status"], "single_axis_candidate_selected")
        self.assertEqual(result["selected_config"], "nopromotion")

    def test_safety_regression_blocks_primary_improvement(self):
        row = candidate_row("nopromotion", mean=-0.02, primary_high=-0.005)
        row["delta_restricted_total_journey_horizon_min_ci_high"] = 0.6

        result = decide(
            pd.DataFrame([row]),
            reference=REFERENCE,
            simple_config=SIMPLE,
            frequency_failure_configs={"nofreq", "rawhistory"},
        )

        self.assertEqual(result["status"], "reference_retained")

    def test_prefers_noninferior_simple_optimizer(self):
        frame = pd.DataFrame([
            candidate_row(SIMPLE, mean=0.002, primary_high=0.009),
        ])

        result = decide(
            frame,
            reference=REFERENCE,
            simple_config=SIMPLE,
            frequency_failure_configs={"nofreq", "rawhistory"},
        )

        self.assertEqual(result["status"], "simpler_optimizer_selected")
        self.assertEqual(result["selected_config"], SIMPLE)

    def test_frequency_control_best_marks_claim_failure(self):
        frame = pd.DataFrame([
            candidate_row("nofreq", mean=-0.001, primary_high=0.01),
            candidate_row("nopromotion", mean=0.002, primary_high=0.01),
        ])

        result = decide(
            frame,
            reference=REFERENCE,
            simple_config=SIMPLE,
            frequency_failure_configs={"nofreq", "rawhistory"},
        )

        self.assertEqual(result["status"], "frequency_claim_failed")
        self.assertIsNone(result["selected_config"])


if __name__ == "__main__":
    unittest.main()
