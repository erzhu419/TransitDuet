import unittest

import pandas as pd

from scripts.decide_freqduet_protocol_v5_screen import (
    ALLOCATION_CONTROLS,
    FREQUENCY_CONTROLS,
    MECHANISM_ABLATIONS,
    PRIMARY,
    REFERENCE,
    REFERENCE_NO_HARM_LIMITS,
    SIMPLE_CONFIG,
    decide,
)


def paired_frame():
    rows = []
    configs = (
        FREQUENCY_CONTROLS | ALLOCATION_CONTROLS
        | MECHANISM_ABLATIONS | {SIMPLE_CONFIG})
    for name in sorted(configs):
        if name in FREQUENCY_CONTROLS:
            mean, low, high = 0.50, 0.20, 0.80
        elif name in ALLOCATION_CONTROLS:
            mean, low, high = 0.30, 0.10, 0.50
        elif name in MECHANISM_ABLATIONS:
            mean, low, high = 0.30, 0.10, 0.50
        else:
            mean, low, high = 0.10, -0.10, 0.20
        row = {
            "candidate": name,
            "reference": REFERENCE,
            f"delta_{PRIMARY}_mean": mean,
            f"delta_{PRIMARY}_ci_low": low,
            f"delta_{PRIMARY}_ci_high": high,
        }
        for metric in REFERENCE_NO_HARM_LIMITS:
            row[f"delta_{metric}_ci_low"] = -0.001
            row[f"delta_{metric}_ci_high"] = 0.001
        rows.append(row)
    return pd.DataFrame(rows)


def summary_frame(projected_sum=0.0):
    return pd.DataFrame([{
        "config": REFERENCE,
        f"{PRIMARY}_mean": 18.0,
        "lower_causal_guard_enabled_mean": 1.0,
        "upper_plan_projected_delta_sum_abs_mean_s_mean": projected_sum,
        "upper_interval_onboard_cost_sum_mean": 0.7,
    }])


class ProtocolV5ScreenDecisionTest(unittest.TestCase):
    def test_selects_simple_optimizer_only_after_frequency_support(self):
        result = decide(paired_frame(), summary_frame())

        self.assertEqual(
            result["status"],
            "frequency_supported_simple_optimizer_candidate",
        )
        self.assertEqual(result["selected_config"], SIMPLE_CONFIG)

    def test_rejects_main_when_a_frequency_control_is_superior(self):
        paired = paired_frame()
        control = sorted(FREQUENCY_CONTROLS)[0]
        mask = paired["candidate"].eq(control)
        paired.loc[mask, f"delta_{PRIMARY}_mean"] = -0.40
        paired.loc[mask, f"delta_{PRIMARY}_ci_low"] = -0.60
        paired.loc[mask, f"delta_{PRIMARY}_ci_high"] = -0.20

        result = decide(paired, summary_frame())

        self.assertEqual(result["status"], "structural_redesign_required")
        self.assertIsNone(result["selected_config"])

    def test_marks_weak_frequency_advantage_inconclusive(self):
        paired = paired_frame()
        control = sorted(FREQUENCY_CONTROLS)[0]
        mask = paired["candidate"].eq(control)
        paired.loc[mask, f"delta_{PRIMARY}_mean"] = 0.10
        paired.loc[mask, f"delta_{PRIMARY}_ci_low"] = -0.10
        paired.loc[mask, f"delta_{PRIMARY}_ci_high"] = 0.30

        result = decide(paired, summary_frame())

        self.assertEqual(result["status"], "frequency_evidence_inconclusive")

    def test_fails_when_zero_sum_plan_invariant_is_violated(self):
        result = decide(paired_frame(), summary_frame(projected_sum=0.01))

        self.assertEqual(result["status"], "implementation_contract_failed")

    def test_marks_layer_allocation_evidence_inconclusive(self):
        paired = paired_frame()
        control = sorted(ALLOCATION_CONTROLS)[0]
        mask = paired["candidate"].eq(control)
        paired.loc[mask, f"delta_{PRIMARY}_mean"] = 0.05
        paired.loc[mask, f"delta_{PRIMARY}_ci_low"] = -0.10
        paired.loc[mask, f"delta_{PRIMARY}_ci_high"] = 0.20

        result = decide(paired, summary_frame())

        self.assertEqual(result["status"], "allocation_evidence_inconclusive")


if __name__ == "__main__":
    unittest.main()
