import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.run_freqduet_protocol_v2_matrix import (
    METRICS,
    OPTIONAL_COST_METRICS,
    PROTOCOL_VERSION,
    V4_SAFETY_METRICS,
    V5_MECHANISM_METRICS,
    V5_SAFETY_METRICS,
    V6_MECHANISM_METRICS,
    V6_SAFETY_METRICS,
    analysis_metrics_for_frame,
    config_fingerprint,
    hierarchical_bootstrap,
    holm_adjusted_pvalues,
    paired_sign_flip_p,
    protocol_version_for_config,
    scenario_contract,
    source_fingerprint,
    validate_common_scenario_tapes,
    validate_evaluation_frame,
    validate_run_manifest,
)


def evaluation_frame(seeds):
    rows = []
    for seed in seeds:
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "eval_seed": int(seed),
            "checkpoint_ep": 59,
            "policy_digest": "abc123",
            "scenario_tape_id": f"tape-{seed}",
        }
        row.update({metric: 1.0 for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


class ProtocolV2MatrixTest(unittest.TestCase):
    def test_analysis_metrics_accept_complete_explicit_cost_views(self):
        frame = evaluation_frame([101, 102])
        for metric in OPTIONAL_COST_METRICS:
            frame[metric] = 1.0

        self.assertEqual(
            analysis_metrics_for_frame(frame),
            METRICS + OPTIONAL_COST_METRICS,
        )

    def test_v4_analysis_requires_and_includes_safety_endpoints(self):
        frame = evaluation_frame([101, 102])
        frame["protocol_version"] = "freqduet-eval-v4"
        with self.assertRaisesRegex(RuntimeError, "safety metrics"):
            analysis_metrics_for_frame(frame)

        for metric in V4_SAFETY_METRICS:
            frame[metric] = 1.0
        self.assertTrue(
            set(V4_SAFETY_METRICS).issubset(analysis_metrics_for_frame(frame)))

    def test_v5_analysis_requires_safety_and_mechanism_endpoints(self):
        frame = evaluation_frame([101, 102])
        frame["protocol_version"] = "freqduet-eval-v5"
        with self.assertRaisesRegex(RuntimeError, "required passenger"):
            analysis_metrics_for_frame(frame)

        strict_metrics = V5_SAFETY_METRICS + V5_MECHANISM_METRICS
        for metric in strict_metrics:
            frame[metric] = 1.0
        self.assertTrue(
            set(strict_metrics).issubset(analysis_metrics_for_frame(frame)))

    def test_v5_evaluation_validation_rejects_missing_mechanism_metric(self):
        frame = evaluation_frame([101, 102])
        frame["protocol_version"] = "freqduet-eval-v5"
        for metric in V5_SAFETY_METRICS + V5_MECHANISM_METRICS:
            frame[metric] = 1.0
        frame = frame.drop(columns=["upper_plan_projected_delta_sum_abs_mean_s"])

        with self.assertRaisesRegex(ValueError, "missing evaluation columns"):
            validate_evaluation_frame(
                frame,
                [101, 102],
                "synthetic-v5",
                protocol_version="freqduet-eval-v5",
            )

    def test_v6_requires_execution_metrics_and_departure_guard_contract(self):
        frame = evaluation_frame([101, 102])
        frame["protocol_version"] = "freqduet-eval-v6"
        for metric in V6_SAFETY_METRICS + V6_MECHANISM_METRICS:
            frame[metric] = 1.0
        frame["lower_causal_guard_evidence_mode"] = "arrival_event_v5"

        with self.assertRaisesRegex(ValueError, "evidence contract"):
            validate_evaluation_frame(
                frame,
                [101, 102],
                "synthetic-v6",
                protocol_version="freqduet-eval-v6",
            )

        frame["lower_causal_guard_evidence_mode"] = (
            "pre_action_departure_v6")
        validate_evaluation_frame(
            frame,
            [101, 102],
            "synthetic-v6",
            protocol_version="freqduet-eval-v6",
        )

    def test_analysis_metrics_reject_mixed_cost_contract(self):
        frame = evaluation_frame([101, 102])
        frame["service_cost_observed"] = [1.0, np.nan]
        frame["service_cost_restricted"] = [1.0, 1.0]

        with self.assertRaisesRegex(RuntimeError, "incomplete explicit"):
            analysis_metrics_for_frame(frame)

    def test_mixed_wait_objectives_use_only_explicit_cost_views(self):
        frame = evaluation_frame([101, 102])
        frame["service_cost_wait_metric"] = ["observed", "restricted"]
        for metric in OPTIONAL_COST_METRICS:
            frame[metric] = 1.0

        metrics = analysis_metrics_for_frame(frame)
        self.assertNotIn("service_cost", metrics)
        self.assertTrue(set(OPTIONAL_COST_METRICS).issubset(metrics))

    def test_mixed_wait_objectives_without_explicit_views_are_rejected(self):
        frame = evaluation_frame([101, 102])
        frame["service_cost_wait_metric"] = ["observed", "restricted"]

        with self.assertRaisesRegex(RuntimeError, "mixes generic"):
            analysis_metrics_for_frame(frame)

    def test_evaluation_frame_requires_exact_unique_seed_rows(self):
        frame = evaluation_frame([101, 101])
        with self.assertRaisesRegex(ValueError, "one row per"):
            validate_evaluation_frame(frame, [101, 102], "synthetic")

    def test_evaluation_frame_rejects_nonfinite_metric(self):
        frame = evaluation_frame([101, 102])
        frame.loc[0, "service_cost"] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_evaluation_frame(frame, [101, 102], "synthetic")

    def test_exact_sign_flip_test_uses_training_seed_as_unit(self):
        self.assertAlmostEqual(
            paired_sign_flip_p(np.array([-1.0, -1.0, -1.0])),
            0.25,
        )

    def test_crossed_bootstrap_shares_eval_resample_across_train_seeds(self):
        frame = pd.DataFrame([
            {"train_seed": train, "eval_seed": evaluation, "metric": value}
            for train in (1, 2)
            for evaluation, value in ((11, 0.0), (12, 10.0))
        ])

        estimates = hierarchical_bootstrap(frame, "metric", draws=256)

        self.assertTrue(set(np.unique(estimates)).issubset({0.0, 5.0, 10.0}))

    def test_crossed_bootstrap_rejects_incomplete_grid(self):
        frame = pd.DataFrame([
            {"train_seed": 1, "eval_seed": 11, "metric": 0.0},
            {"train_seed": 1, "eval_seed": 12, "metric": 1.0},
            {"train_seed": 2, "eval_seed": 11, "metric": 2.0},
        ])

        with self.assertRaisesRegex(ValueError, "complete"):
            hierarchical_bootstrap(frame, "metric", draws=10)

    def test_scenario_tape_is_common_across_training_seeds(self):
        frame = pd.DataFrame([
            {"train_seed": 1, "eval_seed": 11, "scenario_tape_id": "a"},
            {"train_seed": 2, "eval_seed": 11, "scenario_tape_id": "b"},
        ])

        with self.assertRaisesRegex(RuntimeError, "common scenario tape"):
            validate_common_scenario_tapes(frame)

    def test_holm_correction_controls_each_comparison_family(self):
        corrected = holm_adjusted_pvalues([0.01, 0.04, 0.03, float("nan")])
        np.testing.assert_allclose(corrected[:3], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(corrected[3]))

    def test_config_fingerprint_covers_inheritance_lineage(self):
        result = config_fingerprint(
            "F_freqduet_protocol_v2_upperdisc_hiro")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertGreater(len(result["lineage"]), 1)
        self.assertTrue(
            result["lineage"][-1].endswith(
                "F_freqduet_protocol_v2_upperdisc_hiro.yaml"))

    def test_protocol_version_is_resolved_through_config_inheritance(self):
        self.assertEqual(
            protocol_version_for_config(
                "F_freqduet_protocol_v2_uppercompact_strict_intervaladd_hiro"),
            PROTOCOL_VERSION,
        )

    def test_source_fingerprint_covers_code_and_base_environment_data(self):
        first = source_fingerprint()
        second = source_fingerprint()

        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(len(first["sha256"]), 64)
        paths = {entry["path"] for entry in first["files"]}
        self.assertIn("runner_v3.py", paths)
        self.assertIn("randomness.py", paths)
        self.assertIn("upper/interval_credit.py", paths)
        self.assertIn("env/data/passenger_OD.xlsx", paths)
        self.assertEqual(first["file_count"], len(first["files"]))

    def test_v6_ablation_configs_share_one_scenario_contract(self):
        names = [
            "main", "nofreq", "rawhistory", "allfreq", "upperonly",
            "loweronly", "swapped", "nobudget", "noguard",
            "noloadcost", "waitonlycredit", "csac",
        ]
        contracts = [
            scenario_contract(f"F_freqduet_protocol_v6_{name}_hiro")
            for name in names
        ]
        self.assertEqual(len({value["sha256"] for value in contracts}), 1)
        self.assertEqual(
            contracts[0]["version"], "freqduet-scenario-contract-v1")

    def test_run_manifest_rejects_source_mismatch_on_resume(self):
        expected = {
            "manifest_version": "freqduet-run-manifest-v2",
            "source_fingerprint": {"sha256": "a" * 64},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol_run_manifest.json"
            path.write_text(
                '{"manifest_version":"freqduet-run-manifest-v2",'
                '"source_fingerprint":{"sha256":"' + "b" * 64 + '"}}')
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_run_manifest(path, expected=expected)


if __name__ == "__main__":
    unittest.main()
