import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from scripts.run_freqduet_protocol_v2_matrix import (
    METRICS,
    OPTIONAL_COST_METRICS,
    PROTOCOL_VERSION,
    analysis_metrics_for_frame,
    config_fingerprint,
    holm_adjusted_pvalues,
    paired_sign_flip_p,
    protocol_version_for_config,
    source_fingerprint,
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
        self.assertIn("upper/interval_credit.py", paths)
        self.assertIn("env/data/passenger_OD.xlsx", paths)
        self.assertEqual(first["file_count"], len(first["files"]))

    def test_run_manifest_rejects_source_mismatch_on_resume(self):
        expected = {
            "manifest_version": "freqduet-run-manifest-v1",
            "source_fingerprint": {"sha256": "a" * 64},
        }
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol_run_manifest.json"
            path.write_text(
                '{"manifest_version":"freqduet-run-manifest-v1",'
                '"source_fingerprint":{"sha256":"' + "b" * 64 + '"}}')
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_run_manifest(path, expected=expected)


if __name__ == "__main__":
    unittest.main()
