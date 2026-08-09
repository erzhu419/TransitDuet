import copy
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.authoritative_evidence_registry import (
    DEFAULT_REGISTRY,
    DEFAULT_REPOSITORY_ROOT,
    build_registry_outputs,
    load_registry,
    validate_registry,
)


class AuthoritativeEvidenceRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = DEFAULT_REPOSITORY_ROOT
        cls.registry_path = cls.root / DEFAULT_REGISTRY

    def test_registered_snapshot_is_hash_verified_and_fail_closed(self):
        records = validate_registry(
            load_registry(self.registry_path), self.root
        )
        self.assertEqual(len(records), 6)
        by_id = {row["evidence_id"]: row for row in records}
        self.assertTrue(
            by_id["mujoco_v12_responsibility_confirmatory"][
                "positive_claim_supported"
            ]
        )
        self.assertTrue(
            by_id["mujoco_v13_behavioral_confirmatory"][
                "manuscript_reportable"
            ]
        )
        self.assertFalse(
            by_id["mujoco_v13_behavioral_confirmatory"][
                "positive_claim_supported"
            ]
        )
        quant = by_id["quant_v74_matched_baseline_confirmatory"]
        self.assertEqual(
            quant["facts"]["primary_status_counts"],
            {
                "supported_improvement": 8,
                "supported_harm": 1,
                "inconclusive": 3,
            },
        )
        self.assertFalse(
            by_id["mujoco_v14_endpoint_aligned_screen"][
                "manuscript_reportable"
            ]
        )
        self.assertFalse(
            by_id["legacy_paper_diagnostics_snapshot"][
                "manuscript_reportable"
            ]
        )

    def test_hash_tampering_is_rejected(self):
        registry = copy.deepcopy(load_registry(self.registry_path))
        registry["records"][0]["artifacts"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_registry(registry, self.root)

    def test_positive_disposition_requires_confirmatory_design(self):
        registry = copy.deepcopy(load_registry(self.registry_path))
        registry["records"][0]["evidence_stage"] = "development"
        with self.assertRaisesRegex(
            ValueError, "positive disposition lacks confirmatory support"
        ):
            validate_registry(registry, self.root)

    def test_build_outputs_preserves_positive_and_negative_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = build_registry_outputs(
                registry_path=self.registry_path,
                repository_root=self.root,
                output_dir=root / "results",
                md_output=root / "ledger.md",
            )
            self.assertEqual(summary["record_count"], 6)
            self.assertEqual(summary["reportable_record_count"], 3)
            self.assertEqual(summary["positive_supported_record_count"], 1)
            self.assertEqual(summary["mixed_or_negative_record_count"], 2)
            self.assertTrue((root / "results" / "summary.json").is_file())
            self.assertTrue((root / "ledger.md").is_file())


if __name__ == "__main__":
    unittest.main()
