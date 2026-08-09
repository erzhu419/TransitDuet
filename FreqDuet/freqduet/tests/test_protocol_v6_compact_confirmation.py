import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.audit_protocol_v6_compact_confirmation import (
    DEFAULT_MATCHED_CONTEXT,
    DEFAULT_PRIMARY,
    DEFAULT_SENSITIVITY,
    evaluate_primary_confirmation,
)


class ProtocolV6CompactConfirmationTest(unittest.TestCase):
    def make_artifacts(self, root: Path, *, overlapping_seeds: bool = False):
        selection_manifest = root / "selection_manifest.json"
        selection_manifest.write_text(json.dumps({
            "stage": "exploratory",
            "independent_confirmation": False,
            "train_seeds": [1, 2],
            "eval_seeds": [11, 13],
        }))
        selection_sha = hashlib.sha256(
            selection_manifest.read_bytes()).hexdigest()
        selection_gate = root / "selection_gate.json"
        selection_gate.write_text(json.dumps({
            "gate_version": "freqduet-v6-incremental-selection-v3",
            "audit_stage": "exploratory",
            "status": "ambiguous_multiple_passes",
            "selected_candidate": None,
            "passing_candidates": [DEFAULT_PRIMARY, DEFAULT_SENSITIVITY],
            "matched_context": DEFAULT_MATCHED_CONTEXT,
            "input_artifacts": {
                "manifest": {
                    "path": str(selection_manifest),
                    "sha256": selection_sha,
                },
            },
            "matrix_provenance": {
                "source_fingerprint_sha256": "a" * 64,
                "scenario_contract_sha256": "b" * 64,
            },
        }))
        aggregate = root / "confirmation"
        aggregate.mkdir()
        (aggregate / "matrix_manifest.json").write_text(json.dumps({
            "stage": "confirmation",
            "independent_confirmation": True,
            "train_seeds": [2 if overlapping_seeds else 3, 5],
            "eval_seeds": [17, 19],
            "run_source_fingerprint": {"sha256": "a" * 64},
            "scenario_contract": {"sha256": "b" * 64},
            "run_git_provenance": {
                "commit": "c" * 40,
                "tracked_dirty": False,
            },
        }))
        return aggregate, selection_gate

    def test_primary_pass_controls_claim_and_sensitivity_is_descriptive(self):
        with TemporaryDirectory() as tmp:
            aggregate, selection = self.make_artifacts(Path(tmp))
            with patch(
                "scripts.audit_protocol_v6_compact_confirmation."
                "evaluate_selection",
                side_effect=[
                    {"status": "unique_pass"},
                    {"status": "no_pass"},
                ],
            ) as evaluate:
                result = evaluate_primary_confirmation(
                    aggregate, selection_gate_path=selection)

        self.assertEqual(result["status"], "primary_confirmed")
        self.assertTrue(result["primary_claim_eligible"])
        self.assertFalse(result["sensitivity_confirmed"])
        self.assertFalse(result["sensitivity_can_rescue_primary"])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(
            evaluate.call_args_list[0].kwargs["expected_stage"],
            "confirmation",
        )

    def test_sensitivity_pass_cannot_rescue_failed_primary(self):
        with TemporaryDirectory() as tmp:
            aggregate, selection = self.make_artifacts(Path(tmp))
            with patch(
                "scripts.audit_protocol_v6_compact_confirmation."
                "evaluate_selection",
                side_effect=[
                    {"status": "no_pass"},
                    {"status": "unique_pass"},
                ],
            ):
                result = evaluate_primary_confirmation(
                    aggregate, selection_gate_path=selection)

        self.assertEqual(result["status"], "primary_not_confirmed")
        self.assertFalse(result["primary_claim_eligible"])
        self.assertTrue(result["sensitivity_confirmed"])

    def test_overlapping_train_seeds_fail_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, selection = self.make_artifacts(
                Path(tmp), overlapping_seeds=True)
            with self.assertRaisesRegex(ValueError, "lineage checks failed"):
                evaluate_primary_confirmation(
                    aggregate, selection_gate_path=selection)


if __name__ == "__main__":
    unittest.main()
