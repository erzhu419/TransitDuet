import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.current_manuscript_audit import (
    DEFAULT_BIBLIOGRAPHY,
    DEFAULT_MANUSCRIPT,
    DEFAULT_READINESS,
    DEFAULT_REGISTRY,
    REPOSITORY_ROOT,
    audit_current_manuscript,
)


class CurrentManuscriptAuditTest(unittest.TestCase):
    def _copy_inputs(self, destination: Path) -> None:
        for relative in (
            DEFAULT_MANUSCRIPT,
            DEFAULT_READINESS,
            DEFAULT_BIBLIOGRAPHY,
            DEFAULT_REGISTRY,
        ):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((REPOSITORY_ROOT / relative).read_bytes())

    def test_current_authoritative_manuscript_passes(self):
        result = audit_current_manuscript()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(
            result["registry_counts"],
            {
                "total": 47,
                "reportable": 4,
                "positive": 2,
                "development": 41,
                "legacy": 2,
            },
        )
        self.assertEqual(len(result["reportable_evidence_ids"]), 4)

    def test_missing_bibliography_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            manuscript = root / DEFAULT_MANUSCRIPT
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8") + "\n[@missing_reference]\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing from bibliography"):
                audit_current_manuscript(repository_root=root)

    def test_legacy_evidence_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            manuscript = root / DEFAULT_MANUSCRIPT
            manuscript.write_text(
                manuscript.read_text(encoding="utf-8")
                + "\nSource: top_journal_unified_matrix_latest\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "legacy or stale evidence"):
                audit_current_manuscript(repository_root=root)

    def test_registry_count_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            registry_path = root / DEFAULT_REGISTRY
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            registry["records"].pop()
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registry count"):
                audit_current_manuscript(repository_root=root)

    def test_duplicate_quant_table_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._copy_inputs(root)
            manuscript = root / DEFAULT_MANUSCRIPT
            text = manuscript.read_text(encoding="utf-8")
            duplicate = (
                "| generic HRL-PPO | lower-LF drift | -0.000006 | "
                "0.65460 | inconclusive |\n"
            )
            text = text.replace(duplicate, duplicate + duplicate, 1)
            manuscript.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "12 contrasts"):
                audit_current_manuscript(repository_root=root)


if __name__ == "__main__":
    unittest.main()
