import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.theory_report import run_theory_report


class TheoryReportTest(unittest.TestCase):
    def test_theory_report_writes_theorems_and_stat_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper.json"
            paper.write_text(json.dumps({
                "statistical_checks": [
                    {"status": "supported", "n_common": 5},
                    {"status": "not_supported", "n_common": 3},
                ]
            }), encoding="utf-8")
            payload = run_theory_report(root / "out", paper)
            self.assertGreaterEqual(len(payload["theorems"]), 4)
            self.assertEqual(payload["statistical_coverage"]["n_checks"], 2)
            self.assertTrue((root / "out" / "theory_report.md").exists())


if __name__ == "__main__":
    unittest.main()
