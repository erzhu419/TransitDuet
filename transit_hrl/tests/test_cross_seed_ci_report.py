import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.cross_seed_ci_report import run_cross_seed_ci_report


class CrossSeedCIReportTest(unittest.TestCase):
    def test_cross_seed_ci_report_marks_paper_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paper = root / "paper.json"
            paper.write_text(json.dumps({
                "statistical_checks": [{
                    "check": "x",
                    "claim": "demo",
                    "metric": "m",
                    "status": "supported",
                    "direction": "increase",
                    "n_common": 12,
                    "delta_mean": 1.0,
                    "delta_ci95_low": 0.5,
                    "delta_ci95_high": 1.5,
                    "improvement_ci95_low": 0.5,
                    "win_rate": 0.9,
                    "sign_p_value": 0.2,
                }]
            }), encoding="utf-8")
            payload = run_cross_seed_ci_report(root / "out", paper, min_pairs=5)
            self.assertEqual(payload["summary"]["n_checks"], 1)
            self.assertEqual(payload["summary"]["n_paper_ready"], 1)
            self.assertTrue((root / "out" / "cross_seed_ci.csv").exists())


if __name__ == "__main__":
    unittest.main()
