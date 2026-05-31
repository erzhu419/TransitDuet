import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.paper_diagnostics import build_claim_matrix, write_report


class PaperDiagnosticsTest(unittest.TestCase):
    def test_claim_matrix_builds_with_missing_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = build_claim_matrix(root / "results", root / "transit")
            self.assertGreaterEqual(len(claims), 1)
            self.assertIn("claim", claims[0])
            write_report(root / "report.md", claims)
            self.assertTrue((root / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
