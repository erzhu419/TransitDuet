import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.manuscript_figures import build_figures


class ManuscriptFiguresTest(unittest.TestCase):
    def test_build_figures_from_committed_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "figures"
            payload = build_figures(Path("transit_hrl/results"), out)

            self.assertEqual(payload["summary"]["figures"], 5)
            figures_dir = out / "figures"
            expected = [
                "fig1_frequency_separated_protocol",
                "fig2_claim_ablation_matrix",
                "fig3_transit_promotion_real_demand",
                "fig4_external_transit_data_coverage",
                "fig5_orderbook_encoder_replay",
            ]
            for stem in expected:
                for suffix in (".svg", ".pdf", ".png", ".tiff"):
                    path = figures_dir / f"{stem}{suffix}"
                    self.assertTrue(path.exists(), f"missing {path}")
                    self.assertGreater(path.stat().st_size, 1000, f"empty {path}")
            source_data = list((out / "source_data").glob("*.csv"))
            self.assertGreaterEqual(len(source_data), 5)


if __name__ == "__main__":
    unittest.main()
