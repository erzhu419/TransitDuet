import csv
import json
import tempfile
import unittest
from pathlib import Path

from freq_hrl.experiments.authoritative_manuscript_figures import (
    DEVELOPMENT_EVIDENCE_IDS,
    LEGACY_TOKENS,
    REPORTABLE_EVIDENCE_IDS,
    build_authoritative_manuscript_figures,
    build_source_rows,
    load_figure_evidence,
    validate_figure_boundary,
)


class AuthoritativeManuscriptFiguresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, cls.records = load_figure_evidence()
        cls.sources = build_source_rows(cls.records)

    def test_main_quantitative_sources_are_reportable_only(self):
        for name in (
            "fig2_mujoco_confirmatory_source.csv",
            "fig3_quant_contrasts_source.csv",
        ):
            rows = self.sources[name]
            self.assertTrue(rows)
            self.assertTrue(
                {row["evidence_id"] for row in rows}
                <= REPORTABLE_EVIDENCE_IDS
            )
            self.assertEqual(
                {row["evidence_stage"] for row in rows}, {"confirmatory"}
            )
            self.assertEqual(
                {row["manuscript_reportable"] for row in rows}, {"true"}
            )

    def test_development_stop_map_is_explicitly_nonreportable(self):
        rows = self.sources["fig_s1_development_stop_map_source.csv"]
        self.assertEqual(
            [row["evidence_id"] for row in rows],
            list(DEVELOPMENT_EVIDENCE_IDS),
        )
        self.assertEqual({row["evidence_stage"] for row in rows}, {"development"})
        self.assertEqual({row["paper_disposition"] for row in rows}, {"development_only"})
        self.assertEqual({row["manuscript_reportable"] for row in rows}, {"false"})
        self.assertEqual(
            {row["fresh_validation_paths_accessed"] for row in rows}, {"false"}
        )

    def test_reportable_membership_change_is_rejected(self):
        shortened = [
            row
            for row in self.records
            if row["evidence_id"] != "mujoco_v12_responsibility_confirmatory"
        ]
        with self.assertRaisesRegex(ValueError, "reportable evidence membership"):
            validate_figure_boundary(shortened)

    def test_package_exports_and_summary_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "figures"
            summary = build_authoritative_manuscript_figures(output_dir=output)
            self.assertEqual(
                set(summary["reportable_evidence_ids"]),
                REPORTABLE_EVIDENCE_IDS,
            )
            self.assertEqual(
                summary["development_evidence_ids"],
                list(DEVELOPMENT_EVIDENCE_IDS),
            )
            for exports in summary["figures"].values():
                self.assertEqual({Path(path).suffix for path in exports}, {".svg", ".pdf", ".png"})
                for relative in exports:
                    self.assertGreater((output / relative).stat().st_size, 1000)
            svg = (output / "figures/fig1_protocol_and_estimands.svg").read_text(encoding="utf-8")
            self.assertIn("<text", svg)
            summary_text = (output / "summary.json").read_text(encoding="utf-8")
            self.assertFalse(any(token in summary_text for token in LEGACY_TOKENS))
            with (output / "source_data/fig3_quant_contrasts_source.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                quant_rows = list(csv.DictReader(handle))
            self.assertEqual(len(quant_rows), 12)
            self.assertEqual(
                {row["status"] for row in quant_rows},
                {"supported_improvement", "supported_harm", "inconclusive"},
            )
            parsed = json.loads(summary_text)
            self.assertEqual(parsed["main_figure_evidence_policy"], "manuscript_reportable_only")


if __name__ == "__main__":
    unittest.main()
