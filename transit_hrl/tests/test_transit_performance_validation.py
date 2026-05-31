import tempfile
import unittest
from pathlib import Path

from freq_transitduet.scripts.transit_performance_validation import (
    aggregate,
    metric_winners,
    paired_rows,
    read_rows,
    write_report,
)


class TransitPerformanceValidationTest(unittest.TestCase):
    def test_paired_validation_respects_metric_directions(self):
        rows = [
            {
                "config": "T_freqhrl_terminal",
                "seed": 1,
                "composite": 1.0,
                "wait": 5.0,
                "demand_attr_score": 0.30,
            },
            {
                "config": "T_freqhrl_terminal",
                "seed": 2,
                "composite": 1.2,
                "wait": 6.0,
                "demand_attr_score": 0.40,
            },
            {
                "config": "T_baseline",
                "seed": 1,
                "composite": 1.4,
                "wait": 7.0,
                "demand_attr_score": 0.10,
            },
            {
                "config": "T_baseline",
                "seed": 2,
                "composite": 1.6,
                "wait": 8.0,
                "demand_attr_score": 0.20,
            },
        ]
        metrics = ["composite", "wait", "demand_attr_score"]

        summary = aggregate(rows, metrics)
        paired = paired_rows(
            rows,
            target="T_freqhrl_terminal",
            metrics=metrics,
            n_boot=100,
            seed=7,
        )
        winners = {
            row["metric"]: row["winner"]
            for row in metric_winners(summary, metrics)
        }

        self.assertEqual(winners["composite"], "T_freqhrl_terminal")
        self.assertEqual(winners["wait"], "T_freqhrl_terminal")
        self.assertEqual(winners["demand_attr_score"], "T_freqhrl_terminal")
        self.assertAlmostEqual(paired[0]["composite_delta_mean"], -0.4)
        self.assertEqual(paired[0]["composite_win_rate"], 1.0)
        self.assertAlmostEqual(paired[0]["demand_attr_score_delta_mean"], 0.2)
        self.assertEqual(paired[0]["demand_attr_score_win_rate"], 1.0)

    def test_report_roundtrip_from_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "per_seed.csv"
            path.write_text(
                "config,seed,logs_dir,composite,wait,cv,overshoot\n"
                "T_freqhrl_terminal,1,logs/a,1.0,5.0,0.40,1.0\n"
                "T_freqhrl_terminal,2,logs/b,1.2,6.0,0.42,1.2\n"
                "T_baseline,1,logs/c,1.4,7.0,0.50,1.4\n"
                "T_baseline,2,logs/d,1.6,8.0,0.52,1.6\n",
                encoding="utf-8",
            )
            metrics = ["composite", "wait", "cv", "overshoot"]
            rows = read_rows(path)
            summary = aggregate(rows, metrics)
            paired = paired_rows(
                rows,
                target="T_freqhrl_terminal",
                metrics=metrics,
                n_boot=100,
                seed=7,
            )
            winners = metric_winners(summary, metrics)
            report = Path(tmp) / "report.md"

            write_report(
                report,
                summary=summary,
                paired=paired,
                winners=winners,
                target="T_freqhrl_terminal",
                metrics=metrics,
                source=path,
            )

            text = report.read_text(encoding="utf-8")
            self.assertIn("Composite winner: `T_freqhrl_terminal`", text)
            self.assertIn("| T_baseline | -0.400", text)


if __name__ == "__main__":
    unittest.main()
