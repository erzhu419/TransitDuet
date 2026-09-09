import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.build_freqduet_protocol_v6_evidence_package import PAPER_CONTROLLER
from scripts.make_freqduet_protocol_v6_result_figures import build_figures


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_rows(phase: str, n_pairs: int) -> list[dict[str, object]]:
    return [
        {
            "phase": phase,
            "paper_controller": PAPER_CONTROLLER,
            "metric": metric,
            "delta_candidate_minus_reference": mean,
            "ci95_low": low,
            "ci95_high": high,
            "n_pairs": n_pairs,
        }
        for metric, mean, low, high in (
            ("passenger_journey_min", -0.3, -0.8, 0.2),
            ("headway_cv", -0.02, -0.04, -0.005),
        )
    ]


def external_rows() -> list[dict[str, object]]:
    rows = []
    values = {
        "passenger_journey_min": (2.5, -3.2, -26.5),
        "headway_cv": (-0.22, -0.07, 0.006),
        "restricted_service_cost": (-0.12, -0.35, -2.54),
    }
    for metric, metric_values in values.items():
        for baseline, mean in zip(
            ("fixed_headway", "rule_holding", "rule_mpc"), metric_values
        ):
            rows.append({
                "paper_controller": PAPER_CONTROLLER,
                "baseline": baseline,
                "metric": metric,
                "delta_learned_minus_baseline": mean,
                "ci95_low": mean - 0.2,
                "ci95_high": mean + 0.2,
                "n_pairs": 64,
            })
    return rows


class ProtocolV6ResultFiguresTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.package = Path(self.temp.name) / "package"
        tables = self.package / "tables"
        write_csv(
            tables / "table1_v8_confirmation.csv",
            paired_rows("v8_independent_confirmation_ep40", 24),
        )
        write_csv(
            tables / "table2_v9_longtrain.csv",
            paired_rows("v9_independent_longtrain_ep200", 64),
        )
        write_csv(tables / "table3_v9_external_baselines.csv", external_rows())
        write_csv(tables / "table4_evidence_decisions.csv", [
            {
                "phase": "v8_independent_confirmation_ep40",
                "decision": "primary_confirmed",
            },
            {
                "phase": "v9_independent_longtrain_ep200",
                "decision": "longtrain_not_confirmed",
            },
        ])
        (self.package / "evidence_status.json").write_text(json.dumps({
            "package_version": "test-v1",
            "protocol": "freqduet-eval-v6",
            "paper_controller": PAPER_CONTROLLER,
            "submission_ready": False,
        }) + "\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_editable_and_review_figure_exports(self) -> None:
        manifest = build_figures(self.package, ("svg", "png"))

        self.assertFalse(manifest["submission_ready"])
        figures = self.package / "figures"
        for stem in (
            "fig2_protocol_v6_confirmation_robustness",
            "fig3_protocol_v6_external_tradeoff",
        ):
            self.assertGreater((figures / f"{stem}.png").stat().st_size, 1000)
            svg = (figures / f"{stem}.svg").read_text()
            self.assertIn("<text", svg)
        package_manifest = json.loads(
            (self.package / "package_manifest.json").read_text()
        )
        self.assertIn(
            "figures/figure_manifest.json", package_manifest["files"]
        )
        captions = (figures / "captions.md").read_text()
        self.assertIn("Both configurations disable the legacy", captions)
        self.assertIn("combined-policy comparison", captions)

    def test_rejects_hidden_v9_failure(self) -> None:
        decisions = self.package / "tables" / "table4_evidence_decisions.csv"
        write_csv(decisions, [
            {
                "phase": "v8_independent_confirmation_ep40",
                "decision": "primary_confirmed",
            },
            {
                "phase": "v9_independent_longtrain_ep200",
                "decision": "primary_confirmed",
            },
        ])

        with self.assertRaisesRegex(ValueError, "longtrain_not_confirmed"):
            build_figures(self.package, ("svg",))


if __name__ == "__main__":
    unittest.main()
