import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.assemble_freqduet_protocol_v6_manuscript import (
    EXPECTED_METHOD,
    METRICS,
    PAPER_CONTROLLER,
    REFERENCE,
    SOURCE_CANDIDATE,
    build_manuscript,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pair_rows(phase: str, candidate: str, pairs: int) -> list[dict[str, object]]:
    values = {
        "passenger_journey_min": -0.25 if pairs == 24 else -1.24,
        "passenger_wait_min": -0.18 if pairs == 24 else -1.01,
        "in_vehicle_min": -0.08 if pairs == 24 else -0.23,
        "headway_cv": -0.022 if pairs == 24 else -0.009,
        "unserved_rate": 0.0002,
        "holding_s_per_trip": -1.7 if pairs == 24 else -37.6,
        "denied_trip_rate": 0.0048 if pairs == 24 else -0.0165,
        "restricted_service_cost": -0.04 if pairs == 24 else -0.11,
    }
    rows = []
    for metric, *_ in METRICS:
        delta = values[metric]
        rows.append({
            "phase": phase,
            "paper_controller": PAPER_CONTROLLER,
            "source_candidate": candidate,
            "reference": REFERENCE,
            "metric": metric,
            "delta_candidate_minus_reference": delta,
            "ci95_low": delta - abs(delta) * 0.5 - 0.01,
            "ci95_high": delta + abs(delta) * 0.5 + 0.01,
            "paired_signflip_p": 0.03125,
            "paired_signflip_p_holm": 0.125,
            "n_pairs": pairs,
        })
    return rows


def external_rows() -> list[dict[str, object]]:
    deltas = {
        "fixed_headway": {
            "passenger_journey_min": 2.47,
            "passenger_wait_min": 1.01,
            "in_vehicle_min": 1.46,
            "headway_cv": -0.219,
            "unserved_rate": -0.0007,
            "holding_s_per_trip": 286.8,
            "denied_trip_rate": 0.641,
            "restricted_service_cost": -0.122,
        },
        "rule_holding": {metric: -0.1 for metric, *_ in METRICS},
        "rule_mpc": {metric: -0.2 for metric, *_ in METRICS},
    }
    deltas["rule_holding"]["passenger_journey_min"] = -3.224
    deltas["rule_mpc"]["passenger_journey_min"] = -26.511
    rows = []
    for baseline, metric_values in deltas.items():
        for metric, *_ in METRICS:
            delta = metric_values[metric]
            rows.append({
                "paper_controller": PAPER_CONTROLLER,
                "baseline": baseline,
                "metric": metric,
                "learned_mean": 1.0,
                "baseline_mean": 1.0 - delta,
                "delta_learned_minus_baseline": delta,
                "ci95_low": delta - 0.1,
                "ci95_high": delta + 0.1,
                "paired_signflip_p": 0.01,
                "paired_signflip_p_holm": 0.03,
                "n_pairs": 64,
            })
    return rows


class ProtocolV6ManuscriptAssemblyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.package = self.root / "package"
        self.out = self.root / "paper"

        write_json(self.package / "evidence_status.json", {
            "protocol": "freqduet-eval-v6",
            "paper_controller": PAPER_CONTROLLER,
            "v8_confirmation_status": "unique_pass",
            "v9_longtrain_status": "longtrain_not_confirmed",
            "submission_ready": False,
            "submission_blocker": "v9_longtrain_not_confirmed",
        })
        method = dict(EXPECTED_METHOD)
        method.update({
            "action_bins_s": [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0],
            "lower_context_features": [
                "load",
                "capacity",
                "queue",
                "speed_residual",
                "shock_age",
                "schedule_slack",
                "regularity_hold_target_norm",
                "regularity_hold_target_valid",
            ],
            "config_lineage": ["config_v2.yaml", "configs_freqduet/current.yaml"],
        })
        write_json(
            self.package / "figures" / "source_data" / "figure1_method_contract.json",
            method,
        )
        write_csv(
            self.package / "tables" / "table1_v8_confirmation.csv",
            pair_rows(
                "v8_independent_confirmation_ep40", SOURCE_CANDIDATE, 24
            ),
        )
        write_csv(
            self.package / "tables" / "table2_v9_longtrain.csv",
            pair_rows(
                "v9_independent_longtrain_ep200", PAPER_CONTROLLER, 64
            ),
        )
        write_csv(
            self.package / "tables" / "table3_v9_external_baselines.csv",
            external_rows(),
        )
        write_csv(
            self.package / "tables" / "table4_evidence_decisions.csv",
            [
                {
                    "phase": "v8_independent_confirmation_ep40",
                    "decision": "primary_confirmed",
                    "claim_eligible": True,
                    "controller": PAPER_CONTROLLER,
                    "train_seeds": 6,
                    "evaluation_seeds": 4,
                    "paired_rollouts": 24,
                },
                {
                    "phase": "v9_independent_longtrain_ep200",
                    "decision": "longtrain_not_confirmed",
                    "claim_eligible": False,
                    "controller": PAPER_CONTROLLER,
                    "train_seeds": 8,
                    "evaluation_seeds": 8,
                    "paired_rollouts": 64,
                },
            ],
        )
        write_csv(
            self.package / "source_artifacts" / "v9" / "frozen_summary.csv",
            [
                {
                    "config": PAPER_CONTROLLER,
                    "trip_launch_rate_mean": 1.0,
                    "trip_completion_rate_mean": 1.0,
                }
            ],
        )
        figures = self.package / "figures"
        for stem in (
            "fig1_protocol_v6_method",
            "fig2_protocol_v6_confirmation_robustness",
            "fig3_protocol_v6_external_tradeoff",
            "fig4_protocol_v6_physical_outcomes",
            "fig5_protocol_v6_external_realism",
        ):
            (figures / f"{stem}.png").write_bytes(b"png")
            (figures / f"{stem}.pdf").write_bytes(b"%PDF-1.4\n")
        (figures / "captions.md").write_text(
            "# Result captions\n\n## Figure 2 | Confirmation\n\nFigure two.\n\n"
            "## Figure 3 | External\n\nFigure three.\n"
        )
        (figures / "supporting_captions.md").write_text(
            "# Supporting captions\n\n## Figure 1 | Method\n\nFigure one.\n\n"
            "## Figure 4 | Physical\n\nFigure four.\n\n"
            "## Figure 5 | Realism\n\nFigure five.\n"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_builds_claim_bounded_manuscript(self) -> None:
        manifest = build_manuscript(self.package, self.out)

        self.assertFalse(manifest["submission_ready"])
        methods = (self.out / "methods.md").read_text()
        results = (self.out / "results.md").read_text()
        discussion = (self.out / "discussion.md").read_text()
        manuscript = (self.out / "manuscript.md").read_text()
        supplement = (self.out / "supplementary.md").read_text()
        self.assertIn(
            "| V9 (200 ep) | 8 x 8 | 64 | Not confirmed | No |",
            supplement,
        )
        methods_flat = " ".join(methods.split())
        manuscript_flat = " ".join(manuscript.split())
        self.assertIn("promotion, leakage-penalty", methods)
        self.assertIn(r"C_R &= \frac{W_R}{10}", methods)
        self.assertIn("does not directly charge holding", methods)
        self.assertIn("[@haarnoja2018soft]", methods)
        self.assertIn("22 physical stops", methods_flat)
        self.assertIn("Each direction is 10.5 km long", methods_flat)
        self.assertIn("20-origin by 14-hour by 20-destination", methods_flat)
        self.assertIn("policy-independent scenario tape", methods_flat)
        self.assertIn("reverse-direction OD intensities at X13--X15 by 0.4", methods_flat)
        self.assertIn("## External comparators", methods)
        self.assertIn("transparent low-fidelity comparator", methods_flat)
        self.assertIn("Both upper and lower actors are deterministic", methods_flat)
        self.assertIn("passed the preregistered V8 gate", methods_flat)
        self.assertIn("longtrain_not_confirmed", results)
        self.assertIn("gate-positive under its preregistered criteria", results)
        self.assertIn("not a familywise-significant effect", results)
        self.assertIn("do not establish that this interface caused", discussion)
        self.assertIn("V9 did not confirm", discussion)
        self.assertIn("Holm-adjusted $p=0.047$", discussion)
        self.assertIn("sign-flip value is not used", discussion)
        self.assertIn("does not establish superiority", discussion)
        self.assertIn("was not re-estimated from the external data", discussion)
        self.assertIn("higher denied-trip rate", results)
        self.assertIn("cannot be interpreted as", results)
        self.assertIn("does not contain a same-stage NoFreq", supplement)
        self.assertIn("## Abstract", manuscript)
        headings = [
            "# Introduction",
            "# Related Work",
            "# Methods",
            "# Results",
            "# Discussion",
            "# Conclusions",
            "# Data and Code Availability",
        ]
        positions = [manuscript.index(heading) for heading in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("Holm-adjusted training-seed sign-flip", manuscript_flat)
        self.assertIn("does not establish a familywise-significant effect", manuscript_flat)
        self.assertIn(
            "figures/fig1_protocol_v6_method.png",
            methods,
        )
        self.assertLess(
            results.index("{#fig:protocol-v6-3}"),
            results.index("{#fig:protocol-v6-4}"),
        )
        captions = (self.out / "figure_captions.md").read_text()
        positions = [captions.index(f"## Figure {index} ") for index in range(1, 6)]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(
            (self.out / "figures" / "fig5_protocol_v6_external_realism.png").is_file()
        )
        self.assertIn(
            "figures/fig5_protocol_v6_external_realism.png",
            manifest["outputs"],
        )
        self.assertTrue((self.out / "tables" / "table1_confirmation_and_robustness.tex").is_file())
        self.assertTrue((self.out / "assembly_manifest.json").is_file())
        self.assertTrue((self.out / "references.bib").is_file())
        self.assertIn(
            "not same-day AFC/APC/AVL",
            (self.out / "availability.md").read_text(),
        )
        self.assertTrue((self.out / "literature_verification.md").is_file())
        self.assertEqual(
            manifest["target_journal"],
            "Transportation Research Part C: Emerging Technologies",
        )
        submission = self.out / "trc_submission"
        self.assertIn(
            "fig1_protocol_v6_method.pdf",
            (submission / "manuscript_body.md").read_text(),
        )
        submission_body = (submission / "manuscript_body.md").read_text()
        self.assertIn("![Method. Figure one.]", submission_body)
        self.assertNotIn("![Figure 1.", submission_body)
        self.assertNotIn(
            "figures/fig1_protocol_v6_method.png",
            (submission / "manuscript_body.md").read_text(),
        )
        self.assertIn("long-training gate", (submission / "README.md").read_text())
        self.assertTrue((submission / "elsarticle-template.tex").is_file())
        self.assertTrue((submission / "supplementary-template.tex").is_file())
        self.assertIn(
            "## S1. Frozen evidence and decision ledger",
            (submission / "supplementary_body.md").read_text(),
        )
        self.assertIn(
            "supplementary.pdf",
            (submission / "build.sh").read_text(),
        )
        self.assertTrue((submission / "build.sh").stat().st_mode & 0o111)
        self.assertIn(
            "passed a preregistered short-training regularity gate",
            (submission / "highlights.txt").read_text(),
        )
        readme = (self.out / "README.md").read_text()
        self.assertIn("submission_ready: false", readme)
        self.assertIn("Holm-adjusted training-seed sign-flip result is $p=0.125$", readme)
        self.assertEqual(
            manifest["assembly_version"],
            "freqduet-protocol-v6-manuscript-v4",
        )

    def test_rejects_missing_v9_failure(self) -> None:
        status = json.loads((self.package / "evidence_status.json").read_text())
        status["v9_longtrain_status"] = "confirmed"
        write_json(self.package / "evidence_status.json", status)

        with self.assertRaisesRegex(ValueError, "V9 negative result is missing"):
            build_manuscript(self.package, self.out)

    def test_rejects_method_switch_drift(self) -> None:
        path = self.package / "figures" / "source_data" / "figure1_method_contract.json"
        method = json.loads(path.read_text())
        method["promotion_enabled"] = True
        write_json(path, method)

        with self.assertRaisesRegex(ValueError, "promotion_enabled"):
            build_manuscript(self.package, self.out)

    def test_rejects_external_controller_mix(self) -> None:
        path = self.package / "tables" / "table3_v9_external_baselines.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["paper_controller"] = "F_freqhrl_unrelated_controller"
        write_csv(path, rows)

        with self.assertRaisesRegex(ValueError, "external paper controller mismatch"):
            build_manuscript(self.package, self.out)


if __name__ == "__main__":
    unittest.main()
