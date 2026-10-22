"""Check figure source rows, scientific units and vector text without rollouts."""

import csv
import json
from pathlib import Path
import subprocess
import unittest
import xml.etree.ElementTree as ET

PAPER = Path(__file__).resolve().parents[1] / "paper"
STEMS = ("fig1_budgeted_replanning", "fig2_pointmaze_confirmation", "fig3_transit_boundary")


class FigureEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = json.loads((PAPER / "plan_validity_evidence.json").read_text())

    def rows(self, stem):
        with (PAPER / "figures" / f"{stem}_source.csv").open() as stream:
            return list(csv.DictReader(stream))

    def resolve(self, source):
        filename, pointer = source.split("#", 1)
        self.assertEqual(filename, "plan_validity_evidence.json")
        value = self.evidence
        for key in pointer.strip("/").split("/"):
            value = value[key]
        return value

    def test_all_quantitative_csv_values_and_intervals_match_evidence(self):
        for stem in STEMS[1:]:
            for row in self.rows(stem):
                source = self.resolve(row["source"])
                kind, metric = row["kind"], row["metric"]
                if kind == "root_t_CI95":
                    value, interval = source["mean"], source["CI95"]
                elif kind.endswith("CI95"):
                    value, interval = source["mean"][metric], source[kind][metric]
                elif kind == "mean_operation_count":
                    value = (source["upper_calls_per_episode_all_modes"] if metric == "upper_calls"
                             else source["score_checks_by_mode"][row["comparison"]])
                    interval = None
                else:
                    value, interval = source, None
                self.assertAlmostEqual(float(row["value_raw"]), value, places=12)
                if interval is not None:
                    self.assertAlmostEqual(float(row["ci_low_raw"]), interval[0], places=12)
                    self.assertAlmostEqual(float(row["ci_high_raw"]), interval[1], places=12)

    def test_timeline_is_saved_first_path_not_synthetic(self):
        pm = self.evidence["pointmaze_stage9_confirmation"]
        example = pm["example_schedule"]
        self.assertEqual(example["root"], min(map(int, pm["root_effects"])))
        rows = self.rows(STEMS[0])
        for mode, steps in example["decision_steps"].items():
            shown = [int(row["value_raw"]) for row in rows
                     if row["kind"] == "observed_call" and row["comparison"] == mode]
            self.assertEqual(shown, [step for step in steps if step < 300])
        for row in rows:
            if row["kind"] == "protocol":
                self.assertEqual(float(row["value_raw"]), pm["policy_contract"][row["metric"]])

    def test_transit_units_halves_and_interval_types(self):
        for row in self.rows(STEMS[2]):
            if not row["kind"].endswith("CI95"):
                continue
            self.assertEqual(float(row["display_scale"]), 60 if row["metric"] == "wait_min" else 1000)
            if row["kind"] in ("prefix_CI95", "conditional_MC_CI95"):
                self.assertEqual((int(row["n"]), row["statistical_unit"]), (400, "native_prefix"))
            else:
                self.assertEqual(row["statistical_unit"], "complete_native_scene_episode")
                self.assertEqual(int(row["n"]), 320 if row["part"] == "all32_scenes" or row["panel"] == "caption" else 160)

    def test_vectors_preserve_text_and_render_checks_pass(self):
        checks = json.loads((PAPER / "figures/render_checks.json").read_text())
        self.assertEqual(len(checks), 3)
        for stem, check in zip(STEMS, checks):
            self.assertEqual(check["width_mm"], 183)
            self.assertTrue(check["text_within_canvas"])
            self.assertGreater(check["nonwhite_fraction"], .01)
            svg = ET.parse(PAPER / "figures" / f"{stem}.svg")
            self.assertGreater(len(svg.findall(".//{http://www.w3.org/2000/svg}text")), 10)
            extracted = subprocess.run(["pdftotext", str(PAPER / "figures" / f"{stem}.pdf"), "-"],
                                       check=True, capture_output=True, text=True).stdout
            self.assertGreater(len(extracted.strip()), 100)


if __name__ == "__main__":
    unittest.main()
