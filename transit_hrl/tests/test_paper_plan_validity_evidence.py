"""Checks for source-derived paper tables; no environment imports or rollouts."""

import importlib.util
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_evidence", ROOT / "paper/build_plan_validity_evidence.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class PaperEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = REPORT.build()

    def test_pointmaze_intervals_match_original_confirmation(self):
        expected = {
            "fixed_minus_candidate_ise": (.505541, .408443, .602640),
            "jitter_minus_candidate_ise": (.608186, .570111, .646260),
            "current_minus_candidate_ise": (.260773, .152216, .369331),
            "candidate_minus_fixed_return": (40.662456, 31.548406, 49.776506),
            "controller_learning_ise": (20.862987, 15.939428, 25.786547),
        }
        paper = (ROOT / "paper/manuscript.md").read_text()
        for name, values in expected.items():
            actual = self.evidence["pointmaze_stage9_confirmation"]["contrasts"][name]
            for got, want in zip((actual["mean"], *actual["CI95"]), values):
                self.assertAlmostEqual(got, want, places=6)
            self.assertIn(f"{actual['mean']:.6f} | [{actual['CI95'][0]:.6f}, {actual['CI95'][1]:.6f}]", paper)

    def test_units_and_failed_closed_loop_decision_are_preserved(self):
        pm = self.evidence["pointmaze_stage9_confirmation"]
        self.assertEqual((pm["root_count"], pm["evaluation_paths"], pm["paired_mode_rollouts"]), (8, 128, 512))
        self.assertEqual(pm["upper_calls_per_episode_all_modes"], 24)
        self.assertEqual(self.evidence["new_training_or_simulation"], 0)
        native = self.evidence["transit_stage304_two_decisions"]
        self.assertEqual(native["independent_scene_units"], 320)
        for key in ("service_closed_loop_supported", "joint_closed_loop_supported", "critic_incremental_cost_supported"):
            self.assertFalse(native[key])
        pair = native["comparisons"]["all32_scenes"]["service_only_vs_raw"]
        self.assertGreater(pair["episode_bootstrap_CI95"]["cost"][1], 0)
        self.assertGreater(pair["paired_normal_CI95"]["wait_min"][1], 0)

    def test_transit_table_rows_match_saved_reports(self):
        text = (ROOT / "paper/manuscript.md").read_text()
        rows = [
            ("transit_stage298_single_intervention", "joint_budgeted_vs_raw", "cost"),
            ("transit_stage298_single_intervention", "joint_budgeted_vs_raw", "wait_min"),
            ("transit_stage301_single_intervention", "service_only_vs_raw", "cost"),
            ("transit_stage301_single_intervention", "service_only_vs_raw", "wait_min"),
            ("transit_stage301_single_intervention", "joint_budgeted_vs_service_only", "cost"),
            ("transit_stage301_single_intervention", "joint_budgeted_vs_service_only", "wait_min"),
        ]
        for study, name, field in rows:
            row = self.evidence[study]["comparisons"][name]
            values = (row["mean"][field], *row["prefix_CI95"][field], *row["conditional_MC_CI95"][field])
            m, low, high, low2, high2 = (v*1e6 for v in values)
            self.assertIn(f"{m:.3f} | [{low:.3f}, {high:.3f}] | [{low2:.3f}, {high2:.3f}]", text)
        current = self.evidence["transit_stage304_two_decisions"]["comparisons"]["all32_scenes"]
        for name, field in (("service_only_vs_raw", "cost"), ("service_only_vs_raw", "service"),
                            ("service_only_vs_raw", "wait_min"), ("joint_budgeted_vs_service_only", "cost")):
            row = current[name]
            values = (row["mean"][field], *row["episode_bootstrap_CI95"][field], *row["paired_normal_CI95"][field])
            m, low, high, low2, high2 = (v*1e6 for v in values)
            self.assertIn(f"{m:.3f} | [{low:.3f}, {high:.3f}] | [{low2:.3f}, {high2:.3f}]", text)

    def test_active_links_and_citations_resolve(self):
        paper = ROOT / "paper"
        for name in ("manuscript.md", "evidence_map.md", "submission_readiness.md"):
            text = (paper / name).read_text()
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((paper / target).is_file(), (name, target))
        cited = set(re.findall(r"@([A-Za-z0-9_]+)", (paper / "manuscript.md").read_text()))
        defined = set(re.findall(r"@\w+\{([^,]+),", (paper / "references.bib").read_text()))
        self.assertFalse(cited-defined)


if __name__ == "__main__":
    unittest.main()
