"""Checks for source-derived paper tables; no environment imports or rollouts."""

from __future__ import annotations

import importlib.util
import json
from math import isclose
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("paper_evidence", ROOT / "paper/build_plan_validity_evidence.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class PaperEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = REPORT.build()

    def test_saved_artifact_matches_original_result_sources(self):
        saved = json.loads((ROOT / "paper/plan_validity_evidence.json").read_text())

        def compare(left, right, at=""):
            self.assertIs(type(left), type(right), at)
            if isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys(), at)
                for key in left:
                    compare(left[key], right[key], f"{at}/{key}")
            elif isinstance(left, list):
                self.assertEqual(len(left), len(right), at)
                for i, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{at}/{i}")
            elif isinstance(left, float):
                self.assertTrue(isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), (at, left, right))
            else:
                self.assertEqual(left, right, at)

        compare(saved, self.evidence)

    def test_pointmaze_intervals_match_original_confirmation(self):
        expected = {
            "fixed_minus_candidate_ise": (0.505541, 0.408443, 0.602640),
            "jitter_minus_candidate_ise": (0.608186, 0.570111, 0.646260),
            "current_minus_candidate_ise": (0.260773, 0.152216, 0.369331),
            "candidate_minus_fixed_return": (40.662456, 31.548406, 49.776506),
            "controller_learning_ise": (20.862987, 15.939428, 25.786547),
        }
        paper = (ROOT / "paper/manuscript.md").read_text()
        for name, values in expected.items():
            actual = self.evidence["pointmaze_stage9_confirmation"]["contrasts"][name]
            for got, want in zip((actual["mean"], *actual["CI95"]), values):
                self.assertAlmostEqual(got, want, places=6)
            self.assertIn(f"{actual['mean']:.6f} | [{actual['CI95'][0]:.6f}, {actual['CI95'][1]:.6f}]", paper)

    def test_unprivileged_and_pareto_tables_match_saved_summaries(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        unpriv = self.evidence["pointmaze_unprivileged_ablation"]
        for key in (
            "periodic_minus_unprivileged_uniform_ise",
            "jitter_minus_unprivileged_uniform_ise",
            "distance_minus_unprivileged_uniform_ise",
            "privileged_minus_unprivileged_uniform_ise",
            "distance_minus_unprivileged_observable_ise",
        ):
            row = unpriv["contrasts"][key]
            self.assertIn(
                f"{row['mean']:.5f} | [{row['CI95'][0]:.5f}, {row['CI95'][1]:.5f}]",
                paper,
            )
        pareto = self.evidence["pointmaze_budget_pareto"]["frontier"]
        for b in ("12", "15", "20", "24", "30", "40"):
            entry = pareto[b]
            gain = entry["distance_minus_unprivileged_ise"]
            self.assertGreater(gain["CI95"][0], 0.0)
            self.assertIn(
                f"{entry['mean_ise']['periodic']:.5f} | {entry['mean_ise']['distance_threshold']:.5f} | "
                f"{entry['mean_ise']['unprivileged_uniform']:.5f} | "
                f"{gain['mean']:.5f} [{gain['CI95'][0]:.5f}, {gain['CI95'][1]:.5f}]",
                paper,
            )

    def test_fewer_call_distance_contrasts_match_manuscript(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        comparisons = self.evidence["pointmaze_budget_pareto"]["cross_budget_call_savings"]
        for key in ("unprivileged_B20_vs_distance_B24", "unprivileged_B24_vs_distance_B30"):
            contrast = comparisons[key]
            low, high = contrast["CI95"]
            self.assertLess(high, 0)
            self.assertIn(f"`{contrast['mean']:.5f}` (95% CI `[{low:.5f}, {high:.5f}]`)", paper)

    def test_route_b_tables_match_saved_summaries(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        sb = self.evidence["pointmaze_strong_baselines"]
        for m_key, c_key in (
            ("native_periodic_hrl", "native_periodic_minus_ridge_ise"),
            ("shared_periodic", "shared_periodic_minus_ridge_ise"),
            ("temple_gate", "temple_minus_ridge_ise"),
            ("distance_threshold", "distance_minus_ridge_ise"),
            ("deliberation_option_critic", "deliberation_oc_minus_ridge_ise"),
            ("generic_mlp_unprivileged", "generic_mlp_minus_ridge_ise"),
        ):
            c = sb["contrasts"][c_key]
            self.assertGreater(c["CI95"][0], 0.0)
            self.assertEqual(c["favorable_roots"], 8)
            self.assertIn(
                f"{sb['mean_ise'][m_key]:.5f} | {c['mean']:.5f} [{c['CI95'][0]:.5f}, {c['CI95'][1]:.5f}] | "
                f"{c['relative_gain']:.2%} | {c['favorable_roots']}/8",
                paper,
            )

        cem = self.evidence["pointmaze_expensive_cem_planner"]["frontier"]
        for b in ("12", "15", "20", "24", "30", "40"):
            f = cem[b]
            g = f["periodic_minus_unprivileged_ise"]
            self.assertGreater(g["CI95"][0], 0.0)
            self.assertIn(
                f"{f['mean_ise']['periodic']:.5f} | {f['mean_ise']['distance_threshold']:.5f} | "
                f"{f['mean_ise']['unprivileged_uniform']:.5f} | "
                f"{g['mean']:.5f} [{g['CI95'][0]:.5f}, {g['CI95'][1]:.5f}] | "
                f"{f['planner_ms_per_episode']['unprivileged_uniform']:.2f} | "
                f"{f['trigger_score_ms_per_episode']['unprivileged_uniform']:.2f}",
                paper,
            )

        rp = self.evidence["reacher_polygonal_confirmation"]
        for m_key, c_key, must_be_positive in (
            ("shared_jitter", "shared_jitter_minus_ridge_ise", True),
            ("best_clock", "best_clock_minus_ridge_ise", True),
            ("shared_periodic", "shared_periodic_minus_ridge_ise", True),
            ("temple_gate", "temple_minus_ridge_ise", True),
            ("distance_threshold", "distance_minus_ridge_ise", False),
            ("deliberation_option_critic", "deliberation_oc_minus_ridge_ise", False),
            ("native_periodic_hrl", "native_periodic_minus_ridge_ise", False),
        ):
            c = rp["contrasts"][c_key]
            if must_be_positive:
                self.assertGreater(c["CI95"][0], 0.0)
            self.assertIn(
                f"{rp['mean_ise'][m_key]:.5f} | {c['mean']:.5f} [{c['CI95'][0]:.5f}, {c['CI95'][1]:.5f}] | "
                f"{c['relative_gain']:+.2%} | {c['favorable_roots']}/8",
                paper,
            )

    def test_cem_timing_components_keep_distinct_conclusions(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        costs = self.evidence["pointmaze_expensive_cem_planner"]["cross_budget_savings"]
        for comparison in ("unprivileged_B20_vs_periodic_B24", "unprivileged_B24_vs_periodic_B30"):
            row = costs[comparison]
            self.assertGreater(row["planner_ms_saved"]["CI95"][0], 0)
            self.assertLess(row["upper_loop_ms_saved"]["CI95"][1], 0)
            for field in ("planner_ms_saved", "upper_loop_ms_saved", "measured_control_ms_saved"):
                mean, (low, high) = row[field]["mean"], row[field]["CI95"]
                self.assertIn(f"`{mean:.2f} ms`", paper)
                self.assertIn(f"`[{low:.2f}, {high:.2f}]`", paper)
            low, high = row["measured_control_ms_saved"]["CI95"]
            self.assertLess(low, 0)
            self.assertGreater(high, 0)
        for comparison in ("unprivileged_B20_vs_distance_B24", "unprivileged_B24_vs_distance_B30"):
            row = costs[comparison]
            for field, places in (("ise_gain", 5), ("upper_loop_ms_saved", 2)):
                mean, (low, high) = row[field]["mean"], row[field]["CI95"]
                self.assertLess(high, 0)
                self.assertIn(f"`{-mean:.{places}f}" + (" ms`" if places == 2 else "`"), paper)
                self.assertIn(f"`[{-high:.{places}f}, {-low:.{places}f}]`", paper)
            self.assertLess(row["measured_control_ms_saved"]["CI95"][0], 0)
            self.assertGreater(row["measured_control_ms_saved"]["CI95"][1], 0)

    def test_second_plant_multiplicity_and_waiting_diagnostics_match_text(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        reacher = self.evidence["reacher_polygonal_confirmation"]
        sensitivity = reacher["ridge_baseline_ise_sensitivity"]
        nominal = sensitivity["paired_two_sided_p"]["best_clock_minus_ridge_ise"]
        adjusted = min(sensitivity["holm_p"].values())
        self.assertEqual(len(sensitivity["holm_p"]), 7)
        self.assertIn(f"p={nominal:.5f}", paper)
        self.assertIn(f"p={adjusted:.4f}", paper)
        waiting = self.evidence["waiting_value_development_diagnostics"]
        for row in waiting["ellipse_reacher_interventions"].values():
            self.assertIn(f"`{row['adaptive_50_step_gain']:.5f}`", paper)
            self.assertIn(f"`{row['full_episode_gain']:+.5f}`", paper)
        for row in waiting["matched_target_comparison"].values():
            self.assertIn(f"{row['wait_over_local_relative_gain']:.2%}", paper)

    def test_yoked_and_paired_supervision_results_match_manuscript(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        report = self.evidence["pointmaze_review_controls"]
        self.assertEqual((report["root_count"], report["paired_paths"], report["calls_per_episode"]), (8, 512, 24))
        self.assertEqual(report["different_yoked_schedules"], 512)
        self.assertTrue(report["contrasts"]["yoked"]["supported"])
        self.assertFalse(report["contrasts"]["single_arm"]["supported"])
        for row in report["contrasts"].values():
            self.assertIn(f"{row['relative_gain']:.2%}", paper)
            for low, high in (row["CI95"], row["CI97p5"]):
                self.assertIn(f"`[{low:.5f}, {high:.5f}]`", paper)

    def test_units_and_failed_closed_loop_decision_are_preserved(self):
        pm = self.evidence["pointmaze_stage9_confirmation"]
        self.assertEqual((pm["root_count"], pm["evaluation_paths"], pm["paired_mode_rollouts"]), (8, 128, 512))
        self.assertEqual(pm["upper_calls_per_episode_all_modes"], 24)
        self.assertEqual(self.evidence["rebuild_training_or_simulation"], 0)
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
            ("transit_stage301_single_intervention", "joint_budgeted_vs_raw", "cost"),
            ("transit_stage301_single_intervention", "joint_budgeted_vs_raw", "wait_min"),
            ("transit_stage301_single_intervention", "joint_budgeted_vs_service_only", "cost"),
            ("transit_stage301_single_intervention", "joint_budgeted_vs_service_only", "wait_min"),
        ]
        for study, name, field in rows:
            row = self.evidence[study]["comparisons"][name]
            values = (row["mean"][field], *row["prefix_CI95"][field], *row["conditional_MC_CI95"][field])
            m, low, high, low2, high2 = (v * 1e6 for v in values)
            self.assertIn(f"{m:.3f} | [{low:.3f}, {high:.3f}] | [{low2:.3f}, {high2:.3f}]", text)
        current = self.evidence["transit_stage304_two_decisions"]["comparisons"]["all32_scenes"]
        for name, field in (
            ("service_only_vs_raw", "cost"),
            ("service_only_vs_raw", "service"),
            ("service_only_vs_raw", "wait_min"),
            ("joint_budgeted_vs_service_only", "cost"),
        ):
            row = current[name]
            values = (row["mean"][field], *row["episode_bootstrap_CI95"][field], *row["paired_normal_CI95"][field])
            m, low, high, low2, high2 = (v * 1e6 for v in values)
            self.assertIn(f"{m:.3f} | [{low:.3f}, {high:.3f}] | [{low2:.3f}, {high2:.3f}]", text)

    def test_applicability_table_matches_saved_summaries(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        results = ROOT / "results"
        reacher = json.loads((results / "reacher_timing_applicability_20261112/summary.json").read_text())
        cartpole = json.loads((results / "cartpole_timing_applicability_20261112/summary.json").read_text())
        self.assertFalse(reacher["development_gate_passed"])
        self.assertFalse(cartpole["qualification_passed"])
        cases = [(f"Reacher {row['optimizer_seed']}", row) for row in reacher["roots"]]
        cases += [
            (f"Cart-pole, LQR position weight {weight}", row)
            for weight, row in zip((10, 30), cartpole["configurations"], strict=True)
        ]
        for label, row in cases:
            contrasts = row["contrasts"]
            gains = [contrasts[key]["relative_gain"] for key in ("waypoint_total", "event_vs_best", "event_vs_rotated")]
            self.assertIn("| " + " | ".join([label, *(f"{value:+.2%}" for value in gains)]) + " |", paper)

    def test_portfolio_table_matches_frozen_comparison(self):
        paper = (ROOT / "paper/manuscript.md").read_text()
        summary = ROOT / "results/pointmaze_stage9_timing_portfolio_20261112/summary.json"
        result = json.loads(summary.read_text())
        self.assertTrue(result["joint_gate_passed"])
        self.assertFalse(result["original_confirmation_released"])
        self.assertIn(f"Stage9: {result['mean_ise']['history']:.5f}", paper)
        for name in ("threshold", "portfolio"):
            row = result["comparisons"][name]
            gain = row["paired_ise_gain"]
            low, high = gain["interval"]
            self.assertIn(
                f"{result['mean_ise'][name]:.5f} | {gain['mean']:.5f} [{low:.5f}, {high:.5f}] | {row['relative_gain']:.2%}",
                paper,
            )

    def test_review_pdf_contains_all_captions_and_references(self):
        pdf = subprocess.check_output(["pdftotext", str(ROOT / "paper/submission.pdf"), "-"], text=True)
        for label, count in (("Table", 10), ("Figure", 5)):
            self.assertEqual(
                re.findall(rf"^\s*{label} (\d+):", pdf, re.MULTILINE),
                [str(n) for n in range(1, count + 1)],
            )
        self.assertIn("References", pdf)
        self.assertIn("0.505541", pdf)
        self.assertIn("0.99422", pdf)
        self.assertIn("4.48852", pdf)
        self.assertNotIn("schulman2017ppo", pdf)

    def test_active_link_files_and_citations_resolve(self):
        paper = ROOT / "paper"
        for name in ("manuscript.md", "evidence_map.md", "submission_readiness.md", "figure_evidence_audit.md", "README.md"):
            text = (paper / name).read_text()
            for target in re.findall(r"\]\(([^)]+)\)", text):
                if not target.startswith(("https://", "http://", "#")):
                    self.assertTrue((paper / target.partition("#")[0]).is_file(), (name, target))
        cited = set(re.findall(r"@([A-Za-z0-9_]+)", (paper / "manuscript.md").read_text()))
        defined = set(re.findall(r"@\w+\{([^,]+),", (paper / "references.bib").read_text()))
        self.assertFalse(cited - defined)
        self.assertGreaterEqual(len(cited), 30)


if __name__ == "__main__":
    unittest.main()
