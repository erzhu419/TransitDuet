from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.plan_validity_target_credit import fit_targets, predict
from freq_hrl.experiments.plan_validity_window_coverage import cached_rows, error, fit_window, training_readout
from scripts import analyze_plan_validity_window_coverage as analyzer
from scripts import run_plan_validity_window_coverage as worker
from scripts import submit_plan_validity_window_coverage_scheduleurm as submitter
from test_plan_validity_target_credit import bank


class WindowCoverageTest(unittest.TestCase):
    def test_dense_df8_exactly_reproduces_the_old_W_fit(self):
        old = bank()
        rules, _ = fit_targets(old, effective_df=8, threshold_quantile=0.75)
        dense = cached_rows(old, 4)
        rule, oof = fit_window(dense, effective_df=8, threshold_quantile=0.75)
        np.testing.assert_allclose(predict(rule, dense), predict(rules["W"], old), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(rule.threshold, rules["W"].threshold, places=12)
        self.assertEqual(training_readout(dense, rule, oof)["fitting_future_count"], 96)

    def test_budgets_match_but_complexity_is_an_independent_factor(self):
        dense = cached_rows(bank(), 4)
        wide = []
        for i in range(4):
            for row in cached_rows(bank(), 1):
                wide.append({**row, "seed": row["seed"] + 8 * i})
        for rows, paths in ((dense, 8), (wide, 32)):
            for df in (2, 8):
                rule, oof = fit_window(rows, effective_df=df, threshold_quantile=0.75)
                result = training_readout(rows, rule, oof)
                self.assertEqual(result["fitting_future_count"], 96)
                self.assertEqual(result["own_independent_B"]["paths"], paths)
                self.assertAlmostEqual(result["effective_df"], df)

    def test_validation_futures_cannot_change_a_fit(self):
        rows = cached_rows(bank(), 1)
        first, _ = fit_window(rows, effective_df=2, threshold_quantile=0.75)
        changed = deepcopy(rows)
        for row in changed:
            for replica in row["B_replicas"]:
                replica["adaptive_window_advantage"] += 1000
        second, _ = fit_window(changed, effective_df=2, threshold_quantile=0.75)
        self.assertEqual(first.to_dict(), second.to_dict())
        metric = error(predict(first, rows), rows, target="W", block="A", independent=False)
        self.assertIsNone(metric["label_mean_variance"])
        self.assertIsNone(metric["noise_corrected_mse"])

    def test_new_seeds_do_not_reuse_the_revealed_holdout(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_target_credit_2026-11-10.json").read_text())
        plan = draft_plan()
        result = submitter.prepare(source, plan, "fixture", Path("protocols/fixture.json"))
        reserved = {s for c in seed_cells(plan) for seeds in c["seed_roles"].values() for s in seeds}
        for cell in result["cells"]:
            fresh = set(cell["new_training_prefix_seeds"] + cell["held_prefix_seeds"])
            self.assertEqual(len(fresh), 32)
            self.assertFalse(fresh & reserved)
            self.assertFalse(fresh & set(cell["previous_held_prefix_seeds"]))
        for phase, jobs in result["jobs"].items():
            self.assertEqual(len(jobs), 4)
            for job in jobs:
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["protect_from_auto_eviction"])
                self.assertTrue(job["result_dir"].endswith("/public"))

    def test_preparing_a_manifest_does_not_launch_tasks(self):
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            path = Path(folder) / "fixture.json"
            with patch.object(submitter, "execute_bulk") as schedule:
                submitter.main(["--source-manifest", str(submitter.ROOT / "protocols/plan_validity_target_credit_2026-11-10.json"),
                    "--run-name", "fixture", "--manifest", str(path)])
                schedule.assert_not_called()
            self.assertEqual(json.loads(path.read_text())["fitting_state_future_pairs_per_arm"], 96)

    def test_factorial_contrasts_use_paired_complete_episodes(self):
        row = {"history": 4, "L": 5, "threshold": 4, "termination_history": 6,
            "dense_df2": 3, "dense_df8": 5, "wide_df2": 2, "wide_df8": 3}
        result = worker.episode_panels([{**row, "seed": s} for s in range(8)])
        for panel in result.values():
            c = panel["factorial_contrasts"]
            self.assertEqual(c["coverage_df2"]["mean"], -1)
            self.assertEqual(c["coverage_df8"]["mean"], -2)
            self.assertEqual(c["complexity_dense"]["mean"], -2)
            self.assertEqual(c["complexity_wide"]["mean"], -1)
            self.assertEqual(c["interaction"]["mean"], 1)

    def test_missing_roots_remain_unknown_and_do_not_clear_hold(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_target_credit_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        with tempfile.TemporaryDirectory() as folder:
            result = analyzer.summarize(manifest, root=Path(folder))
        self.assertEqual(len(result["missing"]), 4)
        self.assertEqual(result["task_means"], {"pointmaze": None, "reacher_tracking": None})
        self.assertFalse(result["confirmation_released"])

    def test_all_rules_are_frozen_before_new_holdout_and_failed_gate_stops_sampling(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_target_credit_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        cell = manifest["cells"][0]
        old = bank()
        for row in old:
            row["seed"] = cell["training_prefix_seeds"][row["seed"]]
        saved, _ = fit_targets(old, effective_df=8, threshold_quantile=0.75)
        original = {m: saved["W"] for m in cell["sources"]}
        records = {"termination_history": {"termination_readiness": {"eligible_for_fresh_confirmation": False}}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cache = root / cell["source_cache_directory"]
            cache.mkdir(parents=True)
            (cache / "result.json").write_text(json.dumps({"status": "complete", "raw_banks": {"training_bank": old}}))
            (cache / "frozen_rules.json").write_text(json.dumps({k: v.to_dict() for k, v in saved.items()}))
            public = root / cell["source_public_result"]
            public.parent.mkdir(parents=True)
            public.write_text(json.dumps({"status": "complete", "optimizer_seed": cell["optimizer_seed"],
                "training_prefix_seeds": cell["training_prefix_seeds"]}))
            directory = root / cell["output_directory"] / "run"
            (directory / "private").mkdir(parents=True)
            gate_path = directory.parent / "precheck/public/result.json"
            gate_path.parent.mkdir(parents=True)
            gate = {"status": "complete", "stage": "precheck_passed", "run_name": "fixture",
                    "optimizer_seed": cell["optimizer_seed"]}
            gate_path.write_text(json.dumps(gate))
            seen = []

            def collect(*args, **kwargs):
                seed = kwargs["seeds"][0]
                seen.append(seed)
                frozen = directory / "private/frozen_rules.json"
                if seed in cell["held_prefix_seeds"]:
                    self.assertEqual(set(json.loads(frozen.read_text())), set(worker.ARMS))
                else:
                    self.assertFalse(frozen.exists())
                a, b = kwargs["fitting_futures"], kwargs["evaluation_futures"]
                return [{**r, "seed": seed, "A_replicas": r["conditional_replicas"][:a],
                         "B_replicas": r["conditional_replicas"][4:4+b]} for r in old[:3]]

            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources", return_value=(records,
                {m: object() for m in original}, original, SimpleNamespace(**cell["expected_configuration"]))), \
                 patch.object(worker, "collect_window_bank", side_effect=collect), \
                 patch.object(worker, "run_episode", return_value=(None, {"tracking_squared_error_integral": 1})):
                result, _ = worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertEqual(seen, cell["new_training_prefix_seeds"] + cell["held_prefix_seeds"])
                self.assertEqual(len(result["episodes"]), 8)
                self.assertFalse(result["termination_development_eligible"])
                for report in result["training_readout"].values():
                    self.assertEqual(report["fitting_future_count"], 96)
                gate_path.write_text(json.dumps({**gate, "status": "failed"}))
                seen.clear()
                with self.assertRaisesRegex(ValueError, "successful matching native precheck"):
                    worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertFalse(seen)


if __name__ == "__main__":
    unittest.main()
