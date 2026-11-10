from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_phase_support as phase
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments.plan_validity_preregistration import draft_plan, seed_cells
from freq_hrl.experiments.plan_validity_window_coverage import cached_rows, fit_window
from freq_hrl.experiments.pointmaze_plan_validity_branching import _task_options
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts import analyze_plan_validity_phase_support as analyzer
from scripts import run_plan_validity_phase_support as worker
from scripts import submit_plan_validity_phase_support_scheduleurm as submitter
from test_plan_validity_second_task import Controller, DenseTask, options
from test_plan_validity_target_credit import bank


def clock_rule():
    weights = np.zeros(42)
    weights[40] = 1
    return TimingRule("clock_input_test", "phase_ridge", threshold=0.5,
        model={"model": {"feature_mean": [0] * 41, "feature_scale": [1] * 41, "weights": weights.tolist()}})


class PhaseSupportTest(unittest.TestCase):
    def test_clock_model_uses_causal_offset_and_matches_batch_prediction(self):
        row = bank()[0]
        first = phase.clock_row({**row, "opportunity_step": 50}, horizon=1200)
        later = phase.clock_row({**row, "opportunity_step": 70}, horizon=1200)
        rule = clock_rule()
        names, features = tuple(row["feature_names"]), np.asarray(row["causal_features"])
        self.assertEqual(rule.score(names, features, 0, 1), 0)
        self.assertEqual(rule.score(names, features, 0.8, 0.5), 0.8)
        np.testing.assert_allclose(phase.predict(rule, [first, later]), [0, 0.8])
        self.assertEqual(phase.feature_matrix([first], clock=False)[0].shape[1], 39)
        self.assertEqual(phase.feature_matrix([first], clock=True)[0].shape[1], 41)
        model = deepcopy(rule.model)
        model["model"]["weights"][40] = 0
        model["model"]["weights"][41] = 1
        remaining = TimingRule("remaining", "phase_ridge", model=model)
        self.assertEqual(remaining.score(names, features, 0, 0.4), 0.4)

    def test_plain_fit_reproduces_old_window_and_B_does_not_change_clock_fit(self):
        rows = [phase.clock_row(r, horizon=1200) for r in cached_rows(bank(), 1)]
        old, _ = fit_window(rows, effective_df=8, threshold_quantile=0.75)
        plain, _ = phase.fit(rows, clock=False, effective_df=8, threshold_quantile=0.75)
        np.testing.assert_allclose(phase.predict(old, rows), phase.predict(plain, rows), rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(old.threshold, plain.threshold, places=12)
        clock, _ = phase.fit(rows, clock=True, effective_df=8, threshold_quantile=0.75)
        changed = deepcopy(rows)
        for row in changed:
            for replica in row["B_replicas"]:
                replica["adaptive_window_advantage"] += 1000
        again, _ = phase.fit(changed, clock=True, effective_df=8, threshold_quantile=0.75)
        self.assertEqual(clock.to_dict(), again.to_dict())
        self.assertAlmostEqual(clock.model["model"]["effective_df_slopes"], 8)
        self.assertEqual(clock.model["model"]["coefficient_count"], 42)

    def test_later_queries_are_visited_before_renewal_and_replay_conditional_futures(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", _task_options(args), args.env_id)
        with patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})), \
             patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))):
            obs = phase.observe(Controller(), clock_rule(), spec=spec, args=args, seed=71, probes_per_path=2)
            self.assertEqual(obs["start_steps"], [50, 100])
            self.assertNotIn(70, obs["available_steps"])
            self.assertIn(55, obs["available_steps"])
            for start, check in zip(obs["start_steps"], obs["check_steps"]):
                self.assertEqual(start // 50, check // 50)
                self.assertIn(check, obs["available_steps"])
                self.assertLess(check % 50, 25)
            rows = phase.sample(Controller(), clock_rule(), obs, [55], spec=spec, args=args,
                                fitting_futures=1, evaluation_futures=1)
        self.assertEqual(obs["factual"]["upper_decision_count"], 6)
        self.assertTrue(all(s % 50 == 15 for s in obs["factual"]["decision_steps"][1:]))
        self.assertEqual(rows[0]["offset_fraction"], 0.2)
        for r in rows[0]["A_replicas"] + rows[0]["B_replicas"]:
            self.assertAlmostEqual(r["full_advantage"], r["adaptive_window_advantage"] + r["tail_advantage"])

    def test_new_holdout_excludes_all_revealed_roles_and_resources_are_unpinned(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_window_coverage_2026-11-10.json").read_text())
        plan = draft_plan()
        result = submitter.prepare(source, plan, "fixture", Path("protocols/fixture.json"))
        reserved = {s for c in seed_cells(plan) for seeds in c["seed_roles"].values() for s in seeds}
        for c in result["cells"]:
            self.assertEqual(len(c["fit_prefix_seeds"]), 32)
            self.assertFalse(set(c["held_prefix_seeds"]) & (reserved | set(c["exposed_held_prefix_seeds"]) | set(c["fit_prefix_seeds"])))
        self.assertEqual(result["effective_df"], 8)
        self.assertEqual(result["fitting_state_future_pairs_per_arm"], 96)
        for jobs in result["jobs"].values():
            for job in jobs:
                self.assertEqual((job["cpu"], job["ram_mb"], job["vram"]), (1, 2048, 0))
                self.assertIsNone(job["require_node"])
                self.assertIsNone(job["preferred_node"])
                self.assertTrue(job["result_dir"].endswith("/public"))
        self.assertFalse(result["confirmation_released"])

    def test_preparation_does_not_submit(self):
        with tempfile.TemporaryDirectory(dir=submitter.ROOT / "protocols") as folder:
            with patch.object(submitter, "execute_bulk") as schedule:
                submitter.main(["--source-manifest", str(submitter.ROOT / "protocols/plan_validity_window_coverage_2026-11-10.json"),
                    "--run-name", "fixture", "--manifest", str(Path(folder) / "manifest.json")])
                schedule.assert_not_called()

    def test_factorial_contrasts_and_missing_roots_keep_the_declared_scope(self):
        row = {"start_plain": 5, "start_clock": 4, "checks_plain": 3, "checks_clock": 1,
            "history": 2, "threshold": 2, "L": 5, "prior_wide_df2": 4, "termination_history": 6}
        result = worker.episode_panels([{**row, "seed": i} for i in range(8)])
        contrasts = result["all"]["factorial_contrasts"]
        self.assertEqual(contrasts["sampling_plain"]["mean"], -2)
        self.assertEqual(contrasts["clock_checks"]["mean"], -2)
        self.assertEqual(contrasts["interaction"]["mean"], -1)
        source = json.loads((submitter.ROOT / "protocols/plan_validity_window_coverage_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        with tempfile.TemporaryDirectory() as folder:
            result = analyzer.summarize(manifest, root=Path(folder))
        self.assertEqual(len(result["missing"]), 4)
        self.assertEqual(result["task_means"], {"pointmaze": None, "reacher_tracking": None})
        self.assertFalse(result["confirmation_released"])

    def test_main_freezes_all_heads_before_shared_held_queries_and_retains_budgets(self):
        source = json.loads((submitter.ROOT / "protocols/plan_validity_window_coverage_2026-11-10.json").read_text())
        manifest = submitter.prepare(source, draft_plan(), "fixture", Path("protocols/fixture.json"))
        cell = manifest["cells"][0]
        old = bank()
        for row in old:
            row["seed"] = cell["training_prefix_seeds"][row["seed"]]
        newer = []
        for seed in cell["new_training_prefix_seeds"]:
            newer.extend({**r, "seed": seed} for r in cached_rows(old[:3], 1))
        starts = [phase.clock_row(r, horizon=1200) for r in cached_rows(old, 1) + newer]
        prior8, _ = fit_window(starts, effective_df=8, threshold_quantile=0.75)
        prior2, _ = fit_window(starts, effective_df=2, threshold_quantile=0.75)
        original = {m: prior8 for m in cell["sources"]}
        records = {"termination_history": {"termination_readiness": {"eligible_for_fresh_confirmation": False}}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            coverage, target = root / cell["coverage_cache_directory"], root / cell["target_cache_directory"]
            coverage.mkdir(parents=True)
            target.mkdir(parents=True)
            (coverage / "result.json").write_text(json.dumps({"status": "complete", "raw_banks": {"new_training_bank": newer}}))
            (coverage / "frozen_rules.json").write_text(json.dumps({"wide_df8": prior8.to_dict(), "wide_df2": prior2.to_dict()}))
            (target / "result.json").write_text(json.dumps({"raw_banks": {"training_bank": old}}))
            (target / "frozen_rules.json").write_text(json.dumps({"L": prior8.to_dict()}))
            public = root / cell["coverage_public_result"]
            public.parent.mkdir(parents=True)
            public.write_text(json.dumps({"status": "complete", "optimizer_seed": cell["optimizer_seed"]}))
            directory = root / cell["output_directory"] / "run"
            (directory / "private").mkdir(parents=True)
            gate = directory.parent / "precheck/public/result.json"
            gate.parent.mkdir(parents=True)
            gate.write_text(json.dumps({"status": "complete", "stage": "precheck_passed", "run_name": "fixture", "optimizer_seed": cell["optimizer_seed"]}))
            seen = []

            def observe(*args, **kw):
                seed = kw["seed"]
                seen.append(seed)
                frozen = directory / "private/frozen_rules.json"
                if seed in cell["held_prefix_seeds"]:
                    self.assertEqual(set(json.loads(frozen.read_text())), set(phase.ARMS))
                else:
                    self.assertFalse(frozen.exists())
                rows = [r for r in starts if r["seed"] == seed] or [dict(r, seed=seed) for r in starts[:3]]
                features = [dict(r, opportunity_step=s) for r in rows for s in (r["opportunity_step"], r["opportunity_step"] + 5)]
                return {"seed": seed, "factual": {"feature_rows": features}, "start_steps": [50, 100, 150],
                    "check_steps": [55, 100, 155], "available_steps": [50, 55, 100, 105, 150, 155]}

            def sample(*args, **kw):
                obs, steps = args[2:4]
                rows = {r["opportunity_step"]: r for r in obs["factual"]["feature_rows"]}
                return [phase.clock_row({**rows[s], "A_replicas": old[0]["conditional_replicas"][:kw["fitting_futures"]],
                    "B_replicas": old[0]["conditional_replicas"][4:4+kw["evaluation_futures"]]}, horizon=1200) for s in steps]

            with patch.object(worker, "ROOT", root), patch.object(worker, "load_sources", return_value=(records,
                {m: object() for m in original}, original, SimpleNamespace(**cell["expected_configuration"]))), \
                 patch.object(worker, "observe", side_effect=observe), patch.object(worker, "sample", side_effect=sample), \
                 patch.object(worker, "run_episode", return_value=(None, {"tracking_squared_error_integral": 1,
                    "queries": [], "decision_steps": list(range(0, 1200, 50))})):
                result, _ = worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertEqual(seen, cell["fit_prefix_seeds"] + cell["held_prefix_seeds"])
                self.assertEqual(len(result["episodes"]), 8)
                self.assertFalse(result["termination_development_eligible"])
                self.assertEqual(result["held_phase_counts"]["later_checks"]["5"], 16)
                for report in result["training_readout"].values():
                    self.assertEqual(report["fitting_future_count"], 96)
                gate.write_text(json.dumps({"status": "failed"}))
                seen.clear()
                with self.assertRaisesRegex(ValueError, "successful matching phase precheck"):
                    worker.run_cell(cell, manifest, "run", directory, lambda _: None)
                self.assertFalse(seen)


if __name__ == "__main__":
    unittest.main()
