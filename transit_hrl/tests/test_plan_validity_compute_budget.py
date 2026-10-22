from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import compute_budget as budget
from freq_hrl.experiments import plan_validity_transfer as shared
from freq_hrl.experiments import pointmaze_budgeted_trigger as trigger
from freq_hrl.experiments import pointmaze_plan_validity_branching as branching
from freq_hrl.experiments.pointmaze_plan_value_qualification import PointMazeRegimeFeatureBuilder
from freq_hrl.experiments.pointmaze_timing_baselines import TimingRule
from scripts import run_plan_validity_compute_budget as runner
from scripts import select_plan_validity_compute_budget as selector
from test_plan_validity_second_task import Controller, DenseTask, options
from test_pointmaze_timing_baselines import fitting_rows


class Clock:
    wall = cpu = 0.0

    def advance(self, wall, cpu):
        self.wall += wall
        self.cpu += cpu


def record(run_id, *, cpu=20, steps=100, deploy_cpu=1, ise=1):
    return {"run_id": run_id, "task": "pointmaze", "optimizer_seed": 71, "method": "history",
            "status": "validation_complete", "accounting_complete": True,
            "hardware_profile": {"cpu": "fixture", "threads": 1},
            "offline_cost": {"cpu_seconds": cpu, "counts": {"environment_steps": steps}},
            "validation_latency": {"cpu_seconds_per_episode": deploy_cpu, "wall_p99_seconds": 0.001},
            "validation_ise": ise}


def select(records, expected=None, **overrides):
    kwargs = {"cpu_cap_seconds": 50, "environment_step_cap": 1000,
              "deployment_episodes": 2, "latency_p99_cap_seconds": 0.01,
              "expected_runs": records if expected is None else expected}
    kwargs.update(overrides)
    return budget.select_with_budget(records, **kwargs)


class BudgetTest(unittest.TestCase):
    def test_nested_scopes_do_not_double_count_and_count_completed_work(self):
        clock, ledger = Clock(), budget.ComputeLedger()
        with patch.object(budget.time, "perf_counter", side_effect=lambda: clock.wall), \
             patch.object(budget.time, "process_time", side_effect=lambda: clock.cpu):
            with ledger.recording():
                clock.advance(3, 2)
                with budget.measured_phase("branch"):
                    clock.advance(11, 7)
                    budget.count_work("environment_steps", 4)
                clock.advance(2, 1)
        summary = ledger.summary()
        self.assertEqual(summary["cpu_seconds"], 10)
        self.assertEqual(summary["wall_seconds"], 16)
        self.assertEqual(summary["phases"]["offline"]["cpu_seconds"], 3)
        self.assertEqual(summary["counts"]["environment_steps"], 4)
        self.assertIsNone(budget.active_ledger())

    def test_failure_keeps_partial_cost_instead_of_zero(self):
        clock, ledger = Clock(), budget.ComputeLedger()
        with patch.object(budget.time, "perf_counter", side_effect=lambda: clock.wall), \
             patch.object(budget.time, "process_time", side_effect=lambda: clock.cpu):
            with self.assertRaises(RuntimeError), ledger.recording():
                clock.advance(4, 2)
                budget.count_work("environment_steps", 7)
                raise RuntimeError("fixture failure")
        self.assertEqual(ledger.summary()["cpu_seconds"], 2)
        self.assertEqual(ledger.summary()["counts"]["environment_steps"], 7)
        self.assertEqual(ledger.summary()["phases"]["offline"]["failures"], 1)

    def test_search_cost_includes_losing_trials_and_ignores_test_outcomes(self):
        rows = [record("cheap", ise=1), record("expensive", ise=0.1, deploy_cpu=10)]
        result = select(rows)
        self.assertEqual(result["selections"][0]["selected_run_id"], "cheap")
        self.assertEqual(result["selections"][0]["search_cpu_seconds"], 40)
        rows[1]["test_ise"] = -1e9
        rows[0]["test_ise"] = 1e9
        self.assertEqual(result, select(rows))
        self.assertEqual(select(rows, environment_step_cap=150)["selections"][0]["status"],
                         "no_budget_feasible_candidate")

    def test_failed_trial_cost_and_missing_or_unmeasured_runs_are_preserved(self):
        valid, failed = record("valid", cpu=30), record("failed", cpu=25)
        failed["status"] = "failed"
        failed.pop("validation_latency")
        self.assertEqual(select([valid, failed])["selections"][0]["status"], "no_budget_feasible_candidate")
        missing = select([valid], expected=[valid, failed])
        self.assertEqual(missing["selections"][0]["missing_run_ids"], ["failed"])
        self.assertFalse(missing["all_groups_budget_feasible"])
        failed["accounting_complete"] = False
        failed.pop("hardware_profile")
        failed.pop("offline_cost")
        self.assertEqual(select([valid, failed])["selections"][0]["status"], "incomplete_accounting")

    def test_hardware_mismatch_duplicates_and_unplanned_runs_are_rejected(self):
        left, right = record("a"), record("b")
        right["hardware_profile"]["threads"] = 2
        with self.assertRaisesRegex(ValueError, "profile"):
            select([left, right])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            select([left, left])
        with self.assertRaisesRegex(ValueError, "roster"):
            select([left, right], expected=[left])

    def test_shared_pilot_cost_is_paid_once_and_amortized_over_the_declared_cohort(self):
        rows, pilot = [record("r1"), record("r2")], record("pilot", cpu=30, steps=200)
        rows[1]["optimizer_seed"] = 72
        pilot["optimizer_seed"], pilot["status"] = 7, "failed"
        result = select(rows, cpu_cap_seconds=40, shared_records=[pilot], shared_expected_runs=[pilot])
        groups = result["selections"]
        self.assertTrue(result["all_groups_budget_feasible"])
        self.assertEqual(sum(row["shared_allocation_cpu_seconds"] for row in groups), 30)
        self.assertTrue(all(row["lifecycle_cpu_seconds"] == 37 for row in groups))
        self.assertTrue(all(row["shared_failed_trial_count"] == 1 for row in groups))
        self.assertFalse(select(rows, cpu_cap_seconds=30, shared_records=[pilot],
                                shared_expected_runs=[pilot])["all_groups_budget_feasible"])
        missing = select(rows, shared_expected_runs=[pilot])
        self.assertTrue(all(row["status"] == "missing_measurement" for row in missing["selections"]))

    def test_latency_quantiles_include_all_steps_and_deadline_misses(self):
        latency = budget.ControlLatency()
        for index in range(100):
            latency.add(0.1 if index == 99 else 0.001, 0.0002, first_step=index in (0, 50))
        report = latency.summary(episodes=2, deadline_seconds=0.01)
        self.assertEqual(report["decisions"], 100)
        self.assertEqual(report["deadline_misses"], 1)
        self.assertAlmostEqual(report["cpu_seconds_per_episode"], 0.01)
        self.assertEqual(report["wall_max_seconds"], 0.1)
        self.assertAlmostEqual(report["wall_p95_seconds"], 0.001)

    def test_runtime_really_limits_native_threadpools(self):
        import torch
        old = torch.get_num_threads()
        with budget.single_thread_runtime() as profile:
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertTrue(all(row[2] == 1 for row in profile["blas"]))
            self.assertTrue(profile["cpu_model"])
        self.assertEqual(torch.get_num_threads(), old)

    def test_single_predictor_fit_is_identical_but_does_not_fit_unused_comparator(self):
        data = fitting_rows()
        ledger = budget.ComputeLedger()
        with ledger.recording():
            both = trigger.fit_budgeted_trigger_predictors(data, alpha_grid=[1.0])
        full_calls = ledger.summary()["counts"]["ridge_fit_calls"]
        ledger = budget.ComputeLedger()
        name = "causal_validity_interactions"
        with ledger.recording():
            one = trigger.fit_budgeted_trigger_predictors(data, alpha_grid=[1.0], predictors=(name,))
        self.assertEqual(one[name], both[name])
        self.assertEqual(2 * ledger.summary()["counts"]["ridge_fit_calls"], full_calls)

    def test_control_latency_excludes_slow_simulator_but_ledger_charges_it(self):
        args, clock, ledger, latency = options(), Clock(), budget.ComputeLedger(), budget.ControlLatency()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)

        class SlowSimulation(DenseTask):
            def step(self, action):
                clock.advance(0.2, 0.1)
                return super().step(action)

        class MeasuredController(Controller):
            def plan_goal(self, state, **kwargs):
                clock.advance(0.0002, 0.0002)
                return super().plan_goal(state, **kwargs)

            def act_conditioned(self, state, **kwargs):
                clock.advance(0.0001, 0.0001)
                return super().act_conditioned(state, **kwargs)

        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: SlowSimulation(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            patch.object(budget.time, "perf_counter", side_effect=lambda: clock.wall),
            patch.object(budget.time, "process_time", side_effect=lambda: clock.cpu),
            ledger.recording(),
        ):
            _, row = shared.run_episode(MeasuredController(), task_spec=spec, seed=71, args=args,
                time_scale=shared.time_contract(args), rule=TimingRule("fixed", "clock"), latency_recorder=latency)
        report, cost = latency.summary(episodes=1, deadline_seconds=0.01), ledger.summary()
        self.assertAlmostEqual(report["cpu_seconds_per_episode"], 300 * 0.0001 + 6 * 0.0002)
        self.assertGreater(cost["cpu_seconds"], 30)
        self.assertEqual(cost["counts"]["environment_steps"], 300)
        self.assertEqual(cost["counts"]["planner_calls"], 6)
        self.assertEqual(row["trigger_score_checks"], 0)
        self.assertNotIn("feature_vector_builds", cost["counts"])

    def test_scalar_thresholds_and_feature_only_bank_preserve_causal_values(self):
        args = options()
        spec = shared.PlanValidityTaskSpec("pointmaze", branching._task_options(args), args.env_id)
        time_scale = shared.time_contract(args)
        task = DenseTask(seed=71, horizon=300, **spec.options)
        observation = task.reset()
        history = PointMazeRegimeFeatureBuilder(time_scale=time_scale)
        history.reset(observation)
        for _ in range(60):
            observation, *_ = task.step(np.array([0.2, -0.1]))
            history.update(observation)
        names, values, _ = branching._causal_plan_features(observation=observation, feature_builder=history,
            subgoal=np.zeros(2), plan_age_steps=35, time_scale=time_scale)
        for feature in ("plan_age_fraction", "tracking_distance", "force_current_norm", "target_velocity_change_norm"):
            self.assertEqual(shared.threshold_score(feature, observation, history, 35, time_scale), values[names.index(feature)])
        query = branching.PlanRenewalOpportunity("neutral_matched", 80, None, None)
        with (
            patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
            patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            patch.object(shared, "renewal_queries", return_value=[query]),
        ):
            features = shared.collect_renewal_features(Controller(), task_spec=spec, args=args, time_scale=time_scale)
            labels = shared.collect_renewal_labels(Controller(), task_spec=spec, args=args, time_scale=time_scale)
        for feature, label in zip(features, labels):
            np.testing.assert_array_equal(feature["causal_features"], label["causal_features"])

    def test_budget_entry_default_is_plan_only_and_declares_expected_runs(self):
        command = ["--optimizer-seed", "71", "--train-seeds", "1", "2", "3", "--selection-seeds", "4",
            "--branch-fit-seeds", "5", "6", "7", "--timing-selection-seeds", "8", "--trigger-eval-seeds", "9",
            "--method", "history"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "plan.json"
            with patch.object(runner, "run_trial", side_effect=AssertionError("no execution")):
                runner.main([*command, "--output", str(output)])
            data = json.loads(output.read_text())
            self.assertEqual(data["status"], "prepared")
            self.assertEqual(data["records"], [])
            self.assertEqual(len(data["protocol"]["expected_runs"]), 2)
            self.assertEqual(list(Path(folder).iterdir()), [output])

    def test_selector_entry_preserves_caps_and_uses_only_compact_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "measured.json", Path(folder) / "selection.json"
            row = record("single")
            source.write_text(json.dumps({"records": [row], "protocol": {"expected_runs": [
                {key: row[key] for key in ("run_id", "task", "optimizer_seed", "method")}]}}))
            selector.main(["--inputs", str(source), "--plans", str(source),
                "--cpu-cap-seconds", "50", "--environment-step-cap", "1000",
                "--deployment-episodes", "2", "--output", str(output)])
            report = json.loads(output.read_text())
            self.assertEqual(report["selections"][0]["selected_run_id"], "single")
            self.assertEqual(report["budget_caps"]["cpu_seconds"], 50)
            self.assertEqual(report["measurement_files"], [str(source)])
            self.assertFalse(report["performance_claim_supported"])

    def test_measured_or_interrupted_attempts_cannot_be_silently_overwritten(self):
        command = ["--optimizer-seed", "71", "--train-seeds", "1", "2", "3", "--selection-seeds", "4",
            "--branch-fit-seeds", "5", "6", "7", "--timing-selection-seeds", "8", "--trigger-eval-seeds", "9",
            "--method", "fixed", "--tasks", "pointmaze"]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "attempt.json"
            runner.main([*command, "--output", str(output)])

            def attempt(args, task):
                self.assertEqual(json.loads(output.read_text())["status"], "running")
                return {"status": "failed", "offline_cost": {"cpu_seconds": 3}}

            with patch.object(runner, "run_trial", side_effect=attempt) as work:
                self.assertEqual(runner.main([*command, "--execute", "--output", str(output)]), 1)
                with self.assertRaises(FileExistsError):
                    runner.main([*command, "--execute", "--output", str(output)])
                self.assertEqual(work.call_count, 1)
            self.assertEqual(json.loads(output.read_text())["records"][0]["offline_cost"]["cpu_seconds"], 3)

    def test_standalone_fixed_has_no_supervision_and_never_accesses_test_paths(self):
        args = options()
        args.method = "fixed"

        @contextmanager
        def runtime():
            yield {"cpu": "fixture", "threads": 1}

        class SavedController(Controller):
            def state_dict(self):
                return {}

        def episode(model, **kwargs):
            self.assertIn(kwargs["seed"], args.timing_selection_seeds)
            return shared.run_episode(model, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            args.output = Path(folder) / "trial.json"
            with (
                patch.object(runner, "single_thread_runtime", runtime),
                patch.object(runner, "train_shared_controller", return_value=({"selected_checkpoint_iteration": 1}, SavedController())),
                patch.object(runner, "collect_renewal_labels", side_effect=AssertionError("fixed uses no labels")),
                patch.object(runner, "run_episode", side_effect=episode),
                patch.object(shared, "_make_task", side_effect=lambda **kw: DenseTask(**{k: v for k, v in kw.items() if k != "env_id"})),
                patch.object(shared, "pointmaze_goal_bounds", return_value=(np.full(2, -2), np.full(2, 2))),
            ):
                report = runner.run_trial(args, "pointmaze")
            self.assertEqual(report["status"], "validation_complete", report.get("error"))
            self.assertEqual(report["offline_cost"]["counts"]["environment_steps"], 600)
            self.assertNotIn("test_rows", report)
            self.assertEqual(report["validation_latency"]["decisions"], 600)


if __name__ == "__main__":
    unittest.main()
