import contextlib
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_rolling_plan import NativeTemperedRollingPlan, schedule_summary
from scripts import run_native_transit_allocation_frontier_stage162 as spec
from scripts.analyze_native_transit_allocation_frontier_stage162 import summarize
from scripts.submit_native_transit_allocation_frontier_stage162_scheduleurm import task_specification
from tests import test_native_transit_rolling_plan as rolling_tests


class NativeAllocationFrontierTest(unittest.TestCase):
    def environment(self, strength, trips=18):
        env, _ = rolling_tests.NativeRollingPlanTest().environment("nominal_plan", trips)
        plan = NativeTemperedRollingPlan(env, strength)
        env._upper_policy_callback = plan
        return env, plan

    def execute(self, strength, trips=18):
        env, plan = self.environment(strength, trips)
        for tick in range(300 + trips * 360 + 180):
            env.current_time = tick
            env._dispatch_due_trips()
        return env, plan

    def test_zero_and_one_reproduce_both_existing_rolling_endpoints(self):
        for strength, mode in ((0, "nominal_plan"), (1, "causal_forecast")):
            _, original = rolling_tests.NativeRollingPlanTest().execute(mode)
            _, actual = self.execute(strength)
            self.assertEqual(actual.summarize(), original.summarize())
            self.assertEqual(actual.blocks, original.blocks)
        for bad in (-.1, 1.1, np.nan, np.inf):
            with self.assertRaises(ValueError): self.environment(bad)

    def test_preference_strength_is_applied_before_the_budget_projection(self):
        points = np.arange(18) * 360 + 300
        for strength in (0, .25, .5, .75, 1):
            env, plan = self.environment(strength)
            trip = env.timetables[0]
            reference = rolling_tests.NativeRollingPlanTest().environment("causal_forecast")[1]
            full, rates = reference.preferred_intervals(points, trip, 180)
            actual, observed_rates = plan.preferred_intervals(points, trip, 180)
            np.testing.assert_allclose(actual, (1-strength)*np.diff(points)+strength*full, atol=0, rtol=0)
            self.assertEqual(observed_rates, rates)

    def test_interior_strengths_keep_commitments_native_goals_and_tail_budgets(self):
        for strength in (.25, .5, .75):
            for trips in (7, 8, 11, 18):
                env, plan = self.environment(strength, trips)
                env.current_time = 180
                env._dispatch_due_trips()
                first = copy.deepcopy(plan.blocks[0])
                env.frequency_tracker.global_state.forecast = lambda t: 100000/(1+t)
                for tick in range(181, 300 + trips * 360 + 180):
                    env.current_time = tick
                    env._dispatch_due_trips()
                self.assertEqual(plan.blocks[0], first)
                ledger = schedule_summary(plan)
                self.assertEqual(ledger["scheduled_trips"], 2*trips)
                self.assertEqual(ledger["release_lateness_mean_s"], 0)
                for s in ledger["schedules"]:
                    np.testing.assert_array_equal(s["planned_s"], s["actual_s"])
                    np.testing.assert_array_equal(s["target_headways_s"][1:], np.diff(s["planned_s"]))

    def cells(self):
        sources, _ = rolling_tests.NativeRollingPlanTest().cells()
        cells = {}
        for key, source in sources.items():
            for row in source["evaluation"]: row["peak_fleet"] = 12
            rows = []
            for scene in spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[key[0]], key[1], preflight=False):
                for condition in spec.CONDITIONS:
                    original = spec.SOURCE_CONTROLS.get(condition, "rolling_forecast")
                    row = copy.deepcopy(next(r for r in source["evaluation"]
                        if r["condition"] == original and r["scene_seed"] == scene))
                    row.update(condition=condition, strength=spec.STRENGTHS.get(condition),
                               source_control_reproduced=condition in spec.SOURCE_CONTROLS)
                    rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": key[0], "scenario": key[1], "software_qualified": True,
                "training_updates": 0, "native_steps": 24*61380, "evaluation": rows}
        return cells, sources

    def test_analysis_rejects_changed_strengths_schedules_outcomes_or_budgets(self):
        cells, sources = self.cells()
        result = summarize(cells, sources)
        self.assertEqual(result["native_steps"], 240*61380)
        self.assertEqual(result["source_control_reproductions"], 120)
        self.assertEqual(len(result["strength_minus_window_forecast"]), 10)
        self.assertEqual(len(result["regime_root_contrasts"]), 50)
        for failure in ("strength", "duplicate", "control", "schedule", "goal", "budget", "flag", "source"):
            broken, refs = copy.deepcopy(cells), copy.deepcopy(sources)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "rolling_1")
            if failure == "strength": row["strength"] = .5
            elif failure == "duplicate": cell["evaluation"].append(copy.deepcopy(row))
            elif failure == "control": row["service_cost_restricted"] += .1
            elif failure == "schedule": row["schedule"]["release_lateness_mean_s"] = 1
            elif failure == "goal": row["schedule"]["schedules"][0]["target_headways_s"][7] += 1
            elif failure == "budget": cell["native_steps"] += 1
            elif failure == "flag": row["source_control_reproduced"] = False
            else: refs[397, "low_noise"]["training_updates"] = 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, refs)

    def test_analysis_does_not_confuse_wait_gain_with_fleet_cost_gain(self):
        cells, sources = self.cells()
        for (root, _), cell in cells.items():
            if root == 397:
                for row in cell["evaluation"]:
                    if row["condition"] == "rolling_05":
                        row["service_cost_restricted"] += 1/12
                        row["restricted_wait_horizon_min"] -= 1
                        row["peak_fleet"] = 13
        summary = summarize(cells, sources)
        row = next(r for r in summary["strength_minus_window_forecast"]
                   if r["root"] == 397 and r["condition"] == "rolling_05")
        self.assertAlmostEqual(row["fleet_cost_delta"], 1/12)
        self.assertAlmostEqual(row["nonfleet_cost_delta"], 0)
        self.assertAlmostEqual(row["restricted_wait_horizon_min"], -1)
        self.assertEqual(row["cost_wins"], 0)

    def test_scheduler_is_unpinned_code_only_with_small_matching_output(self):
        task = task_specification("test_frontier", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertEqual(task["local_result_dir"], task["result_dir"])
        self.assertTrue(task["result_dir"].endswith("/cells/low_noise/seed_397"))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])

    def test_worker_qualifies_endpoints_before_interior_with_no_extra_episodes(self):
        cells, sources = self.cells()
        calls = []

        def episode(root, scenario, scene, raw, checkpoint, condition):
            calls.append(condition)
            row = next(r for r in cells[root, scenario]["evaluation"]
                if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"]

        output = io.StringIO()
        with patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(output):
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS)*4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 24*61380)
        self.assertEqual(write.call_args.args[1]["training_updates"], 0)
        lines = output.getvalue().splitlines()
        qualified = next(i for i, line in enumerate(lines) if line == "ALLOCATION_FRONTIER_FIRST_SCENE_SOURCE_QUALIFIED")
        self.assertIn("rolling_1/", lines[qualified-1])
        self.assertIn("rolling_025/", lines[qualified+1])


if __name__ == "__main__":
    unittest.main()
