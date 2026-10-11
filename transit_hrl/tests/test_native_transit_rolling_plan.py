import copy
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core.time_allocation import budgeted_time_points, projected_time_points
from freq_hrl.domains.transit.native_rolling_plan import NativeRollingServicePlan, schedule_summary
from scripts import run_native_transit_rolling_authority_stage161 as spec
from scripts.analyze_native_transit_rolling_authority_stage161 import summarize
from scripts.submit_native_transit_rolling_authority_stage161_scheduleurm import task_specification
from tests import test_native_transit_service_plan as service_tests


class NativeRollingPlanTest(unittest.TestCase):
    def environment(self, condition, trips=18):
        env, _ = service_tests.ServicePlanTest().environment("nominal_plan", trips_per_direction=trips)
        plan = NativeRollingServicePlan(env, condition)
        env._upper_policy_callback = plan
        return env, plan

    def execute(self, condition, trips=18):
        env, plan = self.environment(condition, trips)
        for tick in range(300 + trips * 360 + 180):
            env.current_time = tick
            env._dispatch_due_trips()
        return env, plan

    def test_projection_reallocates_without_freezing_at_an_existing_bound(self):
        nominal = np.array([0, 240, 600, 1080])
        preferred = [200, 300, 580]
        np.testing.assert_array_equal(budgeted_time_points(nominal, preferred, minimum=240, maximum=480), nominal)
        projected = projected_time_points(nominal, preferred, minimum=240, maximum=480)
        np.testing.assert_array_equal(np.diff(projected), [250, 350, 480])
        np.testing.assert_array_equal(projected[[0, -1]], nominal[[0, -1]])
        for values in ([0, 0, 0], [-1000, 1000, 10000], [240, 360, 480]):
            p = projected_time_points(nominal, values, minimum=240, maximum=480)
            self.assertTrue(np.all(np.diff(p) >= 240))
            self.assertTrue(np.all(np.diff(p) <= 480))
            self.assertEqual(p[-1], nominal[-1])

    def test_real_dispatch_reproduces_nominal_and_executes_rolling_goals(self):
        nominal_env, _ = service_tests.ServicePlanTest().execute("nominal_plan", trips=18)
        for condition in ("nominal_plan", "causal_forecast", "forecast_reversed"):
            env, plan = self.execute(condition)
            summary = plan.summarize()
            self.assertEqual(summary["queries"], 36)
            self.assertEqual(summary["upper_decisions"], 6)
            self.assertEqual(summary["daily_endpoint_error_max_s"], 0)
            self.assertEqual(summary["release_lateness_mean_s"], 0)
            for tt, nominal_tt in zip(env.timetables, nominal_env.timetables):
                self.assertEqual(tt._actual_launch_time, tt._freqduet_scheduled_launch)
                if condition == "nominal_plan":
                    self.assertEqual(tt._actual_launch_time, nominal_tt._actual_launch_time)
                    self.assertEqual(tt.target_headway, nominal_tt.target_headway)
            if condition != "nominal_plan":
                self.assertGreater(summary["window_endpoint_shift_abs_mean_s"], 0)

    def test_later_observations_replan_only_uncommitted_departures(self):
        env, plan = self.environment("causal_forecast")
        env.current_time = 180
        env._dispatch_due_trips()
        first = copy.deepcopy(plan.blocks[0])
        tentative = {t.launch_turn: t._freqduet_scheduled_launch for t in env.timetables
                     if not t.direction and t.launch_turn not in plan._targets}
        env.frequency_tracker.global_state.forecast = lambda t: 100000 / (1 + t)
        for tick in range(181, 300 + 18 * 360 + 180):
            env.current_time = tick
            env._dispatch_due_trips()
        self.assertEqual(plan.blocks[0], first)
        self.assertTrue(any(t._freqduet_scheduled_launch != tentative[t.launch_turn]
                            for t in env.timetables if t.launch_turn in tentative))
        plan.summarize()
        env.timetables[0]._freqduet_scheduled_launch += 1
        with self.assertRaises(RuntimeError): plan.summarize()

    def test_tail_blocks_keep_daily_budget_and_cross_window_bounds(self):
        for count in (7, 8, 11, 13):
            _, plan = self.execute("causal_forecast", trips=count)
            summary = plan.summarize()
            self.assertEqual(summary["scheduled_trips"], 2 * count)
            for s in summary["schedules"]:
                self.assertEqual(s["planned_s"][-1], s["nominal_s"][-1])
                np.testing.assert_array_equal(s["target_headways_s"][1:], np.diff(s["planned_s"]))

    def cells(self):
        sources = {}
        cells = {}
        schedules = []
        for direction in (False, True):
            times = (np.arange(131) * 360 + 300 + 180 * direction).astype(float).tolist()
            schedules.append({"direction": direction, "nominal_s": times, "planned_s": times,
                              "actual_s": times, "target_headways_s": [360.] * 131})
        schedule = {"queries": 262, "scheduled_trips": 262, "upper_decisions": 44,
            "committed_unchanged": True, "daily_endpoint_error_max_s": 0,
            "window_endpoint_shift_abs_mean_s": 0., "planned_shift_abs_mean_s": 0., "schedules": schedules}
        for root in spec.ROOTS:
            source_rows = []
            for scenario in spec.SCENARIOS:
                rows = []
                for scene in spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False):
                    common = {k: 1. for k in spec.native.source.source_spec.authority.routing.METRICS}
                    common.update(ep=300, done_reason="evaluation_horizon", N_fleet=12,
                                  simulation_end_time_s=61380, passengers_generated=100,
                                  passengers_unserved=0, trips_completed=262, ep_steps=61380)
                    for condition in ("forecast", "nominal"):
                        source_rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene, **common})
                    for condition in spec.CONDITIONS:
                        rows.append({"condition": condition, "scenario": scenario, "scene_seed": scene,
                            **common, "source_control_reproduced": condition in {
                                "window_forecast", "window_nominal", "rolling_nominal"}, "schedule": copy.deepcopy(schedule)})
                cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                    "seed": root, "scenario": scenario, "software_qualified": True, "training_updates": 0,
                    "native_steps": 20 * 61380, "evaluation": rows}
            sources[root] = {"evaluation": source_rows}
        return cells, sources

    def test_analysis_rejects_changed_daily_or_decision_budgets_and_source_controls(self):
        cells, sources = self.cells()
        result = summarize(cells, sources)
        self.assertEqual(result["native_steps"], 200 * 61380)
        self.assertEqual(result["source_control_reproductions"], 120)
        for failure in ("duplicate", "endpoint", "goal", "decisions", "control", "commitment"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "rolling_nominal")
            if failure == "duplicate": cell["evaluation"].append(copy.deepcopy(row))
            elif failure == "endpoint": row["schedule"]["schedules"][0]["planned_s"][-1] += 1
            elif failure == "goal": row["schedule"]["schedules"][0]["target_headways_s"][7] += 1
            elif failure == "decisions": row["schedule"]["upper_decisions"] += 1
            elif failure == "control": row["service_cost_restricted"] += .1
            else: row["schedule"]["committed_unchanged"] = False
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources)

    def test_scheduler_is_unpinned_code_only_and_syncs_matching_small_directory(self):
        task = task_specification("test_rolling", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["local_result_dir"], task["result_dir"])
        self.assertTrue(task["result_dir"].endswith("/cells/low_noise/seed_397"))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])

    def test_worker_checks_three_controls_before_extended_authority(self):
        cells, sources = self.cells()
        rows = cells[397, "low_noise"]["evaluation"]
        registered = sources[397]
        for r in registered["evaluation"]: r["execution"] = {"queries": 262}
        calls = []

        def episode(root, scenario, scene, raw, checkpoint, condition):
            calls.append(condition)
            row = next(r for r in rows if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"]

        output = io.StringIO()
        with patch.object(spec, "source_cell", return_value=registered), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                                           "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(output):
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS) * 4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 20 * 61380)
        self.assertIn("ROLLING_AUTHORITY_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
