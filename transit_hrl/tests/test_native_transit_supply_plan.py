import contextlib
import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_supply_plan import NativeSupplyRollingPlan, occupancy_peak, STRENGTHS
from scripts import run_native_transit_supply_allocation_stage163 as spec
from scripts.analyze_native_transit_supply_allocation_stage163 import summarize
from scripts.submit_native_transit_supply_allocation_stage163_scheduleurm import task_specification
from tests import test_native_transit_allocation_frontier as frontier_tests


class NativeSupplyPlanTest(unittest.TestCase):
    def environment(self, condition="supply_aware", trips=18):
        env, _ = frontier_tests.NativeAllocationFrontierTest().environment(0, trips)
        env.routes = [SimpleNamespace(distance=500., speed_limit=5.) for _ in range(20)]
        env.completed_trip_observations = {}
        env._peak_concurrent = 0
        plan = NativeSupplyRollingPlan(env, condition)
        env._upper_policy_callback = plan
        return env, plan

    def test_joint_occupancy_ties_and_active_buses(self):
        self.assertEqual(occupancy_peak([], []), 0)
        self.assertEqual(occupancy_peak([0, 10], [10, 20]), 1)
        self.assertEqual(occupancy_peak([0, 0, 5], [10, 20, 6]), 3)
        env, plan = self.environment(trips=3)
        for tt in env.timetables: tt.launched = True
        env.bus_all = [SimpleNamespace(on_route=True, direction=False, effective_route=env.routes[10:], travel_distance=2500.)]
        self.assertEqual(plan.predicted_peak({}, 600), 1)
        env._peak_concurrent = 3
        self.assertEqual(plan.predicted_peak({}, 600), 3)
        env._peak_concurrent = 0
        for tt in env.timetables: tt.launched = False
        self.assertEqual(plan.predicted_peak({t.launch_turn: 600. for t in env.timetables}, 600), 7)

    def test_completed_trip_ewma_uses_only_observed_times_once(self):
        env, plan = self.environment()
        env.completed_trip_observations = {
            0: {"direction": False, "launch_time_s": 300., "completion_time_s": 1500.},
            1: {"direction": True, "launch_time_s": 480., "completion_time_s": 1980.},
            2: {"direction": False, "launch_time_s": 660., "completion_time_s": 2460.}}
        plan.observe_completions(2500)
        self.assertEqual(plan.duration_mean, {False: 1320., True: 1500.})
        plan.observe_completions(2600)
        self.assertEqual(plan.duration_mean, {False: 1320., True: 1500.})
        self.assertEqual(len(plan.seen_completions), 3)
        env.completed_trip_observations[3] = {"direction": True, "launch_time_s": 1000., "completion_time_s": 3000.}
        with self.assertRaises(RuntimeError): plan.observe_completions(2600)

    def test_supply_term_can_change_selection_and_zero_recovers_tentative_spacing(self):
        decisions = []
        for condition in ("demand_only", "supply_aware"):
            env, plan = self.environment(condition)
            trip = env.timetables[0]
            nominal = np.array([t.launch_time for t in env.timetables if not t.direction])
            # Previously compressed tentative intervals, with unchanged endpoints.
            nominal[1:5] -= np.array([30, 60, 60, 30])
            env.frequency_tracker.global_state.forecast = lambda t: 1 + t*t
            seen = []

            def peak(points, now):
                seen.append(list(points.values()))
                return 12 if len(seen) == 1 else 15

            with patch.object(plan, "predicted_peak", side_effect=peak):
                plan.preferred_intervals(nominal, trip, 180)
            decisions.append(plan.decisions[-1])
            np.testing.assert_array_equal(np.diff(seen[0]), [360]*17)
            self.assertEqual([c["strength"] for c in decisions[-1]["candidates"]], list(STRENGTHS))
        self.assertNotEqual(decisions[0]["selected"], 0)
        self.assertEqual(decisions[1]["selected"], 0)

    def test_native_dispatch_keeps_tail_budgets_and_irrevocable_commitments(self):
        for condition in ("demand_only", "supply_aware"):
            for trips in (7, 11, 18):
                env, plan = self.environment(condition, trips)
                env.current_time = 180
                env._dispatch_due_trips()
                first = copy.deepcopy(plan.blocks[0])
                env.frequency_tracker.global_state.forecast = lambda t: 100000/(1+t)
                for tick in range(181, 300+trips*360+180):
                    env.current_time = tick
                    env._dispatch_due_trips()
                self.assertEqual(plan.blocks[0], first)
                schedule = plan.summarize()
                self.assertEqual(schedule["queries"], trips*2)
                self.assertEqual(schedule["release_lateness_mean_s"], 0)
                for s in schedule["schedules"]:
                    np.testing.assert_array_equal(s["planned_s"], s["actual_s"])
                    np.testing.assert_array_equal(s["target_headways_s"][1:], np.diff(s["planned_s"]))

    def cells(self):
        sources, _ = frontier_tests.NativeAllocationFrontierTest().cells()
        cells = {}
        for key, source in sources.items():
            rows = []
            scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[key[0]], key[1], preflight=False)
            for scene in scenes:
                for condition in spec.CONDITIONS:
                    ref_condition = condition if condition in spec.CONTROLS else "window_forecast"
                    row = copy.deepcopy(next(r for r in source["evaluation"] if r["condition"] == ref_condition and r["scene_seed"] == scene))
                    supply = None
                    if condition not in spec.CONTROLS:
                        decisions = [{"time_s": i*1000., "direction": bool(i%2), "selected": 0,
                            "observed_completions": i, "candidates": [{"strength": s, "wait_cost": 1.+s,
                                "headway_cost": s, "predicted_peak": 12, "fleet_cost": 0., "score": 1.+s} for s in STRENGTHS]}
                            for i in range(44)]
                        supply = {"condition": condition, "decisions": decisions, "choice_counts": [44, 0, 0, 0, 0],
                            "completed_trip_count": 43, "trip_duration_mean_s": {"False": 1800., "True": 1800.}}
                    row.update(condition=condition, supply=supply, source_control_reproduced=condition in spec.CONTROLS)
                    rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": key[0], "scenario": key[1], "software_qualified": True,
                "training_updates": 0, "native_steps": 20*61380, "evaluation": rows}
        return cells, sources

    def test_analysis_rejects_score_selection_schedule_and_budget_changes(self):
        cells, sources = self.cells()
        result = summarize(cells, sources)
        self.assertEqual(result["native_steps"], 200*61380)
        self.assertEqual(result["source_control_reproductions"], 120)
        self.assertEqual(len(result["paired_contrasts"]), 6)
        json_diagnostics = result["supply_diagnostics"]
        self.assertTrue(all(d["decisions"] == 880 and d["candidate_fleet_cost_distinguishes"] == 0
            and d["fleet_term_changes_wait_only_argmin"] == 0 for d in json_diagnostics))
        for failure in ("score", "selection", "candidate", "count", "clock", "control", "schedule", "duplicate", "budget"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "supply_aware")
            if failure == "score": row["supply"]["decisions"][0]["candidates"][0]["score"] += 1
            elif failure == "selection": row["supply"]["decisions"][0]["selected"] = 1
            elif failure == "candidate": row["supply"]["decisions"][0]["candidates"][1]["strength"] = .2
            elif failure == "count": row["supply"]["choice_counts"][0] -= 1
            elif failure == "clock": row["supply"]["decisions"][-1]["time_s"] = 0
            elif failure in {"control", "schedule"}:
                ref = next(r for r in cell["evaluation"] if r["condition"] == "rolling_1")
                if failure == "control": ref["service_cost_restricted"] += 1
                else: ref["schedule"]["release_lateness_mean_s"] = 1
            elif failure == "duplicate": cell["evaluation"].append(copy.deepcopy(row))
            else: cell["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources)

    def test_scheduler_and_worker_qualify_controls_before_supply_no_extra_episodes(self):
        task = task_specification("test_supply", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertEqual(task["local_result_dir"], task["result_dir"])
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])
        cells, sources = self.cells()
        calls = []

        def episode(root, scenario, scene, raw, checkpoint, condition):
            calls.append(condition)
            row = next(r for r in cells[root, scenario]["evaluation"] if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"], row["supply"]

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
        self.assertEqual(write.call_args.args[1]["native_steps"], 20*61380)
        self.assertIn("SUPPLY_ALLOCATION_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
