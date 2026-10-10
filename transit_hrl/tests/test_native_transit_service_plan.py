import copy
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.core.time_allocation import budgeted_time_points
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan
from scripts import run_native_transit_service_plan_stage156 as spec
from scripts.analyze_native_transit_service_plan_stage156 import summarize
from scripts.submit_native_transit_service_plan_stage156_scheduleurm import task_specification

sys.path.insert(0, str(spec.NATIVE))
from env.sim import env_bus


class ServicePlanTest(unittest.TestCase):
    def environment(self, condition, *, trips_per_direction=6):
        env = env_bus.__new__(env_bus)
        env.timetables = sorted([SimpleNamespace(launch_time=300 + i * 360 + direction * 180,
            launch_turn=i * 2 + direction, direction=bool(direction), launched=False, target_headway=360)
            for i in range(trips_per_direction) for direction in (0, 1)], key=lambda tt: tt.launch_time)
        env.bus_all = []
        env._n_fleet_target, env._fleet_buffer = 12, 3
        env._upper_dispatch_lookahead_s = 120
        env._last_dispatch_time = {True: -9999, False: -9999}
        env._build_upper_state = lambda trip: [env.current_time]
        env._compute_dispatch_proxy_reward = lambda trip: None
        env.launch_bus = lambda trip: None
        env.frequency_tracker = SimpleNamespace(bin_interval_s=60,
            global_state=SimpleNamespace(forecast=lambda t: 10 + t))
        plan = NativeServicePlan(env, condition)
        env._upper_policy_callback = plan
        return env, plan

    def execute(self, condition, *, trips=6):
        env, plan = self.environment(condition, trips_per_direction=trips)
        for tick in range(300 + trips * 360 + 180):
            env.current_time = tick
            env._dispatch_due_trips()
        return env, plan

    def test_allocation_preserves_endpoints_count_and_bounded_intervals(self):
        nominal = np.arange(6) * 360 + 300
        for amplitude in (-1000, -120, 0, 120, 1000):
            preferred = 360 + amplitude * np.linspace(-1, 1, 5)
            planned = budgeted_time_points(nominal, preferred, minimum=240, maximum=480)
            self.assertEqual(len(planned), 6)
            self.assertEqual(planned[0], nominal[0])
            self.assertEqual(planned[-1], nominal[-1])
            self.assertTrue(np.all(np.diff(planned) >= 240))
            self.assertTrue(np.all(np.diff(planned) <= 480))
        np.testing.assert_array_equal(budgeted_time_points(nominal, [360] * 5,
            minimum=240, maximum=480), nominal)
        front = budgeted_time_points(nominal, [240, 300, 360, 420, 480], minimum=240, maximum=480)
        self.assertEqual(np.max(np.abs(front - nominal)), 180)

    def test_common_interval_offset_is_not_service_allocation(self):
        nominal = np.array([0, 300, 660, 1080])
        planned = budgeted_time_points(nominal, np.diff(nominal) + 70, minimum=240, maximum=480)
        np.testing.assert_array_equal(planned, nominal)
        with self.assertRaises(ValueError):
            budgeted_time_points([0, 100, 200], [300, 300], minimum=240, maximum=480)

    def test_real_dispatch_executes_changed_intervals_and_lower_goals(self):
        for condition in ("nominal_plan", "frontload", "backload", "causal_forecast", "forecast_reversed"):
            env, plan = self.execute(condition)
            ledger = plan.summarize()
            self.assertEqual(ledger["endpoint_error_max_s"], 0)
            self.assertEqual(ledger["release_lateness_mean_s"], 0)
            self.assertEqual(ledger["actual_target_error_abs_mean_s"], 0)
            self.assertEqual(ledger["queries"], 12)
            self.assertEqual(ledger["blocks"], 2)
            self.assertEqual(ledger["changed_headway_count"] == 0, condition == "nominal_plan")
            for tt in env.timetables:
                self.assertEqual(tt._actual_launch_time, tt._freqduet_scheduled_launch)
                self.assertEqual(tt.target_headway, plan._targets[tt.launch_turn])

    def test_causal_forecast_commitment_is_not_rewritten_by_later_observations(self):
        env, plan = self.environment("causal_forecast")
        env.current_time = 180
        env._dispatch_due_trips()
        first = copy.deepcopy(plan.blocks[0])
        env.frequency_tracker.global_state.forecast = lambda t: 100000 / (1 + t)
        for tick in range(181, 2700):
            env.current_time = tick
            env._dispatch_due_trips()
        self.assertEqual(plan.blocks[0], first)
        self.assertEqual([tt._freqduet_scheduled_launch for tt in env.timetables if not tt.direction], first["planned_s"])

    def test_forecast_and_reversal_have_matched_opposite_adjustments(self):
        _, forward = self.execute("causal_forecast")
        _, reverse = self.execute("forecast_reversed")
        for a, b in zip(forward.blocks, reverse.blocks):
            nominal = np.asarray(a["nominal_s"])
            np.testing.assert_array_equal(np.asarray(a["planned_s"]) - nominal,
                                          nominal - np.asarray(b["planned_s"]))

    def test_fleet_cap_delays_are_recorded_without_replanning(self):
        env, plan = self.environment("frontload")
        env._fleet_buffer = 0
        env._n_fleet_target = 1
        env.bus_all = [SimpleNamespace(on_route=True)]
        for tick in range(400):
            env.current_time = tick
            env._dispatch_due_trips()
        first = copy.deepcopy(plan.blocks[0])
        env.bus_all.clear()
        for tick in range(400, 2700):
            env.current_time = tick
            env._dispatch_due_trips()
        self.assertEqual(plan.blocks[0], first)
        self.assertGreater(plan.summarize()["release_lateness_mean_s"], 0)

    def test_tail_blocks_and_interblock_budget_are_preserved(self):
        for count in (7, 8, 11):
            env, plan = self.execute("frontload", trips=count)
            ledger = plan.summarize()
            self.assertEqual(ledger["queries"], count * 2)
            self.assertEqual(ledger["endpoint_error_max_s"], 0)
            self.assertEqual(ledger["actual_target_error_abs_mean_s"], 0)

    def cells(self):
        cells = {}
        for root in spec.ROOTS:
            rows = []
            for condition in spec.CONDITIONS:
                for scenario in spec.contract()["scenarios"]:
                    for scene in spec.source_spec.scene_seeds(root, scenario, preflight=False):
                        row = {"condition": condition, "scenario": scenario, "scene_seed": scene,
                            **{k: 1. for k in spec.source_spec.authority.routing.METRICS},
                            "N_fleet": 12, "simulation_end_time_s": 61380, "passengers_generated": 100,
                            "lower_action_mean": 5, "scheduled_trips": 12, "blocks": [], "service_plan": None}
                        if condition != "source_baseline":
                            _, plan = self.execute(condition)
                            row.update(blocks=plan.blocks, service_plan=plan.summarize())
                        rows.append(row)
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": root, "method": spec.METHOD, "software_qualified": True,
                "baseline_reproduced": True, "neutral_reproduced": True, "training_updates": 0,
                "native_steps": len(rows) * 61380, "evaluation": rows}
        return cells

    def test_analysis_separates_authority_from_learning(self):
        cells = self.cells()
        result = summarize(cells)
        self.assertEqual(result["native_steps"], 240 * 61380)
        self.assertEqual(result["training_updates"], 0)
        self.assertIn("not_learned_upper", result["stage"])
        for failure in ("duplicate", "endpoint", "lower_goal", "budget", "pairing"):
            broken = copy.deepcopy(cells)
            cell = broken[spec.ROOTS[0]]
            row = next(r for r in cell["evaluation"] if r["condition"] == "frontload")
            if failure == "duplicate":
                cell["evaluation"].append(copy.deepcopy(row))
            elif failure == "endpoint":
                row["blocks"][0]["planned_s"][-1] += 1
            elif failure == "lower_goal":
                row["blocks"][0]["target_headways_s"][1] += 10
            elif failure == "budget":
                cell["native_steps"] += 1
            else:
                row["passengers_generated"] += 1
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                summarize(broken)

    def test_dispatch_is_code_only_unpinned_and_frozen(self):
        task = task_specification("test_service_plan", spec.ROOTS[0])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertIsNone(task.get("require_node"))
        self.assertIn("run_native_transit_service_plan_stage156.py", task["cmd"])
        self.assertEqual([p.split("/")[-1] for p in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])
        with self.assertRaisesRegex(RuntimeError, "training update"):
            spec.reject_training()


if __name__ == "__main__":
    unittest.main()
