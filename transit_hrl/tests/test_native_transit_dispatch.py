import copy
import sys
from types import SimpleNamespace
import unittest

from scripts import run_native_transit_dispatch_stage150 as spec
from scripts.analyze_native_transit_dispatch_stage150 import summarize
from scripts.submit_native_transit_dispatch_stage150_scheduleurm import task_specification

sys.path.insert(0, str(spec.NATIVE))
from env.sim import env_bus


class NativeDispatchTest(unittest.TestCase):
    def environment(self, command, *, nominal=300, lookahead=120, direct=True):
        env = env_bus.__new__(env_bus)
        trip = SimpleNamespace(launch_time=nominal, launch_turn=0, direction=True,
                               launched=False, target_headway=360)
        env.timetables = [trip]
        env.bus_all = []
        env._n_fleet_target, env._fleet_buffer = 12, 3
        env._upper_dispatch_lookahead_s = lookahead
        env._last_dispatch_time = {True: -9999, False: -9999}
        env._build_upper_state = lambda trip: [env.current_time]
        env._compute_dispatch_proxy_reward = lambda trip: None
        env.launch_bus = lambda trip: None
        queries = []
        def callback(state, trip):
            queries.append((trip.launch_turn, env.current_time))
            self.assertEqual(state, [env.current_time])
            if direct:
                trip._original_launch = trip.launch_time
                trip._delta_t = command
            return 360
        env._upper_policy_callback = callback
        return env, trip, queries

    def at(self, env, time):
        env.current_time = time
        env._dispatch_due_trips()

    def test_signed_commands_commit_once_and_release_at_actual_time(self):
        for command in (-120, -60, 0, 60, 120):
            with self.subTest(command=command):
                env, trip, queries = self.environment(command)
                self.at(env, 179)
                self.assertFalse(queries)
                self.at(env, 180)
                self.assertEqual(queries, [(0, 180)])
                due = 300 + command
                if due > 180:
                    self.assertFalse(trip.launched)
                    self.at(env, due - 1)
                self.at(env, due)
                self.assertEqual(trip._actual_launch_time, due)
                self.at(env, 500)
                self.assertEqual(queries, [(0, 180)])
                self.assertEqual(trip.launch_time, 300)
                ledger = spec.dispatch_ledger(env, queries)
                self.assertEqual(ledger["actual_shift_mean_s"], command)
                self.assertEqual(ledger["release_lateness_mean_s"], 0)

    def test_nominal_query_reproduces_original_negative_command_failure(self):
        env, trip, queries = self.environment(-60, lookahead=0)
        self.at(env, 240)
        self.assertFalse(queries)
        self.at(env, 300)
        self.assertEqual(trip._actual_launch_time, 300)
        ledger = spec.dispatch_ledger(env, queries)
        self.assertEqual(ledger["advance_count"], 0)
        self.assertEqual(ledger["commitment_clipped_count"], 1)

    def test_service_start_clips_to_observed_clock_not_negative_time(self):
        env, trip, queries = self.environment(-120, nominal=30)
        self.at(env, 0)
        self.assertEqual(trip._actual_launch_time, 0)
        self.assertEqual(queries, [(0, 0)])
        self.assertEqual(spec.dispatch_ledger(env, queries)["commitment_clipped_count"], 1)

    def test_fleet_cap_delays_execution_without_requerying_upper(self):
        env, trip, queries = self.environment(-60)
        env._n_fleet_target, env._fleet_buffer = 0, 1
        env.bus_all = [SimpleNamespace(on_route=True)]
        self.at(env, 180)
        self.at(env, 240)
        self.assertFalse(trip.launched)
        env.bus_all.clear()
        self.at(env, 261)
        self.assertEqual(trip._actual_launch_time, 261)
        self.assertEqual(queries, [(0, 180)])
        self.assertEqual(spec.dispatch_ledger(env, queries)["release_lateness_mean_s"], 21)

    def test_no_callback_or_v1_headway_cannot_launch_before_nominal(self):
        for callback in (False, True):
            env, trip, queries = self.environment(0, direct=False)
            if not callback:
                env._upper_policy_callback = None
            self.at(env, 180)
            self.assertFalse(trip.launched)
            self.at(env, 300)
            self.assertEqual(trip._actual_launch_time, 300)
            self.assertEqual(len(queries), int(callback))

    def test_ledger_rejects_duplicate_queries_and_premature_launch(self):
        env, trip, queries = self.environment(60)
        self.at(env, 180)
        self.at(env, 360)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            spec.dispatch_ledger(env, queries + queries)
        trip._actual_launch_time = 359
        with self.assertRaisesRegex(RuntimeError, "committed release"):
            spec.dispatch_ledger(env, queries)

    def cells(self):
        cells = {}
        for root in spec.ROOTS:
            rows = [{"condition": condition, "scenario": "low_noise", "scene_seed": root,
                "passengers_generated": 100, "N_fleet": 12, "simulation_end_time_s": 61380,
                "dispatch": {"advance_count": 1 if condition == "signed_minus60" else 0},
                **{metric: 1 for metric in spec.frontier.source_spec.routing.METRICS}}
                for condition in spec.CONDITIONS]
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": root, "software_qualified": True, "baseline_reproduced": True,
                "zero_dispatch_reproduced": True, "training_updates": 0,
                "native_steps": 5 * 61380, "evaluation": rows}
        return cells

    def test_summary_is_execution_qualification_not_training_evidence(self):
        result = summarize(self.cells())
        self.assertEqual(result["native_steps"], 10 * 61380)
        self.assertEqual(result["training_updates"], 0)
        self.assertIn("frozen_HIRO_trained_lower", result["stage"])
        self.assertEqual(len(result["panels"]), 10)

    def test_merge_rejects_missing_root_duplicate_and_changed_scene(self):
        for failure in ("missing_root", "duplicate", "scene"):
            cells = self.cells()
            if failure == "missing_root":
                cells.pop(spec.ROOTS[0])
            elif failure == "duplicate":
                cells[spec.ROOTS[0]]["evaluation"].append(copy.deepcopy(cells[spec.ROOTS[0]]["evaluation"][0]))
            else:
                cells[spec.ROOTS[0]]["evaluation"][0]["scene_seed"] += 1
            with self.assertRaises(ValueError):
                summarize(cells)

    def test_scheduler_stages_code_only_and_has_no_node_binding(self):
        task = task_specification("test_signed_dispatch", 217)
        self.assertEqual([path.split("/")[-1] for path in task["stage_input_paths"]],
                         ["scripts", "freq_hrl", "native_freqduet"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(len(task["allowed_nodes"]), 6)
        self.assertIn("run_native_transit_dispatch_stage150.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
