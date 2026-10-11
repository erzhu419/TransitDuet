import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_candidate_value import NativeCandidateValuePlan
from freq_hrl.domains.transit.native_value_context import NativeValueContextAudit
from scripts import run_native_transit_value_context_stage177 as spec
from scripts.analyze_native_transit_value_context_stage177 import summarize
from scripts.submit_native_transit_value_context_stage177_scheduleurm import task_specification
from tests.test_native_transit_action_value import summary_fixture


def contexts_fixture(rows):
    contexts = []
    for row in rows:
        row["context"]["route_speed_mean_mps"] = 7.
        buses = []
        for direction in (False, True):
            for i in range(row["context"]["active"][str(direction)]):
                count = row["context"]["onboard"][str(direction)] if i == 0 else 0
                buses.append({"trip_id": len(buses), "direction": direction,
                    "launch_time_s": row["time_s"]-1000., "elapsed_s": 1000.,
                    "remaining_fraction": .4, "mean_total_s": 1800., "predicted_remaining_s": 720.,
                    "load_fraction": count/100., "onboard_count": count, "capacity": 100,
                    "speed_mps": 7., "holding_s": 0., "dwelling_s": 0.})
        contexts.append({**{k: copy.deepcopy(row[k]) for k in
            ("time_s", "direction", "native_state_v2", "nominal_s", "candidates")},
            "index": row["probe_index"], "greedy": row["original_choice"], "historical_peak": 14,
            "active_buses": buses, "route_speed_mps": [7., 7.],
            "queue_by_station": [[i, d, row["context"]["queues"][str(d)]] for i, d in enumerate((False, True))]})
    decisions = [{"selected": 2, "candidates": [{"score": abs(i-2)} for i in range(6)],
        "time_s": i*100, "direction": bool(i % 2), "historical_peak": 14} for i in range(44)]
    return contexts, {"scene_seed": rows[0]["scene_seed"], "supply": {"decisions": decisions}}


class ValueContextTest(unittest.TestCase):
    def test_observer_reads_preaction_buses_without_changing_choice_or_rng(self):
        plan = object.__new__(NativeValueContextAudit)
        bus = SimpleNamespace(on_route=True, trip_id=7, direction=True, launch_time=600,
            effective_route=[SimpleNamespace(distance=1000)], travel_distance=400,
            passengers=[1, 2], capacity=50, current_speed=7., holding_time=2., dwelling_time=3.)
        plan.env = SimpleNamespace(current_time=1000, _peak_concurrent=14, bus_all=[bus],
            stations=[SimpleNamespace(station_id=2, direction=True, waiting_passengers=[1])],
            routes=[SimpleNamespace(speed_limit=7.)])
        plan.contexts, plan.decisions = [], [None]*8
        plan._incoming = {"time_s": 1000., "direction": True}
        plan.trip_duration = lambda d: 1800.
        rng = np.random.get_state()
        with patch.object(NativeCandidateValuePlan, "select_candidate", return_value=4):
            self.assertEqual(plan.select_candidate([{"score": 1}]), 4)
        self.assertEqual(plan.contexts[0]["active_buses"][0]["predicted_remaining_s"], 1080.)
        self.assertEqual(plan.contexts[0]["active_buses"][0]["onboard_count"], 2)
        self.assertNotIn("actual_remaining_s", plan.contexts[0]["active_buses"][0])
        np.testing.assert_array_equal(rng[1], np.random.get_state()[1])
        self.assertEqual(rng[2:], np.random.get_state()[2:])
        self.assertEqual((bus.travel_distance, bus.holding_time), (400, 2.))
        plan.decisions.append(None)
        with patch.object(NativeCandidateValuePlan, "select_candidate", return_value=2):
            plan.select_candidate([{"score": 1}])
        self.assertEqual(len(plan.contexts), 2)
        plan.decisions.append(None)
        with patch.object(NativeCandidateValuePlan, "select_candidate", return_value=2):
            plan.select_candidate([{"score": 1}])
        self.assertEqual(len(plan.contexts), 2)

    def test_contexts_match_all_existing_label_states_counts_and_fleet_floor(self):
        rows = summary_fixture()["training_data"][:6]
        contexts, registered = contexts_fixture(rows)
        spec.qualify_contexts(contexts, rows, registered)
        for failure in ("index", "choice", "state", "floor", "queue", "load", "age", "remaining", "future"):
            changed = copy.deepcopy(contexts)
            c = changed[0]
            if failure == "index": c["index"] += 1
            elif failure == "choice": c["greedy"] = 0
            elif failure == "state": c["native_state_v2"][0] += 1
            elif failure == "floor": c["historical_peak"] += 1
            elif failure == "queue": c["queue_by_station"][0][2] += 1
            elif failure == "load": c["active_buses"][0]["onboard_count"] += 1
            elif failure == "age": c["active_buses"][0]["elapsed_s"] += 1
            elif failure == "remaining": c["active_buses"][0]["predicted_remaining_s"] += 1
            else: c["active_buses"][0]["launch_time_s"] = c["time_s"]+1
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                spec.qualify_contexts(changed, rows, registered)

    def test_merge_reuses_labels_and_calls_exact_native_source_qualification(self):
        summary = summary_fixture()
        cells, sources = {}, {}
        for root in spec.ROOTS:
            for scenario in spec.SCENARIOS:
                rows = [r for r in summary["training_data"] if (r["root"], r["scenario"]) == (root, scenario)]
                contexts, registered = contexts_fixture(rows)
                sources[root, scenario] = registered
                observer = {"probe": contexts[-1], "supply": registered["supply"]}
                observer["policy_trace"] = spec.shared.policy_trace(observer["supply"])
                cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                    "software_qualified": True, "seed": root, "scenario": scenario,
                    "scene_seed": registered["scene_seed"], "training_updates": 0, "native_steps": 61380,
                    "observer": observer, "contexts": contexts}
        with patch.object(spec.shared, "qualify") as qualify, patch.object(spec.shared, "validate_scoring"):
            result = summarize(cells, summary, sources, {str(r): {} for r in spec.ROOTS}, {str(r): {} for r in spec.ROOTS})
        self.assertEqual(qualify.call_count, 10)
        self.assertEqual((result["new_native_steps"], result["paired_labels_reused"]), (613800, 360))
        self.assertEqual(len(result["training_data"]), 60)
        for row in result["training_data"]:
            original = next(r for r in summary["training_data"] if
                (r["root"], r["scene_seed"], r["probe_index"]) == (row["root"], row["scene_seed"], row["probe_index"]))
            self.assertEqual({k: v for k, v in row.items() if k != "occupancy_context"}, original)

    def test_tasks_are_ten_unique_unbound_cpu_replays_not_branch_reruns(self):
        tasks = [task_specification("context_test", r, s) for r in spec.ROOTS for s in spec.SCENARIOS]
        self.assertEqual(len({t["signature"] for t in tasks}), 10)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 10)
        for t in tasks:
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertEqual(len(t["stage_input_paths"]), 4)
            self.assertIn("--training-data", t["cmd"])


if __name__ == "__main__":
    unittest.main()
