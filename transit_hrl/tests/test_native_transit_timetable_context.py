import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_timetable_context import (
    NativeTimetableContextAudit, event_features, pending_timetable, qualify_timetable)
from freq_hrl.domains.transit.native_value_context import NativeValueContextAudit
from scripts import run_native_transit_timetable_context_stage185 as spec
from scripts.analyze_native_transit_timetable_context_stage185 import summarize
from scripts.submit_native_transit_timetable_context_stage185_scheduleurm import task_specification
from tests.test_native_transit_floor_value import fixture as old_fixture
from tests.test_native_transit_value_context import contexts_fixture


def add_curves(row):
    row["time_s"] = 8300.
    row["nominal_s"] = (8640.+np.arange(10)*360.).tolist()
    curves = []
    for i, c in enumerate(row["candidates"]):
        prefix = (8640.+np.arange(7)*(300+i*15)).tolist()
        c["commitment"]["prefix_planned_s"] = prefix
        c["score"] = abs(i-2)
        curves.append([*prefix, *np.linspace(prefix[-1], row["nominal_s"][-1], 4)[1:].tolist()])
    trips = [{"trip_id": 10+i, "direction": row["direction"], "nominal_s": t,
        "scheduled_s": t, "committed": False, "target_headway_s": 360.} for i, t in enumerate(row["nominal_s"])]
    trips += [{"trip_id": i, "direction": direction, "nominal_s": t, "scheduled_s": t,
        "committed": True, "target_headway_s": 360.} for i, direction, t in
        ((0, row["direction"], 8460.), (1, not row["direction"], 8520.))]
    return {"pending_trips": sorted(trips, key=lambda t: (t["nominal_s"], t["direction"])),
        "candidate_trip_ids": list(range(10, 20)), "candidate_planned_s": curves}


def fixture():
    rows = spec.prior.source.source.source.training_rows(*old_fixture())
    cells, sources = {}, {}
    for root in spec.ROOTS:
        for scenario in spec.SCENARIOS:
            scenes = sorted({r["scene_seed"] for r in rows if (r["root"], r["scenario"]) == (root, scenario)})
            for index, scene in enumerate(scenes):
                group = [r for r in rows if (r["root"], r["scenario"], r["scene_seed"]) == (root, scenario, scene)]
                tables = [add_curves(r) for r in group]
                contexts, registered = contexts_fixture(group)
                for c, row, table in zip(contexts, group, tables):
                    row["occupancy_context"] = {k: copy.deepcopy(c[k]) for k in spec.OCCUPANCY_KEYS}
                    c["timetable_context"] = table
                metrics = {k: 0. for k in spec.shared.METRICS}
                registered.update(metrics, scenario=scenario, N_fleet=12, simulation_end_time_s=61380)
                observer = {k: registered[k] for k in spec.OBSERVER_KEYS if k not in
                    ("source_control_reproduced", "window_ledger_qualified")}
                observer.update(source_control_reproduced=True, window_ledger_qualified=True)
                cells[root, scenario, index] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                    "software_qualified": True, "seed": root, "scenario": scenario, "scene_index": index,
                    "scene_seed": scene, "training_updates": 0, "native_steps": 61380, "observer": observer,
                    "policy_trace": spec.shared.policy_trace(registered["supply"]), "contexts": contexts}
                sources[root, scenario, index] = registered
    return cells, rows, sources


class TimetableContextTest(unittest.TestCase):
    def test_pending_table_includes_both_directions_and_committed_not_launched(self):
        def trip(tid, direction, launched=False):
            return SimpleNamespace(launch_turn=tid, direction=direction, launched=launched,
                launch_time=1000+tid*180, _freqduet_scheduled_launch=1020+tid*180, target_headway=360.)
        trips = [trip(2, True), trip(1, False), trip(0, True, True)]
        plan = SimpleNamespace(env=SimpleNamespace(timetables=trips), _targets={1: 360.})
        records = pending_timetable(plan)
        self.assertEqual([t["trip_id"] for t in records], [1, 2])
        self.assertEqual([t["committed"] for t in records], [True, False])
        self.assertEqual(records[0]["scheduled_s"], 1200.)

    def test_capture_is_passive_and_keeps_every_candidate_curve(self):
        plan = object.__new__(NativeTimetableContextAudit)
        trips = [SimpleNamespace(launch_turn=i, direction=True, launched=False, launch_time=1000+i*360,
            target_headway=360.) for i in range(3)]
        plan.env = SimpleNamespace(timetables=trips)
        plan._targets, plan.decisions, plan.contexts = {}, [None]*8, []
        plans = [[1000., 1300., 1720.], [1000., 1360., 1720.]]
        def preferred(nominal, trip, now):
            plan._audit_remaining = trips
            for proposed in plans:
                self.assertEqual(plan.candidate_fleet_cost(12, dict(enumerate(proposed)), now), (.25, {"original": True}))
            self.assertEqual(plan.select_candidate([{}, {}]), 1)
            return [360., 360.], [1., 1.]
        def selected(candidates):
            plan.contexts.append({"index": 8})
            return 1
        before, rng = copy.deepcopy([vars(t) for t in trips]), np.random.get_state()
        with patch.object(NativeValueContextAudit, "preferred_intervals", side_effect=preferred), \
                patch.object(NativeValueContextAudit, "candidate_fleet_cost", return_value=(.25, {"original": True})), \
                patch.object(NativeValueContextAudit, "select_candidate", side_effect=selected):
            self.assertEqual(plan.preferred_intervals([1000., 1360., 1720.], trips[0], 880.), ([360., 360.], [1., 1.]))
        self.assertEqual(plan.contexts[0]["timetable_context"]["candidate_planned_s"], plans)
        self.assertEqual([vars(t) for t in trips], before)
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1])
        self.assertEqual(np.random.get_state()[2:], rng[2:])

    def test_sequence_preserves_tail_and_other_direction_and_ignores_outcomes_ids(self):
        row = spec.prior.source.source.source.training_rows(*old_fixture())[0]
        row["timetable_context"] = add_curves(row)
        x = event_features(row)
        self.assertEqual(x.shape, (6, 12, 6))
        changed = copy.deepcopy(row)
        changed.update(labels=[], root=0, scenario="unused", scene_seed=0)
        table = changed["timetable_context"]
        for t in table["pending_trips"]:
            t["trip_id"] += 999
            t["actual_future_launch_s"] = -1000.
        table["candidate_trip_ids"] = [i+999 for i in table["candidate_trip_ids"]]
        np.testing.assert_array_equal(event_features(changed), x)
        changed = copy.deepcopy(row)
        changed["timetable_context"]["candidate_planned_s"][0][8] += 30.
        self.assertFalse(np.array_equal(event_features(changed)[0], x[0]))
        np.testing.assert_array_equal(event_features(changed)[1:], x[1:])
        changed = copy.deepcopy(row)
        other = next(t for t in changed["timetable_context"]["pending_trips"] if t["direction"] != row["direction"])
        other["scheduled_s"] += 30.
        self.assertTrue(all(not np.array_equal(a, b) for a, b in zip(event_features(changed), x)))
        np.testing.assert_array_equal(x[:, :2, 2], x[:, :2, 1])

    def test_qualification_requires_full_prefix_geometry_and_old_occupancy(self):
        cells, rows, sources = fixture()
        key = next(iter(cells))
        cell, registered = cells[key], sources[key]
        group = [r for r in rows if (r["root"], r["scenario"], r["scene_seed"]) ==
            (cell["seed"], cell["scenario"], cell["scene_seed"])]
        spec.qualify_contexts(cell["contexts"], group, registered)
        for failure in ("prefix", "endpoint", "ids", "anchor", "interval", "occupancy"):
            changed = copy.deepcopy(cell["contexts"])
            table = changed[0]["timetable_context"]
            if failure == "prefix": table["candidate_planned_s"][0][1] += 1.
            elif failure == "endpoint": table["candidate_planned_s"][0][-1] += 1.
            elif failure == "ids": table["candidate_trip_ids"].pop()
            elif failure == "anchor": table["pending_trips"][-1]["scheduled_s"] += 1.
            elif failure == "interval": table["candidate_planned_s"][0][-2] -= 1000.
            else: changed[0]["active_buses"][0]["speed_mps"] += 1.
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                spec.qualify_contexts(changed, group, registered)

    def test_join_preserves_all_twenty_groups_and_720_labels(self):
        cells, rows, sources = fixture()
        before = copy.deepcopy(rows)
        result = summarize(cells, rows, sources)
        self.assertEqual((result["source_control_reproductions"], result["training_states"], result["paired_labels_reused"]), (20, 120, 720))
        self.assertEqual((result["new_native_steps"], result["event_count_range"]), (1227600, [12, 12]))
        self.assertEqual(rows, before)
        by_key = {(r["root"], r["scene_seed"], r["probe_index"]): r for r in rows}
        for row in result["training_data"]:
            self.assertEqual({k: v for k, v in row.items() if k != "timetable_context"},
                by_key[row["root"], row["scene_seed"], row["probe_index"]])
        for failure in ("missing_scene", "wrong_scene", "outcome", "trace", "learning"):
            changed = copy.deepcopy(cells)
            key = next(iter(changed))
            if failure == "missing_scene": changed.pop(key)
            elif failure == "wrong_scene": changed[key]["scene_index"] = 2
            elif failure == "outcome": changed[key]["observer"]["ep_reward"] += 1.
            elif failure == "trace": changed[key]["policy_trace"][0]["selected"] = 0
            else: changed[key]["training_updates"] = 1
            with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed, rows, sources)

    def test_twenty_unbound_tasks_only_sync_compact_cells(self):
        tasks = [task_specification("unit_timetable", r, s, i) for r in spec.ROOTS for s in spec.SCENARIOS for i in (0, 1)]
        self.assertEqual(len({t["signature"] for t in tasks}), 20)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 20)
        for t in tasks:
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertEqual(len(t["stage_input_paths"]), 4)
            self.assertIn("--scene-index", t["cmd"])
            self.assertNotIn("_raw", t["result_dir"])


if __name__ == "__main__":
    unittest.main()
