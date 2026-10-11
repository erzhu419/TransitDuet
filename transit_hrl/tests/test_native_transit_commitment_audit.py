import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from freq_hrl.domains.transit.native_commitment_audit import (
    NativeCommitmentAudit, window_ensemble_peaks, observed_window_peak, fleet_cost)
from scripts.analyze_native_transit_commitment_audit_stage167 import validate_audits
from scripts.submit_native_transit_commitment_audit_stage167_scheduleurm import task_specification
from tests import test_native_transit_distributional_supply as distributional_tests


class CommitmentAuditTest(unittest.TestCase):
    def test_prefix_cut_excludes_future_launches_and_keeps_historical_peak(self):
        durations = np.array([[20., 20., 20.], [3., 3., 3.]])
        np.testing.assert_array_equal(window_ensemble_peaks([0., 5., 10.], durations, 10., 0), [2, 1])
        np.testing.assert_array_equal(window_ensemble_peaks([0., 5., 10.], durations, 5., 4), [4, 4])
        self.assertEqual(fleet_cost({"12": 32, "14": 32}), 1/6)

    def test_window_observations_include_active_trips_and_exclude_boundary_launches(self):
        trips = [{"launch_time_s": 0., "completion_time_s": 10.},
            {"launch_time_s": 5., "completion_time_s": 15.},
            {"launch_time_s": 10., "completion_time_s": 20.}]
        self.assertEqual(observed_window_peak(trips, 6., 10., 0), 2)
        self.assertEqual(observed_window_peak(trips, 10., 15., 0), 2)
        self.assertEqual(observed_window_peak(trips, 20., 30., 4), 4)

    def test_passive_audit_preserves_full_scores_actions_dispatch_and_next_anchor(self):
        helper = distributional_tests.DistributionalSupplyTest()
        env, base, _ = helper.environment()
        audit_env, reference, _ = helper.environment()
        audit_env.protocol = SimpleNamespace(evaluation_end_time_s=8000)
        audit = NativeCommitmentAudit(audit_env, reference.model)
        audit_env._upper_policy_callback = audit
        for tick in range(300+18*360+180):
            env.current_time = audit_env.current_time = tick
            env._dispatch_due_trips()
            audit_env._dispatch_due_trips()
        self.assertEqual(audit.summarize(), base.summarize())
        self.assertEqual(audit.supply_summary(), base.supply_summary())
        self.assertEqual(len(audit.audits), len(audit.decisions))
        scheduled = {t.launch_turn: t._freqduet_scheduled_launch for t in audit_env.timetables}
        for i, record in enumerate(audit.audits):
            chosen = record["candidates"][record["selected"]]
            nxt = next((a for a in audit.audits[i+1:] if a["direction"] == record["direction"]), None)
            self.assertEqual(chosen["window_end_s"], nxt["time_s"] if nxt is not None else 8000)
            self.assertEqual(chosen["next_trip_id"], nxt["proposed_trip_ids"][0] if nxt is not None else None)
            for tid, planned in zip(record["proposed_trip_ids"][:7], record["selected_full_planned_s"][:7]):
                self.assertEqual(scheduled[tid], planned)
            self.assertTrue(all(sum(c["prefix_peak_counts"].values()) == 64 for c in record["candidates"]))

    def row(self):
        trips = [{"trip_id": i, "direction": bool(i%2), "launch_time_s": float(i), "completion_time_s": float(i+1)} for i in range(3)]
        decisions, audits = [], []
        for i in range(3):
            ids = [0, 2] if i == 0 else [i]
            candidates = [{"fleet_cost": 0., "score": 1.+j} for j in range(5)]
            decision = {"time_s": float(i), "direction": bool(i%2), "historical_peak": int(i > 0),
                "selected": 0, "candidates": candidates}
            decisions.append(decision)
            audits.append({**{k: v for k, v in decision.items() if k != "candidates"},
                "proposed_trip_ids": ids, "selected_full_planned_s": [float(t) for t in ids],
                "candidates": [{"next_trip_id": 2 if i == 0 else None, "window_end_s": 2. if i == 0 else 10.,
                    "prefix_planned_s": [float(t) for t in ids], "prefix_peak_counts": {"12": 64},
                    "prefix_fleet_cost": 0., "prefix_wait_cost": 1.+j, "prefix_score": 1.+j,
                    "full_fleet_cost": 0.} for j in range(5)]})
        return {"simulation_end_time_s": 10, "peak_fleet": 1, "completed_trips": trips,
            "supply": {"decisions": decisions}, "audits": audits,
            "schedule": {"scheduled_trips": 3, "schedules": [
                {"direction": False, "actual_s": [0., 2.], "planned_s": [0., 2.]},
                {"direction": True, "actual_s": [1.], "planned_s": [1.]}]}}

    def test_analysis_rejects_changed_boundaries_scores_anchors_and_trip_accounting(self):
        row = self.row()
        panels = validate_audits(row)
        self.assertEqual(len(panels), 3)
        self.assertTrue(all(p["actual_window_peak"] == 1 and p["actual_window_fleet_cost"] == 0 for p in panels))
        json.dumps(panels)
        for failure in ("boundary", "score", "histogram", "selected", "anchor", "ledger", "history"):
            broken = copy.deepcopy(row)
            if failure == "boundary": broken["audits"][0]["candidates"][0]["window_end_s"] += 1
            elif failure == "score": broken["audits"][0]["candidates"][0]["prefix_score"] += 1
            elif failure == "histogram": broken["audits"][0]["candidates"][0]["prefix_peak_counts"] = {"12": 63}
            elif failure == "selected": broken["audits"][0]["selected"] = 1
            elif failure == "anchor": broken["audits"][0]["selected_full_planned_s"][1] += 1
            elif failure == "ledger": broken["completed_trips"][0]["completion_time_s"] = 2.5
            else: broken["supply"]["decisions"][2]["historical_peak"] = broken["audits"][2]["historical_peak"] = 2
            with self.subTest(failure=failure), self.assertRaises(ValueError): validate_audits(broken)

    def test_scheduler_uses_all_nodes_new_inputs_and_four_episode_worker(self):
        task = task_specification("test_commitment_audit", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        self.assertIn("run_native_transit_commitment_audit_stage167.py", task["cmd"])
        self.assertIn("--duration-model", task["cmd"])


if __name__ == "__main__":
    unittest.main()
