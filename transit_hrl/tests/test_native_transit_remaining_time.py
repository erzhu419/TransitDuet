import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit.native_remaining_time import NativeRemainingTimeAudit, fit_remaining_model
from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from scripts import run_native_transit_remaining_time_stage171 as spec
from scripts.analyze_native_transit_remaining_time_stage171 import summarize
from scripts.submit_native_transit_remaining_time_stage171_scheduleurm import task_specification
from tests import test_native_transit_distributional_supply as distributional_tests
from tests import test_native_transit_duration_calibration as duration_tests


def sample(now, trip, mean=2000.):
    fraction = np.clip(1-(now-trip["launch_time_s"])/(trip["completion_time_s"]-trip["launch_time_s"]), 0., 1.)
    return {"trip_id": trip["trip_id"], "direction": trip["direction"], "launch_time_s": trip["launch_time_s"],
        "elapsed_s": now-trip["launch_time_s"], "remaining_fraction": float(fraction),
        "mean_total_s": mean, "predicted_remaining_s": max(mean*fraction, 1.),
        "load_fraction": .2, "speed_mps": 3., "holding_s": 0., "dwelling_s": 0.}


class RemainingTimeTest(unittest.TestCase):
    def test_passive_snapshot_does_not_change_window_preferences_or_candidate_scores(self):
        env, original, _ = distributional_tests.DistributionalSupplyTest().environment()
        env.protocol = SimpleNamespace(evaluation_end_time_s=61380)
        env.bus_all = [SimpleNamespace(on_route=True, trip_id=0, direction=False, launch_time=60.,
            effective_route=env.routes[10:], travel_distance=2500., passengers=list(range(10)), capacity=50,
            current_speed=3., holding_time=15., dwelling_time=0.)]
        audited = NativeRemainingTimeAudit(env, original.model)
        control = NativeIncumbentWindowPlan(env, original.model, "incumbent_only")
        nominal = np.asarray([t.launch_time for t in env.timetables if not t.direction])
        first, _ = audited.preferred_intervals(nominal, env.timetables[0], 180.)
        second, _ = control.preferred_intervals(nominal, env.timetables[0], 180.)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(audited.decisions, control.decisions)
        s = audited.snapshots[0]["active"][0]
        self.assertEqual(s["elapsed_s"], 120.)
        self.assertEqual(s["remaining_fraction"], .5)
        self.assertEqual(s["predicted_remaining_s"], 1500.)
        self.assertNotIn("actual_remaining_s", s)

    def cells(self):
        sources = duration_tests.DurationCalibrationTest().cells()
        cells = {}
        for key, source in sources.items():
            fit = source["calibration"][0]
            ledger = sorted(({"trip_id": tid, "direction": d == "True", "launch_time_s": launch,
                "completion_time_s": completion} for d, r in fit["directions"].items()
                for tid, launch, completion in zip(r["trip_ids"], r["launch_s"], r["completion_s"])), key=lambda t: t["trip_id"])
            rows = []
            for condition in spec.CONDITIONS:
                row = copy.deepcopy(fit)
                row.update(condition=condition, scene_seed=fit["scene_seed"] if condition == "fit_audit" else spec.check_scene(*key),
                    source_control_reproduced=condition != "check_control", snapshots=[], completed_trips=copy.deepcopy(ledger))
                if condition != "check_control":
                    for now in np.linspace(180, 46000, 44):
                        active = [sample(now, t, 1600.) for t in ledger if t["launch_time_s"] < now <= t["completion_time_s"]]
                        row["snapshots"].append({"time_s": float(now), "direction": False, "active": active})
                    spec.label_snapshots(row["snapshots"], ledger)
                rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": key[0],
                "scenario": key[1], "software_qualified": True, "training_updates": 0,
                "native_steps": 3*61380, "evaluation": rows}
        return cells, sources

    def test_fit_and_fresh_check_accounting_and_reject_changed_labels_or_controls(self):
        cells, sources = self.cells()
        result = summarize(cells, sources)
        self.assertEqual(result["source_control_reproductions"], 20)
        self.assertEqual(result["native_steps"], 30*61380)
        self.assertEqual(len(result["regime_panels"]), 10)
        self.assertTrue(all(r["corrected"]["mae_s"] < r["progress"]["mae_s"] for r in result["root_summary"]))
        models = result["models"]
        for cell in cells.values():
            for snapshot in cell["evaluation"][2]["snapshots"]:
                for s in snapshot["active"]:
                    s["actual_remaining_s"] += 1000.
        fits = [cells[397, s]["evaluation"][0] for s in spec.SCENARIOS]
        self.assertEqual(fit_remaining_model(fits), models["397"])
        cells, sources = self.cells()
        for failure in ("label", "ledger", "missing_active", "physics", "schedule", "scene", "budget"):
            broken = copy.deepcopy(cells)
            c = broken[397, "low_noise"]
            r = c["evaluation"][2]
            snapshot = next(s for s in r["snapshots"] if s["active"])
            if failure == "label": snapshot["active"][0]["actual_remaining_s"] += 1
            elif failure == "ledger": r["completed_trips"][0]["completion_time_s"] += 1
            elif failure == "missing_active": snapshot["active"].pop()
            elif failure == "physics": r["service_cost_restricted"] += 1
            elif failure == "schedule": r["schedule"]["window_endpoint_shift_abs_mean_s"] += 1
            elif failure == "scene": r["scene_seed"] += 1
            else: c["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources)

    def test_pre_drive_snapshot_can_complete_in_the_same_tick(self):
        trip = {"trip_id": 0, "direction": False, "launch_time_s": 100., "completion_time_s": 200.}
        snapshots = [{"time_s": 200., "direction": False, "active": [sample(200., trip)]}]
        spec.label_snapshots(snapshots, [trip])
        self.assertEqual(snapshots[0]["active"][0]["actual_remaining_s"], 0.)
        with self.assertRaises(ValueError): spec.label_snapshots([{**snapshots[0], "time_s": 201.}], [trip])

    def test_scheduler_and_worker_collect_three_episodes_with_fresh_inputs_only(self):
        task = task_specification("test_remaining_time", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        cells, sources = self.cells()
        rows = cells[397, "low_noise"]["evaluation"]
        model = distributional_tests.DistributionalSupplyTest().model()
        with patch.object(Path, "read_text", return_value=json.dumps({"397": model})), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=rows) as run, \
                patch.object(spec, "raw_directory", return_value=Path("unused")), patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--duration-model", "unused/models.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual([c.args[5] for c in run.call_args_list], [True, False, True])
        self.assertEqual(write.call_args.args[1]["native_steps"], 3*61380)
        self.assertIn("NATIVE_REMAINING_TIME_COMPLETE", output.getvalue())
        old = {spec.source.scene_seed(r, s, split) for r in spec.ROOTS for s in spec.SCENARIOS for split in ("fit", "check")}
        self.assertFalse(old.intersection({spec.check_scene(r, s) for r in spec.ROOTS for s in spec.SCENARIOS}))


if __name__ == "__main__":
    unittest.main()
