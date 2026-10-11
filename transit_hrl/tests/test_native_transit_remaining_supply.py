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

from freq_hrl.domains.transit.native_remaining_time import NativeRemainingSupplyPlan, active_sample
from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from scripts import run_native_transit_remaining_supply_stage172 as spec
from scripts.analyze_native_transit_remaining_supply_stage172 import summarize
from scripts.submit_native_transit_remaining_supply_stage172_scheduleurm import task_specification
from tests import test_native_transit_distributional_supply as distributional_tests
from tests import test_native_transit_incumbent_window as incumbent_tests


class RemainingSupplyTest(unittest.TestCase):
    def test_active_shift_changes_only_active_samples_and_zero_recovers_exact_selector(self):
        env, reference, _ = distributional_tests.DistributionalSupplyTest().environment()
        env.protocol = SimpleNamespace(evaluation_end_time_s=61380)
        env.bus_all = [SimpleNamespace(on_route=True, trip_id=0, direction=False, launch_time=60.,
            effective_route=env.routes[10:], travel_distance=2500., passengers=list(range(10)), capacity=50,
            current_speed=3., holding_time=15., dwelling_time=0.)]
        old = NativeIncumbentWindowPlan(env, reference.model, "rolling_plus_incumbent")
        zero = NativeRemainingSupplyPlan(env, reference.model, {"coefficients": [0.]*11})
        corrected = NativeRemainingSupplyPlan(env, reference.model, {"coefficients": [.05]+[0.]*10})
        planned = {t.launch_turn: t.launch_time for t in env.timetables}
        starts, original = old.sampled_intervals(planned, 180.)
        zero_starts, identity = zero.sampled_intervals(planned, 180.)
        _, changed = corrected.sampled_intervals(planned, 180.)
        np.testing.assert_array_equal(starts, zero_starts)
        np.testing.assert_array_equal(original, identity)
        np.testing.assert_array_equal(changed[:, 1:], original[:, 1:])
        np.testing.assert_allclose(changed[:, 0], original[:, 0]+100., rtol=0, atol=1e-10)
        nominal = np.asarray([t.launch_time for t in env.timetables if not t.direction])
        a, _ = old.preferred_intervals(nominal, env.timetables[0], 180.)
        b, _ = zero.preferred_intervals(nominal, env.timetables[0], 180.)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(old.supply_summary(), zero.supply_summary())
        self.assertEqual(old.audits, zero.audits)
        self.assertEqual(zero.snapshots[0]["active"][0]["correction_s"], 0.)
        corrected.preferred_intervals(nominal, env.timetables[0], 180.)
        self.assertEqual(corrected.snapshots[0]["active"][0]["correction_s"], 100.)
        self.assertNotIn("actual_remaining_s", corrected.snapshots[0]["active"][0])

    def cells(self):
        previous, _, models = incumbent_tests.IncumbentWindowTest().cells()
        pack = {"duration_models": {str(r): m for r, m in models.items()},
            "remaining_models": {str(r): {"coefficients": [.05]+[0.]*10} for r in spec.ROOTS}}
        cells, sources = {}, {}
        for (root, scenario), cell in previous.items():
            w = copy.deepcopy(next(r for r in cell["evaluation"] if r["condition"] == "window_forecast"))
            w["scene_seed"] = spec.source.check_scene(root, scenario)
            ledger = [{"trip_id": tid, "direction": bool(tid % 2)} for tid in range(262)]
            w["completed_trips"] = ledger
            sources[root, scenario] = {"evaluation": [{**w, "condition": "check_control"}]}
            original = copy.deepcopy(next(r for r in cell["evaluation"] if r["condition"] == "rolling_plus_incumbent"))
            rows = []
            for condition in spec.CONDITIONS:
                row = copy.deepcopy(w if condition == "window_forecast" else original)
                row.update(condition=condition, scene_seed=w["scene_seed"], source_control_reproduced=condition in spec.CONTROLS,
                    remaining_corrections=[])
                if condition in spec.CONDITIONS[2:]:
                    for decision in row["supply"]["decisions"]:
                        now = decision["time_s"]
                        bus = SimpleNamespace(trip_id=1, direction=True, launch_time=now-120.,
                            effective_route=[SimpleNamespace(distance=500.)]*10, travel_distance=2500.,
                            passengers=[1]*10, capacity=50, current_speed=3., holding_time=0., dwelling_time=0.)
                        mean = decision["duration_estimates_s"]["True"]
                        sample = active_sample(bus, now, mean)
                        shift = 0. if condition == "zero_remaining_correction" else 100.
                        sample.update(correction_s=shift, original_sample_mean_s=mean*.5,
                            corrected_sample_mean_s=mean*.5+shift, actual_remaining_s=mean*.5+100.)
                        row["remaining_corrections"].append({"time_s": now, "direction": decision["direction"], "active": [sample]})
                if condition == "corrected_remaining": row["service_cost_restricted"] -= .01
                rows.append(row)
            cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": root,
                "scenario": scenario, "software_qualified": True, "training_updates": 0,
                "native_steps": 4*61380, "evaluation": rows}
        return cells, sources, pack

    def analyze(self, cells, sources, pack):
        return summarize(cells, sources, pack, {"models": pack["remaining_models"]}, pack["duration_models"])

    def test_analyzer_separates_prediction_accuracy_from_native_outcomes_and_checks_controls(self):
        cells, sources, pack = self.cells()
        result = self.analyze(cells, sources, pack)
        self.assertEqual(result["source_control_reproductions"], 20)
        self.assertEqual(result["window_ledger_qualified_episodes"], 30)
        self.assertEqual(result["native_steps"], 40*61380)
        self.assertEqual(len(result["regime_root_contrasts"]), 20)
        self.assertTrue(all(d["corrected_sample_mean_mae_s"] == 0 for d in result["active_diagnostics"]))
        json.dumps(result)
        for failure in ("control", "zero_scores", "zero_schedule", "model", "correction", "samples", "choice", "scene", "budget"):
            broken = copy.deepcopy(cells)
            c = broken[397, "low_noise"]
            zero, changed = c["evaluation"][2:]
            if failure == "control": c["evaluation"][0]["ep_reward"] += 1
            elif failure == "zero_scores": zero["supply"]["decisions"][0]["candidates"][0]["score"] += 1
            elif failure == "zero_schedule": zero["schedule"]["window_endpoint_shift_abs_mean_s"] += 1
            elif failure == "model":
                altered = copy.deepcopy(pack)
                altered["remaining_models"]["397"]["coefficients"][0] += .1
                with self.assertRaises(ValueError):
                    summarize(broken, sources, altered, {"models": pack["remaining_models"]}, pack["duration_models"])
                continue
            elif failure == "correction": changed["remaining_corrections"][0]["active"][0]["correction_s"] += 1
            elif failure == "samples": changed["remaining_corrections"][0]["active"][0]["original_sample_mean_s"] += 1
            elif failure == "choice": changed["supply"]["decisions"][0]["selected"] = 0
            elif failure == "scene": changed["scene_seed"] += 1
            else: c["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                self.analyze(broken, sources, pack)

    def test_worker_preserves_source_and_zero_control_before_corrected_and_stages_fresh_inputs(self):
        task = task_specification("test_remaining_supply", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        cells, sources, pack = self.cells()
        rows, calls = cells[397, "low_noise"]["evaluation"], []
        def episode(root, scenario, scene, raw, checkpoint, condition, duration, remaining):
            calls.append(condition)
            row = next(r for r in rows if r["condition"] == condition)
            return row, row["schedule"], row["supply"], [], [], row["remaining_corrections"]
        def panels(row):
            return next(r for r in rows if r["condition"] == row["condition"])["window_panels"]
        with patch.object(Path, "read_text", return_value=json.dumps(pack)), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), patch.object(spec, "validate_audits", side_effect=panels), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--models", "unused/models.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS))
        self.assertEqual(write.call_args.args[1]["native_steps"], 4*61380)
        self.assertIn("REMAINING_SUPPLY_ZERO_CONTROL_QUALIFIED", output.getvalue())
        self.assertTrue(all("completed_trips" not in r for r in write.call_args.args[1]["evaluation"]))


if __name__ == "__main__":
    unittest.main()
