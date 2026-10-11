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

from freq_hrl.domains.transit.native_distributional_supply_plan import (
    NativeDistributionalSupplyPlan, ensemble_peaks, residual_draws, MEMBERS, FLEET_MODEL)
from freq_hrl.domains.transit.native_supply_plan import occupancy_peak
from scripts import run_native_transit_distributional_supply_stage166 as spec
from scripts.analyze_native_transit_distributional_supply_stage166 import summarize
from scripts.submit_native_transit_distributional_supply_stage166_scheduleurm import task_specification
from tests import test_native_transit_calibrated_supply as calibrated_tests
from tests import test_native_transit_duration_calibration as duration_tests


class DistributionalSupplyTest(unittest.TestCase):
    def model(self, root=397, residuals=(-300., 300.)):
        return {"root": root, "members": MEMBERS, "ensemble_seed": 166000+root,
            "fit_scene_seeds": [spec.calibration.scene_seed(root, s, "fit") for s in spec.SCENARIOS],
            "duration_prior_s": {"False": 2000., "True": 1900.},
            "centered_residuals_s": {"False": list(residuals), "True": list(residuals)}}

    def environment(self, residuals=(-300., 300.), trips=18):
        env, mean = calibrated_tests.CalibratedSupplyTest().environment(trips=trips)
        model = self.model(residuals=residuals)
        model["duration_prior_s"] = {"False": 3000., "True": 2800.}
        plan = NativeDistributionalSupplyPlan(env, model)
        env._upper_policy_callback = plan
        return env, plan, mean

    def test_batched_occupancy_matches_scalar_ties_and_historical_floor(self):
        starts = np.array([0., 5., 10., 12.])
        durations = np.array([[10., 5., 10., 1.], [4., 3., 2., 3.], [30., 30., 30., 30.]])
        np.testing.assert_array_equal(ensemble_peaks(starts, durations),
            [occupancy_peak(starts, starts+d) for d in durations])
        np.testing.assert_array_equal(ensemble_peaks(starts, durations, 3), [3, 3, 4])
        np.testing.assert_array_equal(ensemble_peaks([], np.empty((MEMBERS, 0)), 4), [4]*MEMBERS)

    def test_common_draws_are_stable_and_do_not_touch_native_global_rng(self):
        np.random.seed(53)
        state = np.random.get_state()
        model = self.model()
        draws = residual_draws(model, [(2, False), (0, False), (1, True)])
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        reordered = residual_draws(model, [(1, True), (0, False), (2, False)])
        for tid in draws: np.testing.assert_array_equal(draws[tid], reordered[tid])
        self.assertEqual(set(draws[0]), {-300., 300.})

    def test_zero_variance_recovers_mean_cost_and_original_ewma(self):
        env, plan, mean = self.environment(residuals=(0., 0.))
        nominal = np.array([t.launch_time for t in env.timetables if not t.direction])
        mean.preferred_intervals(nominal, env.timetables[0], 180.)
        plan.preferred_intervals(nominal, env.timetables[0], 180.)
        for new, old in zip(plan.decisions[-1]["candidates"], mean.decisions[-1]["candidates"]):
            self.assertEqual({k: v for k, v in new.items() if k != "ensemble_peak_counts"}, old)
            self.assertEqual(new["ensemble_peak_counts"], {str(old["predicted_peak"]): MEMBERS})
        self.assertEqual(plan.decisions[-1]["selected"], mean.decisions[-1]["selected"])
        env.completed_trip_observations[0] = {"direction": False, "launch_time_s": 300., "completion_time_s": 1500.}
        plan.observe_completions(1500.)
        self.assertEqual(plan.trip_duration(False), 1200.)
        self.assertEqual(plan.trip_duration(True), 2800.)

    def test_variability_cost_is_expectation_after_nonlinearity_and_active_progress(self):
        env, plan, _ = self.environment(trips=7)
        # Fit variation alone crosses the target despite all mean peaks being 1.
        env._n_fleet_target = 1
        for t in env.timetables: t.launched = True
        env.bus_all = [SimpleNamespace(on_route=True, direction=False, trip_id=0,
            effective_route=env.routes[10:], travel_distance=2500.)]
        future = env.timetables[2]
        future.launched = False
        env._peak_concurrent = 0
        cost, diag = plan.candidate_fleet_cost(1, {future.launch_turn: 1500.}, 0.)
        expected = np.array([occupancy_peak([0., 1500.], [
            (3000.+v)*.5, 1500.+3000.+w]) for v, w in zip(plan.residuals[0], plan.residuals[future.launch_turn])])
        self.assertEqual(cost, np.mean(np.maximum(expected-1, 0)**2))
        self.assertGreater(cost, 0.)
        self.assertEqual(sum(diag["ensemble_peak_counts"].values()), MEMBERS)
        env._peak_concurrent = 4
        floor_cost, floor = plan.candidate_fleet_cost(4, {future.launch_turn: 1500.}, 0.)
        self.assertEqual(floor, {"ensemble_peak_counts": {"4": MEMBERS}})
        self.assertEqual(floor_cost, 9.)

    def test_distributional_dispatch_preserves_commitments_and_lower_goals(self):
        env, plan, _ = self.environment()
        env.current_time = 180
        env._dispatch_due_trips()
        first = copy.deepcopy(plan.blocks[0])
        env.frequency_tracker.global_state.forecast = lambda t: 100000/(1+t)
        for tick in range(181, 300+18*360+180):
            env.current_time = tick
            env._dispatch_due_trips()
        self.assertEqual(plan.blocks[0], first)
        schedule = plan.summarize()
        self.assertEqual(schedule["release_lateness_mean_s"], 0)
        for row in schedule["schedules"]:
            np.testing.assert_array_equal(np.diff(row["planned_s"]), row["target_headways_s"][1:])

    def test_model_fit_ignores_all_check_outcomes(self):
        cells = duration_tests.DurationCalibrationTest().cells()
        def read(path):
            return json.dumps(cells[397, path.parent.parent.name])
        with patch.object(Path, "read_text", read):
            fitted = spec.fit_model(397)
            for cell in cells.values():
                for trips in cell["calibration"][1]["directions"].values():
                    trips["completion_s"] = (np.asarray(trips["launch_s"])+9000.).tolist()
            self.assertEqual(spec.fit_model(397), fitted)
        self.assertEqual(fitted["duration_prior_s"], {"False": 1820., "True": 2020.})
        self.assertEqual(len(fitted["centered_residuals_s"]["False"]), 655)

    def cells(self):
        sources, _, _ = calibrated_tests.CalibratedSupplyTest().cells()
        models, cells = {r: self.model(r, residuals=(0., 0.)) for r in spec.ROOTS}, {}
        for key, source in sources.items():
            rows = []
            scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[key[0]], key[1], preflight=False)
            for scene in scenes:
                for condition in spec.CONDITIONS:
                    row = copy.deepcopy(next(r for r in source["evaluation"] if r["scene_seed"] == scene
                        and r["condition"] == (condition if condition in spec.CONTROLS else "calibrated_supply")))
                    row.update(condition=condition, source_control_reproduced=condition in spec.CONTROLS)
                    if condition == "distributional_supply":
                        row["supply"].update(fleet_model=FLEET_MODEL, ensemble_members=MEMBERS,
                            ensemble_seed=models[key[0]]["ensemble_seed"], fit_scene_seeds=models[key[0]]["fit_scene_seeds"])
                        for d in row["supply"]["decisions"]:
                            for c in d["candidates"]: c["ensemble_peak_counts"] = {"12": MEMBERS}
                    rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": key[0], "scenario": key[1], "software_qualified": True, "training_updates": 0,
                "fleet_model": FLEET_MODEL, "native_steps": 12*61380, "evaluation": rows}
        return cells, sources, models

    def test_analysis_rejects_score_histogram_selection_model_and_control_changes(self):
        cells, sources, models = self.cells()
        result = summarize(cells, sources, models)
        self.assertEqual(result["source_control_reproductions"], 80)
        self.assertEqual(result["native_steps"], 120*61380)
        self.assertEqual(len(result["paired_contrasts"]), 4)
        json.dumps(result)
        for failure in ("histogram", "cost", "selection", "model", "floor", "control", "selector", "budget"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "distributional_supply")
            if failure == "histogram": row["supply"]["decisions"][0]["candidates"][0]["ensemble_peak_counts"] = {"12": MEMBERS-1}
            elif failure == "cost": row["supply"]["decisions"][0]["candidates"][0]["fleet_cost"] = 1.
            elif failure == "selection": row["supply"]["decisions"][0]["selected"] = 1
            elif failure == "model": row["supply"]["ensemble_seed"] += 1
            elif failure == "floor": row["supply"]["decisions"][0]["candidates"][0]["ensemble_peak_counts"] = {"-1": MEMBERS}
            elif failure == "control": next(r for r in cell["evaluation"] if r["condition"] == "window_forecast")["service_cost_restricted"] += 1.
            elif failure == "selector": next(r for r in cell["evaluation"] if r["condition"] == "calibrated_supply")["supply"]["choice_counts"][0] += 1
            else: cell["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)): summarize(broken, sources, models)

    def test_scheduler_and_worker_use_new_inputs_and_controls_first(self):
        task = task_specification("test_distributional", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        self.assertEqual(Path(task["stage_input_paths"][-1]), spec.ROOT/"results/test_distributional/inputs")
        cells, sources, models = self.cells()
        calls = []
        def episode(root, scenario, scene, raw, checkpoint, condition, model):
            calls.append(condition)
            row = next(r for r in cells[root, scenario]["evaluation"] if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"], row["supply"]
        with patch.object(Path, "read_text", return_value=json.dumps({str(r): m for r, m in models.items()})), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--duration-model", "unused/model.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS)*4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 12*61380)
        self.assertIn("DISTRIBUTIONAL_SUPPLY_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
