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

from freq_hrl.domains.transit.native_calibrated_supply_plan import NativeCalibratedSupplyPlan
from scripts import run_native_transit_calibrated_supply_stage165 as spec
from scripts.analyze_native_transit_calibrated_supply_stage165 import summarize
from scripts.submit_native_transit_calibrated_supply_stage165_scheduleurm import task_specification
from tests import test_native_transit_supply_plan as supply_tests


class CalibratedSupplyTest(unittest.TestCase):
    def environment(self, condition="supply_aware", trips=18):
        env, _ = supply_tests.NativeSupplyPlanTest().environment(condition, trips)
        plan = NativeCalibratedSupplyPlan(env, condition, {False: 3000., True: 2800.})
        env._upper_policy_callback = plan
        return env, plan

    def test_prior_changes_only_pre_completion_duration_then_original_ewma(self):
        env, plan = self.environment()
        self.assertEqual([plan.trip_duration(d) for d in (False, True)], [3000., 2800.])
        old = supply_tests.NativeSupplyPlanTest().environment()[1]
        self.assertEqual(old.trip_duration(False), 1040.)
        env.completed_trip_observations[0] = {"direction": False, "launch_time_s": 300., "completion_time_s": 1500.}
        plan.observe_completions(1500)
        self.assertEqual(plan.trip_duration(False), 1200.)
        self.assertEqual(plan.trip_duration(True), 2800.)
        env.completed_trip_observations[2] = {"direction": False, "launch_time_s": 660., "completion_time_s": 2460.}
        plan.observe_completions(2500)
        self.assertEqual(plan.trip_duration(False), 1320.)
        for invalid in ({False: 2000.}, {False: 2000., True: np.nan}, {False: 0., True: 2000.}):
            with self.assertRaises(ValueError): NativeCalibratedSupplyPlan(env, "supply_aware", invalid)

    def test_real_occupancy_prior_and_demand_only_dispatch_neutrality(self):
        env, plan = self.environment("demand_only")
        old_env, old = supply_tests.NativeSupplyPlanTest().environment("demand_only")
        self.assertGreater(plan.predicted_peak({}, 180), old.predicted_peak({}, 180))
        for tick in range(300+18*360+180):
            env.current_time = old_env.current_time = tick
            env._dispatch_due_trips()
            old_env._dispatch_due_trips()
        self.assertEqual(plan.blocks, old.blocks)
        self.assertEqual(plan.summarize(), old.summarize())
        self.assertEqual(plan.supply_summary()["choice_counts"], old.supply_summary()["choice_counts"])
        first = plan.decisions[0]
        self.assertEqual(first["prior_directions_in_use"], ["False", "True"])
        self.assertEqual(first["duration_estimates_s"], {"False": 3000., "True": 2800.})

    def test_supply_dispatch_preserves_commitments_and_native_goals(self):
        env, plan = self.environment()
        env.current_time = 180
        env._dispatch_due_trips()
        first = copy.deepcopy(plan.blocks[0])
        env.frequency_tracker.global_state.forecast = lambda t: 100000/(1+t)
        for tick in range(181, 300+18*360+180):
            env.current_time = tick
            env._dispatch_due_trips()
        self.assertEqual(plan.blocks[0], first)
        self.assertEqual(plan.summarize()["release_lateness_mean_s"], 0)
        for schedule in plan.summarize()["schedules"]:
            np.testing.assert_array_equal(np.diff(schedule["planned_s"]), schedule["target_headways_s"][1:])

    def cells(self):
        sources, _ = supply_tests.NativeSupplyPlanTest().cells()
        priors = {r: {False: 2000., True: 1900.} for r in spec.ROOTS}
        cells = {}
        for (root, scenario), source in sources.items():
            rows = []
            scenes = spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[root], scenario, preflight=False)
            for scene in scenes:
                for condition in spec.CONDITIONS:
                    row = copy.deepcopy(next(r for r in source["evaluation"]
                        if r["condition"] == spec.SOURCE_CONDITIONS[condition] and r["scene_seed"] == scene))
                    row.update(condition=condition, source_control_reproduced=condition in spec.CONTROLS)
                    if condition.startswith("calibrated_"):
                        supply = row["supply"]
                        supply.update(initial_duration_prior_s={str(d): v for d, v in priors[root].items()},
                            duration_model="fit_only_prior_until_first_completion_then_original_EWMA")
                        for i, decision in enumerate(supply["decisions"]):
                            decision.update(historical_peak=0, prior_directions_in_use=["False", "True"] if i == 0 else [],
                                duration_estimates_s=supply["initial_duration_prior_s"] if i == 0 else {"False": 1900., "True": 1800.})
                    rows.append(row)
            cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": root, "scenario": scenario, "software_qualified": True, "training_updates": 0,
                "native_steps": 16*61380, "duration_prior_s": {str(d): v for d, v in priors[root].items()}, "evaluation": rows}
        return cells, sources, priors

    def test_analysis_rejects_prior_source_control_selection_and_budget_errors(self):
        cells, sources, priors = self.cells()
        result = summarize(cells, sources, priors)
        json.dumps(result)
        self.assertEqual(result["native_steps"], 160*61380)
        self.assertEqual(result["source_control_reproductions"], 120)
        self.assertEqual(len(result["paired_contrasts"]), 6)
        for failure in ("prior", "control", "legacy", "selection", "score", "restart_prior", "schedule", "duplicate", "budget"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "calibrated_supply")
            if failure == "prior": row["supply"]["initial_duration_prior_s"]["True"] += 1
            elif failure == "control": next(r for r in cell["evaluation"] if r["condition"] == "calibrated_demand")["service_cost_restricted"] += 1
            elif failure == "legacy": next(r for r in cell["evaluation"] if r["condition"] == "legacy_supply")["supply"]["boundary"] = "changed"
            elif failure == "selection": row["supply"]["decisions"][0]["selected"] = 1
            elif failure == "score": row["supply"]["decisions"][0]["candidates"][0]["score"] += 1
            elif failure == "restart_prior": row["supply"]["decisions"][2]["prior_directions_in_use"] = ["False"]
            elif failure == "schedule": next(r for r in cell["evaluation"] if r["condition"] == "window_forecast")["schedule"]["release_lateness_mean_s"] = 1
            elif failure == "duplicate": cell["evaluation"].append(copy.deepcopy(row))
            else: cell["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)): summarize(broken, sources, priors)

    def test_scheduler_stages_only_run_specific_inputs_unpinned(self):
        task = task_specification("test_calibrated_supply", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertEqual(task["result_dir"], task["local_result_dir"])
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        self.assertEqual(Path(task["stage_input_paths"][-1]), spec.ROOT/"results/test_calibrated_supply/inputs")
        self.assertIn("--calibration-summary", task["cmd"])

    def test_worker_runs_three_source_controls_before_new_supply(self):
        cells, sources, priors = self.cells()
        calls = []

        def episode(root, scenario, scene, raw, checkpoint, condition, prior):
            calls.append(condition)
            self.assertEqual(prior, priors[root])
            row = next(r for r in cells[root, scenario]["evaluation"] if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"], row["supply"]

        with patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec, "duration_prior", return_value=priors[397]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--calibration-summary", "unused/input.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS)*4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 16*61380)
        self.assertIn("CALIBRATED_SUPPLY_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
