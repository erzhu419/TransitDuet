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

from scripts import run_native_transit_duration_calibration_stage164 as spec
from scripts.analyze_native_transit_duration_calibration_stage164 import summarize
from scripts.submit_native_transit_duration_calibration_stage164_scheduleurm import task_specification
from tests import test_native_transit_rolling_plan as rolling_tests
from tests import test_native_transit_supply_plan as supply_tests
from tests import test_native_transit_residual_plan as residual_tests


class DurationCalibrationTest(unittest.TestCase):
    def cells(self):
        sources, _ = rolling_tests.NativeRollingPlanTest().cells()
        cells = {}
        for (root, scenario), cell in sources.items():
            rows = []
            for split in spec.SPLITS:
                row = copy.deepcopy(cell["evaluation"][0])
                directions = {}
                for direction in (False, True):
                    schedule = next(s for s in row["schedule"]["schedules"] if s["direction"] == direction)
                    duration = 1800 + 200*direction + 10*spec.SCENARIOS.index(scenario)
                    if split == "check": duration += 50
                    directions[str(direction)] = {"trip_ids": list(range(int(direction), 262, 2)),
                        "launch_s": schedule["actual_s"],
                        "completion_s": (np.asarray(schedule["actual_s"])+duration).tolist()}
                row.update(split=split, scene_seed=spec.scene_seed(root, scenario, split),
                    initial_prior_s={"False": 800., "True": 900.}, directions=directions)
                rows.append(row)
            cells[root, scenario] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "seed": root, "scenario": scenario, "software_qualified": True,
                "training_updates": 0, "native_steps": 2*61380, "calibration": rows}
        return cells

    def test_fit_excludes_all_check_outcomes_and_prior_captured_once(self):
        cells = self.cells()
        result = summarize(cells)
        self.assertEqual(result["duration_prior_s"]["397"], {"False": 1820., "True": 2020.})
        self.assertEqual(result["native_steps"], 20*61380)
        json.dumps(result)
        for cell in cells.values():
            for trips in cell["calibration"][1]["directions"].values():
                trips["completion_s"] = (np.asarray(trips["launch_s"])+9000).tolist()
        self.assertEqual(summarize(cells)["duration_prior_s"], result["duration_prior_s"])
        env, _ = supply_tests.NativeSupplyPlanTest().environment()
        plan = spec.DurationCalibrationPlan(env)
        with patch.object(spec.NativeResidualPlan, "__call__", return_value=360.) as parent:
            plan(None, env.timetables[0])
            self.assertEqual(plan.initial_prior, {"False": 1040., "True": 1040.})
            for route in env.routes: route.speed_limit = 1.
            plan(None, env.timetables[1])
            self.assertEqual(plan.initial_prior, {"False": 1040., "True": 1040.})
            self.assertEqual(parent.call_count, 2)

    def test_passive_duration_capture_preserves_zero_residual_dispatch_and_goals(self):
        env, _ = residual_tests.NativeResidualPlanTest().environment([0, 0])
        env.routes = [SimpleNamespace(distance=500., speed_limit=5.) for _ in range(20)]
        plan = spec.DurationCalibrationPlan(env)
        env._upper_policy_callback = plan
        for tick in range(2700):
            env.current_time = tick
            env._dispatch_due_trips()
        reference_env, reference = residual_tests.NativeResidualPlanTest().execute([0, 0])
        self.assertEqual(plan.blocks, reference.blocks)
        self.assertEqual(plan.queries, reference.queries)
        self.assertEqual([(t._actual_launch_time, t.target_headway) for t in env.timetables],
            [(t._actual_launch_time, t.target_headway) for t in reference_env.timetables])

    def test_analysis_rejects_split_scene_duration_and_physical_budget_changes(self):
        cells = self.cells()
        for failure in ("split", "scene", "future", "duration", "duplicate", "launch", "goal", "budget", "prior"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = cell["calibration"][0]
            trips = row["directions"]["False"]
            if failure == "split": row["split"] = "check"
            elif failure == "scene": row["scene_seed"] += 1
            elif failure == "future": trips["completion_s"][0] = 61381
            elif failure == "duration": trips["completion_s"][0] = trips["launch_s"][0]
            elif failure == "duplicate": trips["trip_ids"][1] = trips["trip_ids"][0]
            elif failure == "launch": trips["launch_s"][0] += 1
            elif failure == "goal": row["schedule"]["schedules"][0]["target_headways_s"][7] += 1
            elif failure == "budget": cell["native_steps"] += 1
            else: row["initial_prior_s"]["False"] = np.nan
            with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(broken)

    def test_fitting_check_seeds_are_disjoint_from_previous_evaluation(self):
        seeds = [spec.scene_seed(r, s, split) for r in spec.ROOTS for s in spec.SCENARIOS for split in spec.SPLITS]
        old = [v for r in spec.ROOTS for s in spec.SCENARIOS for v in
            spec.native.source.source_spec.scene_seeds(spec.native.LOWER_ROOT[r], s, preflight=False)]
        self.assertEqual(len(set(seeds)), 20)
        self.assertFalse(set(seeds).intersection(old))

    def test_scheduler_and_worker_use_two_remote_episodes_not_rl_training(self):
        task = task_specification("test_duration", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 3072)
        self.assertEqual(task["result_dir"], task["local_result_dir"])
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])
        rows = self.cells()[397, "low_noise"]["calibration"]
        with patch.object(spec.source, "source_cell"), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=rows) as run, \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual([c.args[2] for c in run.call_args_list], list(spec.SPLITS))
        self.assertEqual(write.call_args.args[1]["native_steps"], 2*61380)
        self.assertEqual(write.call_args.args[1]["training_updates"], 0)
        self.assertIn("NATIVE_DURATION_CALIBRATION_COMPLETE", output.getvalue())


if __name__ == "__main__":
    unittest.main()
