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

from freq_hrl.domains.transit.native_incumbent_window_plan import NativeIncumbentWindowPlan
from freq_hrl.domains.transit.native_commitment_plan import NativeCommitmentScoringPlan
from freq_hrl.domains.transit.native_service_plan import NativeServicePlan
from freq_hrl.domains.transit.native_rolling_plan import schedule_summary
from scripts import run_native_transit_incumbent_window_stage169 as spec
from scripts.analyze_native_transit_incumbent_window_stage169 import summarize
from scripts.submit_native_transit_incumbent_window_stage169_scheduleurm import task_specification
from tests import test_native_transit_distributional_supply as distributional_tests
from tests import test_native_transit_commitment_scoring as scoring_tests


class IncumbentWindowTest(unittest.TestCase):
    def test_forced_primitive_reproduces_window_dispatch_goals_and_all_endpoints(self):
        for count in (11, 131):
            old_env, _, _ = distributional_tests.DistributionalSupplyTest().environment(trips=count)
            new_env, new_model, _ = distributional_tests.DistributionalSupplyTest().environment(trips=count)
            old = NativeServicePlan(old_env, "causal_forecast")
            new_env.protocol = SimpleNamespace(evaluation_end_time_s=61380)
            new = NativeIncumbentWindowPlan(new_env, new_model.model, "incumbent_only")
            old_env._upper_policy_callback, new_env._upper_policy_callback = old, new
            for tick in range(300+count*360+180):
                old_env.current_time = new_env.current_time = tick
                old_env._dispatch_due_trips()
                new_env._dispatch_due_trips()
            self.assertEqual(schedule_summary(old), new.summarize())
            self.assertTrue(all(c["strength"] is None for d in new.decisions for c in d["candidates"]))
            self.assertEqual(new.supply_summary()["choice_counts"], [len(new.decisions)])
            if count == 131:
                self.assertEqual(new.summarize()["upper_decisions"], 44)

    def test_augmented_pool_preserves_five_scores_and_appends_local_window_geometry(self):
        old_env, original, _ = distributional_tests.DistributionalSupplyTest().environment()
        env, model, _ = distributional_tests.DistributionalSupplyTest().environment()
        old_env.protocol = env.protocol = SimpleNamespace(evaluation_end_time_s=8000)
        old = NativeCommitmentScoringPlan(old_env, original.model, "prefix_wait_full_fleet")
        new = NativeIncumbentWindowPlan(env, model.model, "rolling_plus_incumbent")
        old_env._upper_policy_callback, env._upper_policy_callback = old, new
        old_env.current_time = env.current_time = 180
        old_env._dispatch_due_trips()
        env._dispatch_due_trips()
        self.assertEqual(new.decisions[0]["candidates"][:5], old.decisions[0]["candidates"])
        candidate = new.decisions[0]["candidates"][-1]
        self.assertIsNone(candidate["strength"])
        prefix = candidate["commitment"]["prefix_planned_s"]
        nominal = [t.launch_time for t in env.timetables if not t.direction]
        self.assertEqual([prefix[0], prefix[5], prefix[6]], [nominal[0], nominal[5], nominal[6]])
        self.assertTrue(np.all(np.diff(prefix) >= 240) and np.all(np.diff(prefix) <= 480))
        for tick in range(181, 300+18*360+180):
            env.current_time = tick
            env._dispatch_due_trips()
        schedule = new.summarize()
        self.assertEqual(sum(new.supply_summary()["choice_counts"]), len(new.decisions))
        scheduled = {t.launch_turn: t._freqduet_scheduled_launch for t in env.timetables}
        for d, audit in zip(new.decisions, new.audits):
            self.assertEqual(d["selected"], min(range(6), key=lambda k: d["candidates"][k]["score"]))
            for tid, p in zip(audit["proposed_trip_ids"][:7], audit["selected_full_planned_s"][:7]):
                self.assertEqual(scheduled[tid], p)
        self.assertEqual(schedule["release_lateness_mean_s"], 0.)

    def cells(self):
        previous, _, _, models = scoring_tests.CommitmentScoringTest().cells()
        cells = {}
        for key, source in previous.items():
            rows = []
            for condition in spec.CONDITIONS:
                for ref in (r for r in source["evaluation"] if r["condition"] == "prefix_wait_full_fleet"):
                    row = copy.deepcopy(next(r for r in source["evaluation"] if r["scene_seed"] == ref["scene_seed"]
                        and r["condition"] == ("window_forecast" if condition in {"window_forecast", "incumbent_only"} else "prefix_wait_full_fleet")))
                    row.update(condition=condition, source_control_reproduced=condition in spec.CONTROLS,
                        incumbent_control_reproduced=condition == "incumbent_only")
                    if condition in {"incumbent_only", "rolling_plus_incumbent"}:
                        row["supply"] = copy.deepcopy(ref["supply"])
                        row["window_panels"] = copy.deepcopy(ref["window_panels"])
                        row["window_ledger_qualified"] = True
                        for d, p in zip(row["supply"]["decisions"], row["window_panels"]):
                            incumbent = copy.deepcopy(d["candidates"][0])
                            incumbent["strength"] = None
                            if condition == "incumbent_only":
                                d["candidates"], d["selected"] = [incumbent], 0
                            else:
                                incumbent["commitment"]["prefix_wait_cost"] = .9
                                incumbent["commitment"]["prefix_score"] = incumbent["score"] = .9
                                d["candidates"].append(incumbent)
                                d["selected"] = p["selected"] = 5
                        row["supply"]["choice_counts"] = [44] if condition == "incumbent_only" else [0, 0, 0, 0, 0, 44]
                        if condition == "rolling_plus_incumbent": row["service_cost_restricted"] -= .01
                    rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": key[0],
                "scenario": key[1], "software_qualified": True, "training_updates": 0,
                "native_steps": 16*61380, "evaluation": rows}
        return cells, previous, models

    def analyze(self, cells, sources, models):
        with patch.object(spec.source.distributional, "fit_model", side_effect=lambda r: models[r]):
            return summarize(cells, sources, models)

    def test_analysis_checks_forced_control_augmented_scores_and_real_protocol_errors(self):
        cells, sources, models = self.cells()
        result = self.analyze(cells, sources, models)
        self.assertEqual(result["source_control_reproductions"], 80)
        self.assertEqual(result["incumbent_control_reproductions"], 40)
        self.assertEqual(result["window_ledger_qualified_episodes"], 120)
        self.assertEqual(result["native_steps"], 160*61380)
        self.assertTrue(all(d["incumbent_choices"] == 880 for d in result["candidate_diagnostics"]))
        for p in result["paired_contrasts"]:
            self.assertAlmostEqual(p["service_cost_restricted"], -.03 if p["reference"] == "window_forecast" else -.01)
        json.dumps(result)
        for failure in ("incumbent_physics", "incumbent_schedule", "scores", "selection", "candidates", "window", "budget"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "rolling_plus_incumbent")
            if failure.startswith("incumbent"):
                forced = next(r for r in cell["evaluation"] if r["condition"] == "incumbent_only")
                if failure == "incumbent_physics": forced["ep_reward"] += 1
                else: forced["schedule"]["window_endpoint_shift_abs_mean_s"] += 1
            elif failure == "scores": row["supply"]["decisions"][0]["candidates"][5]["score"] += .1
            elif failure == "selection": row["supply"]["decisions"][0]["selected"] = 0
            elif failure == "candidates": row["supply"]["decisions"][0]["candidates"][5]["strength"] = .5
            elif failure == "window": row["window_panels"][0]["window_end_s"] += 1
            else: cell["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                self.analyze(broken, sources, models)

    def test_worker_preserves_controls_and_forced_qualification_before_augmented_pool(self):
        cells, sources, models = self.cells()
        cell = cells[397, "low_noise"]
        calls = []
        def episode(root, scenario, scene, raw, checkpoint, condition, model):
            calls.append(condition)
            row = next(r for r in cell["evaluation"] if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"], row["supply"], [], []
        def panels(row):
            return next(r for r in cell["evaluation"] if r["condition"] == row["condition"] and r["scene_seed"] == row["scene_seed"])["window_panels"]
        with patch.object(Path, "read_text", return_value=json.dumps({str(r): m for r, m in models.items()})), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "validate_audits", side_effect=panels), patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--duration-model", "unused/model.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS)*4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 16*61380)
        self.assertIn("INCUMBENT_WINDOW_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())

    def test_scheduler_is_unpinned_and_stages_only_fresh_inputs(self):
        task = task_specification("test_incumbent_window", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        self.assertIn("run_native_transit_incumbent_window_stage169.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
