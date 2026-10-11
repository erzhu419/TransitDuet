import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from freq_hrl.domains.transit.native_commitment_plan import NativeCommitmentScoringPlan, HORIZONS
from scripts import run_native_transit_commitment_scoring_stage168 as spec
from scripts.analyze_native_transit_commitment_scoring_stage168 import summarize
from scripts.submit_native_transit_commitment_scoring_stage168_scheduleurm import task_specification
from tests.test_native_transit_distributional_supply import DistributionalSupplyTest


class CommitmentScoringTest(unittest.TestCase):
    def test_each_horizon_executes_its_score_and_keeps_committed_anchors(self):
        for condition, (prefix_wait, prefix_fleet) in HORIZONS.items():
            env, reference, _ = DistributionalSupplyTest().environment()
            env.protocol = SimpleNamespace(evaluation_end_time_s=8000)
            plan = NativeCommitmentScoringPlan(env, reference.model, condition)
            env._upper_policy_callback = plan
            for tick in range(300+18*360+180):
                env.current_time = tick
                env._dispatch_due_trips()
            schedule = plan.summarize()
            self.assertEqual(schedule["release_lateness_mean_s"], 0)
            scheduled = {t.launch_turn: t._freqduet_scheduled_launch for t in env.timetables}
            for d, audit in zip(plan.decisions, plan.audits):
                for c in d["candidates"]:
                    p = c["commitment"]
                    self.assertNotIn("proposed_planned_s", p)
                    self.assertEqual(c["score"], (p["prefix_wait_cost"] if prefix_wait else c["wait_cost"])
                        + (p["prefix_fleet_cost"] if prefix_fleet else c["fleet_cost"]))
                self.assertEqual(d["selected"], min(range(5), key=lambda k: d["candidates"][k]["score"]))
                for t, p in zip(audit["proposed_trip_ids"][:7], audit["selected_full_planned_s"][:7]):
                    self.assertEqual(scheduled[t], p)

    def cells(self):
        original, _, models = DistributionalSupplyTest().cells()
        cells, sources = {}, {}
        changes = {"full_wait_prefix_fleet": .04, "prefix_wait_full_fleet": -.02, "prefix_wait_prefix_fleet": -.01}
        for key, cell in original.items():
            refs = [copy.deepcopy(r) for r in cell["evaluation"] if r["condition"] == "distributional_supply"]
            for r in refs:
                r.update(audits=[], completed_trips=[])
            sources[key] = {"evaluation": refs}
            rows = []
            for condition in spec.CONDITIONS:
                for ref in refs:
                    row = copy.deepcopy(next(r for r in cell["evaluation"] if r["scene_seed"] == ref["scene_seed"]
                        and r["condition"] == ("window_forecast" if condition == "window_forecast" else "distributional_supply")))
                    row.update(condition=condition, source_control_reproduced=condition in spec.CONTROLS)
                    if condition != "window_forecast":
                        row.update(window_ledger_qualified=True, window_panels=[])
                        decisions = row["supply"]["decisions"]
                        for i, d in enumerate(decisions):
                            future = next((n for n in decisions[i+1:] if n["direction"] == d["direction"]), None)
                            end = future["time_s"] if future is not None else row["simulation_end_time_s"]
                            peak = future["historical_peak"] if future is not None else row["peak_fleet"]
                            row["window_panels"].append({**{k: d[k] for k in ("time_s", "direction", "selected")},
                                "window_end_s": end, "actual_window_peak": peak,
                                "actual_window_fleet_cost": max(peak-12, 0)**2/12, "selected_prefix_fleet_cost": 0.})
                            if condition in HORIZONS:
                                for c in d["candidates"]:
                                    c["commitment"] = {"prefix_peak_counts": {"12": 64}, "prefix_fleet_cost": 0.,
                                        "prefix_wait_cost": c["wait_cost"], "prefix_score": c["wait_cost"],
                                        "full_fleet_cost": c["fleet_cost"], "window_end_s": end}
                        row["service_cost_restricted"] += changes.get(condition, 0.)
                    rows.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": key[0],
                "scenario": key[1], "software_qualified": True, "training_updates": 0,
                "native_steps": 20*61380, "evaluation": rows}
        return cells, sources, original, models

    def analyze(self, cells, sources, windows, models):
        def panels(ref):
            full = next(r for r in cells[397, "low_noise"]["evaluation"] if r["condition"] == "full_wait_full_fleet")
            return full["window_panels"]
        with patch.object(spec.distributional, "fit_model", side_effect=lambda r: models[r]), \
                patch("scripts.analyze_native_transit_commitment_scoring_stage168.validate_audits", side_effect=panels):
            return summarize(cells, sources, windows, models)

    def test_analysis_qualifies_factorial_and_rejects_actual_protocol_errors(self):
        cells, sources, windows, models = self.cells()
        result = self.analyze(cells, sources, windows, models)
        self.assertEqual(result["source_control_reproductions"], 80)
        self.assertEqual(result["window_ledger_qualified_episodes"], 160)
        self.assertEqual(result["native_steps"], 200*61380)
        self.assertEqual(len(result["paired_contrasts"]), 12)
        self.assertEqual(len(result["regime_root_contrasts"]), 60)
        effect = result["factorial_effects"][0]
        self.assertAlmostEqual(effect["prefix_wait_main_effect"]["service_cost_restricted"], -.035)
        self.assertAlmostEqual(effect["prefix_fleet_main_effect"]["service_cost_restricted"], .025)
        self.assertAlmostEqual(effect["interaction"]["service_cost_restricted"], -.03)
        json.dumps(result)
        for failure in ("score", "histogram", "floor", "selection", "boundary", "ledger", "control", "budget", "duplicate"):
            broken = copy.deepcopy(cells)
            cell = broken[397, "low_noise"]
            row = next(r for r in cell["evaluation"] if r["condition"] == "prefix_wait_prefix_fleet")
            d = row["supply"]["decisions"][0]
            if failure == "score": d["candidates"][0]["score"] += .1
            elif failure == "histogram": d["candidates"][0]["commitment"]["prefix_peak_counts"] = {"12": 63}
            elif failure == "floor": d["candidates"][0]["commitment"]["prefix_peak_counts"] = {"-1": 64}
            elif failure == "selection": d["selected"] = 1
            elif failure == "boundary": d["candidates"][0]["commitment"]["window_end_s"] += 1
            elif failure == "ledger": row["window_ledger_qualified"] = False
            elif failure == "control": next(r for r in cell["evaluation"] if r["condition"] == "window_forecast")["service_cost_restricted"] += .1
            elif failure == "budget": cell["native_steps"] += 1
            else: cell["evaluation"].append(copy.deepcopy(row))
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                self.analyze(broken, sources, windows, models)

    def test_worker_runs_controls_first_and_keeps_only_compact_audit_results(self):
        cells, sources, windows, models = self.cells()
        cell = cells[397, "low_noise"]
        calls = []
        def episode(root, scenario, scene, raw, checkpoint, condition, model):
            calls.append(condition)
            row = next(r for r in cell["evaluation"] if r["condition"] == condition and r["scene_seed"] == scene)
            return row, row["schedule"], row["supply"], [], []
        def panels(row):
            ref = next(r for r in cell["evaluation"] if r["condition"] == row["condition"] and r["scene_seed"] == row["scene_seed"])
            return ref["window_panels"]
        with patch.object(Path, "read_text", return_value=json.dumps({str(r): m for r, m in models.items()})), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.source, "source_cell", return_value=windows[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch("scripts.analyze_native_transit_commitment_audit_stage167.validate_audits", side_effect=panels), \
                patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--duration-model", "unused/model.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, list(spec.CONDITIONS)*4)
        self.assertEqual(write.call_args.args[1]["native_steps"], 20*61380)
        self.assertTrue(all("audits" not in r and "completed_trips" not in r for r in write.call_args.args[1]["evaluation"]))
        self.assertIn("COMMITMENT_SCORING_FIRST_SCENE_SOURCE_QUALIFIED", output.getvalue())

    def test_scheduler_stages_only_new_inputs_and_uses_all_six_nodes(self):
        task = task_specification("test_commitment_scoring", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        self.assertIn("run_native_transit_commitment_scoring_stage168.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
