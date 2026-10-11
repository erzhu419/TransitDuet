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

from freq_hrl.domains.transit.native_candidate_value import NativeCandidateValuePlan
from scripts import run_native_transit_candidate_value_stage173 as spec
from scripts.analyze_native_transit_candidate_value_stage173 import ranking_pairs, summarize
from scripts.submit_native_transit_candidate_value_stage173_scheduleurm import task_specification
from tests import test_native_transit_distributional_supply as distributional_tests
from tests import test_native_transit_remaining_supply as remaining_tests


class CandidateValueTest(unittest.TestCase):
    def test_single_override_preserves_scores_and_prefix_and_resumes_greedy(self):
        env, model, _ = distributional_tests.DistributionalSupplyTest().environment()
        env.protocol = SimpleNamespace(evaluation_end_time_s=8000)
        baseline = NativeCandidateValuePlan(env, model.model, 0)
        nominal = np.asarray([t.launch_time for t in env.timetables if not t.direction])
        snapshot = {"cost_before": .4, "rng": {"scenario_seed": 11}}
        with patch("freq_hrl.domains.transit.native_candidate_value.causal_snapshot", return_value=snapshot):
            baseline._incoming_state = [0.]*16
            baseline.preferred_intervals(nominal, env.timetables[0], 180.)
            forced = (baseline.probe["greedy"]+1) % 6
            plan = NativeCandidateValuePlan(env, model.model, 0, forced_candidate=forced, reference=baseline)
            plan._incoming_state = [0.]*16
            plan.preferred_intervals(nominal, env.timetables[0], 180.)
            self.assertEqual(plan.probe["selected"], forced)
            self.assertEqual(plan.probe["candidates"], baseline.probe["candidates"])
            self.assertTrue(plan.probe["physical_state_matched"] and plan.probe["rng_matched"])
            self.assertEqual(plan.select_candidate([{"score": 2.}, {"score": 1.}]), 1)
            for failure in ("prefix", "rng", "cost"):
                reference = copy.deepcopy(baseline)
                if failure == "prefix": reference.prefix[0]["upper_state"][0] = 1.
                elif failure == "rng": reference.snapshot["rng"]["scenario_seed"] = 12
                else: reference.snapshot["cost_before"] += .1
                plan = NativeCandidateValuePlan(env, model.model, 0, forced_candidate=forced, reference=reference)
                plan._incoming_state = [0.]*16
                with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                    plan.preferred_intervals(nominal, env.timetables[0], 180.)

    def cells(self):
        sources, _, pack = remaining_tests.RemainingSupplyTest().cells()
        cells = {}
        for key, registered in sources.items():
            original = registered["evaluation"][1]
            baseline = copy.deepcopy(original)
            index = spec.probe_index(registered)
            d = baseline["supply"]["decisions"][index]
            baseline.update(condition="observer", source_control_reproduced=True,
                policy_trace=spec.policy_trace(baseline["supply"]),
                probe={"index": index, "time_s": d["time_s"], "direction": d["direction"], "trip_id": 44,
                    "upper_state": [0.]*16, "nominal_s": [0., 360.], "cost_before": .4,
                    "greedy": d["selected"], "selected": d["selected"], "candidates": copy.deepcopy(d["candidates"]),
                    "prefix_decisions_qualified": index+1, "physical_state_matched": False, "rng_matched": False})
            branches = []
            for candidate in spec.CANDIDATES:
                row = copy.deepcopy(baseline)
                row.update(condition=f"candidate_{candidate}", source_control_reproduced=candidate == d["selected"])
                row["probe"].update(selected=candidate, physical_state_matched=True, rng_matched=True)
                row["policy_trace"][index]["selected"] = row["window_panels"][index]["selected"] = candidate
                row["service_cost_restricted"] -= .01*(5-candidate)
                del row["supply"]
                branches.append(row)
            cells[key] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": key[0],
                "scenario": key[1], "software_qualified": True, "training_updates": 0, "native_steps": 7*61380,
                "probe_index": index, "observer": baseline, "branches": branches,
                "corrected_probe": registered["evaluation"][3]["supply"]["decisions"][index]}
        return cells, sources, pack

    def test_analyzer_reports_realized_regret_without_deployment_or_independence_claim(self):
        cells, sources, pack = self.cells()
        result = summarize(cells, sources, pack, pack)
        self.assertEqual((result["source_control_reproductions"], result["paired_interventions"]), (20, 60))
        self.assertEqual((result["native_steps"], result["window_ledger_qualified_episodes"]), (70*61380, 70))
        self.assertTrue(all(p["realized_best_candidate"] == 0 for p in result["state_panels"]))
        self.assertTrue(all(abs(p["original_realized_regret"]-.05) < 1e-10 for p in result["state_panels"]))
        self.assertTrue(all(r["states_with_peak_discrimination"] == 0 for r in result["root_diagnostics"]))
        self.assertEqual(ranking_pairs([0.]*6, [0.]*6)["outcome_ties"], 15)
        json.dumps(result)
        for failure in ("source", "identity", "candidate", "state", "rng", "prefix", "continuation", "window", "budget", "probe"):
            broken = copy.deepcopy(cells)
            c = broken[397, "low_noise"]
            branch = c["branches"][0]
            if failure == "source": c["observer"]["ep_reward"] += 1
            elif failure == "identity": c["branches"][5]["service_cost_restricted"] += .1
            elif failure == "candidate": c["branches"][1]["probe"]["selected"] = 0
            elif failure == "state": branch["probe"]["upper_state"][0] = 1
            elif failure == "rng": branch["probe"]["rng_matched"] = False
            elif failure == "prefix": branch["policy_trace"][0]["selected"] = 0
            elif failure == "continuation": branch["policy_trace"][-1]["selected"] = 0
            elif failure == "window": branch["window_panels"][22]["window_end_s"] += 1
            elif failure == "budget": c["native_steps"] += 1
            else: c["probe_index"] = 8
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources, pack, pack)

    def test_validator_allows_only_the_registered_single_choice_override(self):
        cells, _, pack = self.cells()
        row = copy.deepcopy(cells[397, "low_noise"]["observer"])
        row["supply"]["decisions"][22]["selected"] = row["window_panels"][22]["selected"] = 0
        row["supply"]["choice_counts"] = [1, 0, 0, 0, 0, 43]
        args = dict(strengths=spec.source.incumbent.CANDIDATES["rolling_plus_incumbent"])
        mapped = {**row, "condition": "prefix_wait_full_fleet"}
        with self.assertRaises(ValueError):
            spec.validate_scoring(mapped, pack["duration_models"]["397"], **args)
        spec.validate_scoring(mapped, pack["duration_models"]["397"], choice_overrides={22: 0}, **args)
        mapped["supply"]["decisions"][23]["selected"] = 0
        with self.assertRaises(ValueError):
            spec.validate_scoring(mapped, pack["duration_models"]["397"], choice_overrides={22: 0}, **args)

    def test_worker_runs_identity_before_alternatives_and_scheduler_stages_only_new_inputs(self):
        task = task_specification("test_candidate_value", 397, "low_noise")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        cells, sources, pack = self.cells()
        cell, calls = cells[397, "low_noise"], []
        def episode(*args, candidate=None, **kwargs):
            calls.append(candidate)
            row = copy.deepcopy(cell["observer"] if candidate is None else cell["branches"][candidate])
            if candidate is not None:
                row["supply"] = copy.deepcopy(cell["observer"]["supply"])
                row["supply"]["decisions"][22]["selected"] = candidate
            return row, SimpleNamespace(probe=cell["observer"]["probe"])
        with patch.object(Path, "read_text", return_value=json.dumps(pack)), \
                patch.object(spec, "source_cell", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec, "episode", side_effect=episode), patch.object(spec, "qualify"), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise",
                    "--models", "unused/models.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, [None, 5, 0, 1, 2, 3, 4])
        self.assertEqual(write.call_args.args[1]["native_steps"], 7*61380)
        self.assertTrue(all("supply" not in r for r in write.call_args.args[1]["branches"]))
        self.assertIn("CANDIDATE_VALUE_IDENTITY_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
