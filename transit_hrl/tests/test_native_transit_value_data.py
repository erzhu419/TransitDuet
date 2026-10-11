import contextlib
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_native_transit_value_data_stage174 as spec
from scripts.analyze_native_transit_value_data_stage174 import summarize
from scripts.submit_native_transit_value_data_stage174_scheduleurm import task_specification
from tests import test_native_transit_candidate_value as candidate_tests
from tests import test_native_transit_incumbent_window as incumbent_tests


class ValueDataTest(unittest.TestCase):
    def cells(self):
        previous, _, pack = candidate_tests.CandidateValueTest().cells()
        cells, sources = {}, {}
        for (root, scenario), old in previous.items():
            for index in spec.PROBES:
                cell = copy.deepcopy(old)
                baseline = cell["observer"]
                baseline["scene_seed"] = 900000000+root*1000+spec.SCENARIOS.index(scenario)*10
                d = baseline["supply"]["decisions"][index]
                baseline["probe"].update(index=index, time_s=d["time_s"], direction=d["direction"],
                    candidates=copy.deepcopy(d["candidates"]), selected=d["selected"], greedy=d["selected"],
                    prefix_decisions_qualified=index+1, native_state_v2=[0.]*16)
                for candidate, row in enumerate(cell["branches"]):
                    row["scene_seed"] = baseline["scene_seed"]
                    row["probe"] = {**copy.deepcopy(baseline["probe"]), "selected": candidate,
                        "physical_state_matched": True, "rng_matched": True}
                    row["policy_trace"] = copy.deepcopy(baseline["policy_trace"])
                    row["window_panels"] = copy.deepcopy(baseline["window_panels"])
                    row["policy_trace"][index]["selected"] = row["window_panels"][index]["selected"] = candidate
                cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract(), probe_index=index,
                    training_context={"queues": {"False": 2, "True": 3}, "active": {"False": 5, "True": 6},
                        "onboard": {"False": 10, "True": 12}, "route_speed_mean_mps": 3.5, "cost_before": .4})
                del cell["corrected_probe"]
                cells[root, scenario, index] = cell
                sources[root, scenario] = copy.deepcopy(baseline)
        return cells, sources, pack["duration_models"]

    def test_source_selection_is_registered_not_outcome_selected_and_excludes_checks(self):
        cells, _, _ = incumbent_tests.IncumbentWindowTest().cells()
        cell = cells[397, "low_noise"]
        rows = [r for r in cell["evaluation"] if r["condition"] == "rolling_plus_incumbent"]
        rows[0]["ep_reward"], rows[-1]["ep_reward"] = -1000., 1000.
        cell["evaluation"].reverse()
        with patch.object(Path, "read_text", side_effect=lambda: json.dumps(cell)):
            self.assertEqual(spec.source_record(397, "low_noise")["scene_seed"], min(r["scene_seed"] for r in rows))
            cell["training_updates"] = 1
            with self.assertRaises(ValueError):
                spec.source_record(397, "low_noise")
            cell["training_updates"] = 0
            for row in rows:
                row["scene_seed"] = spec.shared.source.source.check_scene(397, "low_noise")
            with self.assertRaises(ValueError):
                spec.source_record(397, "low_noise")

    def test_context_is_preaction_only_and_labels_preserve_physical_cost_decomposition(self):
        context = spec.training_context({"buses": [
            {"on_route": True, "direction": False, "passengers": [1, 2]},
            {"on_route": True, "direction": True, "passengers": [3]},
            {"on_route": False, "direction": True, "passengers": [4, 5]}],
            "queues": [[0, False, [1, 2]], [1, False, [3]], [2, True, [4]]],
            "routes": [[0, 2.], [1, 4.]], "cost_before": .4})
        self.assertEqual(context, {"queues": {"False": 3, "True": 1}, "active": {"False": 1, "True": 1},
            "onboard": {"False": 2, "True": 1}, "route_speed_mean_mps": 3., "cost_before": .4})
        cells, _, _ = self.cells()
        cell = cells[397, "low_noise", 8]
        cell["observer"]["peak_fleet"] = 14
        row = cell["branches"][0]
        row.update(peak_fleet=15, service_cost_restricted=cell["observer"]["service_cost_restricted"]+.7,
            ep_reward=cell["observer"]["ep_reward"]+123.,
            restricted_wait_horizon_min=cell["observer"]["restricted_wait_horizon_min"]+.2)
        data = spec.records(cell)
        label = data["labels"][0]
        self.assertAlmostEqual(label["actual_cost_delta"], .7)
        self.assertAlmostEqual(label["physical_return_delta"], -70.)
        self.assertEqual(label["episode_reward_delta"], 123.)
        self.assertAlmostEqual(label["fleet_cost_delta"], 5/12)
        self.assertAlmostEqual(label["nonfleet_cost_delta"], .7-5/12)
        self.assertAlmostEqual(label["wait_delta_min"], .2)
        self.assertEqual((data["native_state_v2"], data["original_choice"]), ([0.]*16, 5))

    def test_analyzer_qualifies_training_not_performance_and_rejects_protocol_changes(self):
        cells, sources, models = self.cells()
        result = summarize(cells, sources, models, models)
        self.assertEqual((result["training_states"], result["paired_interventions"]), (30, 180))
        self.assertEqual((result["native_steps"], result["source_control_reproductions"]), (210*61380, 60))
        self.assertEqual((result["training_updates"], result["window_ledger_qualified_episodes"]), (0, 210))
        self.assertEqual(len(result["training_data"]), 30)
        self.assertTrue(all(r["states_with_better_realized_action"] == 15 for r in result["root_diagnostics"]))
        json.dumps(result)
        for failure in ("source", "scene", "probe", "v2", "context", "identity", "continuation", "rng", "budget"):
            broken = copy.deepcopy(cells)
            c = broken[397, "low_noise", 8]
            if failure == "source": c["observer"]["ep_reward"] += 1
            elif failure == "scene": c["observer"]["scene_seed"] = spec.shared.source.source.check_scene(397, "low_noise")
            elif failure == "probe": c["probe_index"] = 22
            elif failure == "v2": c["observer"]["probe"]["native_state_v2"] = [0.]*5
            elif failure == "context": c["training_context"]["cost_before"] += .1
            elif failure == "identity": c["branches"][5]["service_cost_restricted"] += .1
            elif failure == "continuation": c["branches"][0]["policy_trace"][-1]["selected"] = 0
            elif failure == "rng": c["branches"][0]["probe"]["rng_matched"] = False
            else: c["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources, models, models)
        with self.assertRaises(ValueError):
            summarize(cells, sources, {}, models)

    def test_worker_qualifies_identity_first_and_scheduler_keeps_unique_probe_paths(self):
        tasks = [task_specification("test_value_data", r, s, i)
            for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
        self.assertEqual(len({t["signature"] for t in tasks}), 30)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 30)
        for task in tasks:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
        cells, sources, models = self.cells()
        cell, calls = cells[397, "low_noise", 8], []
        snapshot = {"buses": [], "queues": [], "routes": [[0, 3.]], "cost_before": .4}
        def episode(*args, candidate=None, **kwargs):
            calls.append(candidate)
            row = copy.deepcopy(cell["observer"] if candidate is None else cell["branches"][candidate])
            if candidate is not None:
                row["supply"] = copy.deepcopy(cell["observer"]["supply"])
                row["supply"]["decisions"][8]["selected"] = candidate
            return row, SimpleNamespace(snapshot=snapshot)
        with patch.object(Path, "read_text", return_value=json.dumps(models)), \
                patch.object(spec, "source_record", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec.shared, "episode", side_effect=episode), patch.object(spec.shared, "qualify"), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), patch.object(spec, "write_json") as write, \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                patch.object(sys, "argv", ["worker", "--seed", "397", "--scenario", "low_noise", "--probe", "8",
                    "--models", "unused/models.json", "--output", "unused/result.json"]), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
        self.assertEqual(calls, [None, 5, 0, 1, 2, 3, 4])
        saved = write.call_args.args[1]
        self.assertEqual(saved["training_context"], spec.training_context(snapshot))
        self.assertTrue(all("supply" not in r for r in saved["branches"]))
        self.assertIn("VALUE_DATA_IDENTITY_SOURCE_QUALIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
