import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from scripts import run_native_transit_value_direction_stage175 as spec
from scripts.analyze_native_transit_value_data_stage174 import summarize as summarize_previous
from scripts.analyze_native_transit_value_direction_stage175 import summarize
from scripts.submit_native_transit_value_direction_stage175_scheduleurm import task_specification
from tests import test_native_transit_value_data as source_tests


class ValueDirectionTest(unittest.TestCase):
    def cells(self):
        fixture = source_tests.ValueDataTest()
        previous_cells, sources, models = fixture.cells()
        previous = summarize_previous(previous_cells, sources, models, models)
        cells, _, _ = fixture.cells(probes=spec.PROBES)
        for c in cells.values():
            c.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract())
        return cells, sources, models, previous

    def test_pair_requires_same_scene_adjacent_index_and_opposite_direction(self):
        cells, _, _, previous = self.cells()
        cell = cells[397, "low_noise", 9]
        paired = spec.paired_reference(397, "low_noise", 9, previous)
        spec.validate_pair(cell, paired)
        for failure in ("direction", "index", "scene", "time"):
            broken = copy.deepcopy(cell)
            if failure == "direction": broken["observer"]["probe"]["direction"] = paired["direction"]
            elif failure == "index": broken["probe_index"] += 1
            elif failure == "scene": broken["observer"]["scene_seed"] += 1
            else: broken["observer"]["probe"]["time_s"] = paired["time_s"]-1
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                spec.validate_pair(broken, paired)
        for failure in ("missing", "duplicate", "unqualified"):
            broken = copy.deepcopy(previous)
            if failure == "missing": broken["training_data"].remove(paired)
            elif failure == "duplicate": broken["training_data"].append(paired)
            else: broken["software_qualified"] = False
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                spec.paired_reference(397, "low_noise", 9, broken)

    def test_merge_preserves_scene_groups_and_separates_new_from_cumulative_budget(self):
        cells, sources, models, previous = self.cells()
        result = summarize(cells, sources, models, models, previous)
        self.assertEqual((result["new_training_states"], result["training_states"]), (30, 60))
        self.assertEqual((result["new_native_steps"], result["native_steps"]), (210*61380, 420*61380))
        self.assertEqual((result["source_control_reproductions"], result["paired_interventions"]), (120, 360))
        self.assertEqual((result["training_updates"], result["window_ledger_qualified_episodes"]), (0, 420))
        self.assertTrue(all(d["direction_states"] == {"False": 15, "True": 15} for d in result["root_diagnostics"]))
        self.assertEqual(len(result["training_data"]), 60)
        self.assertEqual(len(result["fitting_groups"]), 10)
        self.assertTrue(all(g["probes"] == [8, 9, 22, 23, 36, 37] for g in result["fitting_groups"]))
        json.dumps(result)
        for failure in ("source_time", "identity", "context", "budget"):
            broken, old = copy.deepcopy(cells), copy.deepcopy(previous)
            cell = broken[397, "low_noise", 9]
            if failure == "source_time": spec.paired_reference(397, "low_noise", 9, old)["time_s"] += 1
            elif failure == "identity": cell["branches"][5]["service_cost_restricted"] += 1
            elif failure == "context": cell["training_context"]["cost_before"] += 1
            else: cell["native_steps"] += 1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(broken, sources, models, models, old)

    def test_worker_checks_opposite_direction_before_native_collection(self):
        cells, sources, models, previous = self.cells()
        cell = cells[397, "low_noise", 9]
        argv = ["worker", "--seed", "397", "--scenario", "low_noise", "--probe", "9",
            "--models", "unused/models.json", "--paired-data", "unused/paired.json", "--output", "unused/result.json"]
        def read(path):
            return json.dumps(previous if path.name == "paired.json" else models)
        with patch.object(Path, "read_text", autospec=True, side_effect=read), \
                patch.object(spec, "source_record", return_value=sources[397, "low_noise"]), \
                patch.object(spec.source, "collect", return_value=cell) as collect, \
                patch.object(spec, "write_json") as write, patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            spec.main()
            self.assertEqual(collect.call_args.args[:4], (397, "low_noise", 9, models["397"]))
            self.assertEqual(write.call_args.args[1]["protocol"], spec.EXPERIMENT_PROTOCOL)
            self.assertIn("NATIVE_BIDIRECTIONAL_VALUE_DATA_COMPLETE", output.getvalue())
            collect.reset_mock()
            paired = spec.paired_reference(397, "low_noise", 9, previous)
            paired["direction"] = cell["observer"]["probe"]["direction"]
            with self.assertRaises(ValueError):
                spec.main()
            collect.assert_not_called()

    def test_scheduler_has_thirty_unique_unpinned_tasks_and_only_fresh_inputs(self):
        tasks = [task_specification("test_value_direction", r, s, i)
            for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
        self.assertEqual(len({t["signature"] for t in tasks}), 30)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 30)
        for task in tasks:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
            self.assertIn("run_native_transit_value_direction_stage175.py", task["cmd"])
            self.assertIn("--paired-data", task["cmd"])
            self.assertNotIn(spec.SOURCE_RUN, task["cmd"])


if __name__ == "__main__":
    unittest.main()
