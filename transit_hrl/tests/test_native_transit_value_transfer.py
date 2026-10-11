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

from scripts import run_native_transit_value_transfer_stage179 as spec
from scripts.analyze_native_transit_value_transfer_stage179 import summarize
from scripts.submit_native_transit_value_transfer_stage179_scheduleurm import task_specification
from tests import test_native_transit_value_data as data_tests
from tests.test_native_transit_value_context import contexts_fixture
from tests import test_native_transit_incumbent_window as incumbent_tests


def fixture():
    cells, sources, models = data_tests.ValueDataTest().cells(probes=spec.PROBES)
    for (root, scenario, index), cell in cells.items():
        cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract())
        cell["observer"]["scene_seed"] += 1
        nominal = cell["observer"]["probe"]["candidates"][5]["commitment"]["prefix_planned_s"]
        cell["observer"]["probe"]["nominal_s"] = copy.deepcopy(nominal)
        for branch in cell["branches"]:
            branch["scene_seed"] = cell["observer"]["scene_seed"]
            branch["probe"]["nominal_s"] = copy.deepcopy(nominal)
        row = spec.records(cell)
        contexts, _ = contexts_fixture([row])
        contexts[0]["historical_peak"] = sources[root, scenario]["supply"]["decisions"][index]["historical_peak"]
        cell["context"] = contexts[0]
    for registered in sources.values(): registered["scene_seed"] += 1
    model = {"method": spec.fitted.PRIMARY, "steps": spec.value.STEPS,
        "training_states": 60, "parameter_count": 1586,
        "training_groups": [[r, sources[r, s]["scene_seed"]-1] for r in spec.ROOTS for s in spec.SCENARIOS],
        "center": [0.]*96, "scale": [1.]*96, "cost_scale": .01,
        "weights": {"0.weight": np.full((16, 96), .001).tolist(), "0.bias": [0.]*16,
            "2.weight": np.full((2, 16), .01).tolist(), "2.bias": [0., 0.]}}
    primary = {"protocol": spec.fitted.EXPERIMENT_PROTOCOL, "contract": spec.fitted.contract(),
        "software_qualified": True, "method": spec.fitted.PRIMARY, "model": model,
        "critic_updates": 18*spec.value.STEPS, "actor_updates": 0, "new_native_steps": 0}
    for cell in cells.values():
        row = spec.records(cell)
        row["occupancy_context"] = {k: cell["context"][k] for k in spec.OCCUPANCY_KEYS}
        cell["prediction"] = spec.frozen_prediction(row, model)
    return cells, sources, models, primary


class ValueTransferTest(unittest.TestCase):
    def test_second_scene_is_registered_not_selected_by_reward(self):
        cells, _, _ = incumbent_tests.IncumbentWindowTest().cells()
        cell = cells[397, "low_noise"]
        rows = sorted((r for r in cell["evaluation"] if r["condition"] == "rolling_plus_incumbent"),
            key=lambda r: r["scene_seed"])
        rows[1]["ep_reward"], rows[3]["ep_reward"] = -1000., 1000.
        cell["evaluation"].reverse()
        with patch.object(Path, "read_text", return_value=json.dumps(cell)):
            self.assertEqual(spec.source_record(397, "low_noise")["scene_seed"], rows[1]["scene_seed"])
            self.assertEqual(spec.labels.source_record(397, "low_noise")["scene_seed"], rows[0]["scene_seed"])

    def test_prediction_is_causal_sealed_before_action_and_preserves_original_choice(self):
        cells, _, _, primary = fixture()
        cell = cells[397, "low_noise", 8]
        row = spec.records(cell)
        row["occupancy_context"] = {k: cell["context"][k] for k in spec.OCCUPANCY_KEYS}
        original = spec.frozen_prediction(row, primary["model"])
        row["labels"] = [{"actual_cost_delta": -10000.}]*6
        row.update(ep_reward=99999., service_cost_restricted=-99.)
        self.assertEqual(spec.frozen_prediction(row, primary["model"]), original)
        plan = object.__new__(spec.FrozenValueTransferAudit)
        plan.decisions, plan.probe_index = [None]*8, 8
        plan.probe, plan.contexts, plan.value_model = cell["observer"]["probe"], [cell["context"]], primary["model"]
        with patch.object(spec.NativeValueContextAudit, "select_candidate", return_value=row["original_choice"]), \
                patch.object(spec.labels, "training_context", return_value=row["context"]), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            plan.snapshot = {}
            self.assertEqual(plan.select_candidate(row["candidates"]), row["original_choice"])
        self.assertEqual(plan.prediction, original)
        self.assertIn("FROZEN_VALUE_PREDICTION_SEALED", out.getvalue())

    def test_frozen_evaluation_qualifies_all_controls_without_fitting_or_deployment(self):
        cells, sources, models, primary = fixture()
        with patch.object(spec.value, "fit_value", side_effect=AssertionError("transfer must not fit")):
            result = summarize(cells, sources, models, models, spec.model_input(primary), primary)
        self.assertEqual((result["evaluation_states"], result["paired_interventions"]), (60, 360))
        self.assertEqual((result["source_control_reproductions"], result["native_steps"]), (120, 420*61380))
        self.assertEqual((result["critic_updates"], result["actor_updates"], result["training_updates"]), (0, 0, 0))
        self.assertNotIn("training_states", result)
        self.assertFalse(result["native_screen_eligible"])
        self.assertEqual(len(result["evaluation_groups"]), 10)
        self.assertEqual(len(result["by_regime"]), 5)
        json.dumps(result)
        for failure in ("prediction", "choice", "floor", "labels", "training_scene", "normalization", "primary"):
            changed, pack = copy.deepcopy(cells), spec.model_input(copy.deepcopy(primary))
            c = changed[397, "low_noise", 8]
            if failure == "prediction": c["prediction"]["predicted_heads"][0][0] += .01
            elif failure == "choice": c["prediction"]["selected"] = (c["prediction"]["selected"]+1)%6
            elif failure == "floor": c["context"]["historical_peak"] += 1
            elif failure == "labels": c["branches"][5]["service_cost_restricted"] += .01
            elif failure == "training_scene": pack["model"]["training_groups"][0][1] += 1
            elif failure == "normalization": pack["model"]["center"][0] += 1
            else: pack["primary"] = "base_regret"
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)):
                summarize(changed, sources, models, models, pack, primary)

    def test_collector_records_prediction_before_identity_then_all_alternatives(self):
        cells, sources, models, primary = fixture()
        cell, events = cells[397, "low_noise", 8], []
        def episode(*args, candidate=None, **kwargs):
            events.append(candidate)
            row = copy.deepcopy(cell["observer"] if candidate is None else cell["branches"][candidate])
            if candidate is not None:
                row["supply"] = copy.deepcopy(cell["observer"]["supply"])
                row["supply"]["decisions"][8]["selected"] = candidate
            else:
                self.assertIs(kwargs["plan_class"].func, spec.FrozenValueTransferAudit)
            return row, SimpleNamespace(snapshot={}, contexts=[cell["context"]], prediction=cell["prediction"])
        def write(path, prediction):
            self.assertEqual(Path(path).name, "prediction.json")
            self.assertEqual(prediction, cell["prediction"])
            events.append("sealed")
        with patch.object(spec, "source_record", return_value=sources[397, "low_noise"]), \
                patch.object(spec.native, "load_source", return_value=(None, Path("unused"))), \
                patch.object(spec.shared, "episode", side_effect=episode), patch.object(spec.shared, "qualify"), \
                patch.object(spec.labels, "training_context", return_value=cell["training_context"]), \
                patch.object(spec, "raw_directory", return_value=Path("unused")), \
                patch.object(spec, "write_json", side_effect=write), \
                patch.dict(sys.modules, {"frequency": SimpleNamespace(DemandFrequencyTracker=None)}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            result = spec.collect(397, "low_noise", 8, models["397"], primary["model"], Path("unused/result.json"))
        self.assertEqual(events, [None, "sealed", 5, 0, 1, 2, 3, 4])
        self.assertEqual(result["prediction"], cell["prediction"])
        self.assertIn("VALUE_TRANSFER_IDENTITY_SOURCE_QUALIFIED", out.getvalue())

    def test_sixty_unique_unbound_tasks_only_stage_code_and_new_small_inputs(self):
        tasks = [task_specification("unit_transfer", r, s, i)
            for r in spec.ROOTS for s in spec.SCENARIOS for i in spec.PROBES]
        self.assertEqual(len({t["signature"] for t in tasks}), 60)
        self.assertEqual(len({t["result_dir"] for t in tasks}), 60)
        for t in tasks:
            self.assertIsNone(t["require_node"])
            self.assertEqual(t["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((t["cpu"], t["ram_mb"]), (1, 3072))
            self.assertEqual([Path(p).name for p in t["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet", "inputs"])
            self.assertIn("--value-model", t["cmd"])


if __name__ == "__main__":
    unittest.main()
