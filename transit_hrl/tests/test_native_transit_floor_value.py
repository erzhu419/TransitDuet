import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn.functional import softplus

from scripts import run_native_transit_floor_value_stage181 as spec
from scripts.analyze_native_transit_floor_value_stage181 import summarize
from scripts.submit_native_transit_floor_value_stage181_scheduleurm import task_specification
from tests.test_native_transit_value_balance import fixture as old_fixture


def fixture():
    first, labels, second = old_fixture()
    rows = spec.source.training_rows(first, labels, second)
    baselines = []
    for group in sorted({spec.paired.scene_group(r) for r in rows}):
        rr = [r for r in rows if spec.paired.scene_group(r) == group]
        baselines.append({"root": group[0], "scene_seed": group[1], "scenario": rr[0]["scenario"],
            "peak_fleet": max(r["occupancy_context"]["historical_peak"] for r in rr)+1})
    return first, labels, second, baselines


class FloorValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_absolute_targets_come_from_source_labels_not_inference_features(self):
        inputs = fixture()
        old = copy.deepcopy(inputs)
        rows = spec.training_rows(*inputs)
        self.assertEqual(len(rows), 120)
        self.assertEqual(inputs, old)
        baseline = {(b["root"], b["scene_seed"]): b for b in inputs[-1]}
        for row in rows:
            peak = baseline[spec.paired.scene_group(row)]["peak_fleet"]
            for label in row["labels"]:
                self.assertAlmostEqual(label["future_fleet_cost"]+spec.value.floor_cost(row),
                    max(peak-12, 0)**2/12+label["fleet_cost_delta"])
        for failure in ("missing_scene", "wrong_scene", "below_floor", "negative_future_cost"):
            changed = copy.deepcopy(inputs)
            if failure == "missing_scene": changed[-1].pop()
            elif failure == "wrong_scene": changed[-1][0]["scenario"] = "other"
            elif failure == "below_floor": changed[-1][0]["peak_fleet"] = 0
            else:
                for row in changed[0]["training_data"]:
                    if row["scene_seed"] == changed[-1][0]["scene_seed"]:
                        row["occupancy_context"]["historical_peak"] = changed[-1][0]["peak_fleet"]
            with self.subTest(failure=failure), self.assertRaises(ValueError): spec.training_rows(*changed)

    def test_floor_and_serialized_physical_units_match_torch_and_preserve_identity(self):
        rows = spec.training_rows(*fixture())[:6]
        rng = torch.get_rng_state().clone()
        models = {m: spec.value.fit_value(rows, m, steps=4) for m in spec.value.METHODS}
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for method, model in models.items():
            self.assertEqual(model["parameter_count"], 1603)
            self.assertGreater(model["parameter_update_l2"], 0)
            p = spec.value.predict(rows[0], json.loads(json.dumps(model)))
            x = torch.as_tensor((spec.occupancy.features(rows[0], "occupancy_regret")-model["center"])/model["scale"], dtype=torch.float32)
            w = {k: torch.tensor(v) for k, v in model["weights"].items()}
            output = torch.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"]
            if method == "floor_absolute": output[:, 0] = softplus(output[:, 0])
            physical = output*torch.tensor(model["target_scale"])
            delta = physical-physical[rows[0]["original_choice"]].clone()
            np.testing.assert_allclose(p["predicted_heads"], delta[:, :2].numpy(), atol=1e-7, rtol=1e-5)
            np.testing.assert_allclose(p["predicted_wait_delta_min"], delta[:, 2].numpy(), atol=1e-7, rtol=1e-5)
            np.testing.assert_array_equal(p["predicted_heads"][rows[0]["original_choice"]], [0., 0.])
            self.assertEqual(p["predicted_wait_delta_min"][rows[0]["original_choice"]], 0.)
            model["weights"]["2.weight"] = np.zeros((3, 16)).tolist()
            model["weights"]["2.bias"] = [-1000., 0., 0.]
            p = spec.value.predict(rows[0], model)
            if method == "floor_absolute":
                np.testing.assert_array_equal(p["absolute_fleet_cost"], [spec.value.floor_cost(rows[0])]*6)
            else:
                self.assertTrue(all(v < p["causal_fleet_floor"] for v in p["absolute_fleet_cost"]))

    def test_predictions_ignore_terminal_labels_and_group_ids(self):
        rows = spec.training_rows(*fixture())[:6]
        model = spec.value.fit_value(rows, "floor_absolute", steps=2)
        changed = copy.deepcopy(rows[0])
        changed.update(root=999, scene_seed=1, scenario="unused", labels=[])
        self.assertEqual(spec.value.predict(rows[0], model), spec.value.predict(changed, model))

    def test_wait_constraint_filters_without_changing_physical_cost_weights(self):
        row = {"original_choice": 2}
        prediction = {"predicted_heads": [[-.5, .1], [-.1, 0.], [0., 0.], [-.05, 0.]],
            "predicted_wait_delta_min": [.01, -.02, 0., -100.]}
        with patch.object(spec.value, "predict", return_value=prediction):
            self.assertEqual(spec.value.choose_action(row, {}, "cost_only"), 0)
            self.assertEqual(spec.value.choose_action(row, {}, "wait_feasible"), 1)
            prediction["predicted_heads"] = [[0., 0.]]*4
            self.assertEqual(spec.value.choose_action(row, {}, "wait_feasible"), 2)

    def test_absolute_supervision_identifies_the_paired_fleet_offset(self):
        rows = spec.training_rows(*fixture())[:6]
        a, b = copy.deepcopy(rows), copy.deepcopy(rows)
        for row in a:
            for label in row["labels"]: label["future_fleet_cost"] = 0.
        for row in b:
            for label in row["labels"]: label["future_fleet_cost"] = 1.
        models = [spec.value.fit_value(rr, "raw_absolute", steps=1) for rr in (a, b)]
        self.assertNotEqual(models[0]["loss_ledger"][0]["absolute_fleet_mse"], models[1]["loss_ledger"][0]["absolute_fleet_mse"])
        self.assertEqual(models[0]["loss_ledger"][0]["paired_mse"], models[1]["loss_ledger"][0]["paired_mse"])
        self.assertNotEqual(models[0]["weights"], models[1]["weights"])

    def test_heldout_fleet_and_wait_targets_cannot_change_own_predictions(self):
        rows = spec.training_rows(*fixture())[:12]
        changed = copy.deepcopy(rows)
        for row in changed[:6]:
            for label in row["labels"]:
                label["future_fleet_cost"] += 100.
                label["wait_delta_min"] *= 100.
        group = spec.paired.scene_group(rows[0])
        predictions = []
        for rr in (rows, changed):
            model = spec.value.fit_value([r for r in rr if spec.paired.scene_group(r) != group], "floor_absolute", steps=3)
            self.assertNotIn(list(group), model["training_groups"])
            predictions.append([spec.value.predict(r, model) for r in rr[:6]])
        self.assertEqual(*predictions)

    def test_two_shared_fits_qualify_four_variants_with_fixed_primary_and_budget(self):
        inputs = fixture()
        actual_fit = spec.value.fit_value
        def short_fit(rows, method):
            model = actual_fit(rows, method, steps=2)
            model["steps"] = spec.occupancy.STEPS
            return model
        with patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            cells = {m: spec.fit_and_validate(*inputs, m) for m in spec.value.METHODS}
        result = summarize(cells)
        self.assertEqual(len(result["variants"]), 4)
        self.assertEqual(result["critic_updates"], 84000)
        self.assertFalse(result["native_deployment_eligible"])
        self.assertEqual(result["third_scene_screen_eligible"], cells["floor_absolute"]["third_scene_screen_eligible"])
        self.assertEqual([len(result["variants"][spec.PRIMARY]["validation"][s]["folds"]) for s in ("scene", "regime", "lower_root")], [20, 5, 2])
        for failure in ("primary", "floor", "wait_selection", "shared_predictions"):
            changed = copy.deepcopy(cells)
            row = changed["floor_absolute"]["validation"]["wait_feasible"]["scene"]["predictions"][0]
            if failure == "primary": changed["raw_absolute"]["third_scene_screen_eligible"] = True
            elif failure == "floor": row["absolute_fleet_cost"][0] = row["causal_fleet_floor"]-1
            elif failure == "wait_selection": row["predicted_wait_delta_min"][row["original_choice"]] = 1.
            else: row["predicted_wait_delta_min"][0] -= 1.
            with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed)

    def test_tasks_keep_two_small_fits_unbound_without_native_runs(self):
        tasks = [task_specification("unit_floor", m) for m in spec.value.METHODS]
        self.assertEqual(len({t["signature"] for t in tasks}), 2)
        for task in tasks:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertEqual(len(task["stage_input_paths"]), 4)
            self.assertIn("--baselines", task["cmd"])
            self.assertNotIn("run_native_transit_value_transfer_stage179.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
