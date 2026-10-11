import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_value_balance_stage180 as spec
from scripts.analyze_native_transit_value_balance_stage180 import summarize
from scripts.submit_native_transit_value_balance_stage180_scheduleurm import task_specification
from tests.test_native_transit_occupancy_value import fixture as first_fixture


def fixture():
    first, labels = first_fixture()
    rows = copy.deepcopy(first["training_data"])
    for row in rows:
        row["scene_seed"] += 100
        for label in row["labels"]:
            label["physical_return_delta"] = -100*label["actual_cost_delta"]
    return first, labels, {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(),
        "software_qualified": True, "evaluation_states": 60, "native_steps": 420*61380,
        "source_control_reproductions": 120, "paired_interventions": 360, "window_ledger_qualified_episodes": 420,
        "training_updates": 0, "actor_updates": 0, "critic_updates": 0, "evaluation_data": rows,
        "evaluation_groups": [{**g, "scene_seed": g["scene_seed"]+100} for g in labels["fitting_groups"]]}


class ValueBalanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_head_scales_use_paired_cost_differences(self):
        targets = np.asarray([[[i*.1, i*.001] for i in range(6)]]*2)
        scale, native = spec.value.training_cost_scales(targets, "joint_native")
        self.assertEqual(scale, native)
        self.assertAlmostEqual(native, np.sqrt(np.mean(targets.sum(-1)**2)))
        heads, total = spec.value.training_cost_scales(targets, "paired_heads")
        a, b = np.triu_indices(6, 1)
        np.testing.assert_allclose(heads, np.sqrt(np.mean((targets[:, a]-targets[:, b])**2, axis=(0, 1))))
        self.assertAlmostEqual(heads[0]/heads[1], 100.)
        self.assertEqual(total, native)
        zeros, _ = spec.value.training_cost_scales(np.zeros((1, 6, 2)), "paired_heads")
        np.testing.assert_array_equal(zeros, [1e-6, 1e-6])

    def test_regret_keeps_physical_head_weights_and_serialized_inference_restores_units(self):
        rows = fixture()[0]["training_data"][:6]
        capture, models = {}, {}
        actual_regret = spec.value.expected_regret
        for normalization in spec.NORMALIZATIONS:
            def regret(predicted, targets):
                capture[normalization] = (predicted.detach().clone(), targets.detach().clone())
                return actual_regret(predicted, targets)
            with patch.object(spec.value, "expected_regret", side_effect=regret):
                models[normalization] = spec.value.fit_value(rows, spec.source.fitted.PRIMARY,
                    steps=1, normalization=normalization)
        joint, heads = (models[n] for n in spec.NORMALIZATIONS)
        units = np.asarray(heads["cost_scale"])/heads["regret_cost_scale"]
        np.testing.assert_allclose(capture["paired_heads"][0].numpy(), capture["joint_native"][0].numpy()*units,
            atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(capture["paired_heads"][1], capture["joint_native"][1])
        model = models["paired_heads"]
        predicted = spec.value.predict_value(rows[0], json.loads(json.dumps(model)))
        x = (spec.value.features(rows[0], model["method"])-model["center"])/model["scale"]
        w = {k: np.asarray(v) for k, v in model["weights"].items()}
        physical = (np.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"])*model["cost_scale"]
        np.testing.assert_array_equal(predicted, physical-physical[rows[0]["original_choice"]])
        np.testing.assert_array_equal(predicted[rows[0]["original_choice"]], [0., 0.])
        self.assertEqual(model["parameter_count"], joint["parameter_count"])

    def test_expanded_training_keeps_all_twenty_groups_and_unmodified_labels(self):
        first, labels, second = fixture()
        old = copy.deepcopy((first, labels, second))
        rows = spec.training_rows(first, labels, second)
        self.assertEqual(len(rows), 120)
        self.assertEqual(len({spec.paired.scene_group(r) for r in rows}), 20)
        self.assertEqual((first, labels, second), old)
        for failure in ("duplicate", "overlap", "cost", "identity", "candidate", "qualification"):
            changed = copy.deepcopy(second)
            row = changed["evaluation_data"][0]
            if failure == "duplicate": changed["evaluation_data"][1] = copy.deepcopy(row)
            elif failure == "overlap": row["scene_seed"] -= 100
            elif failure == "cost": row["labels"][0]["actual_cost_delta"] += 1
            elif failure == "identity": row["labels"][2]["wait_delta_min"] = 1
            elif failure == "candidate": row["labels"].pop()
            else: changed["software_qualified"] = False
            with self.subTest(failure=failure), self.assertRaises(ValueError): spec.training_rows(first, labels, changed)

    def test_heldout_labels_never_change_head_scales_or_value_predictions(self):
        rows = fixture()[0]["training_data"][:12]
        def fit(train):
            return spec.value.fit_value(train, spec.source.fitted.PRIMARY, steps=3, normalization="paired_heads")
        a = spec.paired.grouped_validation(rows, "scene", fit_model=fit, predictor=spec.value.predict_value)
        changed = copy.deepcopy(rows)
        for row in changed[:6]:
            for label in row["labels"]:
                for head in spec.paired.HEADS: label[head] *= 100.
        b = spec.paired.grouped_validation(changed, "scene", fit_model=fit, predictor=spec.value.predict_value)
        self.assertEqual([p["predicted_heads"] for p in a["predictions"][:6]],
            [p["predicted_heads"] for p in b["predictions"][:6]])
        held = list(spec.paired.scene_group(rows[0]))
        self.assertNotIn(held, a["folds"][0]["training_groups"])

    def test_matched_fits_cannot_swap_primary_or_authorize_native_deployment(self):
        first, labels, second = fixture()
        actual_fit = spec.value.fit_value
        def short_fit(rows, method, *, normalization):
            model = actual_fit(rows, method, steps=2, normalization=normalization)
            model["steps"] = spec.value.STEPS
            return model
        with patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            cells = {n: spec.fit_and_validate(first, labels, second, n) for n in spec.NORMALIZATIONS}
        result = summarize(cells)
        self.assertEqual(result["critic_updates"], 84000)
        self.assertFalse(result["native_deployment_eligible"])
        self.assertEqual(result["third_scene_screen_eligible"], cells[spec.PRIMARY]["third_scene_screen_eligible"])
        self.assertEqual([len(cells[spec.PRIMARY]["validation"][s]["folds"]) for s in ("scene", "regime", "lower_root")], [20, 5, 2])
        changed = copy.deepcopy(cells)
        changed["joint_native"]["third_scene_screen_eligible"] = True
        with self.assertRaises(ValueError): summarize(changed)
        changed = copy.deepcopy(cells)
        changed[spec.PRIMARY]["validation"]["scene"]["predictions"][0]["scene_seed"] += 1
        with self.assertRaises(ValueError): summarize(changed)

    def test_two_matched_tasks_keep_code_and_compact_inputs_unbound(self):
        tasks = [task_specification("unit_balance", n) for n in spec.NORMALIZATIONS]
        self.assertEqual(len({t["result_dir"] for t in tasks}), 2)
        for task in tasks:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertEqual(len(task["stage_input_paths"]), 4)
            self.assertIn("--normalization", task["cmd"])


if __name__ == "__main__":
    unittest.main()
