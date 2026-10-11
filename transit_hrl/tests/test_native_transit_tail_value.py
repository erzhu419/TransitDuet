import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_tail_value_stage184 as spec
from scripts.analyze_native_transit_tail_value_stage184 import summarize
from scripts.submit_native_transit_tail_value_stage184_scheduleurm import task_specification
from tests.test_native_transit_floor_value import fixture as old_fixture


def fixture():
    inputs = old_fixture()
    for bank in (inputs[0]["training_data"], inputs[1]["training_data"], inputs[2]["evaluation_data"]):
        for row in bank:
            for i, candidate in enumerate(row["candidates"]):
                candidate.update(wait_cost=.3+i*.01, headway_cost=i*.02)
    return inputs


class TailValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_tail_summaries_separate_equal_prefix_actions(self):
        row = spec.source.source.source.training_rows(*fixture())[0]
        row["candidates"][4] = copy.deepcopy(row["candidates"][3])
        row["candidates"][3].update(wait_cost=.30090945882613707, headway_cost=.02295145763233575)
        row["candidates"][4].update(wait_cost=.3008926870563994, headway_cost=.030588462198637856)
        base, tail = spec.value.features(row), spec.value.features(row, "tail_summary")
        np.testing.assert_array_equal(base[3], base[4])
        np.testing.assert_array_equal(tail[:, :96], base)
        self.assertEqual(tail.shape, (6, 98))
        self.assertFalse(np.array_equal(tail[3], tail[4]))
        d = spec.alias_diagnostics([row])
        self.assertEqual(len(d["old_distinct_outcome_aliases"]), 1)
        self.assertEqual(d["remaining_distinct_outcome_aliases"], [])

    def test_initial_predictor_is_preserved_and_added_weights_learn(self):
        rows = spec.source.source.source.training_rows(*fixture())[:6]
        models = [spec.value.fit_value(rows, "floor_absolute", steps=0, objective="proper_mse", representation=r)
            for r in ("occupancy", "tail_summary")]
        old, new = models
        self.assertEqual((old["parameter_count"], new["parameter_count"]), (1603, 1635))
        self.assertNotIn("representation", old)
        for key in ("0.bias", "2.weight", "2.bias"):
            self.assertEqual(old["weights"][key], new["weights"][key])
        np.testing.assert_array_equal(np.asarray(new["weights"]["0.weight"])[:, :96], old["weights"]["0.weight"])
        np.testing.assert_array_equal(np.asarray(new["weights"]["0.weight"])[:, 96:], np.zeros((16, 2)))
        for key in ("predicted_heads", "predicted_wait_delta_min", "absolute_fleet_cost"):
            np.testing.assert_allclose(spec.value.predict(rows[0], old)[key], spec.value.predict(rows[0], new)[key], atol=1e-12)
        trained = spec.value.fit_value(rows, "floor_absolute", steps=4, objective="proper_mse", representation="tail_summary")
        self.assertTrue(np.any(np.asarray(trained["weights"]["0.weight"])[:, 96:] != 0))

    def test_training_only_scaling_and_serialized_inference_ignore_outcomes(self):
        rows = spec.source.source.source.training_rows(*fixture())[:12]
        group = spec.paired.scene_group(rows[0])
        train = [r for r in rows if spec.paired.scene_group(r) != group]
        model = spec.value.fit_value(train, "floor_absolute", steps=3, objective="proper_mse", representation="tail_summary")
        raw = np.stack([spec.value.features(r, "tail_summary") for r in train])
        np.testing.assert_array_equal(model["center"], raw.mean(axis=(0, 1)))
        expected_scale = raw.std(axis=(0, 1))
        expected_scale[expected_scale < 1e-8] = 1.
        np.testing.assert_array_equal(model["scale"], expected_scale)
        changed = copy.deepcopy(rows[0])
        changed.update(labels=[], root=999, scene_seed=0, scenario="unseen")
        original = spec.value.predict(rows[0], model)
        self.assertEqual(original, spec.value.predict(changed, json.loads(json.dumps(model))))
        self.assertEqual(original["predicted_heads"][rows[0]["original_choice"]], [0., 0.])
        self.assertEqual(original["predicted_wait_delta_min"][rows[0]["original_choice"]], 0.)
        self.assertTrue(all(c >= original["causal_fleet_floor"] for c in original["absolute_fleet_cost"]))
        for c in changed["candidates"]:
            c.update(wait_cost=1e6, headway_cost=1e6)
        self.assertEqual(model["center"], raw.mean(axis=(0, 1)).tolist())
        with self.assertRaises(ValueError):
            spec.value.fit_value(train, "floor_absolute", steps=1, representation="tail_summary", validation_rows=train)

    def test_same_step_group_pairing_and_fixed_primary(self):
        inputs, actual_fit = fixture(), spec.value.fit_value
        def short_fit(rows, method, *, steps=1500, **kwargs):
            model = actual_fit(rows, method, steps=2 if steps == 1500 else steps, **kwargs)
            if steps == 1500: model["steps"] = 1500
            return model
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2)), patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            stop = spec.source.source.fit_and_validate(*inputs)
            control = spec.source.fit_and_validate(*inputs, stop)
            cell = spec.fit_and_validate(*inputs, control)
        result = summarize(cell, control)
        self.assertEqual(cell["frozen_selected_steps"], control["frozen_selected_steps"])
        self.assertEqual(result["critic_updates"], control["critic_updates"])
        self.assertEqual(result["new_control_updates"], 0)
        self.assertFalse(result["native_deployment_eligible"])
        for failure in ("steps", "groups", "targets", "architecture", "primary"):
            changed = copy.deepcopy(cell)
            if failure == "steps": changed["fit_ledger"][0]["steps"] += 1
            elif failure == "groups": changed["fit_ledger"][0]["training_groups"].pop()
            elif failure == "targets": changed["fit_ledger"][0]["target_scale"][0] += 1
            elif failure == "architecture": changed["fit_ledger"][0]["parameter_count"] = 1603
            else: changed["third_scene_screen_eligible"] = not changed["third_scene_screen_eligible"]
            with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed, control)

    def test_scheduler_is_one_small_unbound_task(self):
        task = task_specification("unit_tail_value")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual(len(task["stage_input_paths"]), 4)
        self.assertIn("run_native_transit_tail_value_stage184.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
