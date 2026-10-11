import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import torch

from scripts import run_native_transit_proper_value_stage183 as spec
from scripts.analyze_native_transit_proper_value_stage183 import summarize
from scripts.submit_native_transit_proper_value_stage183_scheduleurm import task_specification
from tests.test_native_transit_floor_value import fixture


class ProperValueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_regret_has_no_gradient_effect_in_proper_mse(self):
        rows = spec.source.source.training_rows(*fixture())[:6]
        models = {}
        for objective in ("decision_regret", "proper_mse"):
            models[objective] = []
            for sign in (-1, 1):
                with patch.object(spec.value.occupancy, "expected_regret", side_effect=lambda p, y: sign*1000*p.sum()):
                    models[objective].append(spec.value.fit_value(rows, "floor_absolute", steps=1, objective=objective))
        self.assertEqual(models["proper_mse"][0]["weights"], models["proper_mse"][1]["weights"])
        self.assertNotEqual(models["decision_regret"][0]["weights"], models["decision_regret"][1]["weights"])
        self.assertEqual(models["proper_mse"][0]["objective"], "proper_mse")
        for key in ("center", "scale", "target_scale", "regret_cost_scale", "parameter_count"):
            self.assertEqual(models["proper_mse"][0][key], models["decision_regret"][0][key])

    def test_frozen_schedule_and_objective_are_the_only_fitting_changes(self):
        inputs, actual_fit = fixture(), spec.value.fit_value
        def short_fit(rows, method, *, steps=1500, validation_rows=None, objective="decision_regret"):
            model = actual_fit(rows, method, steps=2 if steps == 1500 else steps,
                validation_rows=validation_rows, objective=objective)
            if steps == 1500: model["steps"] = 1500
            return model
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2)), patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            control = spec.source.fit_and_validate(*inputs)
            cell = spec.fit_and_validate(*inputs, control)
            result = summarize(cell, control)
            self.assertEqual(cell["frozen_selected_steps"], [r["selected_steps"] for r in control["selection_ledger"]])
            self.assertEqual(cell["critic_updates"], control["refit_critic_updates"])
            self.assertEqual(result["new_control_updates"], 0)
            self.assertFalse(result["native_deployment_eligible"])
            for failure in ("budget", "groups", "steps", "scales", "primary"):
                changed = copy.deepcopy(cell)
                if failure == "budget": changed["critic_updates"] += 1
                elif failure == "groups": changed["fit_ledger"][0]["training_groups"].pop()
                elif failure == "steps": changed["fit_ledger"][0]["steps"] += 1
                elif failure == "scales": changed["fit_ledger"][0]["target_scale"][0] += 1
                else: changed["third_scene_screen_eligible"] = not changed["third_scene_screen_eligible"]
                with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed, control)
            changed = copy.deepcopy(control)
            changed["selection_ledger"][0]["outer_training_groups"] = []
            with self.assertRaises(ValueError): spec.fit_and_validate(*inputs, changed)

    def test_proper_value_still_ignores_outcome_fields_during_inference(self):
        rows = spec.source.source.training_rows(*fixture())[:6]
        model = spec.value.fit_value(rows, "floor_absolute", steps=2, objective="proper_mse")
        row = copy.deepcopy(rows[0])
        row.update(labels=[], root=0, scenario="other", scene_seed=0)
        self.assertEqual(spec.value.predict(row, model), spec.value.predict(rows[0], model))

    def test_one_small_unbound_scheduler_task_has_no_new_inner_fitting(self):
        task = task_specification("unit_proper_value")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertEqual(len(task["stage_input_paths"]), 4)
        self.assertIn("run_native_transit_proper_value_stage183.py", task["cmd"])


if __name__ == "__main__":
    unittest.main()
