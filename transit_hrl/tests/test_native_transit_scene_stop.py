import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_scene_stop_stage182 as spec
from scripts.analyze_native_transit_scene_stop_stage182 import summarize
from scripts.submit_native_transit_scene_stop_stage182_scheduleurm import task_specification
from tests.test_native_transit_floor_value import fixture


class SceneStopTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_inner_split_is_outcome_independent_and_keeps_whole_scenes(self):
        rows = spec.source.training_rows(*fixture())
        train, validation = spec.inner_split(rows)
        tg, vg = ({spec.paired.scene_group(r) for r in rr} for rr in (train, validation))
        self.assertEqual((len(tg), len(vg), len(train), len(validation)), (16, 4, 96, 24))
        self.assertTrue(tg.isdisjoint(vg))
        self.assertEqual({r["root"] for r in train}, {397, 401})
        self.assertEqual({r["root"] for r in validation}, {397, 401})
        changed = copy.deepcopy(rows)
        for row in changed: row["labels"] = []
        a, b = spec.inner_split(changed[::-1])
        self.assertEqual({spec.paired.scene_group(r) for r in a}, tg)
        self.assertEqual({spec.paired.scene_group(r) for r in b}, vg)

    def test_checkpoint_validation_does_not_update_weights_or_scaling(self):
        rows = spec.source.training_rows(*fixture())
        train, validation = spec.inner_split(rows)
        altered = copy.deepcopy(validation)
        for row in altered:
            for label in row["labels"]:
                label["future_fleet_cost"] += 1.
                label["wait_delta_min"] *= 100.
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2, 3)):
            models = [spec.value.fit_value(train, "floor_absolute", steps=3, validation_rows=v)
                for v in (None, validation, altered)]
        for model in models[1:]:
            for key in ("weights", "center", "scale", "target_scale", "regret_cost_scale", "loss_ledger"):
                self.assertEqual(model[key], models[0][key])
            self.assertEqual([c["step"] for c in model["checkpoint_ledger"]], [1, 2, 3])
        self.assertNotEqual(models[1]["checkpoint_ledger"], models[2]["checkpoint_ledger"])
        for c in models[1]["checkpoint_ledger"]:
            self.assertAlmostEqual(c["validation_loss"], c["absolute_fleet_mse"]+c["paired_mse"], places=6)
        with self.assertRaises(ValueError):
            spec.value.fit_value(train, "floor_absolute", steps=1, validation_rows=train[:6])

    def test_checkpoint_is_minimum_predictive_loss_with_earlier_tie(self):
        ledger = [{"step": s, "validation_loss": v} for s, v in [(100, 2.), (300, 1.), (600, 1.), (1500, 3.)]]
        self.assertEqual(spec.selected_checkpoint(ledger), 300)
        self.assertEqual(spec.selected_checkpoint(ledger[::-1]), 300)

    def test_selected_refit_restarts_and_uses_all_outer_training_scenes(self):
        inputs = fixture()
        actual_fit, calls = spec.value.fit_value, []
        def short_fit(rows, method, *, steps=1500, validation_rows=None):
            model = actual_fit(rows, method, steps=steps if validation_rows is None else 2,
                validation_rows=validation_rows)
            if validation_rows is not None: model["steps"] = 1500
            calls.append((copy.deepcopy(rows), steps, validation_rows is not None, model))
            return model
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2)), patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            result = spec.fit_and_validate(*inputs)
        self.assertEqual(len(calls), 56)
        self.assertEqual(len(result["selection_ledger"]), 28)
        self.assertEqual(result["inner_critic_updates"], 42000)
        self.assertEqual(result["critic_updates"], 42000+sum(r["selected_steps"] for r in result["selection_ledger"]))
        for i in range(0, len(calls), 2):
            inner_rows, _, validation, inner = calls[i]
            refit_rows, steps, validation_refit, refit = calls[i+1]
            self.assertTrue(validation)
            self.assertFalse(validation_refit)
            self.assertLess(len(inner_rows), len(refit_rows))
            self.assertEqual(steps, spec.selected_checkpoint(inner["checkpoint_ledger"]))
            expected = actual_fit(refit_rows, "floor_absolute", steps=steps)
            self.assertEqual(refit["weights"], expected["weights"])

    def test_outer_heldout_labels_cannot_choose_their_checkpoint(self):
        inputs = fixture()
        rows = spec.source.training_rows(*inputs)
        held = spec.paired.scene_group(rows[0])
        train = [r for r in rows if spec.paired.scene_group(r) != held]
        test = [r for r in rows if spec.paired.scene_group(r) == held]
        it, iv = spec.inner_split(train)
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2)):
            inner = spec.value.fit_value(it, "floor_absolute", steps=2, validation_rows=iv)
            step = spec.selected_checkpoint(inner["checkpoint_ledger"])
            model = spec.value.fit_value(train, "floor_absolute", steps=step)
        changed = copy.deepcopy(test)
        for row in changed:
            for label in row["labels"]:
                label.update(future_fleet_cost=100., wait_delta_min=100., actual_cost_delta=100.)
        self.assertNotIn(list(held), model["training_groups"])
        self.assertNotIn(list(held), inner["validation_groups"])
        self.assertEqual([spec.value.predict(r, model) for r in test], [spec.value.predict(r, model) for r in changed])

    def test_nested_analysis_preserves_primary_source_groups_and_actual_budget(self):
        inputs = fixture()
        actual_fit = spec.value.fit_value
        def short_fit(rows, method, *, steps=1500, validation_rows=None):
            model = actual_fit(rows, method, steps=2 if steps == 1500 else steps, validation_rows=validation_rows)
            if steps == 1500: model["steps"] = 1500
            return model
        with patch.object(spec.value, "CHECKPOINT_STEPS", (1, 2)), patch.object(spec.value, "fit_value", side_effect=short_fit), contextlib.redirect_stdout(io.StringIO()):
            control = spec.source.fit_and_validate(*inputs, "floor_absolute")
            cell = spec.fit_and_validate(*inputs)
            result = summarize(cell, control)
            self.assertEqual(result["new_control_updates"], 0)
            self.assertEqual(len(result["variants"]), 4)
            self.assertFalse(result["native_deployment_eligible"])
            self.assertEqual(result["third_scene_screen_eligible"], cell["third_scene_screen_eligible"])
            for failure in ("inner_roles", "wrong_step", "outer_scene", "budget", "primary"):
                changed = copy.deepcopy(cell)
                if failure == "inner_roles": changed["selection_ledger"][0]["inner_validation_groups"] = []
                elif failure == "wrong_step": changed["selection_ledger"][0]["selected_steps"] = 1500
                elif failure == "outer_scene": changed["validation"]["cost_only"]["scene"]["predictions"][0]["scene_seed"] += 1
                elif failure == "budget": changed["critic_updates"] += 1
                else: changed["third_scene_screen_eligible"] = not changed["third_scene_screen_eligible"]
                with self.subTest(failure=failure), self.assertRaises(ValueError): summarize(changed, control)
        json.dumps(result)

    def test_scheduler_runs_only_one_unbound_nested_critic_task(self):
        task = task_specification("unit_scene_stop")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertTrue(task["result_dir"].endswith("unit_scene_stop/fit"))
        self.assertEqual(len(task["stage_input_paths"]), 4)
        self.assertIn("OPENBLAS_NUM_THREADS=1", task["cmd"])


if __name__ == "__main__":
    unittest.main()
