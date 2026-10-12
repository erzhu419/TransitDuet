import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_fleet_residual_stage193 as spec
from scripts.analyze_native_transit_fleet_residual_stage193 import qualify, summarize
from scripts.submit_native_transit_fleet_residual_stage193_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_expected_value import fixture


def fleet_model(rows, method):
    m = fake_model(rows, "tail_summary")
    m.update(method=method, parameter_count=1601, target_scale=1.,
        objective="absolute_fleet_MSE_plus_paired_fleet_MSE", initialization="shared_scalar_hidden_zero_fleet_projection")
    return m


def predictions(rows):
    return [{**r, "fleet_absolute_mse": 0., "selected_fleet_delta_error": 0.} for r in fake_predictions(rows)]


class FleetResidualTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bank, _, _, _ = bank_fixture()
        cls.targets, _ = spec.source.build_targets(cls.bank, fixture(cls.bank))
        cls.registered = spec.manifest(spec.source.manifest(cls.bank["manifest"]))

    def test_zero_projection_is_exact_prior_or_zero_and_labels_are_not_inputs(self):
        rows = self.bank["training_data"][:6]
        rng = torch.get_rng_state().clone()
        models = {m: spec.value.fit_fleet(rows, m, steps=0) for m in spec.value.METHODS}
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertEqual(models["fleet_from_zero"]["weights"], models["fleet_residual"]["weights"])
        for method, model in models.items():
            self.assertEqual(model["parameter_count"], 1601)
            expected = spec.value.prior_future(rows[0]) if method == "fleet_residual" else np.zeros(6)
            np.testing.assert_array_equal(spec.value.predict_future(rows[0], model), expected)
            changed = copy.deepcopy(rows[0])
            changed.update(labels=[], root=999, scene_seed=-1, scenario="unused")
            np.testing.assert_array_equal(spec.value.predict_future(changed, model), expected)

    def test_zero_prior_can_learn_and_numpy_matches_training_clamp(self):
        rows = copy.deepcopy(self.bank["training_data"][:6])
        for row in rows:
            for c in row["candidates"]: c["fleet_cost"] = 0.
            for label in row["labels"]: label["future_fleet_cost"] = .2
        models = {m: spec.value.fit_fleet(rows, m, steps=4) for m in spec.value.METHODS}
        self.assertEqual(models["fleet_from_zero"]["weights"], models["fleet_residual"]["weights"])
        for model in models.values():
            self.assertGreater(model["parameter_update_l2"], 0.)
            p = spec.value.predict_future(rows[0], json.loads(json.dumps(model)))
            self.assertTrue(np.all(p > 0))
            x = torch.tensor((spec.value.floor.features(rows[0], "tail_summary")-model["center"])/model["scale"], dtype=torch.float32)
            w = {k: torch.tensor(v) for k, v in model["weights"].items()}
            output = (torch.tanh(x@w["0.weight"].T+w["0.bias"])@w["2.weight"].T+w["2.bias"]).ravel()
            np.testing.assert_allclose(p, output.clamp_min(0).numpy()*model["target_scale"], atol=1e-8, rtol=1e-5)
            model["weights"]["2.weight"] = np.zeros((1, 16)).tolist()
            model["weights"]["2.bias"] = [-100.]
            np.testing.assert_array_equal(spec.value.predict_future(rows[0], model), np.zeros(6))

    def test_only_fleet_changes_and_exact_identity_is_retained(self):
        rows = self.bank["training_data"][:6]
        reference = spec.value.floor.fit_value(rows, "floor_absolute", steps=0, objective="proper_mse", representation="tail_summary")
        models = {m: spec.value.fit_fleet(rows, m, steps=0) for m in spec.value.METHODS}
        for variant in spec.value.VARIANTS:
            original = spec.value.floor.predict(rows[0], reference)
            result = spec.value.predict(rows[0], reference, variant, models)
            self.assertEqual(result["predicted_wait_delta_min"], original["predicted_wait_delta_min"])
            np.testing.assert_array_equal(np.asarray(result["predicted_heads"])[:, 1], np.asarray(original["predicted_heads"])[:, 1])
            np.testing.assert_array_equal(result["predicted_heads"][rows[0]["original_choice"]], [0., 0.])
            self.assertTrue(np.all(np.asarray(result["absolute_fleet_cost"]) >= result["causal_fleet_floor"]))

    def test_worker_holds_out_all_callbacks_and_does_not_retrain_controls(self):
        fold = self.registered["folds"][0]
        train, _, _ = spec.source.source.split_rows(self.bank, fold["split"], fold["index"])
        reference = fake_model(train, "tail_summary")
        with patch.object(spec.value, "fit_fleet", side_effect=lambda rows, method, **_: fleet_model(rows, method)) as fit, \
                patch.object(spec.value, "evaluate", side_effect=lambda rows, *args: predictions(rows)):
            cell, _ = spec.fit_fold(self.bank, self.targets, fold, reference)
        self.assertEqual(fit.call_count, 2)
        for call in fit.call_args_list:
            self.assertEqual(len(call.args[0]), 594)
            self.assertNotIn(tuple(fold["held_out"]), {spec.source.source.paired.scene_group(r) for r in call.args[0]})
        qualify(cell, fold, self.registered)
        for failure in ("groups", "controls", "selector"):
            broken = copy.deepcopy(cell)
            if failure == "groups": broken["fits"]["fleet_residual"]["training_groups"].append(fold["held_out"])
            elif failure == "controls": broken["predictions"]["fleet_residual"]["mean8"]["cost_only"][0]["predicted_wait_delta_min"][0] = 10.
            else: broken["predictions"]["fleet_residual"]["mean8"]["wait_feasible"][0]["selected"] = -1
            with self.subTest(failure=failure), self.assertRaises(ValueError): qualify(broken, fold, self.registered)

    def test_report_keeps_all_comparisons_and_reserved_screen_closed(self):
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.source.source.split_rows(self.bank, fold["split"], fold["index"])
            keys = {spec.source.state_key(r) for r in self.targets}
            test = [r for r in test if spec.source.state_key(r) in keys]
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "fits": {m: fleet_model(train, m) for m in spec.value.METHODS},
                "training_mean_states": 30-len(test), "evaluation_mean_states": len(test),
                "predictions": {v: {l: {p: predictions(test) for p in spec.source.source.value.POLICIES}
                    for l in ("original", "mean8")} for v in spec.value.VARIANTS},
                "critic_updates": 200, "actor_updates": 0, "new_native_steps": 0,
                "fit_wall_s": {m: 1. for m in spec.value.METHODS}}
        result = summarize(cells, self.registered)
        self.assertEqual((result["new_fits"], result["reused_fits"], result["critic_updates"]), (34, 17, 3400))
        self.assertEqual((result["evaluation_states"], result["evaluation_prefix_scene_groups"]), (30, 10))
        self.assertEqual(len(result["variants"]), 8)
        self.assertFalse(result["expected_target_development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        cells.pop(next(iter(cells)))
        with self.assertRaises(ValueError): summarize(cells, self.registered)

    def test_scheduler_only_syncs_compact_receipts_and_unbinds_all_six_nodes(self):
        tasks = [task_specification("unit_fleet", fold=f) for f in self.registered["folds"]]
        self.assertEqual(len({t["signature"] for t in tasks}), 17)
        for task in [*tasks, task_specification("unit_fleet")]:
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
