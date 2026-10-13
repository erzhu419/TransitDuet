import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_mean_cost_stage215 as spec
from scripts.submit_native_transit_mean_cost_stage215_scheduleurm import task_specification
from tests import test_native_transit_mean_wait as wait_fixture
from tests.test_native_transit_nonlinear_wait import prediction_rows


class MeanCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wait_fixture.MeanWaitTest.setUpClass()
        fixture = wait_fixture.MeanWaitTest
        cls.bank, cls.masses, cls.keys = fixture.bank, fixture.masses, fixture.keys
        cls.registered = spec.manifest(fixture.registered)
        cls.targets = [r for r in cls.bank["training_data"] if spec.state_key(r) in cls.keys]

    def prior(self, fold):
        train, test, _ = spec.base.coverage.split_rows(self.bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in self.keys]
        model = spec.base.cost.fit_value(train, sample_weights=spec.previous.source.feasibility.sample_weights(train, self.masses))
        frozen = {spec.state_key(r): {**p, **spec.base.cost.predict(r, model)} for r, p in zip(test, prediction_rows(test))}
        predictions = {}
        for name, variant in spec.WAIT_HEADS.items():
            waits = [{"predicted_harm_mass_share": [0. if i == r["original_choice"] else .5 for i in range(6)],
                "predicted_feasibility_score": [0. if name == "glm" or i == r["original_choice"] else .6 for i in range(6)]} for r in test]
            predictions[variant] = spec.evaluate(test, frozen, waits)
        return {"frozen_cost_fit": {k: v for k, v in model.items() if k != "coefficients"}, "predictions": predictions}

    def test_same_labels_reproduce_costs_scales_weights_and_wait_masks(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        before = copy.deepcopy(prior)
        cell, models = spec.fit_fold(self.bank, self.targets, self.bank["training_data"], self.masses, fold, prior)
        spec.qualify(cell, fold, self.registered)
        self.assertEqual(models["old_cost"], models["mean_cost"])
        self.assertEqual(prior, before)
        for name, variant in spec.WAIT_HEADS.items():
            self.assertEqual(cell["predictions"][f"old_cost_{name}"], prior["predictions"][variant])
            self.assertEqual(cell["predictions"][f"mean_cost_{name}"], prior["predictions"][variant])
        self.assertEqual(cell["closed_form_solves"], 2)
        self.assertEqual(models["mean_cost"]["sample_weight_sum"], 2280-48)

    def test_outer_mean_labels_cannot_change_models_or_actions(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        changed = copy.deepcopy(self.bank["training_data"])
        for row in changed:
            if spec.base.credit.group_key(row, fold["split"]) == fold["held_out"]:
                for label in row["labels"]:
                    label["fleet_cost_delta"] += 1000.
                    label["nonfleet_cost_delta"] -= 1000.
        cell, models = spec.fit_fold(self.bank, self.targets, self.bank["training_data"], self.masses, fold, prior)
        other, refit = spec.fit_fold(self.bank, self.targets, changed, self.masses, fold, prior)
        self.assertEqual(models, refit)
        self.assertEqual(cell["predictions"], other["predictions"])

    def test_training_cost_labels_change_only_new_heads_not_wait_or_controls(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        changed = copy.deepcopy(self.bank["training_data"])
        for row in changed:
            if spec.base.credit.group_key(row, fold["split"]) != fold["held_out"]:
                for label in row["labels"]:
                    if label["candidate"] != row["original_choice"]:
                        label["fleet_cost_delta"] *= -2.
                        label["actual_cost_delta"] = label["fleet_cost_delta"]+label["nonfleet_cost_delta"]
        cell, models = spec.fit_fold(self.bank, self.targets, changed, self.masses, fold, prior)
        spec.qualify(cell, fold, self.registered)
        self.assertNotEqual(models["mean_cost"]["coefficients"], models["old_cost"]["coefficients"])
        for name, variant in spec.WAIT_HEADS.items():
            self.assertEqual(cell["predictions"][f"old_cost_{name}"], prior["predictions"][variant])
        for field in ("target_scale", "sample_weight_sum"):
            bad = copy.deepcopy(cell)
            bad["fits"]["mean_cost"][field] = [99.]*3 if field == "target_scale" else 99999.
            with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)
        bad = copy.deepcopy(cell)
        bad["predictions"][spec.PRIMARY][0]["predicted_feasibility_score"] = list(bad["predictions"][spec.PRIMARY][0]["predicted_feasibility_score"])
        bad["predictions"][spec.PRIMARY][0]["predicted_feasibility_score"][1] += .1
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)

    def test_mismatched_old_cost_predictor_stops_the_comparison(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        prior["predictions"][spec.previous.NEW_GLM][0]["predicted_heads"][1][0] += .1
        with self.assertRaisesRegex(ValueError, "reproduce"):
            spec.fit_fold(self.bank, self.targets, self.bank["training_data"], self.masses, fold, prior)

    def test_identity_does_not_pass_mean_or_disjoint_future_cost_gates(self):
        cells, panels = {}, {}
        for row in self.targets:
            labels = {l["candidate"]: l for l in row["labels"]}
            values = np.asarray([[labels[i][h] for h in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for i in range(6)])
            panels[spec.state_key(row)] = {"values": np.repeat(values[None], 8, axis=0), "original_choice": row["original_choice"]}
        for fold in self.registered["folds"]:
            rows = [r for r in self.targets if spec.base.credit.group_key(r, fold["split"]) == fold["held_out"]]
            cells[fold["split"], fold["index"]] = {"predictions": {v: prediction_rows(rows) for v in spec.VARIANTS}}
        with patch.object(spec, "qualify"):
            result = spec.summarize(cells, self.registered, panels)
        self.assertFalse(result["development_gate"] or result["native_deployment_eligible"])
        for variant in result["variants"].values():
            self.assertFalse(variant["gates"]["cost"])
            self.assertTrue(variant["gates"]["wait"])
            self.assertTrue(all(not gate["cost"] for gate in variant["half_gates"].values()))

    def test_scheduler_budget_unbound_small_result_only(self):
        for task in (task_specification("unit_mean_cost"), task_specification("unit_mean_cost", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
            self.assertIn("py310/bin/python", task["cmd"])
        self.assertEqual(spec.contract()["budget"], {"new_cost_fits": 47, "control_cost_refits": 47,
            "closed_form_solves": 94, "operational_wait_fits": 0, "critic_optimizer_updates": 0,
            "actor_updates": 0, "new_native_episodes": 0})


if __name__ == "__main__":
    unittest.main()
