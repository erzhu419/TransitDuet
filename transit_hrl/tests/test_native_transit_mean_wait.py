import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_native_transit_mean_wait_stage213 as spec
from scripts.submit_native_transit_mean_wait_stage213_scheduleurm import task_specification
from tests import test_native_transit_decision_wait as decision
from tests.test_native_transit_nonlinear_wait import prediction_rows


class MeanWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        decision.DecisionWaitTest.setUpClass()
        cls.bank = decision.DecisionWaitTest.bank
        p = spec.source.manifest(decision.DecisionWaitTest.registered)
        p = spec.previous.previous.previous.manifest(p)
        p = spec.previous.previous.manifest(p)
        cls.control_manifest = spec.previous.manifest(p)
        cls.registered = spec.manifest(spec.collection.manifest(cls.control_manifest))
        cls.keys = {spec.state_key(r) for r in cls.registered["mean_states"]}
        original = spec.source.build_masses(cls.bank["training_data"], {})
        for m in original:
            if spec.state_key(m) in cls.keys: m["draw_count"] = 8
        cls.original = {"protocol": spec.source.EXPERIMENT_PROTOCOL, "contract": spec.source.contract(), "targets": original}
        cls.prepared = {"protocol": spec.collection.EXPERIMENT_PROTOCOL, "contract": spec.collection.contract(),
            "targets": [copy.deepcopy(m) if spec.state_key(m) in cls.keys else {**copy.deepcopy(m), "draw_count": 8, "sample_weight": 1} for m in original],
            "new_mean_labels": [r for r in cls.bank["training_data"] if spec.state_key(r) not in cls.keys]}
        cls.masses = spec.checked_masses(cls.bank, cls.prepared, cls.original, cls.registered)

    def prior(self, fold):
        train, test, _ = spec.source.source.coverage.split_rows(self.bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in self.keys]
        groups = [g for g in self.registered["groups"] if spec.source.source.credit.group_key(g, fold["split"]) != fold["held_out"]]
        partition = spec.stopping.grouping.partition_groups(groups, {(r["root"], r["scene_seed"]) for r in train if spec.state_key(r) in self.keys})
        return {"frozen_cost_fit": {"training_groups": sorted([[g["root"], g["scene_seed"]] for g in groups])},
            "predictions": {v: prediction_rows(test) for v in spec.CONTROLS.values()},
            "selections": {v: {"partition": partition} for v in spec.CORRECTIONS.values()}}

    def test_precision_upgrade_keeps_weights_evaluation_and_folds_exact(self):
        self.assertEqual(sum(spec.source.feasibility.sample_weights(self.bank["training_data"], self.masses)), 2280)
        self.assertTrue(all(m["draw_count"] == 8 for m in self.masses.values()))
        for k in ("groups", "mean_states", "folds"):
            self.assertEqual(self.registered[k], self.control_manifest[k])
        bad = copy.deepcopy(self.prepared)
        new = next(m for m in bad["targets"] if spec.state_key(m) not in self.keys)
        del new["sample_weight"]
        with self.assertRaises(ValueError):
            spec.checked_masses(self.bank, bad, self.original, self.registered)
        bad = copy.deepcopy(self.prepared)
        next(m for m in bad["targets"] if spec.state_key(m) in self.keys)["positive_wait_min"][0] += 1
        with self.assertRaises(ValueError):
            spec.checked_masses(self.bank, bad, self.original, self.registered)

    def test_outer_labels_cannot_affect_fits_or_selections_and_controls_are_frozen(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        old = copy.deepcopy(prior)
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        changed = copy.deepcopy(self.masses)
        for r in targets:
            if spec.source.source.credit.group_key(r, fold["split"]) == fold["held_out"]:
                changed[spec.state_key(r)]["positive_wait_min"] = [999.]*6
        with patch.object(spec.stopping, "CHECKPOINTS", (0, 2)), \
                patch.object(spec.source.feasibility, "fit_wait", wraps=spec.source.feasibility.fit_wait) as glm:
            registered = {**self.registered, "contract": spec.contract()}
            cell, models = spec.fit_fold(self.bank, targets, self.masses, fold, prior, registered)
            spec.qualify(cell, fold, registered)
            other, refit = spec.fit_fold(self.bank, targets, changed, fold, prior, registered)
            self.assertEqual(glm.call_count, 10)  # One outer and four shared inner references per call.
            self.assertEqual(models, refit)
            self.assertEqual(cell["selections"], other["selections"])
            self.assertEqual(cell["predictions"], other["predictions"])
            for v, k in spec.CONTROLS.items(): self.assertEqual(cell["predictions"][v], prior["predictions"][k])
            for v in (spec.NEW_GLM, *spec.CORRECTIONS):
                self.assertEqual(cell["fits"][v]["training_mean_states"], cell["fits"][v]["training_states"])
                for row in cell["predictions"][v]:
                    base = next(r for r in cell["predictions"]["old_glm"] if spec.state_key(r) == spec.state_key(row))
                    self.assertEqual(row["predicted_heads"], base["predicted_heads"])
            bad = copy.deepcopy(cell)
            bad["selections"][spec.PRIMARY]["inner_fits"][0]["validation_mean_states"] += 6
            with self.assertRaises(ValueError): spec.qualify(bad, fold, registered)
            bad = copy.deepcopy(cell)
            bad["fits"][spec.NEW_GLM]["sample_weight_sum"] += 7
            with self.assertRaises(ValueError): spec.qualify(bad, fold, registered)
        self.assertEqual(prior, old)

    def test_inner_reference_excludes_its_own_validation_targets(self):
        fold = self.registered["folds"][0]
        prior = self.prior(fold)
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        held = set(map(tuple, prior["selections"]["bounded_only"]["partition"][0]))
        changed = copy.deepcopy(self.masses)
        for row in self.bank["training_data"]:
            if (row["root"], row["scene_seed"]) in held:
                m = changed[spec.state_key(row)]
                m["positive_wait_min"] = [0. if i == row["original_choice"] else 100. for i in range(6)]
        fit, captured = spec.source.feasibility.fit_wait, []
        def record(*args, **kwargs):
            result = fit(*args, **kwargs)
            captured.append(result)
            return result
        with patch.object(spec.stopping, "CHECKPOINTS", (0,)), patch.object(spec.source.feasibility, "fit_wait", side_effect=record):
            spec.fit_fold(self.bank, targets, self.masses, fold, prior, self.registered)
            spec.fit_fold(self.bank, targets, changed, fold, prior, self.registered)
        self.assertEqual(captured[1], captured[6])
        self.assertNotEqual(captured[0]["coefficients"], captured[5]["coefficients"])

    def test_identity_choices_cannot_pass_cost_gate_in_any_split_or_future_half(self):
        rows = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        panels = {}
        for row in rows:
            labels = {l["candidate"]: l for l in row["labels"]}
            values = np.asarray([[labels[i][f] for f in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for i in range(6)])
            panels[spec.state_key(row)] = {"values": np.repeat(values[None], 8, axis=0), "original_choice": row["original_choice"]}
        cells = {}
        for fold in self.registered["folds"]:
            held = [r for r in rows if spec.source.source.credit.group_key(r, fold["split"]) == fold["held_out"]]
            cells[fold["split"], fold["index"]] = {"predictions": {v: prediction_rows(held) for v in spec.VARIANTS}}
        empty = {scope: {v: {s: [] for s in spec.SPLITS} for v in spec.VARIANTS} for scope in ("training", "held")}
        result = spec.variant_summary(cells, self.registered, empty, panels, spec.PRIMARY)
        self.assertFalse(result["gates"]["cost"])
        self.assertTrue(result["gates"]["wait"])
        self.assertTrue(all(not g["cost"] for g in result["half_gates"].values()))

    def test_scheduler_budget_small_results_unbound_and_python310(self):
        for task in (task_specification("unit_means"), task_specification("unit_means", {"split": "scene", "index": 0})):
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertIsNone(task["require_node"])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual((spec.contract()["budget"]["new_outer_glm_fits"], spec.contract()["budget"]["new_inner_glm_fits"]), (47, 188))
        for path in (spec.ROOT/"scripts/run_native_transit_mean_wait_stage213.py",
                spec.ROOT/"scripts/submit_native_transit_mean_wait_stage213_scheduleurm.py"):
            ast.parse(path.read_text(), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
