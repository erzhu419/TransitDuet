import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_nested_wait_stage203 as spec
from scripts.submit_native_transit_nested_wait_stage203_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_sequence_value import fixture


class NestedWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"]
        cls.bank = bank_fixture()[0]
        cls.previous = spec.source.manifest(spec.source.source.manifest(spec.source.source.source.manifest(
            spec.mean.manifest(cls.bank["manifest"]))))
        cls.registered = spec.manifest(cls.previous)
        cls.keys = {spec.state_key(r) for r in cls.registered["mean_states"]}

    def prior_for(self, fold):
        train, test, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in self.keys]
        return {"frozen_cost_fit": {**fake_model(train, "tail_summary"), "parameter_count": 294, "steps": 0},
            "fits": {}, "predictions": {v: fake_predictions(test) for v in spec.source.VARIANTS}}

    def test_penalty_path_reuses_features_and_only_changes_last96_diagonal_entries(self):
        rows, weights = self.rows[:12], [8]*6+[1]*6
        with patch.object(spec.wait, "features", wraps=spec.wait.features) as extract:
            models = spec.wait.fit_wait_path(rows, "conditional_wait", sample_weights=weights,
                interaction_ridges=spec.INTERACTION_RIDGES)
        self.assertEqual(extract.call_count, len(rows))
        model = models[1]
        raw = np.stack([spec.wait.features(r, "conditional_wait") for r in rows])
        x = (raw-model["center"])/model["scale"]
        targets = np.asarray([[l["wait_delta_min"] for l in r["labels"]] for r in rows])
        a, b = np.triu_indices(6, 1)
        z, y = (x[:, a]-x[:, b]).reshape(-1, 198), (targets[:, a]-targets[:, b]).reshape(-1)/model["target_scale_min"]
        w = np.repeat(weights, 15)
        expected = np.linalg.solve(z.T@(w[:, None]*z)/w.sum()+np.diag([.1]*102+[1.]*96), z.T@(w*y)/w.sum())
        np.testing.assert_allclose(model["coefficients"], expected, atol=1e-12, rtol=0)
        default = spec.wait.fit_wait(rows, "conditional_wait", sample_weights=weights)
        self.assertEqual(models[0], default)
        norms = [np.linalg.norm(m["coefficients"][102:]) for m in models]
        self.assertTrue(all(a >= b-1e-12 for a, b in zip(norms, norms[1:])))

    def test_inner_batched_error_matches_single_prediction_physical_pair_mse(self):
        train, test = self.rows[:12], self.rows[12:18]
        model = spec.wait.fit_wait(train, "conditional_wait", sample_weights=[1]*12, interaction_ridge=10.)
        raw = np.stack([spec.wait.features(r, "conditional_wait") for r in test])
        prediction = np.asarray([spec.wait.predict_wait(r, model) for r in test])
        target = np.asarray([[l["wait_delta_min"] for l in r["labels"]] for r in test])
        a, b = np.triu_indices(6, 1)
        expected = np.mean((prediction[:, a]-prediction[:, b]-(target[:, a]-target[:, b]))**2)
        self.assertAlmostEqual(spec.heldout_error(raw, test, model), expected, places=12)

    def test_selection_uses_worst_regime_then_equal_mean_then_stronger_exact_tie(self):
        scores = [{"interaction_ridge": p, "worst_regime_pair_mse_min2": worst,
            "equal_regime_mean_pair_mse_min2": average} for p, worst, average in [(0.1, 2., .1), (1., 1., .8), (10., 1., .9), (100., 1., .8)]]
        self.assertEqual(spec.choose_penalty(scores), 100.)
        for score in scores: score.update(worst_regime_pair_mse_min2=0., equal_regime_mean_pair_mse_min2=0.)
        self.assertEqual(spec.choose_penalty(scores), 100.)

    def test_own_inner_heldout_labels_do_not_change_its_models_or_normalization(self):
        rows = self.rows
        held = sorted({r["scenario"] for r in rows})[0]
        changed = copy.deepcopy(rows)
        for row in changed:
            if row["scenario"] == held:
                for label in row["labels"]: label["wait_delta_min"] += label["candidate"]*100
        actual, captured = spec.wait.fit_wait_path, []
        def capture(*args, **kwargs):
            result = actual(*args, **kwargs)
            captured.append(result)
            return result
        keys = {spec.state_key(r) for r in rows}
        with patch.object(spec.wait, "fit_wait_path", side_effect=capture):
            a = spec.select_within_training(rows, keys)
            b = spec.select_within_training(changed, keys)
        self.assertEqual(captured[0], captured[5])
        self.assertEqual(a["inner"][0]["normalization"], b["inner"][0]["normalization"])
        held_groups = {spec.paired.scene_group(r) for r in rows if r["scenario"] == held}
        self.assertFalse({tuple(g) for g in captured[0][0]["training_groups"]} & held_groups)

    def test_outer_heldout_labels_do_not_choose_penalty_or_change_final_model(self):
        fold = self.registered["folds"][0]
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        prior = self.prior_for(fold)
        before = copy.deepcopy(prior)
        cell, model = spec.fit_fold(self.bank, targets, fold, prior)
        spec.qualify(cell, fold, self.registered)
        changed = copy.deepcopy(targets)
        for row in changed:
            if list(spec.paired.scene_group(row)) == fold["held_out"]:
                for label in row["labels"]: label["wait_delta_min"] += label["candidate"]*100
        other, refit = spec.fit_fold(self.bank, changed, fold, prior)
        self.assertEqual(model, refit)
        self.assertEqual(cell["selection"], other["selection"])
        self.assertEqual(before, prior)
        for variant in spec.source.VARIANTS: self.assertEqual(cell["predictions"][variant], prior["predictions"][variant])
        bad = copy.deepcopy(cell)
        bad["selection"]["inner"][0]["training_groups"].append(bad["selection"]["inner"][0]["evaluation_groups"][0])
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)
        bad = copy.deepcopy(cell)
        bad["predictions"]["regime_selected_wait"][0]["predicted_heads"][0][0] += 1
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)

    def test_all47_outer_folds_have_disjoint_inner_groups_and_exact967_solve_budget(self):
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
            test = [r for r in test if spec.state_key(r) in self.keys]
            inner = []
            for regime in sorted({r["scenario"] for r in train}):
                fitting = [r for r in train if r["scenario"] != regime]
                evaluation = [r for r in train if r["scenario"] == regime and spec.state_key(r) in self.keys]
                inner.append({"held_regime": regime, "training_groups": sorted(map(list, {spec.paired.scene_group(r) for r in fitting})),
                    "evaluation_groups": sorted(map(list, {spec.paired.scene_group(r) for r in evaluation})),
                    "training_states": len(fitting), "training_mean_states": sum(spec.state_key(r) in self.keys for r in fitting),
                    "evaluation_mean_states": len(evaluation), "new_closed_form_fits": 4,
                    "pair_mse_min2": {str(p): 0. for p in spec.INTERACTION_RIDGES}})
            prior = self.prior_for(fold)
            model = {"training_groups": sorted(map(list, {spec.paired.scene_group(r) for r in train})),
                "training_states": len(train), "sample_weight_sum": len(train)+7*(240-len(test)),
                "parameter_count": 198, "steps": 0, "ridge": .1, "interaction_ridge": 100.}
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "fit": model, "frozen_cost_fit": prior["frozen_cost_fit"],
                "selection": {"inner": inner, "scores": spec.combine_scores(inner), "selected_ridge": 100.},
                "predictions": {**prior["predictions"], "regime_selected_wait": fake_predictions(test)},
                "new_closed_form_fits": 4*len(inner)+1, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0}
        result = spec.summarize(cells, self.registered, targets)
        self.assertEqual(result["new_closed_form_fits"], 967)
        self.assertEqual(result["selected_penalties"], {"scene": {"100.0": 40}, "regime": {"100.0": 5}, "lower_root": {"100.0": 2}})
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_scheduler_keeps_fits_unbound_and_raw_data_remote(self):
        for task in (task_specification("unit_nested_wait"), task_specification("unit_nested_wait", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
