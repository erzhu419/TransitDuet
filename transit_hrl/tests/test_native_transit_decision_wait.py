import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_decision_wait_stage204 as spec
from scripts.submit_native_transit_decision_wait_stage204_scheduleurm import task_specification
from tests.test_native_transit_coverage_value import bank_fixture, fake_model, fake_predictions
from tests.test_native_transit_sequence_value import fixture


class DecisionWaitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"]
        cls.bank = bank_fixture()[0]
        old = spec.source.source.manifest(spec.source.source.source.manifest(
            spec.source.source.source.source.manifest(spec.mean.manifest(cls.bank["manifest"]))))
        cls.previous = spec.source.manifest(old)
        cls.registered = spec.manifest(cls.previous)
        cls.keys = {spec.state_key(r) for r in cls.registered["mean_states"]}

    def prior_for(self, fold):
        train, test, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
        test = [r for r in test if spec.state_key(r) in self.keys]
        return {"frozen_cost_fit": {**fake_model(train, "tail_summary"), "parameter_count": 294, "steps": 0},
            "predictions": {v: fake_predictions(test) for v in spec.source.VARIANTS}}

    def test_decision_criterion_prioritizes_wait_then_cost_instead_of_MSE(self):
        scores = [{"interaction_ridge": p, "wait_violation_min": w, "worst_root_equal_regime_cost": c,
            "equal_root_cost": c, "worst_regime_pair_mse_min2": mse}
            for p, w, c, mse in [(.1, .01, -.5, 0.), (1., 0., -.1, 1.), (10., 0., -.2, 2.), (100., 0., -.2, 2.)]]
        self.assertEqual(spec.choose_penalty(scores), 100.)
        for score in scores: score["wait_violation_min"] = .01
        scores[1]["wait_violation_min"] = .001
        self.assertEqual(spec.choose_penalty(scores), 1.)

    def test_cached_inner_cost_predictions_match_original_ridge_in_physical_units(self):
        train, test = self.rows[:12], self.rows[12:18]
        model = spec.cost.fit_value(train, sample_weights=[8]*6+[1]*6)
        raw = np.stack([spec.wait.features(r, "conditional_wait") for r in test])
        heads = spec.cost_predictions(raw, test, model)
        for row, current in zip(test, heads):
            np.testing.assert_allclose(current, spec.cost.predict(row, model)["predicted_heads"], atol=1e-12, rtol=0)
            np.testing.assert_array_equal(current[row["original_choice"]], [0., 0.])

    def test_decision_metrics_use_predictions_for_selection_and_selected_labels_for_outcomes(self):
        row = {"root": 397, "original_choice": 0, "labels": [
            {"candidate": 0, "actual_cost_delta": 0., "wait_delta_min": 0.},
            {"candidate": 1, "actual_cost_delta": .2, "wait_delta_min": .3},
            {"candidate": 2, "actual_cost_delta": -.1, "wait_delta_min": -.5}]}
        heads = np.asarray([[[0., 0.], [-2., 0.], [-1., 0.]]])
        result = spec.decision_metrics([row], heads, np.asarray([[0., -.01, -.1]]))
        self.assertEqual(result["by_root"]["397"]["mean_actual_cost_delta"], .2)
        self.assertEqual(result["by_root"]["397"]["mean_wait_delta_min"], .3)
        result = spec.decision_metrics([row], heads, np.asarray([[0., .01, -.1]]))
        self.assertEqual(result["overall"]["mean_actual_cost_delta"], -.1)
        self.assertEqual(result["overall"]["mean_wait_delta_min"], -.5)

    def test_own_inner_heldout_labels_do_not_affect_either_cost_or_wait_models(self):
        rows, changed = self.rows, copy.deepcopy(self.rows)
        held = sorted({r["scenario"] for r in rows})[0]
        for row in changed:
            if row["scenario"] == held:
                for label in row["labels"]:
                    if label["candidate"] != row["original_choice"]:
                        for field in ("fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min"): label[field] += 100
                        label["actual_cost_delta"] = label["fleet_cost_delta"]+label["nonfleet_cost_delta"]
        fit_cost, fit_wait, costs, waits = spec.cost.fit_value, spec.wait.fit_wait_path, [], []
        def capture_cost(*args, **kwargs):
            model = fit_cost(*args, **kwargs)
            costs.append(model)
            return model
        def capture_wait(*args, **kwargs):
            models = fit_wait(*args, **kwargs)
            waits.append(models)
            return models
        keys = {spec.state_key(r) for r in rows}
        with patch.object(spec.cost, "fit_value", side_effect=capture_cost), patch.object(spec.wait, "fit_wait_path", side_effect=capture_wait):
            a = spec.select_within_training(rows, keys)
            b = spec.select_within_training(changed, keys)
        self.assertEqual(costs[0], costs[5])
        self.assertEqual(waits[0], waits[5])
        self.assertEqual(a["inner"][0]["cost_fit"], b["inner"][0]["cost_fit"])
        self.assertEqual(a["inner"][0]["normalization"], b["inner"][0]["normalization"])

    def test_outer_heldout_labels_do_not_select_penalty_or_change_final_fit_and_costs(self):
        fold = self.registered["folds"][0]
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        prior = self.prior_for(fold)
        before = copy.deepcopy(prior)
        cell, model = spec.fit_fold(self.bank, targets, fold, prior)
        spec.qualify(cell, fold, self.registered)
        changed = copy.deepcopy(targets)
        for row in changed:
            if list(spec.paired.scene_group(row)) == fold["held_out"]:
                for label in row["labels"]:
                    if label["candidate"] != row["original_choice"]:
                        for field in ("fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min"): label[field] += 100
                        label["actual_cost_delta"] = label["fleet_cost_delta"]+label["nonfleet_cost_delta"]
        other, refit = spec.fit_fold(self.bank, changed, fold, prior)
        self.assertEqual(model, refit)
        self.assertEqual(cell["selection"], other["selection"])
        self.assertEqual(prior, before)
        for variant in spec.source.VARIANTS: self.assertEqual(cell["predictions"][variant], prior["predictions"][variant])
        bad = copy.deepcopy(cell)
        inner = bad["selection"]["inner"][0]
        inner["cost_fit"]["training_groups"].append(inner["evaluation_groups"][0])
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)
        bad = copy.deepcopy(cell)
        bad["predictions"]["decision_selected_wait"][0]["predicted_heads"][0][0] += 1
        with self.assertRaises(ValueError): spec.qualify(bad, fold, self.registered)

    def test_all47_folds_preserve_root_regime_populations_and_exact1197_solve_budget(self):
        targets = [r for r in self.bank["training_data"] if spec.state_key(r) in self.keys]
        cells = {}
        for fold in self.registered["folds"]:
            train, test, _ = spec.coverage.split_rows(self.bank, fold["split"], fold["index"])
            test = [r for r in test if spec.state_key(r) in self.keys]
            inner = []
            for regime in sorted({r["scenario"] for r in train}):
                fitting = [r for r in train if r["scenario"] != regime]
                evaluation = [r for r in train if r["scenario"] == regime and spec.state_key(r) in self.keys]
                membership = sorted(map(list, {spec.paired.scene_group(r) for r in fitting}))
                def metrics(rs):
                    return {"states": len(rs), "changed_choices": 0, "mean_actual_cost_delta": 0., "mean_wait_delta_min": 0.}
                decision = {"overall": metrics(evaluation), "by_root": {str(r): metrics([s for s in evaluation if s["root"] == r])
                    for r in sorted({s["root"] for s in evaluation})}}
                inner.append({"held_regime": regime, "training_groups": membership,
                    "evaluation_groups": sorted(map(list, {spec.paired.scene_group(r) for r in evaluation})),
                    "training_states": len(fitting), "training_mean_states": sum(spec.state_key(r) in self.keys for r in fitting),
                    "evaluation_mean_states": len(evaluation), "new_closed_form_fits": 5,
                    "cost_fit": {"training_groups": membership, "training_states": len(fitting), "parameter_count": 294,
                        "steps": 0, "ridge": .1, "sample_weight_sum": len(fitting)+7*sum(spec.state_key(r) in self.keys for r in fitting)},
                    "decisions": {str(p): decision for p in spec.INTERACTION_RIDGES},
                    "pair_mse_min2": {str(p): 0. for p in spec.INTERACTION_RIDGES}})
            prior = self.prior_for(fold)
            model = {"training_groups": sorted(map(list, {spec.paired.scene_group(r) for r in train})),
                "training_states": len(train), "sample_weight_sum": len(train)+7*(240-len(test)), "parameter_count": 198,
                "steps": 0, "ridge": .1, "interaction_ridge": 100.}
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "software_qualified": True, "fold": fold, "fit": model, "frozen_cost_fit": prior["frozen_cost_fit"],
                "selection": {"inner": inner, "scores": spec.combine_scores(inner), "selected_ridge": 100., "inner_wait_feasible": True},
                "predictions": {**prior["predictions"], "decision_selected_wait": fake_predictions(test)},
                "new_closed_form_fits": 5*len(inner)+1, "critic_optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0}
        result = spec.summarize(cells, self.registered, targets)
        self.assertEqual(result["new_closed_form_fits"], 1197)
        self.assertEqual(result["inner_wait_feasible_folds"], {"scene": 40, "regime": 5, "lower_root": 2})
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])

    def test_scheduler_is_unbound_and_syncs_no_raw_models_or_bank(self):
        for task in (task_specification("unit_decision_wait"), task_specification("unit_decision_wait", {"split": "scene", "index": 0})):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])


if __name__ == "__main__":
    unittest.main()
