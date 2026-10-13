import ast
import copy
import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_native_transit_wait_feasibility_stage206 as spec
from scripts.submit_native_transit_wait_feasibility_stage206_scheduleurm import task_specification
from tests.test_native_transit_decision_wait import DecisionWaitTest
from tests.test_native_transit_coverage_value import fake_predictions
from tests.test_native_transit_sequence_value import fixture


class WaitFeasibilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = fixture()[0]["training_data"][:12]

    def test_draw_masses_preserve_rare_large_harm_instead_of_majority_sign(self):
        positive, negative = spec.feasibility.wait_masses(np.array([[-.1], [-.1], [-.1], [1.]]))
        self.assertAlmostEqual(positive[0], .25)
        self.assertAlmostEqual(negative[0], .075)
        self.assertGreater((positive-negative)[0], 0.)
        doubled = np.tile(np.array([[-.1], [-.1], [-.1], [1.]]), (2, 1))
        for a, b in zip((positive, negative), spec.feasibility.wait_masses(doubled)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)

    def test_analytic_gradient_and_false_safe_cost_orientation(self):
        rng = np.random.default_rng(206)
        x, beta = rng.normal(size=(7, 4)), rng.normal(size=4)
        positive, negative, weights = rng.uniform(size=(3, 7))
        loss, gradient = spec.feasibility.objective(beta, x, positive, negative, weights, 2.)
        for i in range(4):
            step = np.eye(4)[i]*1e-6
            a = spec.feasibility.objective(beta+step, x, positive, negative, weights, 2.)[0]
            b = spec.feasibility.objective(beta-step, x, positive, negative, weights, 2.)[0]
            self.assertAlmostEqual(gradient[i], (a-b)/2e-6, places=6)
        self.assertGreater(loss, 0.)
        _, symmetric = spec.feasibility.objective(np.zeros(1), np.ones((1, 1)), np.ones(1), np.ones(1), np.ones(1), 1.)
        _, asymmetric = spec.feasibility.objective(np.zeros(1), np.ones((1, 1)), np.ones(1), np.ones(1), np.ones(1), 2.)
        self.assertEqual(symmetric[0], 0.)
        self.assertLess(asymmetric[0], 0.)

    def test_training_scales_draw_counts_and_causal_identity_inference(self):
        targets = spec.build_masses(self.rows, {})
        for row in targets[:6]: row["draw_count"] = 8
        masses = {spec.state_key(r): r for r in targets}
        model = spec.feasibility.fit_wait(self.rows, masses, "asymmetric_feasibility")
        self.assertEqual((model["parameter_count"], model["sample_weight_sum"], model["training_mean_states"]), (199, 54., 6))
        raw = np.stack([spec.source.wait.features(r, "conditional_wait") for r in self.rows])
        np.testing.assert_array_equal(model["center"], raw.mean(axis=(0, 1)))
        weights = [8]*6+[1]*6
        absolute = [sum(np.asarray(r["positive_wait_min"])+r["negative_wait_min"])/5 for r in targets]
        self.assertAlmostEqual(model["target_scale_min"], max(np.average(absolute, weights=weights), 1e-6))
        row, changed = self.rows[0], copy.deepcopy(self.rows[0])
        changed.update(root=-1, scenario="unused", scene_seed=-1, labels=[])
        for bus in changed["occupancy_context"]["active_buses"]: bus["actual_remaining_s"] = 99999.
        prediction = spec.feasibility.predict(row, model)
        self.assertEqual(prediction, spec.feasibility.predict(changed, model))
        self.assertEqual(prediction["predicted_feasibility_score"][row["original_choice"]], 0.)
        self.assertEqual(prediction["predicted_harm_mass_share"][row["original_choice"]], 0.)

    def test_count_weights_equivalent_to_replicating_supported_single_draws(self):
        targets = spec.build_masses(self.rows, {})
        masses = {spec.state_key(r): r for r in targets}
        raw = np.zeros((len(self.rows), 6, 198))
        with patch.object(spec.feasibility.wait, "features", side_effect=lambda r, v: raw[0]):
            symmetric = spec.feasibility.fit_wait(self.rows, masses, "symmetric_feasibility")
            asymmetric = spec.feasibility.fit_wait(self.rows, masses, "asymmetric_feasibility")
        self.assertGreater(asymmetric["coefficients"][-1], symmetric["coefficients"][-1])
        for target in targets: target["draw_count"] = 8
        with patch.object(spec.feasibility.wait, "features", side_effect=lambda r, v: raw[0]):
            repeated = spec.feasibility.fit_wait(self.rows, masses, "symmetric_feasibility")
        np.testing.assert_allclose(symmetric["coefficients"], repeated["coefficients"], rtol=0, atol=1e-12)

    def test_build_masses_joins_draws_and_rejects_nonzero_identity(self):
        row = self.rows[0]
        values = np.zeros((8, 6, 4))
        alternative = (row["original_choice"]+1)%6
        values[:, alternative, 3] = [1., -.5]*4
        panel = {"original_choice": row["original_choice"], "values": values}
        targets = spec.build_masses(self.rows, {spec.state_key(row): panel})
        self.assertEqual(targets[0]["draw_count"], 8)
        self.assertEqual((targets[0]["positive_wait_min"][alternative], targets[0]["negative_wait_min"][alternative]), (.5, .25))
        self.assertEqual(targets[1]["draw_count"], 1)
        panel["values"][:, row["original_choice"], 3] = .1
        with self.assertRaises(ValueError): spec.build_masses(self.rows, {spec.state_key(row): panel})

    def test_frozen_cost_choice_threshold_and_identity_ties(self):
        prediction = {"predicted_heads": [[0., 0.], [-2., 0.], [-1., 0.]], "predicted_feasibility_score": [0., .1, -.1]}
        self.assertEqual(spec.choose(prediction, 0), 2)
        prediction["predicted_heads"] = [[0., 0.]]*3
        self.assertEqual(spec.choose(prediction, 0), 0)

    def test_worker_excludes_own_heldout_wait_masses_and_keeps_exact_cost_heads(self):
        DecisionWaitTest.setUpClass()
        bank, previous = DecisionWaitTest.bank, DecisionWaitTest.registered
        registered = spec.manifest(previous)
        fold = registered["folds"][0]
        train, test, _ = spec.source.coverage.split_rows(bank, fold["split"], fold["index"])
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        targets = [r for r in bank["training_data"] if spec.state_key(r) in keys]
        test = [r for r in test if spec.state_key(r) in keys]
        prepared = spec.build_masses(bank["training_data"], {})
        for r in prepared:
            if spec.state_key(r) in keys: r["draw_count"] = 8
        masses = {spec.state_key(r): r for r in prepared}
        prior = {"frozen_cost_fit": {"training_groups": sorted(map(list, {spec.source.paired.scene_group(r) for r in train}))},
            "predictions": {spec.source.contract()["primary"]: fake_predictions(test)}}
        before = copy.deepcopy(prior)
        cell, models = spec.fit_fold(bank, targets, masses, fold, prior)
        spec.qualify(cell, fold, registered)
        changed = copy.deepcopy(masses)
        for row in targets:
            if list(spec.source.paired.scene_group(row)) == fold["held_out"]:
                changed[spec.state_key(row)]["positive_wait_min"] = [999.]*6
        other, refit = spec.fit_fold(bank, targets, changed, fold, prior)
        self.assertEqual(models, refit)
        self.assertEqual(cell["predictions"], other["predictions"])
        self.assertEqual(prior, before)
        for variant in spec.feasibility.FALSE_SAFE_COSTS:
            for old, new in zip(prior["predictions"][spec.source.contract()["primary"]], cell["predictions"][variant]):
                self.assertEqual(old["predicted_heads"], new["predicted_heads"])
                self.assertNotIn("predicted_wait_delta_min", new)
        bad = copy.deepcopy(cell)
        bad["predictions"]["asymmetric_feasibility"][0]["predicted_heads"][0][0] = 99.
        with self.assertRaises(ValueError): spec.qualify(bad, fold, registered)

    def test_scheduler_server_only_unbound_scoped_and_python310(self):
        for task in (task_specification("unit_feasibility", operation="prepare"), task_specification("unit_feasibility", {"split": "scene", "index": 0}),
                task_specification("unit_feasibility", operation="report")):
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertNotIn("_raw", task["result_dir"])
        for path in (spec.ROOT/"freq_hrl/domains/transit/native_wait_feasibility.py",
                spec.ROOT/"scripts/run_native_transit_wait_feasibility_stage206.py",
                spec.ROOT/"scripts/submit_native_transit_wait_feasibility_stage206_scheduleurm.py"):
            ast.parse(path.read_text(), feature_version=(3, 10))
        self.assertEqual(spec.contract()["budget"], dict(new_wait_fits=94, new_cost_fits=0, actor_updates=0, new_native_episodes=0))

    def test_all47_folds_keep240_states_and_identity_cannot_pass_cost_or_half_gates(self):
        DecisionWaitTest.setUpClass()
        bank, previous = DecisionWaitTest.bank, DecisionWaitTest.registered
        registered = spec.manifest(previous)
        keys = {spec.state_key(r) for r in registered["mean_states"]}
        panels = {spec.state_key(r): {**r, "values": np.zeros((8, 6, 4))}
            for r in bank["training_data"] if spec.state_key(r) in keys}
        cells = {}
        for fold in registered["folds"]:
            train, test, _ = spec.source.coverage.split_rows(bank, fold["split"], fold["index"])
            test = [r for r in test if spec.state_key(r) in keys]
            membership = sorted(map(list, {spec.source.paired.scene_group(r) for r in train}))
            fits = {v: {"training_groups": membership, "training_states": len(train), "training_mean_states": 240-len(test),
                "sample_weight_sum": len(train)+7*(240-len(test)), "parameter_count": 199, "false_safe_cost": cost,
                "ridge": .1, "max_iterations": 200, "solver_success": True, "optimizer_iterations": 0}
                for v, cost in spec.feasibility.FALSE_SAFE_COSTS.items()}
            predictions = [{**r, "predicted_feasibility_score": [0.]*6,
                "predicted_harm_mass_share": [0. if i == r["original_choice"] else .5 for i in range(6)]} for r in fake_predictions(test)]
            cells[fold["split"], fold["index"]] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "software_qualified": True,
                "fold": fold, "fits": fits, "predictions": {v: predictions for v in spec.VARIANTS}, "new_wait_fits": 2,
                "new_cost_fits": 0, "actor_updates": 0, "new_native_episodes": 0, "wait_optimizer_iterations": 0,
                "frozen_cost_fit": {"training_groups": membership}}
        result = spec.summarize(cells, registered, panels)
        self.assertEqual(result["new_wait_fits"], 94)
        for variant in result["variants"].values():
            self.assertFalse(variant["gates"]["cost"])
            for validation in variant["validation"].values(): self.assertEqual(validation["overall"]["states"], 240)
            for half in variant["future_halves"].values(): self.assertFalse(half["gates"]["LL"]["cost"])
        self.assertFalse(result["development_gate"] or result["third_scene_screen_eligible"] or result["native_deployment_eligible"])
        with self.assertRaises(ValueError): spec.summarize({}, registered, panels)


if __name__ == "__main__":
    unittest.main()
