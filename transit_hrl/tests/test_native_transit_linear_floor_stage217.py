import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_native_transit_linear_floor_stage217 as spec
from scripts.submit_native_transit_linear_floor_stage217_scheduleurm import task_specification
from tests import test_native_transit_mean_wait as fixture


def ridge_reference(raw, targets, weights):
    left, right = np.triu_indices(6, 1)
    scale = np.asarray([2., 3., 4.])
    z = (raw[:, left]-raw[:, right]).reshape(-1, raw.shape[-1])
    y = ((targets[:, left]-targets[:, right])/scale).reshape(-1, 3)
    w = np.repeat(weights, 15)
    w = w/w.sum()
    coefficients = np.linalg.solve(z.T@(w[:, None]*z)+.1*np.eye(raw.shape[-1]), z.T@(w[:, None]*y))
    return {"center": [0.]*raw.shape[-1], "scale": [1.]*raw.shape[-1], "target_scale": scale.tolist(),
        "coefficients": coefficients.tolist(), "ridge": .1,
        "final_pair_loss": float(np.sum(w[:, None]*(z@coefficients-y)**2)/3)}


class LinearFloorTest(unittest.TestCase):
    def test_representable_values_have_zero_floor_and_ridge_is_the_excess(self):
        rng = np.random.default_rng(217)
        raw = rng.normal(size=(12, 6, 3))
        beta = rng.normal(size=(3, 3))
        targets = raw@beta
        weights = np.asarray([8]*4+[1]*8)
        reference = ridge_reference(raw, targets, weights)
        model, diagnostic = spec.project(raw, targets, weights, reference)
        np.testing.assert_allclose(np.asarray(model["coefficients"])*model["target_scale"], beta, atol=1e-12, rtol=0)
        self.assertEqual(diagnostic["numerical_rank"], 3)
        self.assertAlmostEqual(diagnostic["projection_cost_pair_mse"], 0.)
        self.assertGreater(diagnostic["ridge_excess_cost_pair_mse"], 0.)
        self.assertAlmostEqual(diagnostic["ridge_cost_pair_mse"], diagnostic["ridge_excess_cost_pair_mse"])

    def test_rank_deficiency_and_unrepresentable_values_have_a_nonzero_floor(self):
        rng = np.random.default_rng(218)
        x = rng.normal(size=(10, 6, 1))
        raw = np.concatenate((x, 2*x, np.zeros_like(x)), axis=-1)
        targets = rng.normal(size=(10, 6, 3))
        weights = np.asarray([8]*3+[1]*7)
        _, d = spec.project(raw, targets, weights, ridge_reference(raw, targets, weights))
        self.assertEqual(d["numerical_rank"], 1)
        self.assertGreater(d["projection_cost_pair_mse"], 0.)
        self.assertAlmostEqual(d["ridge_cost_pair_mse"], d["projection_cost_pair_mse"]+d["ridge_excess_cost_pair_mse"])

    def test_changed_weights_cannot_pass_the_archived_training_loss_check(self):
        rng = np.random.default_rng(219)
        raw, targets = rng.normal(size=(10, 6, 3)), rng.normal(size=(10, 6, 3))
        weights = np.asarray([8]*3+[1]*7)
        with self.assertRaisesRegex(ValueError, "data/weights"):
            spec.project(raw, targets, np.ones(10), ridge_reference(raw, targets, weights))

    def test_linear_witness_separates_feasible_infeasible_empty_and_solver_failure(self):
        d = np.asarray([[1., 0.], [-1., 1.]])
        result = spec.linear_witness(d, np.ones(2))
        self.assertTrue(result["feasible"])
        self.assertTrue(np.all(d@result["coefficients"] >= 1.-1e-7))
        impossible = spec.linear_witness(np.asarray([[1.], [-1.]]), np.ones(2))
        self.assertFalse(impossible["feasible"])
        self.assertEqual(impossible["status"], 0)
        self.assertGreater(impossible["dual_lower_bound"], 0.)
        self.assertGreater(impossible["margin_violation_mean"], 0.)
        empty = spec.linear_witness(np.empty((0, 2)), np.empty(0))
        self.assertTrue(empty["feasible"])
        self.assertEqual(empty["linear_programs"], 0)
        with patch.object(spec, "linprog", return_value=SimpleNamespace(status=4, message="numerical problem", nit=0)):
            failure = spec.linear_witness(d, np.ones(2))
            self.assertIsNone(failure["feasible"])
            self.assertIsNone(failure["coefficients"])
        zero = spec.linear_witness(np.zeros((2, 1)), np.ones(2))
        self.assertFalse(zero["feasible"])
        self.assertEqual(zero["linear_programs"], 0)

    def test_training_only_constraints_exclude_identity_and_half_disagreements(self):
        row = {"root": 397, "scene_seed": 7, "probe_index": 8, "original_choice": 0}
        values = np.zeros((8, 6, 4))
        values[:4, 1, 0], values[4:, 1, 0] = -1., 2.
        values[:, 1, 3], values[:4, 2, 3], values[4:, 2, 3] = -1., 1., -2.
        panels = {spec.state_key(row): {"values": values}}
        cost = np.zeros((1, 6, 2))
        cost[0, 1, 0] = 1.
        wait = np.zeros((1, 6, 3))
        wait[..., 2] = 1.
        a = spec.stable_constraints([row], panels, cost, wait, np.asarray([True]), "first_last")
        b = spec.stable_constraints([row], panels, cost, wait, np.asarray([True]), "even_odd")
        self.assertEqual(a["cost_order"][0].shape, (0, 2))
        self.assertEqual(len(a["wait_sign"][0]), 4)
        self.assertEqual(len(b["cost_order"][0]), 5)
        self.assertEqual(len(b["wait_sign"][0]), 5)
        none = spec.stable_constraints([row], panels, cost, wait, np.asarray([False]), "first_last")
        self.assertTrue(all(len(x) == 0 for x, _ in none.values()))

    def test_outer_labels_do_not_change_projection_or_feasibility_programs(self):
        fixture.MeanWaitTest.setUpClass()
        f = fixture.MeanWaitTest
        registered = spec.source.manifest(f.registered)
        rows, masses, fold = f.bank["training_data"], f.masses, registered["folds"][0]
        masks = spec.previous.masks(rows, fold, f.keys)
        train = [r for r, keep in zip(rows, masks["training_all"]) if keep]
        ridge = spec.source.base.cost.fit_value(train, sample_weights=spec.wait_source.source.feasibility.sample_weights(train, masses))
        glm = {"center": [0.]*198, "scale": [1.]*198, "coefficients": [0.]*199}
        batch = spec.wait_source.neural.batches(rows)
        heads, _, shares, scores = spec.previous.inference(rows, batch, ridge, glm, "glm")
        archived = [{**r, "predicted_heads": heads[i].tolist(), "predicted_harm_mass_share": shares[i].tolist(),
            "predicted_feasibility_score": scores[i].tolist()} for i, r in enumerate(rows) if masks["held"][i]]
        cell = {"predictions": {spec.source.PRIMARY: archived}}
        panels = {}
        for r in rows:
            if spec.state_key(r) not in f.keys: continue
            labels = sorted(r["labels"], key=lambda l: l["candidate"])
            values = np.asarray([[l[h] for h in ("actual_cost_delta", "fleet_cost_delta", "nonfleet_cost_delta", "wait_delta_min")] for l in labels])
            panels[spec.state_key(r)] = {"values": np.repeat(values[None], 8, axis=0)}
        a, d, model, witnesses = spec.diagnose_fold(rows, masses, panels, registered, fold, cell, ridge, glm, batch)
        changed, changed_panels = copy.deepcopy(rows), copy.deepcopy(panels)
        for i in np.flatnonzero(masks["held"]):
            r = changed[i]
            for l in r["labels"]:
                if l["candidate"] != r["original_choice"]:
                    l["fleet_cost_delta"] += 100.
                    l["actual_cost_delta"] += 100.
                    changed_panels[spec.state_key(r)]["values"][:, l["candidate"], [0, 1]] += 100.
        b, other_d, other_model, other_witnesses = spec.diagnose_fold(changed, masses, changed_panels, registered, fold, cell, ridge, glm, batch)
        self.assertEqual(model, other_model)
        self.assertEqual(d, other_d)
        self.assertEqual(witnesses, other_witnesses)
        self.assertEqual(a["projection_glm"]["training_reference"], b["projection_glm"]["training_reference"])
        self.assertNotEqual(a["projection_glm"]["held"], b["projection_glm"]["held"])
        self.assertEqual(model["sample_weight_sum"], 2232.)
        self.assertEqual(model["parameter_count"], 294)

    def test_scheduler_budget_single_thread_unbound_and_no_model_pull(self):
        task = task_specification("unit_linear_floor")
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
        self.assertNotIn("_raw", task["result_dir"])
        self.assertEqual(spec.contract()["budget"], {"cost_projection_fits": 47, "max_linear_programs": 188,
            "optimizer_updates": 0, "actor_updates": 0, "new_native_episodes": 0})


if __name__ == "__main__":
    unittest.main()
