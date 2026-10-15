import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize

from freq_hrl.domains.transit import native_variance_critic as learning
from scripts import run_native_transit_variance_critic_stage243 as spec
from scripts.submit_native_transit_variance_critic_stage243_scheduleurm import task_specification


def fixture():
    rng = np.random.default_rng(243)
    n = 18
    rows = [{"root": 397 if i < 9 else 401, "scene_seed": i//3, "probe_index": i%3} for i in range(n)]
    x = rng.normal(size=(n, 6, 210))
    originals = rng.integers(6, size=n)
    components = rng.normal(size=(n, 6, 2))*.02
    data = {"x": x, "components": components, "heads": np.stack((components.sum(axis=-1), rng.normal(size=(n, 6))*.01), axis=-1),
        "originals": originals, "roots": np.array([r["root"] for r in rows]), "raw_logits": -np.ones((n, 6))}
    data["raw_choices"] = learning.joint.binding.policy.selected(data["heads"][..., 0], data["raw_logits"], originals)
    draws = rng.normal(size=(n, 8, 6, 4))*.015
    draws[..., 1] = (rng.random((n, 8, 6)) < .12)*.2
    draws[..., 2] += (x[..., 0]*.02)[:, None]
    draws[..., 0] = draws[..., 1]+draws[..., 2]
    values = draws[:, :4].mean(axis=1)[..., (0, 3)]
    cached = learning.strength.lowrank.fit_path(rows, data, values, "original")[0]
    return rows, data, draws, cached


class VarianceCriticTest(unittest.TestCase):
    def test_split_identity_capacity_and_frozen_raw(self):
        rows, data, draws, cached = fixture()
        with patch.object(learning.joint, "fit_reference", side_effect=AssertionError("Raw retraining")):
            models, fit = learning.fit_paths(rows, data, draws[:, :4], cached)
        spec.qualify_fit(fit)
        np.testing.assert_allclose(data["components"].sum(axis=-1), data["heads"][..., 0])
        for arm in learning.ARMS:
            for key, pred in learning.candidates(data, models[arm]).items():
                np.testing.assert_allclose(pred["components"].sum(axis=-1), pred["cost"], atol=1e-12)
                if key == "zero": np.testing.assert_array_equal(pred["chosen"], data["raw_choices"])
                if arm in ("combined", "split"):
                    np.testing.assert_array_equal(pred["cost"], learning.strength.base.costs(data, cached, **learning.strength.CANDIDATES[key]))
        for f in fit["fits"]:
            self.assertLessEqual(f["norm"], f["radius"]+1e-10)
            self.assertEqual(f["total_cost_function_df"], f["retained_rank"])
        self.assertLessEqual(fit["combined_max_abs_difference"], 1e-12)

    def test_gls_matches_independent_constrained_optimizer_and_zero_radius(self):
        rng = np.random.default_rng(9)
        z = rng.normal(size=(40, 3))
        target, weights = rng.normal(size=(40, 2)), np.exp(rng.normal(size=(40, 2)))
        initial = np.zeros((2, 3))
        initial[:, 0] = .1
        theta, fit = learning.solve_components(z, target, weights, initial, .2)
        def loss(v): return np.mean(np.sum(weights*(z@v.reshape(2, 3).T-target)**2, axis=1))
        oracle = minimize(loss, initial.ravel(), method="SLSQP", constraints=[{
            "type": "ineq", "fun": lambda v: .2**2-np.sum(v.reshape(2, 3).sum(axis=0)**2)}],
            options={"ftol": 1e-12, "maxiter": 200})
        self.assertTrue(oracle.success)
        self.assertAlmostEqual(loss(theta.ravel()), oracle.fun, places=10)
        np.testing.assert_allclose(theta.ravel(), oracle.x, atol=1e-6)
        equal, fit = learning.solve_components(z, target, weights, np.zeros((2, 3)), 0.)
        np.testing.assert_array_equal(equal.sum(axis=0), np.zeros(3))
        self.assertLessEqual(fit["optimality"], 1e-8)

    def test_paired_noise_cancels_common_future_shocks_and_pools_empty_groups(self):
        rows, data, draws, cached = fixture()
        first = learning.noise_targets(rows, data, draws[:, :4], cached["scales"][0])
        changed = draws[:, :4].copy()
        shock = np.arange(4)[None, :, None]*.1
        changed[..., 1] += shock
        changed[..., 2] += shock
        second = learning.noise_targets(rows, data, changed, cached["scales"][0])
        for a, b in zip(first[:3], second[:3]): np.testing.assert_allclose(a, b, atol=1e-12)
        empty = draws[:, :4].copy()
        empty[:3, :, :, 1] = 0
        _, precision, gamma, ledger = learning.noise_targets(rows, data, empty, cached["scales"][0])
        self.assertTrue(np.isfinite(precision).all())
        self.assertTrue(((gamma >= 0) & (gamma <= 1)).all())
        self.assertGreater(ledger["pooled_variance_of_mean"][1], 0)
        empty[..., 1] = 0
        _, precision, _, ledger = learning.noise_targets(rows, data, empty, cached["scales"][0])
        np.testing.assert_array_equal(precision[..., 1], np.ones((18, 6)))
        self.assertEqual(ledger["zero_observed_variance_heads"], ["risk"])

    def test_group_signal_excludes_own_prefix_and_fit_cannot_read_unused_futures(self):
        rows, data, draws, cached = fixture()
        base = learning.noise_targets(rows, data, draws[:, :4], cached["scales"][0])[3]
        altered = draws.copy()
        altered[:3, :4, 1:, 2] += .2
        changed = learning.noise_targets(rows, data, altered[:, :4], cached["scales"][0])[3]
        np.testing.assert_allclose(base["groups"][0]["prior_signal_energy"], changed["groups"][0]["prior_signal_energy"])
        self.assertNotEqual(base["groups"][1]["prior_signal_energy"], changed["groups"][1]["prior_signal_energy"])
        first = learning.fit_paths(rows, data, draws[:, :4], cached)
        unused = draws.copy()
        unused[:, 4:] += 10000.
        self.assertEqual(first, learning.fit_paths(rows, data, unused[:, :4], cached))
        bad = draws[:, :4].copy()
        bad[..., 0] += 1
        with self.assertRaisesRegex(ValueError, "do not sum"): learning.fit_paths(rows, data, bad, cached)

    def test_physical_cost_accounting_and_root_aggregation(self):
        rows, data, draws, cached = fixture()
        models, _ = learning.fit_paths(rows, data, draws[:, :4], cached)
        selection = {a: {f: "3@1" for f in learning.FILTERS} for a in learning.ARMS}
        pred = learning.predict(data, models, selection)
        first = learning.metrics(data, draws[:, 4:], pred)
        self.assertEqual(first["all"]["cases"], 18)
        for piece in (first["all"], *first["by_root"].values()):
            for p in piece["variants"].values(): self.assertAlmostEqual(p["delta_cost"], p["delta_service"]+p["delta_risk"], places=12)
        np.testing.assert_array_equal(pred["combined_pooled_safe"]["chosen"], pred["split_pooled_safe"]["chosen"])
        self.assertEqual(learning.aggregate([first, first])["all"]["variants"], first["all"]["variants"])
        poisoned = draws[:, 4:].copy()
        poisoned[:, :, 1:, 3] += 100
        self.assertNotEqual(first, learning.metrics(data, poisoned, pred))

    def test_contract_and_scheduler_do_not_stage_caches_or_pin_nodes(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c["budget"]["new_raw_reference_fits"], 0)
        self.assertEqual(c["budget"]["new_native_episodes"], 0)
        for key in (spec.PREFLIGHT, None):
            task = task_specification("unit_variance", key)
            self.assertEqual((task["cpu"], task["ram_mb"]), (1, 3072))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
            self.assertFalse(any("results" in str(p) for p in task["stage_input_paths"]))


if __name__ == "__main__": unittest.main()
