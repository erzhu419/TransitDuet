from copy import deepcopy
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_regularized_pair_diagnostic as diagnostic
from scripts import pointmaze_regularized_pair_diagnostic_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from tests.test_pointmaze_averaged_label_diagnostic import cached_pair


def baseline(pairs):
    return {"rows": [{"seed": p["seed"], "check_step": p["check_step"],
                      "repeat_mean": float(np.mean(p["replicate_tail_advantages"])),
                      "repeat_variance": float(np.var(p["replicate_tail_advantages"], ddof=1)),
                      "replicates": len(p["replicate_tail_advantages"]),
                      "single_draw_label": p["replicate_tail_advantages"][0],
                      "single_draw_prediction": 0.1, "averaged_prediction": 0.2}
                     for p in pairs]}


class RegularizedPairDiagnosticTest(unittest.TestCase):
    def test_unit_mean_loss_ridge_has_known_solution_and_fixed_strength(self):
        train, query = [cached_pair(1)], [cached_pair(2)]
        pred, fit = diagnostic.fit_pair_ridge(train, query, label="averaged")
        np.testing.assert_allclose(pred, [0.1], atol=1e-12)
        self.assertAlmostEqual(fit["effective_degrees_of_freedom"], 0.5)
        self.assertEqual(fit["parameter_count"], 40)
        self.assertEqual(fit["alpha"], 1.0)
        duplicate, _ = diagnostic.fit_pair_ridge(train * 3, query, label="averaged")
        np.testing.assert_allclose(duplicate, pred, atol=1e-12)
        single, control = diagnostic.fit_pair_ridge(train, query, label="single_draw")
        np.testing.assert_allclose(single, [0.05], atol=1e-12)
        for key in ("feature_mean", "feature_scale", "training_paths"):
            self.assertEqual(fit[key], control[key])
        np.testing.assert_allclose(fit["feature_mean"][:3], [0.15, 0.5, 0.])
        np.testing.assert_allclose(fit["feature_scale"][:3], [0.05, 1., 1.])

    def test_shared_value_structure_is_antisymmetric_and_zero_at_identity_or_terminal(self):
        train, forward = [cached_pair(1)], cached_pair(2)
        reverse, identity, terminal = deepcopy(forward), deepcopy(forward), deepcopy(forward)
        reverse["now_endpoint"], reverse["wait_endpoint"] = reverse["wait_endpoint"], reverse["now_endpoint"]
        identity["wait_endpoint"] = deepcopy(identity["now_endpoint"])
        for arm in ("now", "wait"):
            terminal[f"{arm}_endpoint"]["features"][1] = 0.
        pred, _ = diagnostic.fit_pair_ridge(train, [forward, reverse, identity, terminal], label="averaged")
        self.assertAlmostEqual(pred[0], -pred[1])
        self.assertEqual(pred[2], 0.)
        self.assertEqual(pred[3], 0.)
        with self.assertRaisesRegex(ValueError, "overlap"):
            diagnostic.fit_pair_ridge(train, train, label="averaged")

    def test_heldout_targets_and_baseline_predictions_never_enter_fit(self):
        pairs = [cached_pair(s, step) for s in (1, 2, 3) for step in (50, 100)]
        kwargs = dict(root=20, fit_seeds=[1, 2, 3], eval_seeds=[4])
        first = diagnostic.qualify(pairs, baseline(pairs), **kwargs)
        changed = deepcopy(pairs)
        for p in changed:
            if p["seed"] == 1:
                p["replicate_tail_advantages"] = [10., 20., 30., 40.]
                p["now_endpoint"]["cost_to_go"] += 100.
        old = baseline(changed)
        for row in old["rows"]:
            row["averaged_prediction"] = 1000.
        second = diagnostic.qualify(changed, old, **kwargs)
        for a, b in zip(first["rows"][:2], second["rows"][:2]):
            for key in ("ridge_single_draw_prediction", "ridge_averaged_prediction"):
                self.assertEqual(a[key], b[key])
        self.assertEqual(first["ridge_fits"], 6)
        self.assertEqual(first["additional_primitive_steps"], 0)
        self.assertEqual(first["critic_optimizer_steps"], 0)
        self.assertTrue(all(f["held_out_path"] not in f["training_paths"] for f in first["folds"]))
        with self.assertRaisesRegex(ValueError, "labels differ"):
            diagnostic.qualify(changed, baseline(pairs), **kwargs)
        with self.assertRaisesRegex(ValueError, "seed roles"):
            diagnostic.qualify(pairs, baseline(pairs), root=20, fit_seeds=[1, 2, 3], eval_seeds=[1])

    def test_gate_requires_zero_and_matched_neural_control_without_clipping(self):
        rows = [{"repeat_mean": 1., "repeat_variance": 4., "replicates": 2,
                 "ridge_single_draw_prediction": 0., "ridge_averaged_prediction": 1.,
                 "stage19_single_draw_prediction": 0., "stage19_averaged_prediction": 0.5}]
        result = diagnostic.contrast_metrics(rows)
        self.assertEqual(result["conditional_mean_mse_estimate"]["ridge_averaged"], -2.)
        self.assertTrue(result["qualification_passed"])
        rows[0]["stage19_averaged_prediction"] = 1.
        self.assertFalse(diagnostic.contrast_metrics(rows)["qualification_passed"])
        rows[0]["ridge_averaged_prediction"] = 3.
        rows[0]["stage19_averaged_prediction"] = 10.
        self.assertFalse(diagnostic.contrast_metrics(rows)["qualification_passed"])

    def test_scheduler_stages_three_caches_without_node_pin(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_pair_ridge", 209011, preflight=False, protocol_spec=spec)
        self.assertEqual(len(spec.input_results(209011, preflight=False)), 3)
        for key, path in spec.input_results(209011, preflight=False).items():
            self.assertIn("--" + key.replace("_", "-"), task["cmd"])
            self.assertIn(str(path.parent), task["stage_input_paths"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)


if __name__ == "__main__":
    unittest.main()
