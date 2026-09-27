from copy import deepcopy
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_averaged_label_diagnostic as diagnostic
from freq_hrl.experiments.pointmaze_continuation_credit import fit_continuation
from scripts import pointmaze_averaged_label_diagnostic_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from tests.test_pointmaze_paired_value_qualification import pair


def cached_pair(seed, step=50):
    row = pair(seed, step)
    for arm in ("now", "wait"):
        ep = row[f"{arm}_endpoint"]
        ep["features"] += [0.0] * 37
        ep["feature_names"] += [f"summary_{i}" for i in range(37)]
    row["replicate_tail_advantages"] = [0.1, 0.3, -0.1, 0.5]
    return row


class AveragedLabelDiagnosticTest(unittest.TestCase):
    def test_cached_join_is_keyed_and_keeps_sources_unchanged(self):
        pairs = [cached_pair(1, 50), cached_pair(1, 100)]
        original = deepcopy(pairs)
        noise = [{"seed": 1, "check_step": 100, "original_tail_advantage": 0.1,
                  "replicate_tail_advantages": [1., 3., 5., 7.]}]
        joined = diagnostic.join_cached_pairs(pairs, noise)
        self.assertEqual(diagnostic.label_targets(joined, "single_draw").tolist(), [1.])
        self.assertEqual(diagnostic.label_targets(joined, "averaged").tolist(), [4.])
        ep = diagnostic.compact_endpoints(joined)
        self.assertEqual(len(ep[0]["features"]), 424)
        self.assertEqual(ep[0]["features"][40:], [0.] * 384)
        self.assertEqual(pairs, original)
        with self.assertRaisesRegex(ValueError, "uniquely match"):
            diagnostic.join_cached_pairs(pairs, noise * 2)
        noise[0]["original_tail_advantage"] = 100.
        with self.assertRaisesRegex(ValueError, "original label differs"):
            diagnostic.join_cached_pairs(pairs, noise)

    def test_override_changes_only_paired_targets_not_normalization(self):
        train = diagnostic.compact_endpoints([cached_pair(1)])
        query = diagnostic.compact_endpoints([cached_pair(2)])
        kwargs = dict(seed=19, epochs=2, objective="paired_contrast")
        original, diag = fit_continuation(train, query, **kwargs)
        repeated, same = fit_continuation(train, query, contrast_targets=[0.1], **kwargs)
        changed, changed_diag = fit_continuation(train, query, contrast_targets=[-10.], **kwargs)
        np.testing.assert_allclose(original, repeated, atol=1e-6, rtol=1e-6)
        self.assertFalse(np.allclose(original, changed))
        for key in ("baseline_cost_rate", "target_scale", "seed", "training_paths", "parameter_count"):
            self.assertEqual(diag[key], same[key])
            self.assertEqual(diag[key], changed_diag[key])
        with self.assertRaisesRegex(ValueError, "one finite label"):
            fit_continuation(train, query, contrast_targets=[1., 2.], **kwargs)
        with self.assertRaisesRegex(ValueError, "one finite label"):
            fit_continuation(train, query, contrast_targets=[1.], seed=19, epochs=2)

    def test_all_heldout_labels_and_costs_are_excluded_from_predictions(self):
        pairs = [cached_pair(s, step) for s in (1, 2, 3) for step in (50, 100)]
        kwargs = dict(root=19, fit_seeds=[1, 2, 3], eval_seeds=[4], epochs=2)
        first = diagnostic.qualify(pairs, **kwargs)
        changed = deepcopy(pairs)
        for row in changed:
            if row["seed"] == 1:
                row["replicate_tail_advantages"] = [10., 20., 30., 40.]
                for arm in ("now", "wait"):
                    row[f"{arm}_endpoint"]["cost_to_go"] += 100.
        second = diagnostic.qualify(changed, **kwargs)
        for a, b in zip(first["rows"][:2], second["rows"][:2]):
            for label in diagnostic.LABELS:
                self.assertEqual(a[f"{label}_prediction"], b[f"{label}_prediction"])
        for a, b in zip(first["folds"][::2], first["folds"][1::2]):
            for key in ("training_paths", "seed", "baseline_cost_rate", "target_scale"):
                self.assertEqual(a[key], b[key])
            self.assertNotIn(a["held_out_path"], a["training_paths"])
            self.assertEqual(a["state_dim"], 424)
            self.assertEqual(a["parameter_count"], 31425)
        self.assertEqual(first["critic_optimizer_steps"], 12)
        self.assertEqual(first["additional_primitive_steps"], 0)
        self.assertEqual(first["controller_training_iterations"], 0)
        self.assertEqual(first["cached_future_pair_labels"], 24)
        with self.assertRaisesRegex(ValueError, "seed roles"):
            diagnostic.qualify(pairs, root=19, fit_seeds=[1, 2, 3], eval_seeds=[3], epochs=2)

    def test_correction_keeps_negative_estimates_and_gate_requires_both_controls(self):
        rows = [{"repeat_mean": mean, "repeat_variance": 2., "replicates": 2,
                 "single_draw_prediction": 2., "averaged_prediction": mean}
                for mean in (1., 3.)]
        result = diagnostic.metrics(rows)
        self.assertEqual(result["conditional_mean_mse_estimate"],
                         {"zero": 4., "single_draw": 0., "averaged": -1.})
        self.assertTrue(result["averaging_qualification_passed"])
        rows[0]["averaged_prediction"] = 3.
        self.assertFalse(diagnostic.metrics(rows)["averaging_qualification_passed"])
        for row in rows:
            row["single_draw_prediction"] = 100.
            row["averaged_prediction"] = 10.
        self.assertFalse(diagnostic.metrics(rows)["averaging_qualification_passed"])

    def test_scheduler_stages_only_cached_sources_and_uses_existing_roots(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_average_labels", 209011, preflight=False, protocol_spec=spec)
        for key, path in spec.input_results(209011, preflight=False).items():
            self.assertIn("--" + key.replace("_", "-"), task["cmd"])
            self.assertIn(str(path.parent), task["stage_input_paths"])
        self.assertIn("--future-replicates 8", task["cmd"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertIsNone(task["require_node"])


if __name__ == "__main__":
    unittest.main()
