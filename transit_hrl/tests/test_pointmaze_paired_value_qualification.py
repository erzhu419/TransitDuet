from copy import deepcopy
import unittest

import torch

from freq_hrl.experiments.pointmaze_continuation_credit import continuation_loss, fit_continuation
from freq_hrl.experiments import pointmaze_paired_value_qualification as qualification
from scripts import pointmaze_paired_value_qualification_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from tests.test_pointmaze_continuation_credit import endpoint


def pair(seed, step=50):
    return {"seed": seed, "check_step": step,
            "now_endpoint": endpoint(seed), "wait_endpoint": endpoint(seed, position=0.2),
            "window_ise_advantage": 0.02, "actual_tail_advantage": 0.1,
            "predicted_tail_advantage": 0.0}


class PairedValueQualificationTest(unittest.TestCase):
    def test_pair_loss_cancels_common_errors_and_preserves_path_weights(self):
        prediction, target = torch.tensor([1., 2., 6., 9.]), torch.tensor([0., 3., 4., 8.])
        weight = torch.full((4,), 0.25)
        loss = continuation_loss(prediction, target, weight, objective="paired_contrast")
        self.assertAlmostEqual(float(loss), 2.5)
        shifted = continuation_loss(prediction + torch.tensor([100., 100., -50., -50.]),
                                    target, weight, objective="paired_contrast")
        self.assertEqual(float(loss), float(shifted))
        self.assertAlmostEqual(float(continuation_loss(prediction, target, weight,
                                                       objective="absolute")), 1.75)

    def test_shared_value_is_antisymmetric_and_control_uses_same_normalization(self):
        train = qualification.endpoints([pair(1), pair(2)])
        q = qualification.endpoints([pair(3)])
        query = [*q, q[1], q[0], q[0], q[0]]
        pred, paired = fit_continuation(train, query, seed=16, epochs=2,
                                        objective="paired_contrast")
        _, absolute = fit_continuation(train, query, seed=16, epochs=2)
        self.assertAlmostEqual(pred[1] - pred[0], -(pred[3] - pred[2]))
        self.assertEqual(pred[5] - pred[4], 0.0)
        for key in ("seed", "training_paths", "training_rows", "baseline_cost_rate", "target_scale"):
            self.assertEqual(paired[key], absolute[key])
        with self.assertRaisesRegex(ValueError, "aligned"):
            fit_continuation([train[0], train[2], train[1], train[3]], query,
                             seed=16, epochs=2, objective="paired_contrast")

    def test_whole_path_is_excluded_and_qualification_never_samples_environment(self):
        pairs = [pair(s, step) for s in (1, 2) for step in (50, 100)]
        kwargs = dict(root=16, fit_seeds=[1, 2], eval_seeds=[3], epochs=2)
        first = qualification.qualify(pairs, **kwargs)
        modified = deepcopy(pairs)
        for row in modified:
            if row["seed"] == 1:
                row["actual_tail_advantage"] += 50
                for arm in ("now", "wait"):
                    row[f"{arm}_endpoint"]["cost_to_go"] += 100
        second = qualification.qualify(modified, **kwargs)
        for a, b in zip(first["rows"][:2], second["rows"][:2]):
            for objective in qualification.OBJECTIVES:
                self.assertEqual(a[f"{objective}_prediction"], b[f"{objective}_prediction"])
        self.assertEqual(first["additional_primitive_steps"], 0)
        self.assertEqual(first["evaluation_paths_used"], 0)
        self.assertEqual(first["critic_optimizer_steps"], 8)
        self.assertTrue(all(f["held_out_path"] not in f["training_paths"] for f in first["folds"]))
        with self.assertRaisesRegex(ValueError, "seed roles"):
            qualification.qualify(pairs, root=16, fit_seeds=[1, 2], eval_seeds=[2], epochs=2)

    def test_gate_requires_all_three_comparisons(self):
        rows = [{"actual_tail_advantage": 1., "stage16_prediction": 0.8,
                 "absolute_prediction": 0.9, "paired_contrast_prediction": 0.5}]
        metrics = qualification.contrast_metrics(rows)
        self.assertLess(metrics["paired_contrast_mse"], metrics["zero_mse"])
        self.assertGreater(metrics["paired_contrast_mse"], metrics["absolute_mse"])
        self.assertGreater(metrics["paired_contrast_mse"], metrics["stage16_mse"])

    def test_scheduler_stages_stage16_not_stage12(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_paired_value", 209011, preflight=False, protocol_spec=spec)
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertIn("pointmaze_continuation_credit_stage16", str(spec.source_result(209011, preflight=False)))
        self.assertIn(str(spec.source_result(209011, preflight=False).parent), task["stage_input_paths"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)


if __name__ == "__main__":
    unittest.main()
