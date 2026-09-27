from copy import deepcopy
import unittest

import numpy as np

from freq_hrl.experiments import pointmaze_continuation_credit as credit
from scripts import pointmaze_continuation_credit_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from tests.test_pointmaze_onecheck_advantage import causal_row


def endpoint(seed, *, remaining=0.5, position=0.1):
    return {"seed": seed, "feature_names": ["position", "remaining_fraction", "budget_spent"],
            "features": [position, remaining, 0.0], "cost_to_go": 2.0 * remaining + position}


class ContinuationCreditTest(unittest.TestCase):
    def test_terminal_value_is_zero_and_overlapping_paths_are_rejected(self):
        train = [endpoint(s, remaining=r) for s in (1, 2) for r in (0.2, 0.8)]
        prediction, diagnostics = credit.fit_continuation(
            train, [endpoint(3, remaining=0.0), endpoint(3)], seed=16, epochs=2,
        )
        self.assertEqual(prediction[0], 0.0)
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertEqual(diagnostics["optimizer_steps"], 2)
        with self.assertRaisesRegex(ValueError, "overlap"):
            credit.fit_continuation(train, [endpoint(1)], seed=16, epochs=2)

    def test_heldout_path_targets_cannot_change_its_continuation_prediction(self):
        traces = [endpoint(s, remaining=r) for s in (1, 2) for r in (0.2, 0.5, 0.8)]
        pairs = [{"seed": s, "now_endpoint": endpoint(s),
                  "wait_endpoint": endpoint(s, position=0.2),
                  "window_ise_advantage": 0.02, "actual_tail_advantage": 0.1}
                 for s in (1, 2)]
        options = dict(root=16, fit_seeds=[1, 2], eval_seeds=[3], epochs=2)
        first, diagnostics = credit.crossfit_continuations(traces, pairs, **options)
        changed_traces, changed_pairs = deepcopy(traces), deepcopy(pairs)
        for row in changed_traces:
            if row["seed"] == 1:
                row["cost_to_go"] += 100.0
        for arm in ("now", "wait"):
            changed_pairs[0][f"{arm}_endpoint"]["cost_to_go"] += 100.0
        changed_pairs[0]["actual_tail_advantage"] += 1.0
        second, _ = credit.crossfit_continuations(changed_traces, changed_pairs, **options)
        for name in ("predicted_now_tail", "predicted_wait_tail", "bootstrap_advantage"):
            self.assertEqual(first[0][name], second[0][name])
        for row in first:
            self.assertAlmostEqual(row["bootstrap_advantage"],
                                   row["window_ise_advantage"] + row["predicted_tail_advantage"])
        self.assertEqual(diagnostics["folds"][0]["training_paths"], [2])
        self.assertEqual(diagnostics["folds"][1]["training_paths"], [1])
        self.assertAlmostEqual(diagnostics["zero_contrast_mse"], 0.01)
        with self.assertRaisesRegex(ValueError, "seed roles"):
            credit.crossfit_continuations(traces, pairs, root=16,
                                          fit_seeds=[1, 2], eval_seeds=[2], epochs=2)

    def test_identical_endpoints_reduce_bootstrap_to_short_credit(self):
        traces = [endpoint(s, remaining=r) for s in (1, 2) for r in (0.2, 0.8)]
        pairs = [{"seed": s, "now_endpoint": endpoint(s), "wait_endpoint": endpoint(s),
                  "window_ise_advantage": 0.02, "actual_tail_advantage": 0.0}
                 for s in (1, 2)]
        rows, _ = credit.crossfit_continuations(traces, pairs, root=16,
                                               fit_seeds=[1, 2], eval_seeds=[3], epochs=2)
        self.assertTrue(all(r["bootstrap_advantage"] == 0.02 for r in rows))

    def test_controls_share_features_alpha_and_threshold_without_cv(self):
        rows = [{**causal_row(s), "window_ise_advantage": s * 0.01,
                 "bootstrap_advantage": s * -0.01} for s in (1, 1, 2, 2)]
        short = credit.fit_trigger(rows, target="window_ise_advantage")
        boot = credit.fit_trigger(rows, target="bootstrap_advantage")
        self.assertEqual(short["feature_names"], boot["feature_names"])
        for fitted in (short, boot):
            self.assertEqual(fitted["threshold"], 0.0)
            self.assertEqual(fitted["model"]["alpha"], 100.0)
            self.assertNotIn("group_count", fitted["model"])

    def test_scheduler_reuses_only_frozen_roots_and_small_source(self):
        self.assertEqual(spec.roots(preflight=False), (209011, 209061))
        task = task_specification("unit_continuation", 209011, preflight=False,
                                  protocol_spec=spec)
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertIn("--pairs-per-seed 12", task["cmd"])
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["cpu"], 1)
        self.assertEqual(task["ram_mb"], 1536)
        self.assertIn(str(spec.source_result(209011, preflight=False).parent),
                      task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
