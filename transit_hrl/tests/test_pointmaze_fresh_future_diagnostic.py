from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import t

from freq_hrl.experiments import pointmaze_fresh_future_diagnostic as diagnostic
from scripts import pointmaze_fresh_future_diagnostic_spec as spec
from scripts.pointmaze_budgeted_trigger_stage9_spec import cell_options
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification


class FreshFutureDiagnosticTest(unittest.TestCase):
    def test_paired_mc_interval_matches_analytic_variance_and_family_correction(self):
        rows = [{"replicate_tail_advantages": a,
                 "predictions": {"zero": 0., "ridge_averaged": 1., "stage19_averaged": 2.}}
                for a in ([0., 2.], [2., 4.])]
        result = diagnostic.fixed_comparisons(rows)
        zero = result["comparisons"]["zero"]
        self.assertEqual(zero["control_minus_candidate_mse"], 3.)
        self.assertAlmostEqual(zero["mc_standard_error"], np.sqrt(2.))
        self.assertEqual(zero["welch_degrees_of_freedom"], 2.)
        np.testing.assert_allclose(zero["bonferroni_mc_interval"],
                                   3 + np.array([-1., 1.]) * t.ppf(0.99375, 2) * np.sqrt(2.))
        self.assertFalse(result["point_gate_passed"])
        self.assertFalse(result["precision_gate_passed"])
        for key, comparison in result["comparisons"].items():
            m = result["conditional_mean_mse_estimate"]
            self.assertAlmostEqual(comparison["control_minus_candidate_mse"], m[key] - m["ridge_averaged"])

    def test_zero_variance_ties_and_negative_corrected_mse(self):
        rows = [{"replicate_tail_advantages": [1., 1., 1.],
                 "predictions": {"zero": 0., "ridge_averaged": 1., "stage19_averaged": 2.}}]
        result = diagnostic.fixed_comparisons(rows)
        self.assertEqual(result["comparisons"]["zero"]["bonferroni_mc_interval"], [1., 1.])
        self.assertTrue(result["precision_gate_passed"])
        rows[0]["predictions"]["stage19_averaged"] = 1.
        self.assertFalse(diagnostic.fixed_comparisons(rows)["precision_gate_passed"])
        rows[0]["replicate_tail_advantages"] = [0., 2.]
        self.assertEqual(diagnostic.fixed_comparisons(rows)["conditional_mean_mse_estimate"]["ridge_averaged"], -1.)

    def test_actual_frozen_cases_keep_predictions_and_use_disjoint_draws(self):
        all_fresh = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                options = cell_options(root, preflight=preflight)
                args = SimpleNamespace(optimizer_seed=root, branch_fit_seeds=list(options["branch_fit"]),
                                       trigger_eval_seeds=list(options["trigger_eval"]),
                                       **spec.input_results(root, preflight=preflight),
                                       **spec.sampling_options(preflight=preflight))
                with patch.object(diagnostic, "replay_source_controller", side_effect=AssertionError("no replay during cache loading")):
                    cases = diagnostic.load_cases(args)
                self.assertEqual(len(cases), 2 if preflight else 16)
                original = json.loads(args.prediction_result.read_text())["cells"][0]["rows"]
                lookup = {(r["seed"], r["check_step"]): r for r in original}
                for case in cases:
                    self.assertFalse(all_fresh.intersection(case["future_seeds"]))
                    all_fresh.update(case["future_seeds"])
                    r = lookup[(case["seed"], case["check_step"])]
                    self.assertEqual(case["predictions"]["ridge_averaged"], r["ridge_averaged_prediction"])
        self.assertEqual(len(all_fresh), 8 + 2 * 16 * 64)

    def test_worker_reuses_draw_in_both_arms_and_rejects_boundary_drift(self):
        def rollout(controller, *, arm, continuation_seed=None, **kwargs):
            wait = arm == "wait_one_check"
            cost = 1.2 if wait else 1.
            return {"prefix": [0.], "score": 1., "decision_steps": [0, 50],
                    "episode_return": 0., "tracking_squared_error_integral": 1., "window_ise": 0.1,
                    "value_trace": [{"step": 100, "features": [2.] if wait else [1.],
                                     "controller_history": [0.],
                                     "cost_to_go": cost + (continuation_seed or 0) * wait}]}

        case = {"seed": 1, "check_step": 50, "future_seeds": [21, 22], "predictions": {},
                "endpoint_pair": {"now_endpoint": {"step": 100, "features": [1.], "cost_to_go": 1.},
                                  "wait_endpoint": {"step": 100, "features": [2.], "cost_to_go": 1.2}}}
        worker = (object(), {"threshold": 0.}, SimpleNamespace(), SimpleNamespace(upper_period_steps=50))
        with patch.object(diagnostic, "_WORKER", worker), patch.object(diagnostic, "rollout_intervention", side_effect=rollout) as mock:
            result = diagnostic.sample_case(case)
            np.testing.assert_allclose(result["replicate_tail_advantages"], [21.2, 22.2])
            self.assertEqual([c.kwargs["continuation_seed"] for c in mock.call_args_list if "continuation_seed" in c.kwargs], [21, 21, 22, 22])
        bad = deepcopy(case)
        bad["endpoint_pair"]["wait_endpoint"]["features"] = [3.]
        with patch.object(diagnostic, "_WORKER", worker), patch.object(diagnostic, "rollout_intervention", side_effect=rollout):
            with self.assertRaisesRegex(RuntimeError, "frozen endpoints"):
                diagnostic.sample_case(bad)

    def test_full_and_preflight_accounting_and_scheduler_parallel_resources(self):
        for preflight, root, reconstruction, replay, cpu, ram in (
            (True, 208001, 5100, 6600, 2, 3072),
            (False, 209011, 4243200, 2515200, 16, 12288),
        ):
            options = cell_options(root, preflight=preflight)
            args = SimpleNamespace(iterations=options["iterations"], horizon=options["horizon"],
                                   checkpoint_evaluation_interval=options["checkpoint_evaluation_interval"],
                                   **{r + "_seeds": options[r] for r in ("train", "selection", "branch_fit", "trigger_eval")})
            self.assertEqual(sum(diagnostic.reconstruction_steps(args).values()), reconstruction)
            sampling = spec.sampling_options(preflight=preflight)
            self.assertEqual(sum(diagnostic.replay_steps(2 if preflight else 16,
                             sampling["future_replicates"], args.horizon).values()), replay)
            task = task_specification("unit_fresh_future", root, preflight=preflight, protocol_spec=spec)
            self.assertEqual(task["cpu"], cpu)
            self.assertEqual(task["ram_mb"], ram)
            self.assertIsNone(task["require_node"])
            self.assertIn(f"--workers {cpu}", task["cmd"])
            for path in spec.input_results(root, preflight=preflight).values():
                self.assertIn(str(path.parent), task["stage_input_paths"])


if __name__ == "__main__":
    unittest.main()
