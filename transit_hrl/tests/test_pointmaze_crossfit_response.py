from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.core import PhysicalTimeScaleContract
from freq_hrl.core.plan_response import CrossFittedPlanResponseCritic, PlanResponseCritic
from freq_hrl.experiments import pointmaze_crossfit_response as crossfit
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments import pointmaze_plan_hold as hold
from freq_hrl.experiments import pointmaze_separate_motion as motion
from freq_hrl.experiments import pointmaze_state_response as state
from freq_hrl.experiments import pointmaze_temporal_plan as temporal
from scripts import pointmaze_crossfit_response_spec as spec
from scripts.submit_pointmaze_deployed_pair_diagnostic_scheduleurm import task_specification
from test_pointmaze_forecast_response import cache_fixture, ImmediatePool, motion_models
from test_pointmaze_history_information import NAMES, measured_sequences
from test_pointmaze_plan_hold import arguments
from test_pointmaze_temporal_plan import FakeController, FakeTask


def fake_stage31(directory):
    args, _, _ = cache_fixture(directory)
    controller = FakeController()
    controller.config = SimpleNamespace(state_encoder="mlp")
    scale = PhysicalTimeScaleContract(dt_seconds=.01, upper_period_seconds=.5, history_seconds=.64, fast_period_seconds=.04)
    cache = {"temporal_seed_roles":temporal.path_roles(208001, preflight=True)}
    loaded = (cache, {"controller_selected_iteration":2}, controller, scale, directory/"controller.pt",
              {"absolute_errors":{"episode_return":0.}})
    bounds = (np.full(2, -2.), np.full(2, 2.))
    with patch.object(hold, "load_controller", return_value=loaded), \
            patch.object(response, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(hold, "_make_task", side_effect=lambda **kwargs:FakeTask()), \
            patch.object(response, "_make_task", side_effect=lambda **kwargs:FakeTask()), \
            patch.object(hold, "pointmaze_goal_bounds", return_value=bounds), \
            patch.object(response, "pointmaze_goal_bounds", return_value=bounds):
        cell = response.run_cell(args)
    args.response_result = directory/"response.json"
    args.response_result.write_text(json.dumps(response._json_ready({"status":"complete", "protocol":{
        "protocol_version":response.PROTOCOL_VERSION, "optimizer_seed":208001}, "cells":[cell]})))
    args.output = directory/"crossfit"/"result.json"
    return args, loaded, bounds


class CrossFittedCriticTest(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(32)
        return rng.normal(size=(12, 4)), rng.normal(size=(12, 5)), np.repeat([1, 2, 3], 4), np.array(hold.HORIZONS)*.01

    def test_held_path_labels_do_not_enter_its_oof_predictions(self):
        x, y, groups, durations = self.fixture()
        fitted = CrossFittedPlanResponseCritic(durations_seconds=durations).fit(x, y, groups=groups)
        changed = y.copy()
        changed[groups == 2] *= 1000
        other = CrossFittedPlanResponseCritic(durations_seconds=durations).fit(x, changed, groups=groups)
        np.testing.assert_equal(fitted.fitted["out_of_fold_rates"][groups == 2],
                                other.fitted["out_of_fold_rates"][groups == 2])
        for group in np.unique(groups):
            mask = groups == group
            fold = PlanResponseCritic(durations_seconds=durations).fit(x[~mask], y[~mask])
            np.testing.assert_equal(fitted.fitted["out_of_fold_rates"][mask], fold.predict_rates(x[mask]))
        self.assertEqual((fitted.fitted["linear_fits"], fitted.fitted["kernel_solves"]), (4, 1))

    def test_centered_kernel_rate_prediction_matches_manual_algebra(self):
        x, y, groups, durations = self.fixture()
        fitted = CrossFittedPlanResponseCritic(durations_seconds=durations).fit(x, y, groups=groups)
        base, model = fitted.linear.fitted, fitted.fitted
        z = (x-base["feature_mean"])/base["feature_scale"]
        kernel = np.exp(-np.sum((z[:, None]-z[None, :])**2, axis=2)/(2*x.shape[1]))
        centerer = np.eye(len(x))-np.ones((len(x), len(x)))/len(x)
        centered = centerer@kernel@centerer
        residual = y/durations-model["out_of_fold_rates"]
        dual = np.linalg.solve(centered+np.eye(len(x)), residual-residual.mean(axis=0))
        np.testing.assert_allclose(model["dual_weights"], dual, rtol=1e-11, atol=1e-11)
        expected = fitted.linear.predict_rates(x)+residual.mean(axis=0)+centered@dual
        np.testing.assert_allclose(fitted.predict_rates(x), expected, rtol=1e-11, atol=1e-11)
        q = x[:2]+.7
        qz = (q-base["feature_mean"])/base["feature_scale"]
        qk = np.exp(-np.sum((qz[:, None]-z[None, :])**2, axis=2)/(2*x.shape[1]))
        qkc = qk-qk.mean(axis=1, keepdims=True)-kernel.mean(axis=0)+kernel.mean()
        np.testing.assert_allclose(fitted.predict_rates(q), fitted.linear.predict_rates(q)+residual.mean(axis=0)+qkc@dual)

    def test_rate_units_and_linear_reference_are_preserved(self):
        x, y, groups, durations = self.fixture()
        fitted = CrossFittedPlanResponseCritic(durations_seconds=durations).fit(x, y, groups=groups)
        scaled = CrossFittedPlanResponseCritic(durations_seconds=durations).fit(x, 2*y, groups=groups)
        np.testing.assert_allclose(scaled.predict_rates(x), 2*fitted.predict_rates(x), atol=1e-12)
        baseline = PlanResponseCritic(durations_seconds=durations).fit(x, y)
        np.testing.assert_equal(fitted.linear.fitted["weights"], baseline.fitted["weights"])
        self.assertEqual(fitted.fitted["kernel_width_squared"], 4)
        self.assertEqual(fitted.fitted["parameter_count"], 25+60+5)

    def test_two_path_constant_design_and_required_fit(self):
        critic = CrossFittedPlanResponseCritic(durations_seconds=[1., 2.])
        with self.assertRaisesRegex(RuntimeError, "has not been fitted"):
            critic.predict_rates(np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "two training paths"):
            critic.fit(np.ones((2, 3)), np.ones((2, 2)), groups=[1, 1])
        critic.fit(np.ones((2, 3)), np.array([[1., 2.], [2., 4.]]), groups=[1, 2])
        self.assertTrue(np.isfinite(critic.predict_rates(np.ones((3, 3)))).all())


class CrossFitResponseTest(unittest.TestCase):
    def test_fit_only_cache_and_query_labels_do_not_change_models(self):
        rows = [{"seed":i+1, "check_step":100, "feature_names":NAMES} for i in range(12)]
        x = measured_sequences(rows)
        rng = np.random.default_rng(32)
        examples = [{**row, "sequence":seq, "candidate_plan":rng.normal(size=2), "curve":rng.normal(size=5)}
                    for row, seq in zip(rows, x)]
        models = motion_models()
        train = {"seeds":np.array([r["seed"] for r in examples[:8]]), "curve":np.stack([r["curve"] for r in examples[:8]])}
        designs = {m:response.design(examples[:8], models, method=m, root=32) for m in response.METHODS}
        before = deepcopy({m:v.fitted for m,v in models.items()})
        predicted, fits, raw, _ = crossfit.fit_response(train, examples[8:], designs, models, root=32)
        changed = deepcopy(examples[8:])
        for row in changed:
            row["curve"] *= 1000
        other, other_fits, other_raw, _ = crossfit.fit_response(train, changed, designs, models, root=32)
        for method in crossfit.METHODS:
            np.testing.assert_equal(predicted[method], other[method])
        for method in response.METHODS:
            np.testing.assert_equal(raw[method]["dual_weights"], other_raw[method]["dual_weights"])
            np.testing.assert_equal(before.get(method, {}).get("weights", []),
                                    models[method].fitted["weights"] if method in models else [])
            self.assertNotIn("dual_weights", fits["crossfit_"+method])
            self.assertEqual(fits["crossfit_"+method]["parameter_count"],
                             160+45 if method.startswith("raw_") else 250+45)
        with self.assertRaisesRegex(ValueError, "paths overlap"):
            crossfit.fit_response(train, examples[:1], designs, models, root=32)

    def test_cached_stage31_training_is_reused_without_query_arrays(self):
        with TemporaryDirectory() as temporary:
            args, _, _ = fake_stage31(Path(temporary))
            train, designs, prior, path = crossfit.load_training(args, selected_iteration=2)
            self.assertEqual(train["curve"].shape, (2, 5))
            self.assertEqual(set(train["seeds"]), set(prior["fit_paths"]))
            with np.load(path, allow_pickle=False) as original:
                arrays = {k:original[k] for k in original.files if k.startswith("train_") or k.endswith("_train_design")}
            np.savez_compressed(path, **arrays)
            other, other_designs, _, _ = crossfit.load_training(args, selected_iteration=2)
            np.testing.assert_equal(train["curve"], other["curve"])
            np.testing.assert_equal(designs["history"], other_designs["history"])

    def test_fresh_full_history_equal_call_roster_and_budget(self):
        for root in (208001, 209011, 209061):
            args = arguments(root)
            preflight = root == 208001
            paths = crossfit.evaluation_paths(root, preflight=preflight)
            old = [s for module in (temporal, hold, state) for s in module.path_roles(root, preflight=preflight).values()]
            old += [motion.evaluation_paths(root, preflight=preflight), response.evaluation_paths(root, preflight=preflight)]
            hold.validate_paths(args, {"fit":[], "evaluation":paths}, {"temporal_seed_roles":dict(enumerate(old))})
            cases = response.query_cases(root, paths, horizon=args.horizon, pairs_per_path=1 if preflight else 15)
            self.assertEqual(len(cases), 2 if preflight else 120)
            for case in cases:
                self.assertGreaterEqual(case["check_step"], 64)
                renew, keep = hold.pair_schedules(case["seed"], case["check_step"], args.horizon)
                self.assertEqual(renew[:-1], keep[:-1])
                self.assertEqual((renew[-1], keep[-1]), (case["check_step"], case["check_step"]+100))

    def test_hand_computed_gates_include_original_linear_history(self):
        rows = [{"curve":np.array(hold.HORIZONS)*.01*v, "predicted_rates":{
                 m:np.full(5, v if m == crossfit.PRIMARY else 0.) for m in crossfit.METHODS}} for v in (1., -1.)]
        metrics = crossfit.summarize(rows)
        self.assertTrue(metrics["prediction_gate_passed"])
        self.assertTrue(metrics["decision_gate_passed"])
        self.assertEqual(len(metrics["settled_ise_benefit_vs_control"]), 15)
        rows[0]["predicted_rates"]["linear_history"][-1] = 1.
        self.assertFalse(crossfit.summarize(rows)["decision_gate_passed"])

    def test_fake_end_to_end_preserves_lower_feedback_and_full_accounting(self):
        with TemporaryDirectory() as temporary:
            args, loaded, bounds = fake_stage31(Path(temporary))
            with patch.object(hold, "load_controller", return_value=loaded), \
                    patch.object(crossfit, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(hold, "_make_task", side_effect=lambda **kwargs:FakeTask()), \
                    patch.object(crossfit, "_make_task", side_effect=lambda **kwargs:FakeTask()), \
                    patch.object(hold, "pointmaze_goal_bounds", return_value=bounds), \
                    patch.object(crossfit, "pointmaze_goal_bounds", return_value=bounds):
                cell = crossfit.run_cell(args)
            self.assertEqual((cell["training_pairs"], cell["evaluation_pairs"]), (2, 2))
            self.assertEqual((cell["linear_fits"], cell["kernel_solves"], cell["critic_fits"], cell["scalar_rhs_count"]), (21, 7, 28, 140))
            self.assertEqual((cell["fresh_pair_primitive_steps"], cell["candidate_proposal_inference_calls"]), (1030, 2))
            self.assertEqual(cell["reused_training_candidate_proposals"], 2)
            for key in ("controller_updates", "motion_updates", "physical_model_updates", "controller_reconstruction_primitive_steps"):
                self.assertEqual(cell[key], 0)
            for row in cell["rows"]:
                self.assertEqual(row["upper_calls_at_horizons"]["keep"][-1], row["upper_calls_at_horizons"]["renew"][-1])

    def test_scheduler_dynamic_pool_and_remote_model_storage(self):
        for preflight in (True, False):
            task = task_specification("unit_crossfit_response", spec.roots(preflight=preflight)[0],
                                      preflight=preflight, protocol_spec=spec)
            self.assertEqual((task["cpu"], task["ram_mb"]), (2, 3072) if preflight else (17, 24576))
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node00{i}" for i in range(1, 7)])
            self.assertFalse(any("_raw" in path for path in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
