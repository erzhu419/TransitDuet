from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import pointmaze_response_deployment as run
from freq_hrl.experiments import pointmaze_forecast_response as response
from freq_hrl.experiments.pointmaze_goal_validation import _training_seed
from scripts import pointmaze_response_deployment_stage34_spec as spec
from scripts import pointmaze_root_response_stage33_spec as source
from scripts import submit_pointmaze_response_deployment_stage34_scheduleurm as submit
from test_pointmaze_forecast_response import motion_models
from test_pointmaze_temporal_plan import FakeController, FakeTask


class ConstantCritic:
    def __init__(self, rate):
        self.rate = rate

    def predict_rates(self, design):
        return np.full((len(design), 5), self.rate)


class MovingProposal(FakeController):
    def plan_goal(self, state, sample):
        return {"action": np.array([.2 + state[0] * .05, -.1])}


def toy_episode(method, *, rate=1., horizon=300, future_shift=0.):
    args = source.arguments(310001, preflight=True)
    args.horizon = horizon
    models = motion_models()
    critics = {m: ConstantCritic(rate) for m in response.METHODS}
    with patch.object(run, "_make_task", return_value=FakeTask(future_shift)), \
            patch.object(run, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
        return run.rollout(MovingProposal(), models, critics, args=args, seed=4590001, method=method)


def toy_result(root):
    rows = []
    for seed in spec.evaluation_paths(root, preflight=False):
        for method in spec.METHODS:
            value = 1. if method == "history" else 2.
            rows.append({"seed": seed, "method": method, "episode_length": 1200,
                         "tracking_squared_error_integral": value, "episode_return": 3. - value,
                         "upper_inference_calls": 10, "executed_plan_count": 10,
                         "discarded_preview_calls": 0, "upper_inference_seconds": .1,
                         "response_inference_seconds": .1, "wall_seconds": 1.})
    return {"status": "complete", "protocol": {"protocol_version": spec.EXPERIMENT_PROTOCOL,
            "optimizer_seed": root, "preflight": False, "contract": spec.contract()},
            "cells": [{"budget": spec.budget(root, preflight=False), "rows": rows}]}


class ResponseDeploymentTest(unittest.TestCase):
    def test_new_paths_and_exact_budget(self):
        inherited, new = set(), set()
        for preflight in (True, False):
            for root in source.roots(preflight=preflight):
                args = source.arguments(root, preflight=preflight)
                inherited.update(s for paths in source.seed_roles(root, preflight=preflight).values() for s in paths)
                inherited.update(_training_seed(optimizer_seed=root, rollout_root=s, iteration=i)
                                 for s in args.train_seeds for i in range(args.iterations))
                paths = spec.evaluation_paths(root, preflight=preflight)
                self.assertFalse(new.intersection(paths))
                new.update(paths)
                self.assertEqual(len(paths), 2 if preflight else 32)
                self.assertEqual(spec.budget(root, preflight=preflight)["total_primitive_steps"], 6300 if preflight else 385200)
        self.assertFalse(new.intersection(inherited))
        self.assertEqual(spec.checks(1200), (100, 250, 400, 550, 700, 850, 1000))
        self.assertEqual(sum(spec.budget(r, preflight=False)["total_primitive_steps"] for r in spec.OPTIMIZER_ROOTS), 3081600)

    def test_immediate_renewal_reuses_paid_preview(self):
        history, arrays, names = toy_episode("history")
        always, other, _ = toy_episode("always_renew")
        self.assertEqual(history["upper_inference_calls"], history["executed_plan_count"])
        self.assertEqual(history["executed_plan_count"], 4)
        self.assertEqual(history["candidate_preview_calls"], 1)
        self.assertEqual(always["candidate_preview_calls"], 0)
        self.assertEqual(history["executed_plan_steps"], always["executed_plan_steps"])
        np.testing.assert_array_equal(arrays["subgoal"], other["subgoal"])
        self.assertEqual(history["lower_inference_calls"], 300)
        self.assertEqual(history["motion_inference_calls"], 1)
        self.assertEqual(arrays["sequence"].shape, (1, 64, len(names)))
        self.assertGreaterEqual(history["upper_inference_seconds"], 0)

    def test_keep_pays_discarded_preview_and_refreshes_delayed_plan(self):
        history, arrays, _ = toy_episode("history", rate=0.)
        always, other, _ = toy_episode("always_keep")
        self.assertFalse(history["decisions"][0]["renew"])
        self.assertEqual(history["upper_inference_calls"], history["executed_plan_count"] + 1)
        self.assertEqual(history["upper_inference_calls"], always["upper_inference_calls"] + 1)
        self.assertEqual(history["discarded_preview_calls"], 1)
        self.assertIn({"step": 200, "kind": "delayed_plan"}, history["upper_calls"])
        np.testing.assert_array_equal(arrays["subgoal"], other["subgoal"])
        self.assertFalse(np.array_equal(arrays["subgoal"][200], history["decisions"][0]["candidate_plan"]))

    def test_full_episode_has_one_execution_per_block_and_an_explicit_tail(self):
        row, arrays, _ = toy_episode("history", rate=-1., horizon=1200)
        self.assertEqual(row["executed_plan_count"], 10)
        self.assertEqual(row["upper_inference_calls"], 17)
        self.assertEqual(row["discarded_preview_calls"], 7)
        for step in spec.checks(1200):
            self.assertEqual([s for s in row["executed_plan_steps"] if step <= s < step + 150], [step + 100])
        self.assertEqual(row["executed_plan_steps"][-1], 1150)
        self.assertEqual(arrays["ise"].shape, (1200,))
        self.assertAlmostEqual(row["tracking_squared_error_integral"], arrays["ise"].sum())
        self.assertAlmostEqual(row["episode_return"], arrays["reward"].sum())

    def test_decision_never_reads_future_observations(self):
        class FutureShift(FakeTask):
            def observation(self):
                observation = super().observation()
                if self.t > 100:
                    observation.target += 3.
                    observation.target_error = observation.target - observation.achieved_goal
                    observation.task_measurement[:2] = observation.target
                return observation
        original, arrays, _ = toy_episode("history")
        args = source.arguments(310001, preflight=True)
        models = motion_models()
        with patch.object(run, "_make_task", return_value=FutureShift()), \
                patch.object(run, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            changed, changed_arrays, _ = run.rollout(MovingProposal(), models, {m: ConstantCritic(1.) for m in response.METHODS},
                                                    args=args, seed=4590001, method="history")
        np.testing.assert_array_equal(arrays["sequence"], changed_arrays["sequence"])
        np.testing.assert_array_equal(original["decisions"][0]["candidate_plan"], changed["decisions"][0]["candidate_plan"])
        self.assertFalse(np.array_equal(arrays["ise"], changed_arrays["ise"]))

    def test_original_frequency_reference_does_not_pay_unused_critics(self):
        row, arrays, _ = toy_episode("fixed50")
        self.assertEqual(row["upper_inference_calls"], 6)
        self.assertEqual(row["executed_plan_steps"], list(range(0, 300, 50)))
        self.assertEqual(row["response_inference_calls"], 0)
        self.assertEqual(row["motion_inference_calls"], 0)
        self.assertEqual(row["candidate_preview_calls"], 0)

    def test_joint_gate_and_root_statistical_unit(self):
        results = [toy_result(root) for root in spec.OPTIMIZER_ROOTS]
        aggregate = run.aggregate(results)
        self.assertTrue(aggregate["deployment_gate_passed"])
        self.assertEqual(len(aggregate["endpoints"]), 16)
        self.assertEqual(aggregate["endpoints"]["ise:lag1_extrapolation"]["adjusted_ci95"], [1., 1.])
        self.assertEqual(aggregate, run.aggregate(list(reversed(results))))
        changed = deepcopy(results)
        for row in changed[0]["cells"][0]["rows"]:
            if row["method"] == "history":
                row["episode_return"] = -100.
        self.assertFalse(run.aggregate(changed)["deployment_gate_passed"])
        with self.assertRaisesRegex(ValueError, "eight-root"):
            run.aggregate(results[:-1])
        changed = deepcopy(results)
        changed[0]["cells"][0]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing episodes"):
            run.aggregate(changed)

    def test_scheduler_places_dynamically_and_never_stages_raw_or_weights(self):
        for preflight in (True, False):
            root = spec.roots(preflight=preflight)[0]
            task = submit.task_specification("unit_deployment", root, preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node00{i}" for i in range(1, 7)])
            self.assertEqual(task["cpu"], 2 if preflight else 17)
            self.assertEqual(task["ram_mb"], 3072 if preflight else 16384)
            self.assertFalse(any("_raw" in p or p.endswith(".pt") for p in task["stage_input_paths"]))
            self.assertFalse(any(p.endswith(".json") for p in task["stage_input_paths"]))


if __name__ == "__main__":
    unittest.main()
