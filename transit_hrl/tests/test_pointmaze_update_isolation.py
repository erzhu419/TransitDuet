import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_update_isolation as isolation
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_joint_renewal_stage35_spec as previous
from scripts import pointmaze_update_isolation_stage36_spec as spec
from scripts.submit_pointmaze_update_isolation_stage36_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask


class ImmediatePool:
    def __init__(self, *, initializer, initargs, **kwargs):
        initializer(*initargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, function, jobs):
        return map(function, jobs)


class UpdateIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(36)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, upper_learning_rate=3e-4, lower_learning_rate=3e-4))

    def batch(self, model):
        args = spec.source.arguments(310001, preflight=True)
        with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            return joint.rollout(model, args, "learned_history", seed=1, sample=True)[0]

    def test_freezes_actor_and_critic_exactly_for_every_treatment(self):
        source = self.controller()
        for method in spec.METHODS:
            model = joint.make_model(source, spec.native_method(method), root=310001)
            batch = self.batch(model) if method != "fixed50" else self.batch(joint.make_model(source, "learned_history", root=310001))
            before = joint.inference_weights(model)
            metrics = isolation.update_components(model, batch, spec.COMPONENTS[method])
            delta = isolation.change_norms(joint.inference_weights(model), before)
            for name, value in delta.items():
                level = name.rsplit("_", 1)[0]
                if level in spec.COMPONENTS[method]:
                    self.assertGreater(value, 0, (method, name))
                    self.assertGreater(metrics[name + "_optimizer_steps"], 0)
                else:
                    self.assertEqual(value, 0, (method, name))
                    self.assertNotIn(name + "_optimizer_steps", metrics)
                    self.assertFalse(getattr(model, name + "_optimizer").state)

    def test_joint_update_matches_existing_unconstrained_ppo(self):
        model = joint.make_model(self.controller(), "learned_history", root=310001)
        reference = joint.make_model(self.controller(), "learned_history", root=310001)
        reference.load_state_dict(copy.deepcopy(model.state_dict()))
        batch = self.batch(model)
        torch.manual_seed(361)
        np.random.seed(361)
        expected = reference.update(batch)
        torch.manual_seed(361)
        np.random.seed(361)
        observed = isolation.update_components(model, batch, spec.COMPONENTS["joint"])
        for name, state in joint.inference_weights(reference).items():
            for key, value in state.items():
                self.assertTrue(torch.equal(value, joint.inference_weights(model)[name][key]), (name, key))
        for key in observed:
            self.assertEqual(observed[key], expected[key])

    def test_fresh_paths_two_cohorts_and_equal_primitive_budgets(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = [seed for values in roles.values() for seed in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                self.assertFalse(set(seeds).intersection(s for values in previous.seed_roles(root, preflight=preflight).values() for s in values))
        self.assertEqual(spec.budget(preflight=False)["total_primitive_steps"] * 40, 54192000)
        self.assertEqual(spec.contract()["primary_cohort"], "final")
        self.assertEqual(spec.contract()["upper_call_cost_in_reward_units"], 1.)

    def test_scheduler_pool_and_source_only_staging(self):
        task = task_specification("unit_stage36", 310011, "gate_only", preflight=False)
        self.assertEqual(task["cpu"], 9)
        self.assertEqual(task["ram_mb"], 12288)
        self.assertEqual(len(task["allowed_nodes"]), 6)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])
        self.assertIn(spec.RUNNER_SCRIPT, task["cmd"])
        self.assertNotIn(previous.RUNNER_SCRIPT, task["cmd"])

    def test_train_keeps_two_cohorts_and_audits_actual_frozen_weights(self):
        args = spec.source.arguments(310001, preflight=True)
        cell = {"selected_checkpoint_iteration": 0, "factual_row": {"decision_steps": list(range(0, args.horizon, 50))}}
        with tempfile.TemporaryDirectory() as directory, patch.object(isolation, "load_controller", return_value=(self.controller(), cell, {})), \
                patch.object(isolation, "ProcessPoolExecutor", ImmediatePool), \
                patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            result = isolation.train(310001, "gate_only", preflight=True, output=Path(directory) / "cell" / "result.json")
            self.assertEqual(set(result["evaluation_rows"]), {"final", "selected"})
            audit = isolation.audit_result(result, raw_path=Path(directory) / "cell_raw")
            self.assertEqual(audit["episodes"], 4)
            for cohort in spec.COHORTS:
                self.assertEqual([r["seed"] for r in result["evaluation_rows"][cohort]], result["seed_roles"]["evaluation"])
            final = torch.load(result["final_checkpoint"], map_location="cpu", weights_only=False)
            self.assertEqual(final["iteration"], 2)
            changed = copy.deepcopy(result)
            changed["trained_parameter_change_norms"]["upper_value"] = .1
            with self.assertRaisesRegex(ValueError, "freeze/update contract"):
                isolation.audit_result(changed, raw_path=Path(directory) / "cell_raw")
            changed = copy.deepcopy(result)
            changed["inference_counts"]["final"]["lower_inference_calls"] -= 1
            with self.assertRaisesRegex(ValueError, "primitive accounting"):
                isolation.audit_result(changed, raw_path=Path(directory) / "cell_raw")

    def test_final_factorial_interaction_and_negative_results_are_retained(self):
        results = []
        for root in spec.OPTIMIZER_ROOTS:
            for method, value in zip(spec.METHODS, (100., 110., 80., 70., 105.)):
                row = {"episode_return": value, "tracking_squared_error_integral": 2.,
                       "upper_inference_calls": 24., "charged_utility": value - 24.}
                results.append({"root": root, "method": method,
                                "evaluation_rows": {"final": [row], "selected": [{**row, "episode_return": 200.}]}})
        summary = isolation.aggregate(results)
        endpoints = summary["primary_endpoints"]
        self.assertEqual(endpoints["gate_only:frozen:return"]["effect"], "positive")
        self.assertEqual(endpoints["controller_only:frozen:return"]["effect"], "negative")
        self.assertEqual(endpoints["interaction:return"]["mean"], -20.)
        self.assertEqual(endpoints["joint:fixed50:return"]["mean"], -35.)
        self.assertEqual(endpoints["joint:fixed50:ise"]["effect"], "inconclusive")
        self.assertEqual(summary["means"]["selected"]["joint"]["episode_return"], 200.)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            isolation.aggregate(results[:-1])


if __name__ == "__main__":
    unittest.main()
