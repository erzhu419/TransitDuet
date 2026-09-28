import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_actor_acceptance as experiment
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, LevelTrajectoryBatch, SMDPPPOConfig
from scripts import pointmaze_actor_acceptance_stage45_spec as spec
from scripts import pointmaze_signed_microstep_stage44_spec as previous
from scripts.submit_pointmaze_actor_acceptance_stage45_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class ActorAcceptanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(45)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1, lower_state_dim=1, upper_action_dim=1, lower_action_dim=1,
            hidden_dim=0, lower_cost_critic=False, init_log_std=0.))
        with torch.no_grad():
            model.lower_actor.net[0].weight.zero_()
            model.lower_actor.net[0].bias.zero_()
        for optimizer in (model.lower_actor_optimizer, model.lower_value_optimizer):
            optimizer.param_groups[0]["lr"] = .1
        return model

    def batch(self, model):
        state, action = np.ones((4, 1), dtype=np.float32), np.array([[-1.], [-1.], [1.], [1.]], dtype=np.float32)
        with torch.no_grad():
            logp, _ = model.lower_actor.log_prob_entropy(torch.from_numpy(state), torch.from_numpy(action))
        return LevelTrajectoryBatch(state=state, action=action, old_logp=logp.numpy(), old_value=np.zeros(4, dtype=np.float32),
            reward=np.array([-1., -1., 1., 1.], dtype=np.float32), done=np.ones(4, dtype=np.float32), duration=np.ones(4, dtype=np.int64))

    def forced_update(self, sign):
        def update(model, batch, kind):
            self.assertEqual(kind, "actor_critic")
            model.lower_actor_optimizer.zero_grad()
            (sign * model.lower_actor.net[0].bias.sum()).backward()
            model.lower_actor_optimizer.step()
            model.lower_value_optimizer.zero_grad()
            model.lower_value.net[0].bias.sum().backward()
            model.lower_value_optimizer.step()
            return {"lower_actor_optimizer_steps": 1, "lower_value_optimizer_steps": 1}
        return update

    def test_rejection_restores_nonempty_adam_and_actor_but_keeps_critic_update(self):
        model = self.model()
        model.lower_actor_optimizer.zero_grad()
        model.lower_actor.net[0].bias.sum().backward()
        model.lower_actor_optimizer.step()
        actor, adam = copy.deepcopy(model.lower_actor.state_dict()), copy.deepcopy(model.lower_actor_optimizer.state_dict())
        value = copy.deepcopy(model.lower_value.state_dict())
        with patch.object(experiment, "lower_update", side_effect=self.forced_update(1.)):
            result = experiment.actor_update(model, self.batch(model), "accepted")
        self.assertFalse(result["accepted"])
        self.assertTrue(result["objective_drop"])
        self.assertEqual(result["objective_deployed"], result["objective_before"])
        self.assertEqual((result["attempted_actor_steps"], result["retained_actor_steps"], result["value_steps"]), (1, 0, 1))
        torch.testing.assert_close(model.lower_actor.state_dict(), actor, atol=0, rtol=0)
        torch.testing.assert_close(model.lower_actor_optimizer.state_dict(), adam, atol=0, rtol=0)
        self.assertFalse(torch.equal(model.lower_value.state_dict()["net.0.bias"], value["net.0.bias"]))
        self.assertTrue(model.lower_value_optimizer.state)

    def test_good_update_is_kept_and_vanilla_keeps_objective_decrease(self):
        for treatment, sign in (("accepted", -1.), ("vanilla", 1.)):
            model = self.model()
            original = copy.deepcopy(model.lower_actor.state_dict())
            with patch.object(experiment, "lower_update", side_effect=self.forced_update(sign)):
                result = experiment.actor_update(model, self.batch(model), treatment)
            self.assertTrue(result["accepted"])
            self.assertEqual(result["objective_deployed"], result["objective_tentative"])
            self.assertEqual(result["retained_actor_steps"], 1)
            self.assertEqual(result["objective_drop"], treatment == "vanilla")
            self.assertFalse(torch.equal(model.lower_actor.state_dict()["net.0.bias"], original["net.0.bias"]))
            self.assertTrue(model.lower_actor_optimizer.state)

    def test_first_batch_pair_compares_training_arrays_not_summaries(self):
        batch = self.batch(self.model())
        self.assertEqual(experiment.first_batch_pair(batch, copy.deepcopy(batch))["transitions"], 4)
        changed = copy.deepcopy(batch)
        changed.old_value[0] += .1
        with self.assertRaisesRegex(AssertionError, "old_value differs"):
            experiment.first_batch_pair(batch, changed)

    def test_existing_checkpoint_native_style_training_pipeline(self):
        original = spec.source.source
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=8, lower_cost_critic=False, epochs=1, minibatch_size=128))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, source_checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, source_checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(source_checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_file.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec.source, "ROOT", directory), patch.object(original, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                for method in spec.METHODS:
                    calibration.train(310001, method, preflight=True, output=spec.source_result(310001, method, preflight=True),
                                      specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 25200)
                self.assertEqual(result["native_trace_audits"], 84)
                self.assertEqual(result["seed_roles"]["training"], [10990001, 10990002])
                self.assertEqual(result["seed_roles"]["evaluation"], [10993001, 10993002])
                self.assertEqual(set(result["first_batch_pairs"]), set(spec.METHODS))
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["method_cost"]["primitive_steps"], 25200)
                self.assertEqual(summary["optimizer_steps"]["attempted_actor_steps"], 48)
                self.assertNotIn("primary_endpoints", summary)

    def test_fresh_streams_full_roster_budget_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for values in roles.values() for s in values}
                self.assertEqual(len(new), sum(len(v) for v in roles.values()))
                for prior in (previous, spec.source, spec.source.source):
                    old = {s for values in prior.seed_roles(root, preflight=preflight).values() for s in values}
                    self.assertFalse(new.intersection(old))
                self.assertFalse(new.intersection(seen))
                seen.update(new)
        self.assertEqual(spec.CI_FAMILY_SIZE, 16)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 15052800)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 12544)
        for preflight in (True, False):
            task = task_specification("unit_stage45", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)
            self.assertIn("Training complete: result.json written", task["cmd"])

    def test_registered_statistics_and_accounting_reject_corrupt_decisions(self):
        results = []
        opt, budget = spec.options(preflight=False), spec.budget(preflight=False)
        for index, root in enumerate(spec.roots(preflight=False)):
            roles = spec.seed_roles(root, preflight=False)
            training = {}
            for method in spec.METHODS:
                training[method] = {}
                for treatment in spec.TREATMENTS:
                    history = training[method][treatment] = []
                    for iteration in range(1, opt["learning_iterations"] + 1):
                        begin = (iteration - 1) * opt["rollouts_per_iteration"]
                        seeds = roles["training"][begin:begin + opt["rollouts_per_iteration"]]
                        reject = treatment == "accepted" and iteration == 2
                        history.append({"iteration": iteration, "primitive_steps": 9600, "accepted": not reject,
                            "objective_before": 1., "objective_tentative": 0. if reject else 2.,
                            "objective_deployed": 1. if reject else 2., "objective_drop": reject,
                            "attempted_actor_steps": 2, "retained_actor_steps": 0 if reject else 2, "value_steps": 2,
                            "policy_drift": {"gaussian_kl": 0. if reject else .01},
                            "inference_counts": {"upper_inference_calls": 8, "lower_inference_calls": 9600, "gate_inference_calls": 8},
                            "rollout_sampling": [{"seed": seed, "policy_seed": spec.policy_seed(root, seed),
                                **{k: v for k, v in spec.rollout_arguments(root, seed, phase="train", mode="training").items() if k != "sample"}} for seed in seeds]})
            evaluation = {}
            for policy in spec.POLICIES:
                value = 100. if policy == "frozen" else 101. + index if policy.endswith(":accepted") else 95. - index
                evaluation[policy] = {str(i): {mode: [{**{k: value for k in spec.METRICS}, "episode_length": 1200,
                    "upper_inference_calls": 1, "lower_inference_calls": 1200, "gate_inference_calls": 1,
                    "seed": seed, "policy_seed": spec.policy_seed(root, seed), "deployment_mode": mode,
                    **{k: v for k, v in spec.rollout_arguments(root, seed, phase="eval", mode=mode).items() if k != "sample"}}
                    for seed in roles["evaluation"]] for mode in spec.MODES}
                    for i in ((0,) if policy == "frozen" else spec.snapshots(preflight=False))}
            results.append({"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(),
                "root": root, "preflight": False, "options": opt, "seed_roles": roles, "budget": budget,
                "training": training, "evaluation_rows": evaluation, "fixed_upper_gate_networks": "passed",
                "first_batch_pairs": {m: {"status": "passed", "transitions": 9600} for m in spec.METHODS},
                "expected_steps_per_update": {m: 2 for m in spec.METHODS}, "native_trace_audits": budget["native_trace_audits"],
                "inference_counts": {phase: {"primitive_steps": budget[key], "lower_inference_calls": budget[key],
                    "upper_inference_calls": budget[key] // 1200, "gate_inference_calls": budget[key] // 1200}
                    for phase, key in (("train", "training_primitive_steps"), ("eval", "evaluation_primitive_steps"))}})
        summary = experiment.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 16})
        for method in spec.METHODS:
            self.assertAlmostEqual(summary["primary_endpoints"][method + ":accepted_vanilla_final"]["mean"], 13.)
            self.assertEqual(summary["primary_endpoints"][method + ":accepted_frozen_final"]["effect"], "positive")
            self.assertEqual(summary["primary_endpoints"][method + ":vanilla_frozen_final"]["effect"], "negative")
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            experiment.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"][spec.METHODS[0]]["accepted"][1]["retained_actor_steps"] = 2
        with self.assertRaisesRegex(ValueError, "decision or optimizer accounting"):
            experiment.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["evaluation_rows"]["frozen"]["0"]["lower_sampled"][0]["lower_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sampling changed"):
            experiment.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
