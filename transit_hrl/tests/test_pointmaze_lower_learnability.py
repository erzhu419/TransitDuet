import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_lower_learnability as experiment
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_lower_learnability_stage51_spec as spec
from scripts.submit_pointmaze_lower_learnability_stage51_scheduleurm import task_specification
from test_pointmaze_state_baseline import StateBaselineTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class LowerLearnabilityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feedback_gain_and_causal_feature_selection(self):
        gain = experiment.feedback_gain()
        np.testing.assert_allclose(gain[:, :2], 5 * np.eye(2), atol=1e-12, rtol=0)
        np.testing.assert_allclose(gain[:, 2:], np.sqrt(11) * np.eye(2), atol=1e-12, rtol=0)
        a, b = np.zeros((4, 4)), np.vstack((np.zeros((2, 2)), np.eye(2)))
        a[:2, 2:] = np.eye(2)
        self.assertTrue(np.all(np.linalg.eigvals(a - b @ gain).real < 0))
        x = torch.zeros((2, 390))
        x[:, :2], x[:, 2:4], x[:, 4:6] = .1, .02, .05
        x[:, -6:-4], x[:, -4:-2] = .2, .03
        k = torch.as_tensor(gain, dtype=x.dtype)
        for goal, error in (("waypoint", .05), ("task", .1)):
            observed = experiment.feedback_mean(x, k, goal)
            torch.testing.assert_close(observed.tanh(), torch.full((2, 2), 5 * error - np.sqrt(11) * .02 - .03))
            changed = x.clone()
            changed[:, 6:-6], changed[:, -2:] = 123., 456.
            torch.testing.assert_close(experiment.feedback_mean(changed, k, goal), observed, atol=0, rtol=0)
        x[:, 4:6] = 100.
        torch.testing.assert_close(experiment.feedback_mean(x, k, "waypoint").tanh(), torch.full((2, 2), .95))

    def test_feedback_actor_preserves_gaussian_sampling_and_source_std(self):
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False))
        teacher = experiment.FeedbackActor(model.lower_actor, experiment.feedback_gain(), "task")
        x = torch.zeros((1, 390))
        torch.testing.assert_close(teacher.distribution(x).stddev, model.lower_actor.distribution(x).stddev)
        torch.manual_seed(51)
        action, _, mean = teacher.forward_with_mean(x, sample=True)
        torch.manual_seed(51)
        expected = teacher.distribution(x).rsample()
        torch.testing.assert_close(action, expected, atol=0, rtol=0)
        torch.testing.assert_close(mean, teacher.distribution(x).mean, atol=0, rtol=0)

    def test_cloning_changes_only_mean_and_equal_sham_budget(self):
        torch.manual_seed(51)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=2, lower_state_dim=2,
            upper_action_dim=1, lower_action_dim=1, hidden_dim=8, lower_cost_critic=False))
        original = copy.deepcopy(model.state_dict())
        x = np.random.default_rng(51).normal(size=(48, 2)).astype(np.float32)
        labels = (.3 * x[:, :1] - .1 * x[:, 1:]).astype(np.float32)
        clone, fit = experiment.clone_mean(model, x, labels, root=310001, goal="task", epochs=32, sham=False)
        sham, control = experiment.clone_mean(model, x, labels, root=310001, goal="task", epochs=32, sham=True)
        self.assertLess(fit["final_label_mse"], fit["initial_label_mse"])
        self.assertEqual(fit["supervised_optimizer_steps"], control["supervised_optimizer_steps"])
        self.assertEqual(fit["shuffle_seed"], control["shuffle_seed"])
        self.assertNotEqual(fit["final_label_mse"], control["final_label_mse"])
        self.assertEqual(model.state_dict()["config"], original["config"])
        torch.testing.assert_close({k: v for k, v in model.state_dict().items() if k != "config"},
                                   {k: v for k, v in original.items() if k != "config"}, atol=0, rtol=0)
        for fitted in (clone, sham):
            for key in original:
                if key not in ("lower_actor", "config"):
                    torch.testing.assert_close(fitted.state_dict()[key], original[key], atol=0, rtol=0)
            torch.testing.assert_close(fitted.lower_actor.log_std, model.lower_actor.log_std, atol=0, rtol=0)

    def test_raw_score_gradient_and_negative_alignment(self):
        model, batch, rewards = StateBaselineTest().data()
        weight = experiment.score_targets(rewards).reshape(-1)
        actor = copy.deepcopy(model.lower_actor).double()
        logp, _ = actor.log_prob_entropy(torch.from_numpy(batch.state).double(), torch.from_numpy(batch.action).double())
        objective = (logp * torch.as_tensor(weight)).sum() / len(rewards)
        expected = torch.cat([g.reshape(-1) for g in torch.autograd.grad(objective, tuple(actor.parameters()))]).numpy()
        actual = experiment.score_gradient(model.lower_actor, batch, weight, scale=batch.size / len(rewards))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        self.assertAlmostEqual(experiment.alignment(actual, -actual)["cosine"], -1.)
        self.assertEqual(experiment.alignment(actual, np.zeros_like(actual))["cosine"], 0.)

    def test_native_style_pipeline_costs_worker_restoration_and_rosters(self):
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=390, lower_state_dim=390,
            upper_action_dim=2, lower_action_dim=2, hidden_dim=8, lower_cost_critic=False,
            lower_value_state_dim=392, promotion_state_dim=394))
        original = copy.deepcopy(controller.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file = directory / "source.json"
            source_file.write_text(json.dumps({"snapshots": {"2": {"checkpoint": "fixture.pt"}}}))
            with patch.object(spec, "source_result", return_value=source_file), \
                    patch.object(experiment, "load_pair", return_value=[controller]), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                result = experiment.train(310001, preflight=True, output=directory / "run/result.json")
                summary = experiment.aggregate([result], preflight=True)
            self.assertEqual(summary["status"], "preflight_passed")
            self.assertEqual(summary["method_cost"]["primitive_steps"], 9600)
            self.assertEqual(summary["native_trace_audits"], 32)
            self.assertEqual(summary["computation"]["supervised_optimizer_steps"], 16)
            self.assertEqual(summary["computation"]["score_backward_calls"], 5)
            self.assertNotIn("primary_endpoints", summary)
            self.assertIsInstance(joint._WORKER[0].lower_actor, type(controller.lower_actor))
            torch.testing.assert_close({k: v for k, v in controller.state_dict().items() if k != "config"},
                                       {k: v for k, v in original.items() if k != "config"}, atol=0, rtol=0)
            bad = copy.deepcopy(result)
            bad["evaluation_rows"].pop("sham_task")
            with self.assertRaisesRegex(ValueError, "roster incomplete"):
                experiment.aggregate([bad], preflight=True)
            bad = copy.deepcopy(result)
            bad["batch_rows"]["B"][0]["seed"] = result["batch_rows"]["A"][0]["seed"]
            with self.assertRaisesRegex(ValueError, "path roster"):
                experiment.aggregate([bad], preflight=True)
            # Expand compact fixture rows only; no additional simulated episodes.
            cells = []
            for root in spec.roots(preflight=False):
                cell = copy.deepcopy(result)
                opt, roles = spec.options(preflight=False), spec.seed_roles(root, preflight=False)
                cell.update(root=root, preflight=False, options=opt, seed_roles=roles,
                            budget=spec.budget(preflight=False), source_checkpoint_iteration=16,
                            native_trace_audits=240)
                for policy, fit in cell["training"].items():
                    goal, sham = policy.split("_")[1], policy.startswith("sham_")
                    fit.update(supervised_optimizer_steps=320, shuffle_seed=spec.shuffle_seed(root, goal),
                               label_seed=spec.label_seed(root, goal) if sham else None)
                cell["computation"].update(supervised_optimizer_steps=1280, teacher_label_states=19200)

                def rows(template, phase, mode, seeds, value=None):
                    output = []
                    for seed in seeds:
                        row = copy.deepcopy(template)
                        row.update(seed=seed, episode_length=1200, lower_inference_calls=1200,
                                   policy_seed=spec.policy_seed(root, seed), deployment_mode=mode)
                        row.update({k: v for k, v in spec.rollout_arguments(root, seed, phase=phase, mode=mode).items() if k != "sample"})
                        if value is not None:
                            row["episode_return"] = value
                        output.append(row)
                    return output

                for phase in ("A", "B"):
                    cell["batch_rows"][phase] = rows(result["batch_rows"][phase][0], phase, "fitting", roles[phase])
                for policy in spec.POLICIES:
                    for mode in spec.MODES:
                        cell["evaluation_rows"][policy][mode] = rows(result["evaluation_rows"][policy][mode][0],
                            "eval", mode, roles["evaluation"], value=100. if policy == "frozen" else 90.)
                for phase in ("A", "B", "eval"):
                    values = cell["batch_rows"][phase] if phase != "eval" else [r for p in cell["evaluation_rows"].values() for v in p.values() for r in v]
                    cell["inference_counts"][phase] = {k: sum(1200 if k == "primitive_steps" else r[k] for r in values)
                                                       for k in result["inference_counts"][phase]}
                cell["credit"]["mc"]["cosine"], cell["credit"]["gae"]["cosine"] = -.2, .3
                cells.append(cell)
            full = experiment.aggregate(cells, preflight=False)
            self.assertEqual(full["method_cost"]["primitive_steps"], 2304000)
            self.assertEqual(full["computation"]["supervised_optimizer_steps"], 10240)
            self.assertEqual(full["primary_endpoints"]["teacher_waypoint_minus_frozen"]["effect"], "negative")
            self.assertEqual(full["primary_endpoints"]["mc_independent_cosine"]["effect"], "negative")
            self.assertEqual(full["primary_endpoints"]["gae_independent_cosine"]["effect"], "positive")

    def test_fresh_paths_budget_dynamic_placement_and_endpoint_signs(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = [s for v in spec.seed_roles(root, preflight=preflight).values() for s in v]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(set(seeds) & seen)
                self.assertFalse(set(seeds) & {s for v in spec.previous.seed_roles(root, preflight=preflight).values() for s in v})
                seen.update(seeds)
            task = task_specification("unit_stage51", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertNotIn("result_dir", task)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
        self.assertEqual(spec.budget(preflight=True)["total_primitive_steps"], 9600)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 2304000)
        means = {"deterministic": {p: {"episode_return": v} for p, v in zip(spec.POLICIES, (10, 9, 13, 8, 7, 16, 14))}}
        endpoints = spec.contrasts(means, {"mc": {"cosine": -.2}, "gae": {"cosine": .3}})
        self.assertEqual(list(endpoints), list(spec.ENDPOINTS))
        self.assertEqual(list(endpoints.values()), [-1, 3, -2, 6, 1, 2, -.2, .3])


if __name__ == "__main__":
    unittest.main()
