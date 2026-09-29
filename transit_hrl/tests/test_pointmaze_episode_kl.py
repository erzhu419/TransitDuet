import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_episode_kl as experiment
from freq_hrl.experiments import pointmaze_episode_credit as episodes
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_episode_kl_stage48_spec as spec
from scripts.submit_pointmaze_episode_kl_stage48_scheduleurm import task_specification
from test_pointmaze_state_baseline import StateBaselineTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class EpisodeKLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_gaussian_episode_sum_not_per_step_mean(self):
        model, batch, rewards = StateBaselineTest().data()
        reference = copy.deepcopy(model.lower_actor)
        with torch.no_grad():
            model.lower_actor.net[-1].bias.add_(.3)
            state = torch.from_numpy(batch.state)
            old, new = reference.distribution(state), model.lower_actor.distribution(state)
            a, b, s, t = (x.numpy().astype(np.float64) for x in (old.mean, new.mean, old.stddev, new.stddev))
        exact = (np.log(t / s) + (s ** 2 + (a - b) ** 2) / (2 * t ** 2) - .5).sum(axis=1).reshape(3, 4).sum(axis=1)
        terms = experiment.policy_terms(model, batch, reference, episodes.score_targets(rewards).reshape(-1), 3)
        np.testing.assert_allclose(terms["episode_kl"], exact, atol=1e-12, rtol=0)
        self.assertGreater(terms["max_episode_kl"], spec.KL_BUDGET)
        self.assertGreater(terms["max_episode_kl"], float(exact.max()) / 4)

    def test_original_controls_identical_and_accepted_adam_matches_scaled_lr_replay(self):
        model, batch, rewards = StateBaselineTest().data()
        for treatment in ("gae", "episode_mc"):
            expected, observed = copy.deepcopy(model), copy.deepcopy(model)
            np.random.seed(spec.shuffle_seed(310001, 1))
            episodes.update(expected, batch, rewards, treatment, specification=spec)
            record = experiment.update(observed, batch, rewards, treatment, root=310001, iteration=1)
            self.assertEqual(record["selected_scale"], 1.)
            for name in experiment.LOWER_STATE:
                torch.testing.assert_close(getattr(observed, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)
        model.lower_actor_optimizer.param_groups[0]["lr"] = .2
        expected, bounded = copy.deepcopy(model), copy.deepcopy(model)
        record = experiment.update(bounded, batch, rewards, "episode_kl", root=310001, iteration=1)
        self.assertTrue(record["accepted"])
        self.assertLess(record["selected_scale"], 1.)
        self.assertLessEqual(record["deployed_terms"]["max_episode_kl"], spec.KL_BUDGET)
        expected.lower_actor_optimizer.param_groups[0]["lr"] *= record["selected_scale"]
        np.random.seed(spec.shuffle_seed(310001, 1))
        original = episodes.update(expected, batch, rewards, "episode_mc", specification=spec)
        expected.lower_actor_optimizer.param_groups[0]["lr"] = .2
        for name in experiment.LOWER_STATE:
            torch.testing.assert_close(getattr(bounded, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)
        steps = original["actor_optimizer_steps"]
        self.assertEqual(experiment.optimizer_steps(record, steps)["actor_optimizer_steps"], steps * len(record["trials"]))

    def test_all_rejected_restore_nonempty_actor_adam_but_retain_one_original_critic_update(self):
        model, batch, rewards = StateBaselineTest().data()
        np.random.seed(48)
        episodes.update(model, batch, rewards, "episode_mc", specification=spec)
        with torch.no_grad():
            batch.old_logp = model.lower_actor.log_prob_entropy(torch.from_numpy(batch.state), torch.from_numpy(batch.action))[0].numpy()
            batch.old_value = model.lower_value(torch.from_numpy(batch.value_state)).numpy()
        before = {name: copy.deepcopy(getattr(model, name).state_dict()) for name in experiment.LOWER_STATE}
        self.assertTrue(model.lower_actor_optimizer.state)
        expected = copy.deepcopy(model)
        np.random.seed(spec.shuffle_seed(310001, 2))
        original = episodes.update(expected, batch, rewards, "episode_mc", specification=spec)
        with patch.object(spec, "KL_BUDGET", 0.), patch.object(spec, "MAX_BACKTRACKS", 1):
            record = experiment.update(model, batch, rewards, "episode_kl", root=310001, iteration=2)
            self.assertFalse(record["accepted"])
            self.assertEqual(record["selected_scale"], 0.)
            self.assertEqual(record["retained_actor_steps"], 0)
            self.assertEqual(record["deployed_terms"]["max_episode_kl"], 0.)
            self.assertEqual(experiment.optimizer_steps(record, original["actor_optimizer_steps"])["value_optimizer_steps"], 2 * original["value_optimizer_steps"])
        for name in ("lower_actor", "lower_actor_optimizer"):
            torch.testing.assert_close(getattr(model, name).state_dict(), before[name], atol=0, rtol=0)
        for name in ("lower_value", "lower_value_optimizer"):
            torch.testing.assert_close(getattr(model, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)

    def test_existing_checkpoint_native_style_pipeline(self):
        original = spec.previous.previous.previous.source.source
        controller = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=8, lower_cost_critic=False, lower_learning_rate=3e-4, epochs=1, minibatch_size=128))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_file, checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_file.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec.previous.previous.previous.source, "ROOT", directory), patch.object(original, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), patch.object(episodes, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                calibration.train(310001, "task_clock", preflight=True, output=spec.source_result(310001, "task_clock", preflight=True),
                                  specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 15600)
                self.assertEqual(result["native_trace_audits"], 52)
                self.assertEqual(set(result["first_batch_pair_details"]["task_clock"]), {"episode_mc", "episode_kl"})
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["retained_optimizer_steps"], {"actor_optimizer_steps": 60, "value_optimizer_steps": 60})
                self.assertGreaterEqual(summary["optimizer_steps"]["actor_optimizer_steps"], 60)
                self.assertNotIn("primary_endpoints", summary)

    def test_fresh_seeds_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for values in roles.values() for s in values}
                self.assertEqual(len(new), sum(len(v) for v in roles.values()))
                for prior in (spec.previous, spec.previous.previous, spec.previous.previous.previous, spec.previous.previous.previous.source.source):
                    self.assertFalse(new.intersection(s for values in prior.seed_roles(root, preflight=preflight).values() for s in values))
                self.assertFalse(new.intersection(seen))
                seen.update(new)
        self.assertEqual(spec.CI_FAMILY_SIZE, 6)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 5836800)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 4864)
        for preflight in (True, False):
            task = task_specification("unit_stage48", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)

    def synthetic_results(self):
        results = StateBaselineTest().synthetic_results()
        for cell in results:
            root, method = cell["root"], "task_clock"
            cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract(), seed_roles=spec.seed_roles(root, preflight=False))
            histories = cell["training"][method]
            histories["episode_kl"] = histories.pop("state_mc")
            for treatment, history in histories.items():
                for row in history:
                    for key in ("settings", "folds", "baseline_optimizer_steps", "gradient_dispersion"):
                        row.pop(key, None)
                    iteration = row["iteration"]
                    seeds = cell["seed_roles"]["training"][(iteration - 1) * 8:iteration * 8]
                    row["actor_credit"] = treatment
                    row["rollout_sampling"] = [{"seed": seed, "policy_seed": spec.policy_seed(root, seed),
                        **{k: v for k, v in spec.rollout_arguments(root, seed, phase="train", mode="training").items() if k != "sample"}} for seed in seeds]
                    def terms(kl):
                        return {"episode_kl": [kl] * 8, "max_episode_kl": kl, "mean_episode_kl": kl,
                                "clipped_surrogate": .1, "actor_objective": .2}
                    trials = [{"scale": 1., **terms(.8), "actor_optimizer_steps": 2, "value_optimizer_steps": 2}]
                    bounded = treatment == "episode_kl"
                    if bounded:
                        trials.append({"scale": .5, **terms(.08), "actor_optimizer_steps": 2, "value_optimizer_steps": 2})
                    row.update(accepted=True, selected_scale=.5 if bounded else 1., trials=trials, before_terms=terms(0.),
                        deployed_terms=terms(.08 if bounded else .8), retained_actor_steps=2, retained_value_steps=2,
                        actor_optimizer_steps=2 * len(trials), value_optimizer_steps=2 * len(trials), kl_check_calls=2 + len(trials))
            evaluation = cell["evaluation_rows"]
            evaluation[method + ":episode_kl"] = evaluation.pop(method + ":state_mc")
            for stages in evaluation.values():
                for stage in stages.values():
                    for mode, rows in stage.items():
                        for row, seed in zip(rows, cell["seed_roles"]["evaluation"]):
                            row.update(seed=seed, policy_seed=spec.policy_seed(root, seed), **spec.rollout_arguments(root, seed, phase="eval", mode=mode))
            pairs = cell["first_batch_pair_details"][method]
            pairs["episode_kl"] = pairs.pop("state_mc")
        return results

    def test_statistics_trial_provenance_and_executed_retained_accounting(self):
        results = self.synthetic_results()
        summary = experiment.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 6})
        self.assertEqual(summary["primary_endpoints"]["bounded_mc_final"]["mean"], 5.)
        self.assertEqual(summary["primary_endpoints"]["bounded_frozen_final"]["effect"], "positive")
        self.assertEqual(summary["optimizer_steps"], {"actor_optimizer_steps": 1024, "value_optimizer_steps": 1024})
        self.assertEqual(summary["retained_optimizer_steps"], {"actor_optimizer_steps": 768, "value_optimizer_steps": 768})
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            experiment.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["episode_kl"][0]["trials"][0]["max_episode_kl"] = .05
        with self.assertRaisesRegex(ValueError, "trial or executed cost changed"):
            experiment.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["episode_kl"][0]["actor_optimizer_steps"] -= 1
        with self.assertRaisesRegex(ValueError, "optimizer or target accounting changed"):
            experiment.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
