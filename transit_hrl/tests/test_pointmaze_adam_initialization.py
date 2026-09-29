import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_adam_initialization as experiment
from freq_hrl.experiments import pointmaze_episode_credit as episodes
from freq_hrl.experiments import pointmaze_episode_kl as bounded
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_adam_initialization_stage49_spec as spec
from scripts.submit_pointmaze_adam_initialization_stage49_scheduleurm import task_specification
import test_pointmaze_episode_kl as old_tests


class AdamInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def warmed(self):
        model, batch, rewards = old_tests.StateBaselineTest().data()
        np.random.seed(49)
        episodes.update(model, batch, rewards, "gae")
        old_tests.EpisodeKLTest().refresh_batch(model, batch)
        return model, batch, rewards

    def assert_lower_equal(self, left, right):
        for name in bounded.LOWER_STATE:
            torch.testing.assert_close(getattr(left, name).state_dict(), getattr(right, name).state_dict(), atol=0, rtol=0)

    def test_first_reset_only_actor_state_matches_actual_adam_path(self):
        model, batch, rewards = self.warmed()
        source = experiment.optimizer_steps(model)
        self.assertTrue(source)
        groups = copy.deepcopy(model.lower_actor_optimizer.state_dict()["param_groups"])
        original_critic = copy.deepcopy(model)
        np.random.seed(spec.shuffle_seed(310001, 1))
        episodes.update(original_critic, batch, rewards, "gae", specification=spec)
        for treatment in spec.TREATMENTS:
            expected, observed = copy.deepcopy(model), copy.deepcopy(model)
            fresh = treatment in spec.FRESH_TREATMENTS
            if fresh:
                expected.lower_actor_optimizer.state.clear()
            bounded.update(expected, batch, rewards, treatment, root=310001, iteration=1, specification=spec)
            record = experiment.update(observed, batch, rewards, treatment, root=310001, iteration=1)
            self.assert_lower_equal(expected, observed)
            self.assertEqual(record["optimizer_initialization"]["reset"], fresh)
            self.assertEqual(record["optimizer_initialization"]["source_steps"], source)
            self.assertEqual(record["optimizer_initialization"]["starting_steps"], [] if fresh else source)
            self.assertEqual(observed.lower_actor_optimizer.state_dict()["param_groups"], groups)
            for name in ("lower_value", "lower_value_optimizer"):
                torch.testing.assert_close(getattr(observed, name).state_dict(), getattr(original_critic, name).state_dict(), atol=0, rtol=0)
            self.assertLessEqual(record["deployed_terms"]["max_episode_kl"], spec.KL_BUDGET)

    def test_fresh_branch_does_not_reset_again_next_round(self):
        model, batch, rewards = self.warmed()
        first = experiment.update(model, batch, rewards, "mc_fresh", root=310001, iteration=1)
        old_tests.EpisodeKLTest().refresh_batch(model, batch)
        expected = copy.deepcopy(model)
        bounded.update(expected, batch, rewards, "mc_fresh", root=310001, iteration=2, specification=spec)
        second = experiment.update(model, batch, rewards, "mc_fresh", root=310001, iteration=2)
        self.assert_lower_equal(expected, model)
        self.assertFalse(second["optimizer_initialization"]["reset"])
        self.assertEqual(second["optimizer_initialization"]["source_steps"], first["optimizer_initialization"]["ending_steps"])
        self.assertEqual(second["optimizer_initialization"]["starting_steps"], first["optimizer_initialization"]["ending_steps"])

    def test_rejected_fresh_actor_remains_empty_but_critic_keeps_original_update(self):
        model, batch, rewards = self.warmed()
        before, expected = copy.deepcopy(model.lower_actor.state_dict()), copy.deepcopy(model)
        np.random.seed(spec.shuffle_seed(310001, 1))
        episodes.update(expected, batch, rewards, "gae", specification=spec)
        with patch.object(spec, "KL_BUDGET", 0.), patch.object(spec, "MAX_BACKTRACKS", 1):
            record = experiment.update(model, batch, rewards, "mc_fresh", root=310001, iteration=1)
            self.assertFalse(record["accepted"])
            self.assertFalse(model.lower_actor_optimizer.state)
            self.assertEqual(record["optimizer_initialization"]["ending_steps"], [])
            steps = record["retained_value_steps"]
            self.assertEqual(bounded.optimizer_steps(record, steps, specification=spec)["actor_optimizer_steps"], 2 * steps)
        torch.testing.assert_close(model.lower_actor.state_dict(), before, atol=0, rtol=0)
        for name in ("lower_value", "lower_value_optimizer"):
            torch.testing.assert_close(getattr(model, name).state_dict(), getattr(expected, name).state_dict(), atol=0, rtol=0)

    def test_existing_warmed_checkpoint_native_style_pipeline(self):
        original = spec.previous.previous.previous.previous.source.source
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
            with patch.object(spec.previous.previous.previous.previous.source, "ROOT", directory), patch.object(original, "source_result", return_value=source_file), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", old_tests.ImmediatePool), patch.object(episodes, "ProcessPoolExecutor", old_tests.ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: old_tests.DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                calibration.train(310001, "task_clock", preflight=True, output=spec.source_result(310001, "task_clock", preflight=True),
                                  specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = experiment.train(310001, preflight=True, output=directory / "experiment/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 20400)
                self.assertEqual(result["native_trace_audits"], 68)
                self.assertEqual(set(result["first_batch_pair_details"]["task_clock"]), set(spec.TREATMENTS[1:]))
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["retained_optimizer_steps"]["actor_optimizer_steps"], 80)
                self.assertEqual(summary["retained_optimizer_steps"]["value_optimizer_steps"], 80)
                self.assertNotIn("primary_endpoints", summary)

    def test_fresh_seeds_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                new = {s for values in roles.values() for s in values}
                self.assertEqual(len(new), sum(len(v) for v in roles.values()))
                for prior in (spec.previous, spec.previous.previous, spec.previous.previous.previous):
                    self.assertFalse(new.intersection(s for values in prior.seed_roles(root, preflight=preflight).values() for s in values))
                self.assertFalse(new.intersection(seen))
                seen.update(new)
        self.assertEqual(spec.CI_FAMILY_SIZE, 9)
        self.assertEqual(spec.KL_BUDGET, .1)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 7680000)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 6400)
        for preflight in (True, False):
            task = task_specification("unit_stage49", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)

    def synthetic_results(self):
        results = old_tests.EpisodeKLTest().synthetic_results()
        opt, budget, method = spec.options(preflight=False), spec.budget(preflight=False), spec.METHODS[0]
        for cell in results:
            root = cell["root"]
            cell.update(protocol=spec.EXPERIMENT_PROTOCOL, contract=spec.contract(), seed_roles=spec.seed_roles(root, preflight=False), budget=budget)
            template = cell["training"][method]["episode_kl"]
            histories = cell["training"][method] = {t: copy.deepcopy(template) for t in spec.TREATMENTS}
            for treatment, history in histories.items():
                source = [5, 5, 5]
                for row in history:
                    first, fresh = row["iteration"] == 1, treatment in spec.FRESH_TREATMENTS
                    start = [] if first and fresh else source
                    end = [s + row["retained_actor_steps"] for s in (start or [0, 0, 0])]
                    row.update(actor_credit=treatment, credit_estimator=spec.CREDITS[treatment],
                        optimizer_initialization={"reset": first and fresh, "source_steps": source, "starting_steps": start, "ending_steps": end})
                    source = end
                    i = row["iteration"] - 1
                    seeds = cell["seed_roles"]["training"][i * opt["rollouts_per_iteration"]:(i + 1) * opt["rollouts_per_iteration"]]
                    row["rollout_sampling"] = [{"seed": seed, "policy_seed": spec.policy_seed(root, seed),
                        **{k: v for k, v in spec.rollout_arguments(root, seed, phase="train", mode="training").items() if k != "sample"}} for seed in seeds]
            paired = next(iter(cell["first_batch_pair_details"][method].values()))
            cell["first_batch_pair_details"][method] = {t: copy.deepcopy(paired) for t in spec.TREATMENTS[1:]}
            template = cell["evaluation_rows"][method + ":episode_kl"]
            frozen = cell["evaluation_rows"]["frozen"]
            cell["evaluation_rows"] = {"frozen": frozen, **{method + ":" + t: copy.deepcopy(template) for t in spec.TREATMENTS}}
            values = {"frozen": 100., "gae": 101., "gae_fresh": 103., "episode_mc": 104., "mc_fresh": 109.}
            for policy, stages in cell["evaluation_rows"].items():
                value = values[policy.split(":")[-1]]
                for stage in stages.values():
                    for mode, rows in stage.items():
                        for row, seed in zip(rows, cell["seed_roles"]["evaluation"]):
                            row.update({k: value for k in spec.METRICS})
                            row.update(seed=seed, policy_seed=spec.policy_seed(root, seed), **spec.rollout_arguments(root, seed, phase="eval", mode=mode))
            cell["native_trace_audits"] = budget["native_trace_audits"]
            cell["inference_counts"] = {phase: {"primitive_steps": budget[k], "lower_inference_calls": budget[k],
                "upper_inference_calls": budget[k] // 1200, "gate_inference_calls": budget[k] // 1200}
                for phase, k in (("train", "training_primitive_steps"), ("eval", "evaluation_primitive_steps"))}
        return results

    def test_factorial_statistics_costs_and_reset_continuity(self):
        results = self.synthetic_results()
        summary = experiment.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 9})
        effects = summary["primary_endpoints"]
        self.assertEqual(effects["fresh_mc_inherited_mc_final"]["mean"], 5.)
        self.assertEqual(effects["credit_reset_interaction_final"]["mean"], 3.)
        self.assertEqual(effects["fresh_mc_frozen_final"]["mean"], 9.)
        self.assertEqual(summary["optimizer_steps"]["actor_optimizer_steps"], 2048)
        self.assertEqual(summary["retained_optimizer_steps"]["actor_optimizer_steps"], 1024)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            experiment.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["mc_fresh"][1]["optimizer_initialization"]["reset"] = True
        with self.assertRaisesRegex(ValueError, "initialization or continuity"):
            experiment.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["training"]["task_clock"]["gae_fresh"][0]["optimizer_initialization"]["starting_steps"] = [5, 5, 5]
        with self.assertRaisesRegex(ValueError, "initialization or continuity"):
            experiment.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
