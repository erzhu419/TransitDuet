import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.experiments import pointmaze_critic_calibration as calibration
from freq_hrl.experiments import pointmaze_critic_clock as clocks
from freq_hrl.experiments import pointmaze_signed_microstep as probe
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_signed_microstep_stage44_spec as spec
from scripts.submit_pointmaze_signed_microstep_stage44_scheduleurm import task_specification
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class SignedMicrostepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(44)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=1, lower_state_dim=1, upper_action_dim=1, lower_action_dim=1,
            hidden_dim=0, lower_cost_critic=False, init_log_std=0.))

    def test_interpolation_includes_log_std_exact_endpoints_and_no_mutation(self):
        before, after = self.model(), self.model()
        with torch.no_grad():
            for parameter in after.lower_actor.parameters():
                parameter.add_(.25)
            for parameter in after.lower_value.parameters():
                parameter.add_(1.)
        original, updated = joint.inference_weights(before), joint.inference_weights(after)
        for scale in (0., *spec.SCALES.values()):
            weights = probe.interpolated_weights(before, after, scale)
            for name in weights:
                if name != "lower_actor":
                    torch.testing.assert_close(weights[name], original[name], rtol=0, atol=0)
            expected = {key: value + scale * (updated["lower_actor"][key] - value) for key, value in original["lower_actor"].items()}
            torch.testing.assert_close(weights["lower_actor"], expected, rtol=0, atol=0)
            self.assertAlmostEqual(weights["lower_actor"]["log_std"].item(), .25 * scale)
        torch.testing.assert_close(probe.interpolated_weights(before, after, 1.)["lower_actor"], updated["lower_actor"], rtol=0, atol=0)
        torch.testing.assert_close(joint.inference_weights(before), original, rtol=0, atol=0)
        torch.testing.assert_close(joint.inference_weights(after), updated, rtol=0, atol=0)
        self.assertFalse(before.lower_actor_optimizer.state)

    def test_shared_reference_and_fixed_levels_reject_actual_control_changes(self):
        pairs = {method: [self.model(), self.model()] for method in spec.METHODS}
        with torch.no_grad():
            pairs[spec.METHODS[1]][0].lower_value.net[0].bias += .5
        weights = probe.policy_weights(pairs)
        self.assertEqual(set(weights), set(spec.POLICIES))
        torch.testing.assert_close(weights[spec.METHODS[1] + ":full"]["lower_value"],
                                   pairs[spec.METHODS[1]][0].lower_value.state_dict(), rtol=0, atol=0)
        with torch.no_grad():
            pairs[spec.METHODS[1]][1].upper_actor.net[0].bias += .1
        with self.assertRaises(AssertionError):
            probe.policy_weights(pairs)

    def test_registered_contrasts_are_signed_and_use_sampled_returns(self):
        means = {"frozen": {"episode_return": 100.}}
        for method in spec.METHODS:
            means.update({method + ":plus_micro": {"episode_return": 101.},
                          method + ":minus_micro": {"episode_return": 99.},
                          method + ":full": {"episode_return": 97.}})
        values = spec.contrasts(means)
        self.assertEqual(tuple(values), spec.ENDPOINTS)
        for method in spec.METHODS:
            self.assertEqual([values[method + ":" + k] for k in ("plus_vs_zero", "minus_vs_zero", "signed_slope", "full_vs_plus")],
                             [1., -1., 16., -4.])

    def test_existing_checkpoint_pipeline_and_native_style_accounting(self):
        original = spec.source.source
        torch.manual_seed(44)
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
                    patch.object(probe, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                for method in spec.METHODS:
                    calibration.train(310001, method, preflight=True, output=spec.source_result(310001, method, preflight=True),
                                      specification=original, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                result = probe.diagnose(310001, preflight=True, output=directory / "diagnostic/result.json")
                self.assertEqual(result["budget"]["total_primitive_steps"], 15600)
                self.assertEqual(result["native_trace_audits"], 52)
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertEqual(result["seed_roles"]["evaluation"], [10893001, 10893002])
                for policies in result["evaluation_rows"].values():
                    for rows in policies.values():
                        self.assertTrue(all(not r["upper_sample"] and not r["gate_sample"] for r in rows))
                summary = probe.aggregate([result], preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(summary["method_cost"]["primitive_steps"], 15600)
                self.assertNotIn("primary_endpoints", summary)

    def test_frozen_streams_budget_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                for previous in (spec.source, spec.source.source):
                    prior = {s for values in previous.seed_roles(root, preflight=preflight).values() for s in values}
                    self.assertFalse(set(roles["evaluation"]).intersection(prior))
                self.assertFalse(set(roles["evaluation"]).intersection(seen))
                seen.update(roles["evaluation"])
        self.assertEqual(spec.CI_FAMILY_SIZE, 16)
        self.assertEqual(8 * spec.budget(preflight=False)["total_primitive_steps"], 3993600)
        self.assertEqual(8 * spec.budget(preflight=False)["native_trace_audits"], 3328)
        for preflight in (True, False):
            task = task_specification("unit_stage44", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertEqual(task["cpu"], 2 if preflight else 9)
            self.assertEqual(task["ram_mb"], 4096 if preflight else 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertNotIn("result_dir", task)
            self.assertNotIn("local_result_dir", task)
            self.assertIn("Training complete: result.json written", task["cmd"])

    def test_paired_root_bootstrap_missing_roots_sampling_and_inference_counts(self):
        results = []
        for index, root in enumerate(spec.roots(preflight=False)):
            evaluation = {}
            for policy in spec.POLICIES:
                shift = 0. if policy == "frozen" else {"plus_micro": 1., "minus_micro": -1., "full": -2.}[policy.split(":")[1]] * (index + 1)
                evaluation[policy] = {mode: [{**{k: 100. + shift for k in spec.METRICS}, "episode_length": 1200,
                    "upper_inference_calls": 1, "lower_inference_calls": 1200, "gate_inference_calls": 1,
                    "seed": seed, "policy_seed": spec.policy_seed(root, seed), "deployment_mode": mode,
                    **{k: v for k, v in spec.rollout_arguments(root, seed, mode=mode).items() if k != "sample"}}
                    for seed in spec.seed_roles(root, preflight=False)["evaluation"]] for mode in spec.MODES}
            traces = spec.budget(preflight=False)["native_trace_audits"]
            results.append({"root": root, "status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
                "contract": spec.contract(), "preflight": False, "seed_roles": spec.seed_roles(root, preflight=False),
                "budget": spec.budget(preflight=False), "optimizer_steps": 0,
                "weight_controls": "shared_before_actors_exact_zero_full_exact_other_networks_unchanged",
                "evaluation_rows": evaluation, "native_trace_audits": traces,
                "inference_counts": {"evaluation": {"primitive_steps": traces * 1200,
                    "upper_inference_calls": traces, "lower_inference_calls": traces * 1200, "gate_inference_calls": traces}}})
        summary = probe.aggregate(results, preflight=False)
        self.assertEqual(summary["independent_statistics"], {"status": "passed", "endpoints": 16})
        for method in spec.METHODS:
            for key, mean in zip(("plus_vs_zero", "minus_vs_zero", "signed_slope", "full_vs_plus"), (4.5, -4.5, 72., -13.5)):
                self.assertAlmostEqual(summary["primary_endpoints"][method + ":" + key]["mean"], mean)
                self.assertEqual(summary["primary_endpoints"][method + ":" + key]["effect"], "positive" if mean > 0 else "negative")
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            probe.aggregate(results[:-1], preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["evaluation_rows"]["frozen"]["lower_sampled"][0]["lower_seed"] += 1
        with self.assertRaisesRegex(ValueError, "sampling changed"):
            probe.aggregate(changed, preflight=False)
        changed = copy.deepcopy(results)
        changed[0]["inference_counts"]["evaluation"]["upper_inference_calls"] += 1
        with self.assertRaisesRegex(ValueError, "inference accounting changed"):
            probe.aggregate(changed, preflight=False)


if __name__ == "__main__":
    unittest.main()
