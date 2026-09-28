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
from freq_hrl.experiments import pointmaze_frozen_execution as execution
from freq_hrl.experiments import pointmaze_warmup_alignment as alignment
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_warmup_alignment_stage41_spec as spec
from scripts import analyze_pointmaze_warmup_alignment_stage41 as analyzer
from scripts.submit_pointmaze_warmup_alignment_stage41_scheduleurm import task_specification
from test_pointmaze_joint_renewal import CountedController, DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class WarmupAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(41)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, epochs=1, minibatch_size=128))

    def test_only_warmup_changes_frozen_execution_and_lower_noise_stays_coupled(self):
        class RecordedController(CountedController):
            def __init__(self):
                super().__init__()
                self.samples = {k: [] for k in ("upper", "lower", "gate")}

            def act_upper(self, state, sample):
                self.samples["upper"].append(sample)
                if sample:
                    torch.randn(7)
                return super().act_upper(state, sample)

            def act_promotion(self, state, sample):
                self.samples["gate"].append(sample)
                if sample:
                    torch.randn(11)
                return super().act_promotion(state, sample)

            def act_lower(self, state, sample):
                self.samples["lower"].append(sample)
                return {**super().act_lower(state, sample), "action": torch.randn(2).numpy() if sample else np.zeros(2)}

        args = spec.source.arguments(310001, preflight=True)
        for mode in ("warmup", "learning"):
            actions = []
            for method in ("intrinsic_sampled", "intrinsic_aligned"):
                model = RecordedController()
                kwargs = spec.rollout_arguments(310001, method, 10590001, phase="train", mode=mode)
                torch.manual_seed(10590001 + 310001)
                with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                        joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                    batch, row, raw = joint.rollout(model, args, "learned_history", seed=10590001, capture=True, **kwargs)
                expected = mode == "warmup" and method.endswith("_sampled")
                self.assertEqual(set(model.samples["upper"]), {expected})
                self.assertEqual(set(model.samples["gate"]), {expected})
                self.assertTrue(all(model.samples["lower"]))
                self.assertEqual(batch.lower.size, args.horizon)
                self.assertEqual(row["upper_sample"], expected)
                actions.append(raw["action"])
            np.testing.assert_array_equal(*actions)

    def test_native_cells_critic_isolation_first_learning_pairs_and_accounting(self):
        controller = self.controller()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source_path, source_checkpoint = directory / "source.json", directory / "source.pt"
            torch.save({"state_dict": controller.state_dict()}, source_checkpoint)
            cell = {"selected_checkpoint_iteration": 0, "controller_checkpoint": str(source_checkpoint),
                    "factual_row": {"decision_steps": [0, 100, 200]}}
            source_path.write_text(json.dumps({"cells": [cell]}))
            with patch.object(spec, "ROOT", directory), patch.object(spec, "source_result", return_value=source_path), \
                    patch.object(calibration, "load_controller", return_value=(controller, cell, {})), \
                    patch.object(calibration, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(joint, "_make_task", side_effect=lambda **kwargs: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                for method in spec.METHODS:
                    output = directory / "results/test/cells" / method / "replicate_310001/result.json"
                    result = calibration.train(310001, method, preflight=True, output=output,
                                               specification=spec, rollout_worker=alignment.worker_rollout)
                    raw = output.parent.with_name(output.parent.name + "_raw")
                    execution.audit_result(result, raw_path=raw, specification=spec)
                    self.assertEqual(result["optimizer_steps"], {"actor_optimizer_steps": 0 if method == "frozen" else 6,
                                                                "value_optimizer_steps": 0 if method == "frozen" else 12})
                    changed = copy.deepcopy(result)
                    changed["training"][2]["rollout_sampling"][0]["upper_sample"] = True
                    with self.assertRaisesRegex(ValueError, "training sampling"):
                        execution.audit_result(changed, raw_path=raw, specification=spec)
                summary = analyzer.analyze("test", preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(len(summary["checkpoint_replays"]), 40)
                self.assertEqual(len(summary["probe_replays"]), 15)
                self.assertEqual(len(summary["warmup_comparisons"]), 2)
                self.assertEqual(sum(a["episodes"] for a in summary["audits"]), 80)
                self.assertEqual(summary["method_cost"]["primitive_steps"], 33000)
                self.assertEqual(summary["verification_cost"]["primitive_steps"], 16500)
                self.assertNotIn("primary_endpoints", summary["aggregate"])
                self.assertNotIn("shared_warmup", summary)
                for row in summary["warmup_comparisons"]:
                    self.assertEqual(row["first_learning_transitions"], 300)
                    self.assertGreater(row["lower_value_parameter_distance"], 0.)

    def test_pair_check_allows_value_difference_but_rejects_actor_or_transition_change(self):
        controller = self.controller()
        args = spec.source.arguments(310001, preflight=True)
        model = joint.make_model(controller, "learned_history", root=310001)
        seed = spec.seed_roles(310001, preflight=True)["training"][2]
        with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
            batch, _, _ = joint.rollout(model, args, "learned_history", seed=seed,
                **spec.rollout_arguments(310001, "task_aligned", seed, phase="train", mode="learning"))
        left, right = model.state_dict(), copy.deepcopy(model.state_dict())
        a, b = batch.lower, copy.deepcopy(batch.lower)
        b.old_value += 1.
        right["lower_value"][next(iter(right["lower_value"]))].flatten()[0] += 1.
        detail = alignment.audit_warmup_pair(left, right, a, b)
        self.assertAlmostEqual(detail["first_learning_value_rmse"], 1., places=6)
        self.assertGreater(detail["lower_value_parameter_distance"], 0.)
        changed = copy.deepcopy(b)
        changed.reward[0] += 1.
        with self.assertRaisesRegex(AssertionError, "first learning episode reward"):
            alignment.audit_warmup_pair(left, right, a, changed)
        right["lower_actor"][next(iter(right["lower_actor"]))].flatten()[0] += 1.
        with self.assertRaises(AssertionError):
            alignment.audit_warmup_pair(left, right, a, b)

    def test_eight_registered_return_contrasts_and_count_bootstrap(self):
        results = []
        for index, root in enumerate(spec.roots(preflight=False)):
            final = dict(zip(spec.METHODS, (100., 104. + index, 110., 90., 98. - index)))
            first = dict(zip(spec.METHODS, (100., 104., 105. + index, 95., 94.)))
            for method in spec.METHODS:
                stages = {}
                for iteration in spec.snapshots(preflight=False):
                    value = (final if iteration == 32 else first)[method]
                    row = {"episode_return": value, "tracking_squared_error_integral": 1.,
                           "upper_inference_calls": 24., "charged_utility": value - 24.}
                    stages[str(iteration)] = {"evaluation_rows": {"deterministic": [row],
                                               "lower_sampled": [{**row, "episode_return": 999.}]}}
                results.append({"root": root, "method": method, "snapshots": stages})
        summary = calibration.aggregate(results, preflight=False, specification=spec)
        self.assertEqual([summary["primary_endpoints"][k]["mean"] for k in spec.ENDPOINTS],
                         [7.5, 10., -10., -5.5, 2.5, 4.5, 4.5, -1.])
        x = np.array([[row["endpoints"][k] for k in spec.ENDPOINTS] for row in summary["root_rows"]])
        indices = np.random.default_rng(np.random.SeedSequence(spec.BOOTSTRAP_SEED)).integers(0, len(x), (spec.BOOTSTRAP_DRAWS, len(x)))
        counts = np.stack([(indices == i).sum(axis=1) for i in range(len(x))], axis=1)
        bounds = np.quantile(counts @ x / len(x), [.05 / 16, 1 - .05 / 16], axis=0)
        for i, key in enumerate(spec.ENDPOINTS):
            np.testing.assert_allclose(bounds[:, i], summary["primary_endpoints"][key]["ci"], atol=1e-10, rtol=0)

    def test_fresh_seed_rosters_budget_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = [s for values in roles.values() for s in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                old = {s for values in spec.previous.seed_roles(root, preflight=preflight).values() for s in values}
                self.assertFalse(set(seeds).intersection(old))
                for reward in ("intrinsic", "task"):
                    learning = [spec.rollout_arguments(root, reward + suffix, roles["training"][0],
                                phase="train", mode="learning") for suffix in ("_sampled", "_aligned")]
                    self.assertEqual(*learning)
                    probes = [spec.rollout_arguments(root, reward + suffix, roles["probe"][0],
                              phase="probe", mode="probe") for suffix in ("_sampled", "_aligned")]
                    self.assertEqual(*probes)
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"], 20064000)
        self.assertEqual(spec.verification_budget(preflight=False)["total_primitive_steps"], 624000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        for method in spec.METHODS:
            task = task_specification("unit_stage41", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("Training complete: result.json written", task["cmd"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])


if __name__ == "__main__":
    unittest.main()
