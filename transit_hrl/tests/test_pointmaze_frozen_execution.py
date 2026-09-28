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
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_frozen_execution_stage40_spec as spec
from scripts import analyze_pointmaze_frozen_execution_stage40 as analyzer
from scripts.submit_pointmaze_frozen_execution_stage40_scheduleurm import task_specification
from test_pointmaze_joint_renewal import CountedController, DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class ExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self):
        torch.manual_seed(40)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=16, lower_cost_critic=False, epochs=1, minibatch_size=128))

    def test_aligned_training_retains_lower_batches_and_coupled_lower_noise(self):
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

        actions = []
        args = spec.source.arguments(310001, preflight=True)
        for method in ("intrinsic_sampled", "intrinsic_aligned"):
            model = RecordedController()
            kwargs = spec.rollout_arguments(310001, method, 10490003, phase="train", mode="learning")
            torch.manual_seed(10490003 + 310001)
            with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                    joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                batch, row, raw = joint.rollout(model, args, "learned_history", seed=10490003, capture=True, **kwargs)
            self.assertEqual(batch.lower.size, args.horizon)
            expected = method.endswith("_sampled")
            self.assertEqual(row["upper_sample"], expected)
            self.assertEqual(row["gate_sample"], expected)
            self.assertTrue(all(model.samples["lower"]))
            self.assertEqual(set(model.samples["upper"]), {expected})
            self.assertEqual(set(model.samples["gate"]), {expected})
            actions.append(raw["action"])
        np.testing.assert_array_equal(*actions)

    def test_matched_warmup_sampling_and_lower_only_deployment(self):
        for root in spec.roots(preflight=False):
            seed = spec.seed_roles(root, preflight=False)["training"][0]
            for reward in ("intrinsic", "task"):
                a, b = reward + "_sampled", reward + "_aligned"
                self.assertEqual(spec.rollout_arguments(root, a, seed, phase="train", mode="warmup"),
                                 spec.rollout_arguments(root, b, seed, phase="train", mode="warmup"))
                for mode in spec.MODES:
                    kwargs = spec.rollout_arguments(root, b, seed, phase="eval", mode=mode)
                    self.assertFalse(kwargs["sample"])
                    self.assertFalse(kwargs["upper_sample"])
                    self.assertFalse(kwargs["gate_sample"])
                    self.assertEqual(kwargs["lower_sample"], mode == "lower_sampled")

    def test_all_cells_native_execution_shared_warmup_and_accounting(self):
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
                cells = {}
                for method in spec.METHODS:
                    output = directory / "results/test/cells" / method / "replicate_310001/result.json"
                    result = calibration.train(310001, method, preflight=True, output=output,
                                               specification=spec, rollout_worker=execution.worker_rollout)
                    cells[method] = result
                    raw = output.parent.with_name(output.parent.name + "_raw")
                    execution.audit_result(result, raw_path=raw)
                    self.assertEqual(result["optimizer_steps"], {"actor_optimizer_steps": 0 if method == "frozen" else 6,
                                                                "value_optimizer_steps": 0 if method == "frozen" else 12})
                    changed = copy.deepcopy(result)
                    changed["training"][2]["rollout_sampling"][0]["upper_sample"] = not method.endswith("_sampled")
                    if method == "frozen":
                        changed["training"][2]["rollout_sampling"][0]["upper_sample"] = False
                    with self.assertRaisesRegex(ValueError, "training sampling"):
                        execution.audit_result(changed, raw_path=raw)
                    changed = copy.deepcopy(result)
                    changed["training"][2]["rollout_sampling"][0]["lower_seed"] += 1
                    with self.assertRaisesRegex(ValueError, "paired lower stream"):
                        execution.audit_result(changed, raw_path=raw)
                summary = analyzer.analyze("test", preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(len(summary["checkpoint_replays"]), 40)
                self.assertEqual(len(summary["probe_replays"]), 15)
                self.assertEqual(len(summary["shared_warmup"]), 2)
                self.assertEqual(sum(a["episodes"] for a in summary["audits"]), 80)
                self.assertEqual(summary["method_cost"]["primitive_steps"], 33000)
                self.assertEqual(summary["verification_cost"]["primitive_steps"], 16500)
                self.assertNotIn("primary_endpoints", summary["aggregate"])
                for reward in ("intrinsic", "task"):
                    self.assertEqual(cells[reward + "_sampled"]["training"][:2], cells[reward + "_aligned"]["training"][:2])
                paired = [torch.load(cells["intrinsic_" + m]["snapshots"]["2"]["checkpoint"], weights_only=False)["state_dict"]
                          for m in ("sampled", "aligned")]
                paired[1]["lower_value_optimizer"]["state"][0]["exp_avg"].flatten()[0] += 1
                with self.assertRaises(AssertionError):
                    analyzer.audit_warmup_pair(*paired)
                (directory / "results/test/qualification_summary.json").unlink()
                (directory / "results/test/cells/task_aligned/replicate_310001/result.json").unlink()
                with self.assertRaises(FileNotFoundError):
                    analyzer.analyze("test", preflight=True)
                self.assertFalse((directory / "results/test/qualification_summary.json").exists())

    def test_final_first_update_contrasts_and_independent_count_bootstrap(self):
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
        tail = .05 / (2 * spec.CI_FAMILY_SIZE)
        bounds = np.quantile(counts @ x / len(x), [tail, 1 - tail], axis=0)
        for i, key in enumerate(spec.ENDPOINTS):
            np.testing.assert_allclose(bounds[:, i], summary["primary_endpoints"][key]["ci"], atol=1e-10, rtol=0)
        with self.assertRaisesRegex(ValueError, "roster incomplete"):
            calibration.aggregate(results[:-1], preflight=False, specification=spec)

    def test_rosters_budgets_and_dynamic_scheduler(self):
        seen = set()
        for preflight in (True, False):
            for root in spec.roots(preflight=preflight):
                seeds = [s for values in spec.seed_roles(root, preflight=preflight).values() for s in values]
                self.assertEqual(len(seeds), len(set(seeds)))
                self.assertFalse(seen.intersection(seeds))
                seen.update(seeds)
                old = {s for values in spec.previous.seed_roles(root, preflight=preflight).values() for s in values}
                self.assertFalse(set(seeds).intersection(old))
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"], 20064000)
        self.assertEqual(spec.verification_budget(preflight=False)["total_primitive_steps"], 624000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        for method in spec.METHODS:
            task = task_specification("unit_stage40", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("Training complete: result.json written", task["cmd"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])


if __name__ == "__main__":
    unittest.main()
