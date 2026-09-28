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
from freq_hrl.rl.smdp_actor_critic import (
    FrequencySeparatedActorCriticPPO, HierarchicalRolloutBuilder, SMDPPPOConfig, concat_level_batches,
)
from scripts import pointmaze_critic_clock_stage42_spec as spec
from scripts import analyze_pointmaze_critic_clock_stage42 as analyzer
from scripts.submit_pointmaze_critic_clock_stage42_scheduleurm import task_specification
from test_pointmaze_joint_renewal import CountedController, DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class CriticClockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def controller(self, hidden_dim=16):
        torch.manual_seed(42)
        return FrequencySeparatedActorCriticPPO(SMDPPPOConfig(
            upper_state_dim=390, lower_state_dim=390, upper_action_dim=2, lower_action_dim=2,
            hidden_dim=hidden_dim, lower_cost_critic=False, epochs=1, minibatch_size=128))

    def test_reward_critic_inputs_do_not_change_actor_and_reach_value_optimizer(self):
        config = SMDPPPOConfig(upper_state_dim=4, lower_state_dim=4, upper_action_dim=2,
                              lower_action_dim=2, lower_value_state_dim=6, lower_cost_critic=False,
                              hidden_dim=0, epochs=1, minibatch_size=4)
        model = FrequencySeparatedActorCriticPPO(config)
        with torch.no_grad():
            model.lower_value.net[0].weight[:, -2:] = torch.tensor([[2., 3.]])
        state = np.zeros(4, dtype=np.float32)
        a, b = np.r_[state, np.float32(0.), np.float32(1.)], np.r_[state, np.float32(1.), np.float32(1.)]
        torch.manual_seed(42)
        first = model.act_lower(state, sample=True, value_state=a)
        torch.manual_seed(42)
        second = model.act_lower(state, sample=True, value_state=b)
        np.testing.assert_array_equal(first["action"], second["action"])
        self.assertEqual(first["logp"], second["logp"])
        self.assertAlmostEqual(second["value"] - first["value"], 2.)
        with self.assertRaisesRegex(ValueError, "lower value state"):
            model.act_lower(state)
        with torch.no_grad():
            model.lower_value.net[0].weight[:, -2:] = 0.
        builder = HierarchicalRolloutBuilder(gamma=config.gamma)
        builder.begin_upper(state=state, action=np.zeros(2), logp=0., value=0.)
        for index in range(2):
            output = model.act_lower(state, sample=False, value_state=a)
            builder.add_lower(state=state, value_state=a, action=output["action"], logp=output["logp"],
                              value=output["value"], reward=1., done=index == 1)
        builder.finish()
        batch = builder.build().lower
        merged = concat_level_batches([batch, batch])
        np.testing.assert_array_equal(merged.value_state, np.vstack((batch.value_state, batch.value_state)))
        before = joint.inference_weights(model)
        metrics = calibration.lower_update(model, merged, "critic")
        self.assertEqual(metrics["lower_actor_optimizer_steps"], 0.)
        self.assertEqual(metrics["lower_value_optimizer_steps"], 1.)
        self.assertNotEqual(float(model.lower_value.net[0].weight[0, -1]), 0.)
        for key, value in before["lower_actor"].items():
            torch.testing.assert_close(value, model.lower_actor.state_dict()[key], rtol=0, atol=0)
        restored = FrequencySeparatedActorCriticPPO(config)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertEqual(restored.act_lower(state, sample=False, value_state=a)["value"],
                         model.act_lower(state, sample=False, value_state=a)["value"])
        changed = copy.deepcopy(batch)
        changed.value_state = None
        with self.assertRaisesRegex(ValueError, "value states"):
            concat_level_batches([batch, changed])
        with self.assertRaisesRegex(ValueError, "explicit value_state"):
            changed.validate(state_dim=4, action_dim=2, level="lower", value_state_dim=6)

    def test_zero_column_expansion_preserves_inherited_actor_gate_and_critic(self):
        for hidden in (0, 16):
            controller = self.controller(hidden)
            original = joint.make_model(controller, "learned_history", root=310001)
            expanded = clocks.make_model(controller, "learned_history", root=310001)
            self.assertEqual(expanded.config.lower_state_dim, 390)
            self.assertEqual(expanded.lower_value_state_dim, 392)
            for name, weights in joint.inference_weights(original).items():
                for key, value in weights.items():
                    actual = getattr(expanded, name).state_dict()[key]
                    if name == "lower_value" and key == "net.0.weight":
                        torch.testing.assert_close(actual[:, -2:], torch.zeros_like(actual[:, -2:]), rtol=0, atol=0)
                        actual = actual[:, :-2]
                    torch.testing.assert_close(value, actual, rtol=0, atol=0)
            state = torch.randn(8, 390)
            with torch.no_grad():
                expected = original.lower_value(state)
                observed = expanded.lower_value(torch.cat((state, torch.rand(8, 2)), dim=1))
            torch.testing.assert_close(expected, observed, atol=1e-6, rtol=0)
            self.assertFalse(expanded.lower_value_optimizer.state)

    def test_clocks_reset_after_renewal_and_context_is_excluded_from_actor(self):
        class RecordedController(CountedController):
            def act_lower(self, state, sample, *, value_state):
                np.testing.assert_array_equal(state, value_state[:-2])
                return super().act_lower(state, sample)

        args = spec.source.arguments(310001, preflight=True)
        batches, traces = [], []
        for method in ("intrinsic_sham", "intrinsic_clock"):
            seed = spec.seed_roles(310001, preflight=True)["training"][0]
            with patch.object(joint, "_make_task", return_value=DenseTask()), patch.object(
                    joint, "pointmaze_goal_bounds", return_value=(np.full(2, -2.), np.full(2, 2.))):
                batch, row, trace = joint.rollout(RecordedController(gate_action=1.), args, "learned_history",
                    seed=seed, capture=True, lower_value_context_builder=clocks.context_builder(method),
                    **spec.rollout_arguments(310001, method, seed, phase="train", mode="warmup"))
            clocks.audit_context(batch.lower, row, trace["lower_value_context"], clock=spec.VALUE_CLOCK[method])
            batches.append(batch.lower)
            traces.append(trace)
        for key in ("state", "action", "reward", "duration", "done", "old_logp"):
            np.testing.assert_array_equal(getattr(batches[0], key), getattr(batches[1], key))
        np.testing.assert_array_equal(traces[1]["lower_value_context"][row["decision_steps"], 0], 0.)
        self.assertEqual(traces[1]["lower_value_context"][0, 1], 1.)
        self.assertEqual(traces[1]["lower_value_context"][-1, 1], np.float32(1. / args.horizon))
        changed = traces[1]["lower_value_context"].copy()
        changed[row["decision_steps"][1], 0] = .25
        with self.assertRaisesRegex(AssertionError, "causal current-option age"):
            clocks.audit_context(batches[1], row, changed, clock=True)

    def test_all_cells_native_style_replay_pairing_and_accounting(self):
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
                        specification=spec, rollout_worker=clocks.worker_rollout, model_factory=clocks.make_model)
                    raw = output.parent.with_name(output.parent.name + "_raw")
                    clocks.audit_result(result, raw_path=raw)
                    self.assertEqual(result["optimizer_steps"], {"actor_optimizer_steps": 0 if method == "frozen" else 6,
                                                                "value_optimizer_steps": 0 if method == "frozen" else 12})
                summary = analyzer.analyze("test", preflight=True)
                self.assertEqual(summary["status"], "preflight_passed")
                self.assertEqual(len(summary["checkpoint_replays"]), 40)
                self.assertEqual(len(summary["probe_replays"]), 15)
                self.assertEqual(len(summary["warmup_comparisons"]), 2)
                self.assertEqual(sum(a["episodes"] for a in summary["audits"]), 80)
                self.assertEqual(summary["method_cost"]["primitive_steps"], 33000)
                self.assertEqual(summary["verification_cost"]["primitive_steps"], 16500)
                self.assertNotIn("primary_endpoints", summary["aggregate"])
                for row in summary["warmup_comparisons"]:
                    self.assertEqual(row["first_learning_transitions"], 300)
                for row in summary["diagnostics"]:
                    if not spec.VALUE_CLOCK[row["method"]]:
                        self.assertEqual(set(row["clock_weight_norm"].values()), {0.})

    def test_eight_return_contrasts_and_independent_count_bootstrap(self):
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

    def test_fresh_streams_matched_sampling_budgets_and_dynamic_scheduler(self):
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
                for mode in ("warmup", "learning"):
                    kwargs = [spec.rollout_arguments(root, method, roles["training"][0], phase="train", mode=mode)
                              for method in spec.METHODS]
                    self.assertTrue(all(k == kwargs[0] for k in kwargs))
                    self.assertFalse(kwargs[0]["upper_sample"])
                    self.assertFalse(kwargs[0]["gate_sample"])
                    self.assertTrue(kwargs[0]["lower_sample"])
        self.assertEqual(40 * spec.budget(preflight=False)["total_primitive_steps"], 20064000)
        self.assertEqual(spec.verification_budget(preflight=False)["total_primitive_steps"], 624000)
        self.assertEqual(spec.CI_FAMILY_SIZE, 8)
        for method in spec.METHODS:
            task = task_specification("unit_stage42", 310011, method, preflight=False)
            self.assertEqual(task["cpu"], 9)
            self.assertEqual(task["ram_mb"], 12288)
            self.assertIsNone(task["require_node"])
            self.assertEqual(len(task["allowed_nodes"]), 6)
            self.assertIn("Training complete: result.json written", task["cmd"])
            self.assertEqual(task["stage_input_paths"], [str(spec.ROOT / "scripts"), str(spec.ROOT / "freq_hrl")])


if __name__ == "__main__":
    unittest.main()
