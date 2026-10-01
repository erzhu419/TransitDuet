import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_normalized_update as experiment
from freq_hrl.experiments import pointmaze_value_targets as values
from freq_hrl.experiments import pointmaze_first_update as guarded
from freq_hrl.experiments import pointmaze_matched_upper as previous
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from scripts import pointmaze_normalized_update_stage65_spec as spec
from scripts.submit_pointmaze_normalized_update_stage65_scheduleurm import task_specification
from test_pointmaze_value_targets import ValueTargetsTest
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class NormalizedUpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ValueTargetsTest.setUpClass()
        cls.helper = ValueTargetsTest()

    def checkpoint(self, fit, *, period=50, arm="zero_train"):
        return {"protocol": spec.source.EXPERIMENT_PROTOCOL, "root": 310001, "period": period, "arm": arm,
            "treatment": fit.treatment, "config": fit.model.state_dict()["config"], "location": fit.location, "scale": fit.scale,
            "value_training_state": fit.model.lower_value.state_dict(),
            "value_optimizer_training_units": fit.model.lower_value_optimizer.state_dict(), "public_value_state": fit.public_state()}

    def test_split_raw_actor_value_matches_guarded_core_with_nonempty_adam(self):
        model, batch = self.helper.data()
        # Start both Adam states before testing exact continuation, not just initialization.
        guarded.guarded_update(model, batch, level="lower", root=310001, period=50, episode_count=1,
            guard_type=guarded.BacktrackingKLGuard)
        for phase in (1, 2):
            core, fit = copy.deepcopy(model), values.ValueFit(copy.deepcopy(model), "gae_raw")
            expected = guarded.guarded_update(core, batch, level="lower", root=310001, period=50, episode_count=1,
                guard_type=guarded.BacktrackingKLGuard)
            actual, targets = experiment.actor_update(fit, batch.lower, root=310001, period=50)
            value = fit.update(batch.lower, targets, root=310001, period=50, iteration=1, phase="train")
            self.assertEqual(actual["guard"], expected["guard"])
            self.assertEqual(actual["kl_mean"], expected["kl_mean"])
            self.assertEqual(actual["actor_optimizer_steps"], expected["optimizer_steps"]["lower_actor_optimizer_steps"])
            self.assertEqual(value["value_optimizer_steps"], expected["optimizer_steps"]["lower_value_optimizer_steps"])
            torch.testing.assert_close(core.state_dict(), fit.model.state_dict(), atol=0, rtol=0)
            model = core

    def test_normalized_checkpoint_resume_keeps_frame_adam_and_reward_unit_gae(self):
        model, batch = self.helper.data()
        fit = values.ValueFit(copy.deepcopy(model), "mc_normalized")
        mc = values.monte_carlo_returns(batch.lower, model.config.gamma)
        fit.initialize_frame(float(mc.mean()), float(mc.std()), batch.lower)
        fit.update(batch.lower, mc, root=310001, period=50, iteration=1)
        payload = self.checkpoint(fit)
        resumed = experiment.restore_fit(model, payload, root=310001, period=50, arm="zero_train", treatment="mc_normalized")
        self.assertTrue(resumed.model.lower_value_optimizer.state)
        for f in (fit, resumed):
            public = experiment.public_model(f)
            state = torch.as_tensor(batch.lower.value_state, dtype=torch.float32)
            with torch.no_grad():
                torch.testing.assert_close(public.lower_value(state), f.location + f.scale * f.model.lower_value(state), atol=2e-4, rtol=1e-6)
            actual, gae = experiment.actor_update(f, batch.lower, root=310001, period=50)
            _, expected = model._gae(batch.lower.reward, batch.lower.done, batch.lower.duration, batch.lower.old_value)
            np.testing.assert_array_equal(gae, expected)
            f.update(batch.lower, mc, root=310001, period=50, iteration=1, phase="train")
            self.assertEqual((f.location, f.scale), (payload["location"], payload["scale"]))
            self.assertGreater(actual["guard"]["retained_actor_steps"], 0)
        torch.testing.assert_close(fit.model.state_dict(), resumed.model.state_dict(), atol=0, rtol=0)
        bad = copy.deepcopy(payload)
        bad["value_training_state"] = bad["public_value_state"]
        with self.assertRaises(AssertionError):
            experiment.restore_fit(model, bad, root=310001, period=50, arm="zero_train", treatment="mc_normalized")

    def test_archive_to_native_pairing_units_cost_and_checkpoint_labels(self):
        model, predictor = self.helper.helper.model(), self.helper.helper.predictor
        initialization = {"config": json.loads(json.dumps(model.config.__dict__))}
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            training, upper_file, source_file = d / "stage57/result.json", d / "stage63/result.json", d / "stage64/result.json"
            with patch.object(previous, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, initialization)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(values, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(spec.source, "training_result", return_value=training), \
                    patch.object(spec.source, "source_result", return_value=upper_file), \
                    patch.object(spec, "upper_result", return_value=upper_file), \
                    patch.object(spec, "source_result", return_value=source_file), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                original = previous.train(310001, preflight=True, output=training)
                reference = self.helper.reference(model, original, training)
                values.write_json(upper_file, reference)
                values.replay(310001, preflight=True, output=source_file)
                reference.update(protocol=spec.source.source.EXPERIMENT_PROTOCOL)
                for p, arms in reference["comparisons"].items():
                    for arm, cell in arms.items():
                        common = copy.deepcopy(model)
                        if arm == "joint_ppo":
                            with torch.no_grad():
                                common.upper_actor.net[-1].bias.add_(.05)
                        path = d / f"upper_{p}_{arm}.pt"
                        torch.save({"protocol": spec.source.source.EXPERIMENT_PROTOCOL, "root": 310001,
                            "period": int(p), "arm": arm, "treatment": "option_credit", "state_dict": common.state_dict()}, path)
                        cell.update(upper_networks_and_Adam_pair="passed", checkpoints={"option_credit": str(path)})
                values.write_json(upper_file, reference)
                result = experiment.train(310001, preflight=True, output=d / "stage65/result.json")
                summary = experiment.aggregate([result], preflight=True)
            self.assertEqual(result["archive_cost"], spec.budget(preflight=True)["archive"])
            self.assertEqual(result["native_evaluation_counts"], spec.budget(preflight=True)["native_evaluation"])
            self.assertEqual(summary["mechanical_gate"], "passed")
            self.assertEqual(summary["native_trace_audits"], 24)
            for p, groups in result["groups"].items():
                self.assertEqual(result["evaluation_rows"][p]["zero_train"]["frozen_lower"], result["evaluation_rows"][p]["clone"])
                for arm, cell in groups.items():
                    for t, update in cell["updates"].items():
                        saved = torch.load(update["checkpoint"], map_location="cpu", weights_only=False)
                        state = saved["training_state_dict"]
                        exported = copy.deepcopy(state["lower_value"])
                        exported["net.4.weight"] *= saved["scale"]
                        exported["net.4.bias"] = exported["net.4.bias"] * saved["scale"] + saved["location"]
                        torch.testing.assert_close(exported, saved["public_inference_weights"]["lower_value"], atol=0, rtol=0)
                        self.assertTrue(state["lower_value_optimizer"]["state"])
                        self.assertTrue(state["lower_actor_optimizer"]["state"])
                        self.assertNotIn("state_dict", saved)
            for mutation in ("cost", "alias", "guard"):
                bad = copy.deepcopy(result)
                if mutation == "cost":
                    bad["archive_cost"]["MC_continuation_updates"] -= 1
                elif mutation == "alias":
                    bad["evaluation_rows"]["50"]["zero_train"]["frozen_lower"][0]["seed"] += 1
                else:
                    bad["groups"]["50"]["zero_train"]["updates"]["mc_normalized"]["actor"]["kl_mean"] = .03
                with self.assertRaises(ValueError):
                    experiment.qualify(bad, preflight=True)

    def test_frozen_roots_endpoints_budget_and_dynamic_placement(self):
        self.assertEqual(spec.roots(preflight=False), spec.source.roots(preflight=False))
        self.assertIn(310037, spec.roots(preflight=False))
        self.assertEqual(len(spec.ENDPOINTS), 12)
        self.assertEqual(spec.budget(preflight=False)["native_evaluation"]["primitive_steps"] * 8, 1843200)
        task = task_specification("unit_stage65", 310011, preflight=False)
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertFalse(task.get("require_node"))
        for root in spec.roots(preflight=False):
            roles = spec.seed_roles(root, preflight=False)
            self.assertFalse(set(roles["evaluation"]).intersection(spec.source.seed_roles(root, preflight=False)["calibration"]))
            self.assertEqual(roles["first_training"], spec.source.seed_roles(root, preflight=False)["first_training_probe"])


if __name__ == "__main__":
    unittest.main()
