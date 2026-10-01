import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_horizon_value as experiment
from freq_hrl.experiments import pointmaze_continuing_credit as continuing
from freq_hrl.experiments.pointmaze_value_targets import ValueFit
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, SMDPPPOConfig
from scripts import pointmaze_horizon_value_stage67_spec as spec
from scripts.submit_pointmaze_horizon_value_stage67_scheduleurm import task_specification, qualification_task


class HorizonValueTest(unittest.TestCase):
    def data(self):
        torch.set_num_threads(1)
        torch.manual_seed(67)
        model = FrequencySeparatedActorCriticPPO(SMDPPPOConfig(upper_state_dim=6, lower_state_dim=6,
            upper_action_dim=2, lower_action_dim=2, lower_cost_critic=False, lower_value_state_dim=6,
            hidden_dim=8, epochs=2, minibatch_size=5))
        state = np.random.default_rng(67).normal(size=(12, 6)).astype(np.float32)
        state[:, -1] = np.tile(np.arange(6, 0, -1) / 6, 2)
        lower = SimpleNamespace(value_state=state, size=len(state))
        lower.old_value = continuing.episode_predictions(model, lower)
        mc = experiment.discounted_mass(np.tile(np.arange(6, 0, -1), 2), model.config.gamma) * (1. + .2 * state[:, 0])
        return model, lower, mc

    def test_mass_rebase_clock_and_terminal_value(self):
        for gamma in (.995, 1.):
            expected = np.asarray([sum(gamma ** k for k in range(n)) for n in range(8)])
            np.testing.assert_allclose(experiment.discounted_mass(np.arange(8), gamma), expected, atol=1e-6)
        model, lower, mc = self.data()
        initial = copy.deepcopy(model.lower_value.state_dict())
        fit = experiment.FactoredValueFit(model, 6)
        self.assertLessEqual(fit.initialize_rate_frame(lower, mc), 2e-4)
        for key, value in initial.items():
            if not key.startswith("net.4"):
                torch.testing.assert_close(value, model.lower_value.state_dict()[key], atol=0, rtol=0)
        np.testing.assert_allclose(fit.predictions(lower)[::6], lower.old_value[::6], atol=2e-4, rtol=0)
        terminal = copy.deepcopy(lower)
        terminal.value_state[:, -1] = 0.
        np.testing.assert_array_equal(fit.predictions(terminal), np.zeros(lower.size))
        prefix = SimpleNamespace(value_state=lower.value_state[:3], size=3)
        np.testing.assert_array_equal(fit.predictions(prefix), fit.predictions(lower)[:3])
        with self.assertRaisesRegex(ValueError, "ordinary ValueNet"):
            fit.public_state()

    def test_rate_targets_adam_resume_and_frozen_noncritic_state(self):
        model, lower, mc = self.data()
        clone = copy.deepcopy(model)
        fit = experiment.FactoredValueFit(model, 6)
        fit.initialize_rate_frame(lower, mc)
        frame = fit.location, fit.scale
        plain = ValueFit(copy.deepcopy(model), "mc_normalized")
        plain.location, plain.scale = frame
        plain.initialized = True
        expected = plain.update(lower, mc / fit.mass(lower.value_state), root=310001, period=50, iteration=1)
        observed = fit.update(lower, mc, root=310001, period=50, iteration=1)
        self.assertEqual(observed, expected)
        torch.testing.assert_close(model.lower_value.state_dict(), plain.model.lower_value.state_dict(), atol=0, rtol=0)
        torch.testing.assert_close(model.lower_value_optimizer.state_dict(), plain.model.lower_value_optimizer.state_dict(), atol=0, rtol=0)
        for name in ("lower_actor", "lower_actor_optimizer", "upper_actor", "upper_value", "upper_actor_optimizer", "upper_value_optimizer"):
            torch.testing.assert_close(getattr(model, name).state_dict(), getattr(clone, name).state_dict(), atol=0, rtol=0)
        saved = copy.deepcopy(fit.checkpoint())
        resumed = experiment.FactoredValueFit.restore(copy.deepcopy(clone), saved)
        np.testing.assert_array_equal(resumed.predictions(lower), fit.predictions(lower))
        for other in (fit, resumed):
            other.update(lower, mc, root=310001, period=50, iteration=2)
        torch.testing.assert_close(resumed.model.lower_value.state_dict(), model.lower_value.state_dict(), atol=0, rtol=0)
        torch.testing.assert_close(resumed.model.lower_value_optimizer.state_dict(), model.lower_value_optimizer.state_dict(), atol=0, rtol=0)
        self.assertEqual((fit.location, fit.scale), frame)
        with self.assertRaises(ValueError):
            fit.initialize_rate_frame(lower, mc)

    def test_roster_costs_marker_dependency_and_credit_hold(self):
        self.assertEqual(spec.roots(preflight=False), spec.source.roots(preflight=False))
        budget = spec.budget(preflight=False)
        self.assertEqual(budget["archive_episodes"], 544)
        self.assertEqual(budget["calibration_updates"], 128)
        self.assertEqual(budget["actor_optimizer_steps"], 0)
        for preflight in (False, True):
            qualification = qualification_task("unit_stage67", preflight=preflight)
            self.assertEqual(len(qualification["wait_for_files"]), len(spec.roots(preflight=preflight)))
            self.assertIsNone(qualification["result_dir"])
            for root in spec.roots(preflight=preflight):
                task = task_specification("unit_stage67", root, preflight=preflight)
                self.assertEqual((task["cpu"], task["ram_mb"]), (3, 3072))
                self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
                self.assertFalse(task.get("require_node"))
        row = {"root": spec.roots(preflight=True)[0], "cost": spec.budget(preflight=True), "groups": {}}
        for period in spec.PERIODS:
            row["groups"][str(period)] = {}
            for arm in spec.TRAIN_POLICIES:
                treatments = {}
                for t in spec.TREATMENTS:
                    treatments[t] = {"value_mc": {"mse": 1.}, "windows": {"last_decile": {"value_mc": {"mse": 1., "bias": 1.}}},
                        "credit_alignment": {"normalized_sign_disagreement": .3},
                        "gradient": {"mean": {"credit_cosine": .7}, "log_std": {"credit_cosine": .5}}}
                row["groups"][str(period)][arm] = {"treatments": treatments}
        with patch.object(experiment, "qualify", return_value=(row, dict.fromkeys(spec.TREATMENTS, 10), [])):
            summary = experiment.aggregate([row], preflight=True)
            self.assertEqual(summary["credit_gate"], "passed")
            row["groups"]["50"]["joint_ppo"]["treatments"][spec.CANDIDATE]["gradient"]["log_std"]["credit_cosine"] = .4
            self.assertEqual(experiment.aggregate([row], preflight=True)["credit_gate"], "failed")


if __name__ == "__main__":
    unittest.main()
