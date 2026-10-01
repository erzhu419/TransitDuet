import copy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_continuing_credit as experiment
from freq_hrl.experiments import pointmaze_first_update as guarded
from freq_hrl.experiments import pointmaze_update_diagnostics as diagnostics
from freq_hrl.experiments import pointmaze_matched_upper as previous
from freq_hrl.experiments import pointmaze_joint_renewal as joint
from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from scripts import pointmaze_episode_credit_stage63_spec as spec
from scripts.submit_pointmaze_episode_credit_stage63_scheduleurm import task_specification
import test_pointmaze_update_diagnostics as archive_data
from test_pointmaze_joint_renewal import DenseTask
from test_pointmaze_update_isolation import ImmediatePool


class ContinuingCreditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        archive_data.UpdateDiagnosticsTest.setUpClass()
        cls.helper = archive_data.UpdateDiagnosticsTest()

    def test_episode_boundaries_preserve_inputs_and_stop_cross_episode_credit(self):
        model = self.helper.model()
        with tempfile.TemporaryDirectory() as directory:
            native, _, _ = self.helper.archived_episode(model, directory)
        batch = concat_hierarchical_batches([native, native])
        original = batch.lower.done.copy()
        values = np.zeros(batch.lower.size, dtype=np.float32)
        continuing = experiment.episode_batch(batch, values, 300)
        np.testing.assert_array_equal(np.flatnonzero(continuing.lower.done), [299, 599])
        np.testing.assert_array_equal(batch.lower.done, original)
        self.assertIs(continuing.upper, batch.upper)
        for key in ("state", "value_state", "action", "reward", "duration", "old_logp"):
            self.assertIs(getattr(continuing.lower, key), getattr(batch.lower, key))
        model.config.gamma, model.config.gae_lambda = .9, 1.
        rewards = np.r_[np.ones(300), np.full(300, 1000.)].astype(np.float32)
        adv, returns = model._gae(rewards, continuing.lower.done, continuing.lower.duration, values)
        expected = np.asarray([(1. - .9 ** (300 - i)) / .1 for i in range(300)])
        np.testing.assert_allclose(adv[:300], expected, atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(adv, returns)
        self.assertEqual(float(adv[299]), 1.)
        self.assertEqual(float(adv[599]), 1000.)

    def test_episode_old_values_match_scalar_native_without_actor_or_state_changes(self):
        model = self.helper.model()
        with tempfile.TemporaryDirectory() as directory:
            batch, _, _ = self.helper.archived_episode(model, directory)
        before = copy.deepcopy(model.state_dict())
        with patch.object(model, "act_lower", side_effect=AssertionError("value probe must not invoke actor")):
            values = experiment.episode_predictions(model, batch.lower)
        np.testing.assert_array_equal(values, batch.lower.old_value)
        after = model.state_dict()
        self.assertEqual(before.pop("config"), after.pop("config"))
        torch.testing.assert_close(after, before, atol=0, rtol=0)

    def test_archive_control_reproduction_shared_upper_checkpoints_and_actual_gae_cost(self):
        model, predictor = self.helper.model(), self.helper.predictor
        source = {"config": json.loads(json.dumps(model.config.__dict__)),
            "checkpoints": {"50": "clone50.pt", "100": "clone100.pt"}}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stage57, stage58, stage59, stage60 = [directory / stage / "result.json"
                for stage in ("stage57", "stage58", "stage59", "stage60")]
            with patch.object(previous, "load_source", side_effect=lambda *a, **kw:
                    ({p: copy.deepcopy(model) for p in ("50", "100")}, predictor, source)), \
                    patch.object(previous, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(guarded, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
                    patch.object(diagnostics.spec, "source_result", return_value=stage57), \
                    patch.object(guarded.spec, "diagnostic_result", return_value=stage58), \
                    patch.object(spec.source, "diagnostic_result", return_value=stage58), \
                    patch.object(spec.source, "rejection_result", return_value=stage59), \
                    patch.object(spec, "source_result", return_value=stage60), \
                    patch.object(joint, "_make_task", side_effect=lambda **kw: DenseTask()), \
                    patch.object(joint, "pointmaze_goal_bounds", return_value=(-2 * np.ones(2), 2 * np.ones(2))):
                previous.train(310001, preflight=True, output=stage57)
                diagnostics.replay(310001, preflight=True, output=stage58)
                guarded.replay(310001, preflight=True, output=stage59)
                reference = guarded.replay(310001, preflight=True, output=stage60, specification=spec.source)
                gae = FrequencySeparatedActorCriticPPO._gae
                calls = []

                def counted(model, *args, **kwargs):
                    calls.append(1)
                    return gae(model, *args, **kwargs)

                with patch.object(joint, "_make_task", side_effect=AssertionError("Stage63 cannot collect native paths")), \
                        patch.object(FrequencySeparatedActorCriticPPO, "_gae", counted):
                    result = experiment.replay(310001, preflight=True, output=directory / "stage63/result.json")
                summary = experiment.aggregate([result], preflight=True)
                self.assertEqual(len(calls), result["cost"]["ppo_gae_calls"] + result["cost"]["diagnostic_gae_calls"])
            for period in spec.PERIODS:
                for arm in spec.TRAIN_POLICIES:
                    cell = result["comparisons"][str(period)][arm]
                    self.assertEqual(cell["updates"]["option_credit"],
                        reference["comparisons"][str(period)][arm]["treatments"]["backtracking_kl"])
                    states = []
                    for treatment, checkpoint in cell["checkpoints"].items():
                        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                        self.assertEqual((saved["protocol"], saved["root"], saved["period"], saved["arm"], saved["treatment"]),
                            (spec.EXPERIMENT_PROTOCOL, 310001, period, arm, treatment))
                        states.append(saved["state_dict"])
                    for kind in ("actor", "value", "actor_optimizer", "value_optimizer"):
                        torch.testing.assert_close(states[0]["upper_" + kind], states[1]["upper_" + kind], atol=0, rtol=0)
        self.assertEqual(summary["cost"], spec.budget(preflight=True))
        self.assertEqual(summary["executed_optimizer_steps"],
            {"upper_actor": 2, "upper_value": 10, "lower_actor": 40, "lower_value": 120})
        self.assertEqual(summary["native_trial_prerequisite"], "not_applicable_preflight")
        for mutation in ("cost", "steps", "upper", "boundary", "control"):
            bad = copy.deepcopy(result)
            cell = bad["comparisons"]["50"]["joint_ppo"]
            if mutation == "cost":
                bad["cost"]["episode_critic_scalar_calls"] += 1
            elif mutation == "steps":
                bad["warmup_optimizer_steps"]["lower_value"] -= 1
            elif mutation == "upper":
                cell["upper_networks_and_Adam_pair"] = "failed"
            elif mutation == "boundary":
                cell["critic_probe"]["episode_done_count"] += 1
            else:
                cell["control_reproduction"] = "failed"
            with self.assertRaises(ValueError):
                experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(result)
        bad["comparisons"]["50"]["joint_ppo"]["critic_probe"]["episode_value_episode_mc"]["explained_variance"] = -1.
        self.assertEqual(experiment.aggregate([bad], preflight=True)["episode_critic_fit_gate"], "failed")

    def test_full_prerequisite_preserves_failed_case_and_does_not_claim_performance(self):
        cells = [{"root": root, "cost": spec.budget(preflight=False)} for root in spec.roots(preflight=False)]

        def qualified(cell, *, preflight):
            failures = [{"root": cell["root"], "period": 100, "arm": "joint_ppo"}] if cell["root"] == 310101 else []
            return {"root": cell["root"]}, {}, {}, [], failures

        with patch.object(experiment, "qualify", side_effect=qualified):
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(summary["mechanical_gate"], "passed")
        self.assertEqual(summary["episode_critic_fit_gate"], "failed")
        self.assertEqual(summary["native_trial_prerequisite"], "hold")
        self.assertEqual(len(summary["root_rows"]), 8)
        self.assertEqual(summary["episode_critic_fit_failures"][0]["root"], 310101)
        self.assertEqual(summary["performance_claim"], "none_archive_only")
        with self.assertRaises(ValueError):
            experiment.aggregate(cells[:-1], preflight=False)

    def test_frozen_cost_and_dynamic_six_node_placement(self):
        b = spec.budget(preflight=False)
        self.assertEqual(8 * b["archive_episodes"], 4352)
        self.assertEqual(8 * b["episode_critic_scalar_calls"], 5222400)
        self.assertEqual(8 * b["warmup_critic_updates"], 1536)
        self.assertEqual(8 * b["diagnostic_updates"], 80)
        self.assertEqual(8 * b["ppo_gae_calls"], 1616)
        self.assertEqual(8 * b["diagnostic_gae_calls"], 176)
        self.assertEqual(8 * b["candidate_checkpoint_writes"], 64)
        self.assertEqual((b["native_steps"], b["new_evaluation_steps"], spec.KL_BUDGET), (0, 0, .02))
        for preflight in (True, False):
            task = task_specification("unit_stage63", spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertIsNone(task["require_node"])
            self.assertEqual(task["allowed_nodes"], [f"node{i:03d}" for i in range(1, 7)])
            self.assertEqual((task["cpu"], task["ram_mb"]), (2, 4096) if preflight else (9, 12288))
            roles = spec.seed_roles(spec.roots(preflight=preflight)[0], preflight=preflight)
            self.assertFalse(set(roles["calibration"]).intersection(roles["first_training_probe"]))


if __name__ == "__main__":
    unittest.main()
