import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_option_residual_train as experiment
from scripts import pointmaze_option_residual_train_stage112_spec as spec
from test_pointmaze_control_response import Float32Task
from test_pointmaze_optional_plan import OptionalPlanTest


class OptionResidualTrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def sources(self):
        models, predictor, _, calibration = OptionalPlanTest().sources()
        return models, predictor, calibration

    def test_branch_gradient_and_fisher_step_only_change_readout(self):
        model = self.sources()[0]["50"]
        actor = experiment.branch(model)
        state = torch.randn(128, 396)
        with torch.no_grad():
            distribution = actor.distribution(state)
            action = distribution.rsample()
            old_logp = distribution.log_prob(action).sum(-1)
        from freq_hrl.rl.smdp_actor_critic import LevelTrajectoryBatch
        batch = LevelTrajectoryBatch(state=state.numpy().astype(np.float32), action=action.numpy().astype(np.float32),
            reward=np.ones(128, dtype=np.float32), duration=np.ones(128, dtype=np.int64),
            done=np.r_[np.zeros(127), 1.].astype(np.float32), old_logp=old_logp.numpy().astype(np.float32),
            old_value=np.zeros(128, dtype=np.float32))
        gradients, score_cost = experiment.residual_actor_gradients(actor, batch,
            {"scenario": np.linspace(-1., 1., 128, dtype=np.float32)}, clip_ratio=.2, chunk_size=64)
        before = copy.deepcopy(actor.base.state_dict())
        cost = {key: 0 for key in ("residual_fisher_batches", "residual_kl_checks",
            "residual_parameter_updates", "training_freeze_checks")}
        row = experiment.residual_update(actor, {"gradients": {"A": gradients["scenario"],
            "B": gradients["scenario"]}, "states": state.numpy(), "score_costs": {}, "signal_rms": {}}, cost=cost)
        self.assertEqual(row["geometry"]["radius_check"], "passed")
        self.assertAlmostEqual(row["geometry"]["exact_kl"], spec.FISHER_RADIUS, places=6)
        torch.testing.assert_close(actor.base.state_dict(), before, atol=0, rtol=0)
        self.assertEqual(score_cost["actor_score_forward_batches"], 2)

    def test_native_worker_preserves_source_and_has_registered_arm_schedules(self):
        models, predictor, calibration = self.sources()
        model = models["50"]
        args = spec.arguments(410011, preflight=True)
        args.horizon = 100
        experiment.source.native.init_worker(model.config, args)
        weights = experiment.source.native.joint.inference_weights(model)
        residual = experiment.branch(model)
        residual_state = residual.state_dict()
        before = copy.deepcopy(model.state_dict())
        with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
                patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
            outputs = {}
            for arm in spec.ARMS:
                batch, row, innovations = experiment.native_episode(weights, residual_state, seed=112095001,
                    noise_seed=112095002, arm=arm, period=50, predictor=predictor, calibration=calibration["50"],
                    args=args, collect=True)
                experiment.check_row(row, period=50, horizon=100)
                self.assertEqual(batch.size, 100)
                self.assertEqual(innovations.shape, (100, 2))
                outputs[arm] = row
            self.assertEqual(outputs["blind"]["upper_calls"], 0)
            self.assertEqual(outputs["forecast"]["upper_calls"], 0)
            self.assertEqual(outputs["learned"]["upper_calls"], 2)
        experiment.source.native.curves.support.assert_frozen(model, before)

    def test_frozen_budget_and_dynamic_scheduler_contract(self):
        budget = spec.budget(preflight=False)
        self.assertEqual(budget["training_episodes"], 6144)
        self.assertEqual(budget["evaluation_episodes"], 384)
        self.assertEqual(budget["native_episodes"], 6528)
        self.assertEqual(budget["native_upper_calls"], 38016)
        self.assertEqual(budget["residual_fisher_batches"], 7200)
        self.assertEqual(spec.budget(preflight=True)["residual_fisher_batches"], 60)
        self.assertEqual(budget["planning_renewals"], 76032)
        self.assertEqual(spec.contract()["decision"],
            "learned_minus_blind_forecast_and_own_blinded_positive_corrected_CI_required_before_upper_learning")
        for root in spec.roots(preflight=False):
            self.assertIsNone(__import__("scripts.submit_pointmaze_option_residual_train_stage112_scheduleurm",
                fromlist=["task_specification"]).task_specification("run", root, preflight=False)["require_node"])


if __name__ == "__main__":
    unittest.main()
