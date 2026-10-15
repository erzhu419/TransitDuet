from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from freqduet_system_credit import Command, Decision, SystemCreditStream
from freqduet_system_learner import (ArrivalMoments, SystemStateEncoder, SystemSAC,
                                    finite_day_target, ingest_day, warm_start_actor)
from lower.cost_replay_buffer import CostReplayBuffer
from lower.resac_lagrangian import RESACLagrangianTrainer


def decision(time=0, bus=0, trip=0):
    return Decision(time, bus, trip, 1, True, (0.2, 0.3), 360, 2, 1)


def trainer(dimension):
    torch.manual_seed(59001)
    return RESACLagrangianTrainer(dimension, hidden_dim=8, ensemble_size=2,
                                 action_bins=(0, 5, 10, 15, 20, 30, 45),
                                 auto_entropy=False, gamma=1.0, beta_ood=0,
                                 weight_reg=0, device="cpu")


class SystemLearningTest(unittest.TestCase):
    def setUp(self):
        self.encoder = SystemStateEncoder(2, 3, 10, 5, 100)

    def test_moments_match_official_arrival_cv(self):
        moments = ArrivalMoments()
        moments.observe([100, 300])
        moments.observe([100, 300, 200, 500])
        values = np.array([100, 300, 200, 500])
        self.assertAlmostEqual(moments.cv_squared, (values.std() / values.mean()) ** 2)
        self.assertEqual(moments.count, 4)

    def test_empty_and_zero_headways_follow_official_metric(self):
        moments = ArrivalMoments()
        self.assertEqual(moments.cv_squared, 0)
        moments.observe([0, 0])
        self.assertEqual(moments.cv_squared, 0)

    def test_zero_hold_pending_is_not_an_empty_slot(self):
        plain = self.encoder.encode(decision(), ArrivalMoments())
        command = Command(0, 1, 3, 1, 0)
        pending = self.encoder.encode(replace(decision(), pending_commands=(command,)), ArrivalMoments())
        self.assertEqual(len(pending), self.encoder.dimension)
        self.assertFalse(np.array_equal(plain, pending))
        self.assertEqual(pending[2 + 11 + 6], 1)

    def test_command_order_identity_and_all_fields_are_retained(self):
        first, second = Command(0, 1, 3, 1, 15), Command(0, 2, 4, 2, 45)
        state = replace(decision(), pending_commands=(first, second))
        a = self.encoder.encode(state, ArrivalMoments())
        b = self.encoder.encode(replace(state, pending_commands=(second, first)), ArrivalMoments())
        self.assertFalse(np.array_equal(a, b))
        for changed in (replace(first, trip_id=5), replace(first, station_id=2),
                        replace(first, time_s=1), replace(first, action_s=20)):
            c = self.encoder.encode(replace(state, pending_commands=(changed, second)), ArrivalMoments())
            self.assertFalse(np.array_equal(a, c))

    def test_finite_day_target_is_not_clipped_or_trip_discounted(self):
        target = finite_day_target(torch.tensor([-1500., 2.]), torch.tensor([0., 1.]),
                                   torch.tensor([-200., 100.]))
        torch.testing.assert_close(target, torch.tensor([-1700., 2.]))
        signed_cost = finite_day_target(torch.tensor([-.2]), torch.tensor([0.]), torch.tensor([-.1]))
        self.assertAlmostEqual(signed_cost.item(), -.3, places=6)

    def test_warm_actor_preserves_distribution_with_nonzero_new_features(self):
        old, new = trainer(2), trainer(self.encoder.dimension)
        warm_start_actor(old.policy_net, new.policy_net, 2)
        encoded = self.encoder.encode(replace(decision(), pending_commands=(Command(0, 1, 3, 1, 45),)),
                                      ArrivalMoments())
        with torch.no_grad():
            a = old.policy_net.dist_info(torch.tensor([[.2, .3]]))[0]
            b = new.policy_net.dist_info(torch.from_numpy(encoded)[None, :])[0]
        torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-6)

    def test_replay_conserves_physical_reward_and_signed_cv_increments(self):
        stream = SystemCreditStream()
        stream.decide(decision(), 0)
        stream.decide(decision(bus=1, trip=1), 15)
        stream.advance(0, 10, 3, 4, 1, 1, 0)
        stream.acknowledge([Command(0, 0, 0, 1, 0), Command(0, 1, 1, 1, 15)])
        stream.finish(10, service_cleared=True, unserved=3, lf_unserved=1)
        replay = CostReplayBuffer(seed=1)
        states = [np.ones(2), np.zeros(2)]
        result = ingest_day(replay, stream, states, [.1, .1], .04, 59)
        self.assertAlmostEqual(result["cv_increment_sum"], -.06)
        self.assertAlmostEqual(result["replay_reward_sum"], -70 / 3600000)
        self.assertEqual([row[5] for row in replay.buffer], [0, 1])
        self.assertEqual([row[6] for row in replay.buffer], [59, 59])
        self.assertEqual(replay.buffer[0][2], 0)
        np.testing.assert_array_equal(replay.buffer[0][4], states[1])

    def test_dual_increases_on_day_violation_and_decreases_when_feasible(self):
        violated, feasible = SystemSAC(trainer(2)), SystemSAC(trainer(2))
        before = float(violated.trainer.lambda_param)
        self.assertGreater(violated.dual_step(.08, .04), before)
        self.assertLess(feasible.dual_step(.02, .04), before)

    def test_learning_changes_actor_and_critics_but_not_episodic_dual(self):
        learner = SystemSAC(trainer(2))
        replay = CostReplayBuffer(seed=1)
        for index in range(16):
            replay.push([index / 16, .2], [0 if index % 2 else 45], -index / 100,
                        -.02 if index % 2 else .01, [.2, .3], index == 15, 0)
        networks = (learner.trainer.policy_net, learner.trainer.q_net, learner.trainer.cost_q_net)
        before = [[p.detach().clone() for p in net.parameters()] for net in networks]
        dual = float(learner.trainer.lambda_param)
        metrics = learner.update(replay, 16)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(float(learner.trainer.lambda_param), dual)
        for net, weights in zip(networks, before):
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(net.parameters(), weights)))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
