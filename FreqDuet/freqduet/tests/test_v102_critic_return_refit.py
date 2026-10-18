import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_freqduet_v102_critic_return_refit import NativeBatch, audit, other_day_index, refit
from lower.resac_lagrangian import RESACLagrangianTrainer


class CriticRefitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(state=[0., 1.], next_state=[1., 0.], action=1, reward=2., cost=.1,
                          done=False, trip=1),
                     dict(state=[1., 0.], next_state=[0., 0.], action=2, reward=3., cost=.2,
                          done=True, trip=1)]
        self.returns = (np.array([5., 3.]), np.array([.3, .2]))

    def test_crossfit_never_uses_the_evaluation_day(self):
        for i, (seed, day) in enumerate(audit.ITEMS):
            held = other_day_index(i)
            self.assertEqual(audit.ITEMS[held][0], seed)
            self.assertNotEqual(audit.ITEMS[held][1], day)
            self.assertEqual(other_day_index(held), i)

    def test_matched_batches_preserve_actions_but_change_only_labels(self):
        td = NativeBatch(self.rows, self.returns, 'td', 17).sample(8)
        mc = NativeBatch(self.rows, self.returns, 'mc', 17).sample(8)
        for i in (0, 1, 6):
            np.testing.assert_array_equal(td[i], mc[i])
        for i, row in enumerate(mc[0]):
            index = int(row[0])
            self.assertEqual(mc[2][i, 0], self.returns[0][index])
            self.assertEqual(mc[3][i, 0], self.returns[1][index])
        np.testing.assert_array_equal(mc[4], 0)
        np.testing.assert_array_equal(mc[5], 1)

    def test_refit_changes_critics_not_original_actor_or_dual(self):
        torch.manual_seed(3)
        trainer = RESACLagrangianTrainer(2, hidden_dim=8, ensemble_size=2,
            action_range=45., action_bins=[0., 5., 10., 15., 20., 30., 45.], device='cpu')
        before = copy.deepcopy(trainer.q_net.state_dict())
        for mode in ('td', 'mc'):
            fitted, report = refit(trainer, self.rows, self.returns, mode, 17, steps=2, batch=4)
            self.assertEqual(report['updates'], 2)
            self.assertTrue(any(not torch.equal(before[k], fitted.q_net.state_dict()[k]) for k in before))
            for k in before:
                torch.testing.assert_close(before[k], trainer.q_net.state_dict()[k], atol=0, rtol=0)

    def test_clipped_mc_is_rejected_instead_of_silently_changing_target(self):
        trainer = RESACLagrangianTrainer(2, hidden_dim=8, ensemble_size=2, device='cpu')
        with self.assertRaises(AssertionError):
            refit(trainer, self.rows, (np.array([101., 0.]), self.returns[1]), 'mc', 17, steps=1, batch=2)


if __name__ == '__main__':
    unittest.main()
