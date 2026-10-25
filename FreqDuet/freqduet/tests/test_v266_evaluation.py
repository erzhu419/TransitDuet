from types import SimpleNamespace as NS
import unittest

import numpy as np

import analyze_freqduet_v266_evaluation as run


class Statistics(unittest.TestCase):
    def setUp(self):
        self.contexts = [dict(actor=a, demand_day=d) for a in (1,2) for d in (10,20)]
        self.weights = run.crossed_weights(self.contexts)

    def test_crossed_weights_preserve_actor_day_and_seed(self):
        np.testing.assert_array_equal(self.weights, run.crossed_weights(self.contexts))
        np.testing.assert_allclose(self.weights.sum(axis=1), 1.)
        self.assertTrue(np.any(self.weights[:,1] == 1.))
        constant = np.ones((4,8,1,1))*7
        counts = np.ones((run.DRAW_COUNT,4,8))
        point, intervals = run.intervals(constant,self.weights,counts)
        np.testing.assert_allclose(point,7.)
        for ci in intervals.values():
            np.testing.assert_allclose(ci,7.)

    def test_paired_draws_shared_across_metrics_and_arms(self):
        base = np.arange(32.).reshape(4,8,1,1)
        values = np.concatenate([base,-3*base],axis=3)
        values = np.concatenate([values,2*values],axis=2)
        counts = np.random.default_rng(253).multinomial(8,np.ones(8)/8,size=(run.DRAW_COUNT,4))
        _, intervals = run.intervals(values,self.weights,counts)
        for ci in intervals.values():
            np.testing.assert_allclose(ci[:,:,1],-3*ci[::-1,:,0])
            np.testing.assert_allclose(ci[:,1],2*ci[:,0])

    def test_negative_signal_and_opposite_half_signs_retained(self):
        values = np.tile(np.r_[np.ones(4),-np.ones(4)][None,:,None,None],(4,1,1,1))
        stats = run.signal_noise(values,values,self.weights)
        self.assertLess(stats['debiased_signal_squared']['mean'][0],0.)
        self.assertEqual(stats['E_half_cross_product']['mean'],[-1.])
        self.assertLess(stats['single_future_snr'][0],0.)
        self.assertEqual(stats['same_nonzero_E_half_sign_pairs'],[0])

    def test_native_aliases_not_counted_as_replication(self):
        values = np.zeros((4,8,1,1))
        stats = run.signal_noise(values,values,self.weights)
        self.assertEqual(stats['informative_context_action_pairs'],[0])
        self.assertEqual(stats['same_nonzero_E_half_sign_pairs'],[0])
        self.assertEqual(stats['single_future_snr'],[None])

    def test_one_active_context_cannot_masquerade_as_crossed_replication(self):
        values = np.zeros((4,8,1,1))
        values[1] = -1.
        _, intervals = run.intervals(values,self.weights,np.ones((run.DRAW_COUNT,4,8)))
        self.assertEqual(intervals['crossed_actor_day'][1,0,0],0.)
        self.assertLess(intervals['conditional_future'][1,0,0],0.)

    def test_joint_gate_keeps_zero_constraint_and_strict_passenger(self):
        stats = dict(passenger_person_s=dict(mean=-1.,ci95=[-2.,-.1]),
            cv_squared=dict(mean=0.,ci95=[0.,0.]))
        self.assertTrue(run.joint_gate(stats,['cv_squared'],True))
        stats['cv_squared']['ci95'][1] = 1e-12
        self.assertFalse(run.joint_gate(stats,['cv_squared'],True))
        stats['passenger_person_s']['mean'] = 0.
        self.assertFalse(run.joint_gate(stats,['cv_squared'],False))

    def test_tensor_uses_locked_S_arm_not_best_E_arm(self):
        bank = NS(METRICS=('passenger_person_s',), evaluation_unit=lambda c,f: (c,f), unit_key=lambda u:str(u))
        locks = [dict(chosen_branch='b2',plan=dict(physical=['native','b0','b2'])) for _ in range(4)]
        rows = {}
        for c in range(4):
            for f in range(8):
                paths = {b:dict(metrics=dict(passenger_person_s=v),delta_vs_native=dict(passenger_person_s=v-100))
                    for b,v in (('native',100),('b0',1),('b2',110))}
                rows[str((c,f))] = dict(paths=paths)
        values, arms = run.effect_tensor(rows,locks,bank)
        self.assertEqual(arms,['locked_primary','native','b0','b2'])
        np.testing.assert_allclose(values[:,:,0],10.)


if __name__ == '__main__':
    unittest.main()
