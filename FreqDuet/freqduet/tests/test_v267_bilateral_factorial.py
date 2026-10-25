from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_freqduet_v267_bilateral_factorial as run


class MatchedFactorial(unittest.TestCase):
    def test_consumed_prefix_grid_is_not_outcome_selected(self):
        self.assertEqual(run.SEEDS,(12011,12049,12071,12097))
        self.assertEqual(run.DAYS,(2071001,2071002,2071003,2071004))
        self.assertEqual(len(run.ITEMS),40)
        self.assertEqual(len(set(run.ITEMS)),40)
        self.assertEqual(sum(m.startswith('fixed_') for s,d,m in run.ITEMS),8)
        self.assertTrue(all(s == run.SEEDS[0] for s,d,m in run.ITEMS if m.startswith('fixed_')))

    def test_four_smoke_paths_reused_without_new_reference_slots(self):
        self.assertEqual(len(run.SMOKE_MAIN),4)
        self.assertEqual(len(run.REMAINING),36)
        self.assertFalse(set(run.REMAINING)&set(run.SMOKE_MAIN))
        self.assertEqual(sorted(x for shard in run.SHARDS for x in shard),sorted(run.REMAINING))
        self.assertEqual([len(s) for s in run.SHARDS],[5,5,5,5,4,4,4,4])

    def test_bilateral_implementation_and_gain_unchanged(self):
        self.assertIs(run.BenchmarkAudit.__call__,run.prior.TradeoffAudit.__call__)
        self.assertEqual(run.prior.RULES['bilateral025'],('bilateral',.25))
        for mode in run.ARMS:
            audit = run.BenchmarkAudit(run.SEEDS[0],mode)
            self.assertEqual(audit.arm,mode)
            self.assertEqual(audit.method,mode.split('_',1)[1])

    def test_zero_uses_existing_legal_action(self):
        class Policy:
            def feasible_action_mask(self,states):
                return torch.ones(len(states),7,dtype=torch.bool)
        for mode in ('learned_zero','fixed_zero'):
            audit = run.BenchmarkAudit(run.SEEDS[0],mode)
            result = audit(Policy(),torch.zeros(1,2),None,None)
            np.testing.assert_array_equal(result,[[1,0,0,0,0,0,0]])

    def test_exact_factorial_interaction(self):
        values = {m:dict.fromkeys(run.METRICS,v) for m,v in zip(run.ARMS,(12.,7.,9.,3.))}
        effects = run.contrasts(values)
        for m in run.METRICS:
            self.assertEqual(effects['holding_at_learned'][m],5.)
            self.assertEqual(effects['holding_at_fixed'][m],6.)
            self.assertEqual(effects['upper_with_bilateral'][m],3.)
            self.assertEqual(effects['upper_without_holding'][m],4.)
            self.assertEqual(effects['interaction'][m],-1.)

    def test_failures_keep_debits_and_unregistered_paths_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for item in run.ITEMS:
                run.claim(root,'test',run.key(item))
            for name in ('fixed_actor_check','external_fixed_reference'):
                run.claim(root,'test',name)
            self.assertEqual(len(list((root/'started').glob('*.json'))),42)
            with self.assertRaises(FileExistsError):
                run.claim(root,'test',run.key(run.ITEMS[0]))
            with self.assertRaises(AssertionError):
                run.claim(root,'test','replacement')

    def test_truncation_retains_unexecuted_commands_as_outcome(self):
        audit = run.BenchmarkAudit(run.SEEDS[0],'learned_bilateral025')
        audit.scalars = (1.,2.,3.)
        runner = NS(lower_trainer=NS(alpha=1.,lambda_param=2.),upper_trainer=NS(alpha=3.),cfg={},
            env=NS(bus_all=[NS(episode_hold_person_seconds=10.,episode_hold_vehicle_seconds=2.) for _ in range(12)],
                _done_reason='measurement_horizon'))
        helpers = run.prior.factorial.prior
        with patch.object(helpers,'physical_contract',return_value={}), \
                patch.object(helpers,'dispatch_metrics',return_value={}):
            audit.finish(runner,[],[dict(action_s=30.)])
        self.assertEqual(audit.execution['executed_s'],24.)
        self.assertEqual(audit.execution['unexecuted_s'],6.)
        self.assertEqual(audit.execution['done_reason'],'measurement_horizon')


if __name__ == '__main__':
    unittest.main()
