import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_aligned_order as experiment
from scripts import pointmaze_aligned_order_stage86_spec as spec
from scripts.submit_pointmaze_aligned_order_stage86_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class AlignedOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_level_datasets_match_paired_and_alternating_not_original_staged(self):
        for root in spec.roots(preflight=False):
            original = spec.source.seed_roles(root,preflight=False)
            corrected = spec.seed_roles(root,preflight=False)
            mapping = corrected['source_chunk_order']
            self.assertEqual(mapping,list(range(0,16,2))+list(range(1,16,2)))
            for slot,chunk in enumerate(corrected['training_chunks']):
                self.assertEqual(chunk,original['training_chunks'][mapping[slot]])
            for method in ('paired_joint','alternating'):
                for actor,slots in (('lower',range(8)),('upper',range(8,16))):
                    old = [c for g in spec.source.training_groups(method,preflight=False) for u in g['updates']
                        if u['actor']==actor for c in u['chunks']]
                    self.assertEqual([mapping[s] for s in slots],old)
            self.assertNotEqual(mapping,list(range(16)))
            old_seeds = original['native_evaluation']
            old_seeds += [s['scenario_seed'] for c in original['training_chunks'] for b in ('A','B') for s in c['credit_'+b]]
            old_seeds += [n for c in original['training_chunks'] for b in ('A','B') for s in c['credit_'+b] for n in s['noise_seeds']]
            self.assertFalse(set(corrected['native_evaluation'])&set(old_seeds))
        pref = spec.seed_roles(310011,preflight=True)
        self.assertEqual(pref['source_chunk_order'],[0,2,1,3])
        self.assertTrue(all(len(c['credit_A'])==2 for c in pref['training_chunks']))

    def test_real_aligned_training_records_source_chunks_and_freezes_inactive_level(self):
        original = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(original.state_dict())
        model = copy.deepcopy(original)
        roles,args = spec.seed_roles(310011,preflight=True),spec.arguments(310011,preflight=True)
        collector,captured = scenario_fixture.ScenarioCreditTest(),{}
        cost = dict.fromkeys(spec.budget(preflight=True),0)

        def collect(current,index):
            captured[index] = experiment.native.joint.inference_weights(current)
            return {b:collector.collect(current,args,roles['training_chunks'][index]['credit_'+b],variant='staged_aligned') for b in ('A','B')}

        history = experiment.order.train_model(model,method='staged',period=50,horizon=args.horizon,preflight=True,
            collect_chunk=collect,cost=cost)
        experiment.annotate_history(history,preflight=True)
        cumulative = experiment.check_history(history,preflight=True)
        self.assertEqual([r['source_credit_chunks'][0] for r in history],[0,2,1,3])
        self.assertEqual([next(iter(r['actors'])) for r in history],['lower','lower','upper','upper'])
        self.assertEqual(cumulative['nominal_by_level'],{'upper':.001,'lower':.001})
        torch.testing.assert_close(captured[0]['upper_actor'],captured[1]['upper_actor'],atol=0,rtol=0)
        torch.testing.assert_close(captured[2]['lower_actor'],captured[3]['lower_actor'],atol=0,rtol=0)
        experiment.order.learning.check_training_freeze(model,before,('upper','lower'))
        experiment.native.curves.support.assert_frozen(original,before)
        for key in ('objective_checks','mc_calls','actor_score_forward_batches','actor_score_backward_batches',
                'fisher_jvp_batches','exact_kl_forward_batches','actor_mean_parameter_updates','collection_freeze_checks'):
            self.assertEqual(cost[key],spec.budget(preflight=True)[key]//2)
        bad = copy.deepcopy(history);bad[1]['source_credit_chunks'] = [1]
        with self.assertRaises(ValueError):experiment.check_history(bad,preflight=True)

    def test_baseline_checkpoint_identity_exact_reload_std_values_and_inactive_upper(self):
        original = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(original.state_dict())
        for method in spec.BASELINES.values():
            changed = copy.deepcopy(original)
            with torch.no_grad():
                for actor in (('lower',) if method=='lower_trained' else ('upper','lower')):
                    next(getattr(changed,actor+'_actor').net.parameters()).add_(.01)
            payload = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'root':310011,'period':50,'method':method,
                'updates':8 if method=='lower_trained' else 16,'credit_chunks':16,
                'weights':experiment.native.joint.inference_weights(changed)}
            model = experiment.baseline_model(payload,original,root=310011,period=50,method=method)
            torch.testing.assert_close(experiment.native.joint.inference_weights(model),payload['weights'],atol=0,rtol=0)
            bad = copy.deepcopy(payload);bad['updates'] += 1
            with self.assertRaises(ValueError):experiment.baseline_model(bad,original,root=310011,period=50,method=method)
            bad = copy.deepcopy(payload);bad['weights']['lower_actor']['log_std'].add_(.01)
            with self.assertRaises(AssertionError):experiment.baseline_model(bad,original,root=310011,period=50,method=method)
            bad = copy.deepcopy(payload);next(iter(bad['weights']['lower_value'].values())).add_(.01)
            with self.assertRaises(AssertionError):experiment.baseline_model(bad,original,root=310011,period=50,method=method)
            if method=='lower_trained':
                bad = copy.deepcopy(payload);next(v for k,v in bad['weights']['upper_actor'].items() if k!='log_std').add_(.01)
                with self.assertRaises(AssertionError):experiment.baseline_model(bad,original,root=310011,period=50,method=method)
        experiment.native.curves.support.assert_frozen(original,before)

    def test_only_aligned_final_checkpoint_and_registered_cost_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['credit_episodes']*8,b['evaluation_episodes']*8,b['native_steps']*8),(8192,3584,14131200))
        self.assertEqual((b['baseline_checkpoint_loads']*8,b['checkpoint_writes']*8,b['actor_mean_parameter_updates']*8),(64,16,256))
        model = feasible_fixture.FeasibleCreditTest().source()
        with tempfile.TemporaryDirectory() as tmp:
            path = experiment.final_checkpoint(model,Path(tmp)/'result.json',root=310011,period=50,history=[{}]*16)
            payload = torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual((payload['protocol'],payload['method'],payload['updates']),(spec.EXPERIMENT_PROTOCOL,'staged_aligned',16))
            self.assertEqual(payload['source_chunk_order'],spec.source_chunk_order(preflight=False))
            self.assertEqual(len(list(Path(tmp).rglob('*.pt'))),1)
        for preflight in (True,False):
            task = task_specification('unit_stage86',310011,preflight=preflight)
            self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
            self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
            self.assertFalse(task.get('require_node'))
            self.assertTrue(task['result_dir'].endswith('/completion'))
            self.assertIsNone(qualification_task('unit_stage86',preflight=preflight)['result_dir'])

    def test_all18_endpoints_share_one_root_family_and_preserve_hold(self):
        self.assertEqual(len(spec.ENDPOINTS),18)
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},'cost':spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c),patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            d = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/36,1-.05/36])
        self.assertIn('not_new_independent_training_replication',d['performance_claim'])
        self.assertIn('Stage67_critic_route_HOLD_unchanged',d['native_trial_prerequisite'])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == '__main__':
    unittest.main()
