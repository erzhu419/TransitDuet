import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_training_order as experiment
from scripts import pointmaze_training_order_stage85_spec as spec
from scripts import pointmaze_actor_swap_stage84_spec as swap_spec
from scripts.submit_pointmaze_training_order_stage85_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class TrainingOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_frozen_orders_match_dual_actor_counts_samples_and_cumulative_KL(self):
        for preflight in (True,False):
            o = spec.options(preflight=preflight)
            count = o['credit_chunks']//2
            n = 2*o['credit_scenarios_per_batch']*o['rollouts_per_scenario']
            for method in spec.METHODS:
                groups = spec.training_groups(method,preflight=preflight)
                self.assertEqual([c for g in groups for c in g['chunks']],list(range(o['credit_chunks'])))
                updates = [u for g in groups for u in g['updates']]
                self.assertEqual([c for u in updates for c in u['chunks']],list(range(o['credit_chunks'])))
                nominal = {a:sum(u['fraction']*spec.FISHER_RADIUS for u in updates if u['actor']==a) for a in ('upper','lower')}
                self.assertAlmostEqual(sum(nominal.values()),count*.001)
                if method != 'lower_trained':
                    for a in ('upper','lower'):
                        self.assertEqual(sum(u['actor']==a for u in updates),count)
                        self.assertAlmostEqual(nominal[a],count*.0005)
                        self.assertTrue(all(n*len(u['chunks'])==n for u in updates))
                else:self.assertEqual(len(updates),count)
            staged = [u['actor'] for g in spec.training_groups('staged',preflight=preflight) for u in g['updates']]
            self.assertEqual(staged,['lower']*count+['upper']*count)

    def test_real_mean_updates_use_registered_collection_policy_and_freeze_other_networks(self):
        original = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(original.state_dict())
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        collector = scenario_fixture.ScenarioCreditTest()
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        all_history = {}
        for method in spec.METHODS:
            model = copy.deepcopy(original)
            observed = {}

            def collect(current,index):
                observed[index] = experiment.native.joint.inference_weights(current)
                return {b:collector.collect(current,args,roles['training_chunks'][index]['credit_'+b],variant=method) for b in ('A','B')}

            history = experiment.train_model(model,method=method,period=50,horizon=args.horizon,preflight=True,
                collect_chunk=collect,cost=cost)
            all_history[method] = history
            summary = experiment.check_history(history,method=method,preflight=True)
            self.assertAlmostEqual(sum(summary['nominal_by_level'].values()),.002)
            active = ('lower',) if method=='lower_trained' else ('upper','lower')
            experiment.learning.check_training_freeze(model,before,active)
            for row in history:
                self.assertLess(max(d['max_abs_old_logp_difference'] for d in row['actors'].values()),1e-5)
            if method in ('paired_joint','lower_trained'):
                for i in (0,2):torch.testing.assert_close(observed[i],observed[i+1],atol=0,rtol=0)
            else:
                self.assertTrue(any(not torch.equal(v,observed[1]['lower_actor'][k]) for k,v in observed[0]['lower_actor'].items()))
                torch.testing.assert_close(observed[0]['upper_actor'],observed[1]['upper_actor'],atol=0,rtol=0)
                if method=='staged':torch.testing.assert_close(observed[2]['lower_actor'],observed[3]['lower_actor'],atol=0,rtol=0)
        for key in ('objective_checks','mc_calls','actor_score_forward_batches','actor_score_backward_batches',
                'fisher_jvp_batches','exact_kl_forward_batches','actor_parameter_perturbations','parameter_part_checks',
                'actor_mean_parameter_updates','policy_updates','training_freeze_checks','collection_freeze_checks'):
            self.assertEqual(cost[key],spec.budget(preflight=True)[key]//2)
        experiment.native.curves.support.assert_frozen(original,before)
        for field in ('credit_chunks','rollout_policy_update_count','credit_episodes_used','allocation'):
            bad = copy.deepcopy(all_history['paired_joint'])
            bad[0][field] = {'lower':1.} if field=='allocation' else [1] if field=='credit_chunks' else bad[0][field]+1
            with self.assertRaises(ValueError):experiment.check_history(bad,method='paired_joint',preflight=True)

    def test_only_final_inference_checkpoint_records_new_protocol_and_update_count(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        with tempfile.TemporaryDirectory() as tmp:
            path = experiment.final_checkpoint(model,Path(tmp)/'result.json',root=310011,period=50,method='staged',history=[{}]*16)
            payload = torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual((payload['protocol'],payload['root'],payload['period'],payload['method'],payload['updates'],payload['credit_chunks']),
                (spec.EXPERIMENT_PROTOCOL,310011,50,'staged',16,16))
            torch.testing.assert_close(payload['weights'],experiment.native.joint.inference_weights(model),atol=0,rtol=0)
            self.assertEqual(len(list(Path(tmp).rglob('*.pt'))),1)
            self.assertFalse(any('optimizer' in k for k in payload['weights']))

    def test_budget_disjoint_rosters_and_dynamic_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['credit_episodes']*8,b['evaluation_episodes']*8),(32768,3072))
        self.assertEqual((b['native_episodes']*8,b['native_steps']*8),(35840,43008000))
        self.assertEqual((b['checkpoint_writes']*8,b['actor_mean_parameter_updates']*8),(64,896))
        all_seeds = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                seeds = [s['scenario_seed'] for c in roles['training_chunks'] for b in ('A','B') for s in c['credit_'+b]]
                seeds += [n for c in roles['training_chunks'] for b in ('A','B') for s in c['credit_'+b] for n in s['noise_seeds']]
                seeds += roles['native_evaluation'];all_seeds.extend(seeds)
                previous = spec.source.seed_roles(root,preflight=preflight)
                old = [s['scenario_seed'] for c in previous['training_rounds'] for b in ('A','B') for s in c['credit_'+b]]
                old += [n for c in previous['training_rounds'] for b in ('A','B') for s in c['credit_'+b] for n in s['noise_seeds']]
                old += previous['native_evaluation']+swap_spec.seed_roles(root,preflight=preflight)['native_evaluation']
                self.assertFalse(set(seeds)&set(old))
                task = task_specification('unit_stage85',root,preflight=preflight)
                self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
                self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
                self.assertFalse(task.get('require_node'))
                self.assertTrue(task['result_dir'].endswith('/completion'))
            self.assertIsNone(qualification_task('unit_stage85',preflight=preflight)['result_dir'])
        self.assertEqual(len(all_seeds),len(set(all_seeds)))

    def test_one22_endpoint_root_bootstrap_family_preserves_critic_hold(self):
        self.assertEqual(len(spec.ENDPOINTS),22)
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},'cost':spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c),patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/44,1-.05/44])
        self.assertEqual(set(result['endpoints']),set(spec.ENDPOINTS))
        self.assertIn('matched_sample_cumulative_nominal_KL',result['performance_claim'])
        self.assertIn('Stage67_critic_route_HOLD_unchanged',result['native_trial_prerequisite'])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == '__main__':
    unittest.main()
