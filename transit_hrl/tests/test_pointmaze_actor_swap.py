import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_actor_swap as experiment
from scripts import pointmaze_actor_swap_stage84_spec as spec
from scripts.submit_pointmaze_actor_swap_stage84_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class ActorSwapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def donors(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        original = experiment.native.joint.inference_weights(model)
        trained = {}
        for method in spec.source.METHODS:
            changed = copy.deepcopy(model)
            with torch.no_grad():
                for actor in spec.source.METHODS[method]:
                    next(getattr(changed,actor+'_actor').net.parameters()).add_(.01 if method == 'joint_trained' else .02)
            trained[method] = experiment.native.joint.inference_weights(changed)
        return model,original,trained

    def payload(self, weights, method='joint_trained'):
        return {'protocol':spec.source.EXPERIMENT_PROTOCOL,'root':310011,'period':50,
            'method':method,'updates':8,'weights':copy.deepcopy(weights)}

    def test_final_checkpoint_identity_and_freeze(self):
        _,original,trained = self.donors()
        for method,weights in trained.items():
            payload = self.payload(weights,method)
            experiment.check_checkpoint(payload,original,root=310011,period=50,method=method)
            for field in ('root','period','updates'):
                bad = copy.deepcopy(payload);bad[field] += 1
                with self.assertRaises(ValueError):experiment.check_checkpoint(bad,original,root=310011,period=50,method=method)
        for name,key in (('upper_actor','log_std'),('lower_actor','log_std'),
                ('upper_value',next(iter(original['upper_value']))),('lower_value',next(iter(original['lower_value']))),
                ('upper_actor',next(k for k in original['upper_actor'] if k != 'log_std'))):
            bad = self.payload(trained['lower_trained'],'lower_trained')
            bad['weights'][name][key].add_(.1)
            with self.assertRaises(AssertionError):experiment.check_checkpoint(bad,original,root=310011,period=50,method='lower_trained')
        with self.assertRaises(ValueError):
            experiment.check_checkpoint(self.payload(original),original,root=310011,period=50,method='joint_trained')

    def test_exact_actor_crosses_do_not_mutate_source_or_donors(self):
        model,original,trained = self.donors()
        before = copy.deepcopy(model.state_dict())
        snapshot = copy.deepcopy(trained)
        composed = experiment.compose_weights(original,trained)
        self.assertEqual(set(composed),set(spec.VARIANTS))
        for variant,(u,l) in spec.COMPOSITIONS.items():
            donors = {'source':original,**trained}
            expected = {**original,'upper_actor':donors[u]['upper_actor'],'lower_actor':donors[l]['lower_actor']}
            torch.testing.assert_close(composed[variant],expected,atol=0,rtol=0)
        composed['joint_upper_lower_only_lower']['upper_actor']['log_std'].add_(1.)
        torch.testing.assert_close(trained,snapshot,atol=0,rtol=0)
        experiment.native.curves.support.assert_frozen(model,before)

    def test_changed_upper_and_lower_share_native_standard_noise_without_training_trace(self):
        model,original,trained = self.donors()
        weights = experiment.compose_weights(original,trained)
        collector = scenario_fixture.ScenarioCreditTest()
        seeds = spec.seed_roles(310011,preflight=True)['native_evaluation'][:2]
        roster = [{'scenario_seed':s,'noise_seeds':[s,s+100]} for s in seeds]
        evaluation = {}
        for variant in spec.VARIANTS:
            pairs = collector.collect(model,spec.arguments(310011,preflight=True),roster,
                collect=False,variant=variant,weights=weights[variant])
            self.assertTrue(all(batch is None for g in pairs for batch,_ in g))
            evaluation[variant] = [g[0][1] for g in pairs]
        observed = experiment.paired_effects(50,evaluation,seeds)
        self.assertEqual(set(observed),{k for k in spec.ENDPOINTS if k.startswith('50/')})
        expected = observed['50/joint_trained_minus_source_upper_joint_lower']-observed['50/joint_upper_lower_only_lower_minus_lower_trained']
        self.assertAlmostEqual(observed['50/upper_by_lower_interaction'],expected)
        bad = copy.deepcopy(evaluation);bad['joint_upper_source_lower'][0]['lower_seed'] += 1
        with self.assertRaises(ValueError):experiment.paired_effects(50,bad,seeds)
        bad = copy.deepcopy(evaluation);bad['joint_upper_source_lower'][0]['upper_standard_noise'][0][0] += .1
        with self.assertRaises(AssertionError):experiment.paired_effects(50,bad,seeds)

    def test_budget_fresh_seeds_completion_only_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['native_episodes']*8,b['native_steps']*8),(3584,4300800))
        self.assertEqual((b['checkpoint_loads']*8,b['policy_updates'],b['checkpoint_writes'],b['credit_episodes']),(32,0,0,0))
        self.assertEqual(len(spec.ENDPOINTS),22)
        all_seeds = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                seeds = spec.seed_roles(root,preflight=preflight)['native_evaluation']
                all_seeds.extend(seeds)
                previous = spec.source.seed_roles(root,preflight=preflight)
                old = previous['native_evaluation']+[s['scenario_seed'] for r in previous['training_rounds'] for b in ('A','B') for s in r['credit_'+b]]
                old += [n for r in previous['training_rounds'] for b in ('A','B') for s in r['credit_'+b] for n in s['noise_seeds']]
                self.assertFalse(set(seeds)&set(old))
                task = task_specification('unit_stage84',root,preflight=preflight)
                self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
                self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
                self.assertFalse(task.get('require_node'))
                self.assertTrue(task['result_dir'].endswith('/completion'))
            q = qualification_task('unit_stage84',preflight=preflight)
            self.assertIsNone(q['result_dir'])
            self.assertEqual(len(q['wait_for_files']),len(spec.roots(preflight=preflight)))
        self.assertEqual(len(all_seeds),len(set(all_seeds)))

    def test_all22_endpoints_share_one_equal_root_family(self):
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},'cost':spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c),patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/44,1-.05/44])
        self.assertEqual(set(result['endpoints']),set(spec.ENDPOINTS))
        self.assertIn('fixed_final_checkpoint_causal',result['performance_claim'])
        self.assertIn('Stage67_critic_route_HOLD_unchanged',result['native_trial_prerequisite'])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == '__main__':
    unittest.main()
