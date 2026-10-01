import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_iterative_mc as experiment
from scripts import pointmaze_iterative_mc_stage83_spec as spec
from scripts.submit_pointmaze_iterative_mc_stage83_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class IterativeMCTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_multiple_fresh_rounds_update_only_registered_means_with_fixed_budget(self):
        original = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(original.state_dict())
        models = {m:copy.deepcopy(original) for m in spec.METHODS}
        collector = scenario_fixture.ScenarioCreditTest()
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        for round_roles in roles['training_rounds']:
            for method,model in models.items():
                prior = copy.deepcopy(model.state_dict())
                batches = {b:collector.collect(model,args,round_roles['credit_'+b],variant=method) for b in ('A','B')}
                row = experiment.update_mean(model,batches,method=method,period=50,horizon=args.horizon,cost=cost)
                self.assertEqual(set(row['actors']),set(spec.METHODS[method]))
                self.assertTrue(.0005 <= row['exact_sum_kl'] <= .002)
                experiment.check_training_freeze(model,prior,spec.METHODS[method])
                for actor,r in row['actors'].items():
                    self.assertEqual(r['geometry']['nominal_fisher_kl'],.001*spec.METHODS[method][actor])
                    self.assertLessEqual(r['max_abs_old_logp_difference'],1e-5)
                    changes = [not torch.equal(v,model.state_dict()[actor+'_actor'][k]) for k,v in prior[actor+'_actor'].items()]
                    self.assertTrue(any(changes))
        expected = {'objective_checks':32,'mc_calls':64,'actor_score_forward_batches':48,'actor_score_backward_batches':144,
            'fisher_jvp_batches':14,'exact_kl_forward_batches':28,'actor_parameter_perturbations':12,
            'actor_mean_parameter_updates':6,'parameter_part_checks':6,'training_freeze_checks':4,'policy_updates':4}
        for k,v in expected.items():self.assertEqual(cost[k],v)
        for method,model in models.items():experiment.check_training_freeze(model,before,spec.METHODS[method])
        experiment.native.curves.support.assert_frozen(original,before)

    def test_training_freeze_rejects_std_value_and_inactive_upper_changes(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(model.state_dict())
        for part in ('std','value','upper'):
            bad = copy.deepcopy(model)
            with torch.no_grad():
                if part == 'std':bad.lower_actor.log_std.add_(.01)
                if part == 'value':next(bad.lower_value.parameters()).add_(.01)
                if part == 'upper':next(bad.upper_actor.net.parameters()).add_(.01)
            with self.assertRaises(AssertionError):experiment.check_training_freeze(bad,before,spec.METHODS['lower_trained'])

    def test_final_inference_weights_reload_without_optimizer_or_intermediate_checkpoint(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        with tempfile.TemporaryDirectory() as tmp:
            path = experiment.final_checkpoint(model,Path(tmp)/'result.json',root=310011,period=50,method='joint_trained',updates=8)
            saved = torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual((saved['protocol'],saved['root'],saved['period'],saved['method'],saved['updates']),
                (spec.EXPERIMENT_PROTOCOL,310011,50,'joint_trained',8))
            torch.testing.assert_close(saved['weights'],experiment.native.joint.inference_weights(model),atol=0,rtol=0)
            self.assertEqual(len(list(Path(tmp).rglob('*.pt'))),1)
            self.assertFalse(any('optimizer' in k for k in saved['weights']))

    def test_fixed_last_update_budget_disjoint_rosters_and_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['native_episodes']*8,b['native_steps']*8),(18432,22118400))
        self.assertEqual((b['credit_episodes']*8,b['evaluation_episodes']*8),(16384,2048))
        self.assertEqual((b['actor_score_forward_batches']*8,b['actor_score_backward_batches']*8),(40960,122880))
        self.assertEqual((b['fisher_jvp_batches']*8,b['exact_kl_forward_batches']*8),(19392,38784))
        self.assertEqual((b['policy_updates']*8,b['actor_mean_parameter_updates']*8,b['checkpoint_writes']*8),(256,384,32))
        self.assertEqual(len(spec.ENDPOINTS),12)
        all_seeds = {}
        for preflight in (True,False):
            all_seeds[preflight] = []
            self.assertEqual(spec.options(preflight=preflight)['updates'],2 if preflight else 8)
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                seeds = [s['scenario_seed'] for r in roles['training_rounds'] for b in ('A','B') for s in r['credit_'+b]]
                seeds += [n for r in roles['training_rounds'] for b in ('A','B') for s in r['credit_'+b] for n in s['noise_seeds']]
                seeds += roles['native_evaluation']; all_seeds[preflight].extend(seeds)
                previous = spec.source.seed_roles(root,preflight=preflight)
                old = [s['scenario_seed'] for b in ('A','B') for s in previous['credit_'+b]]
                old += [n for b in ('A','B') for s in previous['credit_'+b] for n in s['noise_seeds']]
                old += previous['native_evaluation']
                self.assertFalse(set(seeds)&set(old))
                task = task_specification('unit_stage83',root,preflight=preflight)
                self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
                self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
                self.assertFalse(task.get('require_node'))
                self.assertTrue(task['result_dir'].endswith('/completion'))
            self.assertEqual(len(all_seeds[preflight]),len(set(all_seeds[preflight])))
            self.assertIsNone(qualification_task('unit_stage83',preflight=preflight)['result_dir'])
        self.assertFalse(set(all_seeds[True])&set(all_seeds[False]))

    def test_all12_final_endpoints_share_one_root_bootstrap_and_separate_critic_hold(self):
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},'cost':spec.budget(preflight=False)}
            for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c),patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/24,1-.05/24])
        self.assertEqual(set(result['endpoints']),set(spec.ENDPOINTS))
        self.assertIn('iterative_MC_mean_learning',result['performance_claim'])
        self.assertIn('Stage67_critic_route_HOLD_unchanged',result['native_trial_prerequisite'])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == '__main__':
    unittest.main()
