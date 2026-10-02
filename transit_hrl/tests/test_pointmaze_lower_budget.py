import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_lower_budget as experiment
from scripts import pointmaze_lower_budget_stage90_spec as spec
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as previous
from scripts.submit_pointmaze_call_weighted_stage87_scheduleurm import task_specification, qualification_task
import test_pointmaze_call_weighted as call_fixture
import test_pointmaze_feasible_credit as feasible_fixture


class LowerBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_only_lower_mean_updates_full_samples_and_exact_reserved_budget(self):
        source = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(source.state_dict())
        args,roles = spec.arguments(310011,preflight=True),spec.seed_roles(310011,preflight=True)
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        collector = call_fixture.CallWeightedTest()
        for p in spec.PERIODS:
            model = copy.deepcopy(source)
            for rr in roles['training_rounds']:
                batches = collector.collect(model,args,rr,p,'lower_matched')
                row = experiment.learning.update_mean(model,batches,method='lower_matched',period=p,horizon=args.horizon,
                    cost=cost,allocation=spec.allocation('lower_matched',p))
                nominal,_ = experiment.call_budget.check_update(row,method='lower_matched',period=p,horizon=args.horizon,
                    preflight=True,protocol=spec)
                self.assertEqual(set(row['actors']),{'lower'})
                self.assertEqual(row['actors']['lower']['gradient_episodes'],8)
                self.assertAlmostEqual(nominal,.001-.0005/p,places=12)
                experiment.learning.check_training_freeze(model,before,('lower',))
            self.assertFalse(torch.equal(model.lower_actor.net.state_dict()['0.weight'],source.lower_actor.net.state_dict()['0.weight']))
        for k in ('actor_mean_parameter_updates','policy_updates','training_freeze_checks','actor_score_forward_batches',
                'actor_score_backward_batches','fisher_jvp_batches','exact_kl_forward_batches'):
            self.assertEqual(cost[k],spec.budget(preflight=True)[k],k)
        experiment.learning.native.curves.support.assert_frozen(source,before)

    def test_only_registered_stage88_final_donors_are_loaded_and_composed(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        original = experiment.learning.native.joint.inference_weights(model)
        donors = {}
        for method,delta in (('joint_call',.01),('lower_trained',.02),('lower_matched',.03)):
            changed = copy.deepcopy(model)
            with torch.no_grad():
                for a in (('upper','lower') if method=='joint_call' else ('lower',)):
                    next(getattr(changed,a+'_actor').net.parameters()).add_(delta)
            donors[method] = experiment.learning.native.joint.inference_weights(changed)
        weights = {'base':original,'zero':original,'lower_matched':donors['lower_matched']}
        paths = {m:str(spec.training_result(310011).parent/'final_weights'/f'period_50_{m}.pt') for m in spec.CHECKPOINT_METHODS}
        training = {'groups':{'50':{'trained':{m:{'checkpoint':path,'evaluation_update':8,'final_freeze_check':'passed'}
            for m,path in paths.items()}}}}
        payloads = {path:{'protocol':spec.source.EXPERIMENT_PROTOCOL,'root':310011,'period':50,'method':m,'updates':8,
            'weights':donors[m]} for m,path in paths.items()}
        cost = dict.fromkeys(spec.budget(preflight=True),0)
        with patch.object(experiment.torch,'load',side_effect=lambda path,**kw:payloads[str(path)]) as load:
            composed,metadata = experiment.prepare_evaluation(training,310011,50,weights,cost)
        self.assertEqual(load.call_count,2)
        self.assertEqual((cost['checkpoint_loads'],cost['checkpoint_freeze_checks'],cost['actor_composition_checks']),(2,2,8))
        self.assertEqual(metadata['reused_checkpoints'],paths)
        all_donors = {'source':original,**donors}
        for variant,(upper,lower) in spec.COMPOSITIONS.items():
            torch.testing.assert_close(composed[variant],{**original,'upper_actor':all_donors[upper]['upper_actor'],
                'lower_actor':all_donors[lower]['lower_actor']},atol=0,rtol=0)
        bad = copy.deepcopy(training);bad['groups']['50']['trained']['joint_call']['evaluation_update']=7
        with self.assertRaises(ValueError):experiment.prepare_evaluation(bad,310011,50,weights,cost)

    def test_source_training_requires_exact_full_stage88_rosters(self):
        training = {'status':'complete','protocol':spec.source.EXPERIMENT_PROTOCOL,'root':310011,'preflight':False,
            'contract':spec.source.contract(),'seed_roles':spec.source.seed_roles(310011,preflight=False)}
        experiment.check_training(training,310011)
        bad = copy.deepcopy(training);bad['seed_roles']['training_rounds'][0]['credit_A'][0]['scenario_seed']+=1
        with self.assertRaises(ValueError):experiment.check_training(bad,310011)
        bad = copy.deepcopy(training);bad['preflight']=True
        with self.assertRaises(ValueError):experiment.check_training(bad,310011)

    def test_budget_and_training_components_exactly_decompose_both_fixed_upper_gaps(self):
        values = dict(zip(spec.VARIANTS,(0.,0.,2.,3.,1.,5.,6.,7.)))
        effects = {f'50/{a}_minus_{b}':values[a]-values[b] for a,b in spec.CONTRAST_PAIRS}
        experiment.check_decomposition(50,effects)
        bad = dict(effects);bad['50/lower_matched_minus_lower_full']+=.01
        with self.assertRaises(AssertionError):experiment.check_decomposition(50,bad)

    def test_one_new_learner_budget_paired_training_fresh_eval_and_unpinned_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['native_episodes']*8,b['native_steps']*8),(12288,14745600))
        self.assertEqual((b['actor_mean_parameter_updates']*8,b['checkpoint_writes']*8,b['checkpoint_loads']*8,b['actor_composition_checks']*8),(128,16,32,128))
        self.assertEqual(spec.METHODS,{'lower_matched':('lower',)})
        self.assertEqual(len(spec.ENDPOINTS),28)
        new_eval = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root,preflight=preflight)
                old = spec.source.seed_roles(root,preflight=False)
                if not preflight:self.assertEqual(roles['training_rounds'],old['training_rounds'])
                else:
                    for current,rr in zip(roles['training_rounds'],old['training_rounds']):
                        for name in ('A','B'):self.assertEqual(current['credit_'+name],rr['credit_'+name][:2])
                new_eval.extend(roles['native_evaluation'])
                self.assertFalse(set(roles['native_evaluation'])&set(call_fixture.seeds(old)))
                self.assertFalse(set(roles['native_evaluation'])&set(previous.seed_roles(root,preflight=preflight)['native_evaluation']))
                task = task_specification('unit_stage90',root,preflight=preflight,protocol_spec=spec)
                self.assertIn(spec.RUNNER_SCRIPT,task['cmd'])
                self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
                self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
                self.assertFalse(task.get('require_node'))
                self.assertTrue(task['result_dir'].endswith('/completion'))
            q = qualification_task('unit_stage90',preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q['cmd'])
            self.assertIsNone(q['result_dir'])
            self.assertEqual(len(q['wait_for_files']),len(spec.roots(preflight=preflight)))
        self.assertEqual(len(new_eval),len(set(new_eval)))

    def test_all28_endpoints_share_one_equal_root_corrected_family(self):
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},
            'cost':spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c),patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.learning.native.np,'quantile',wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False)
        self.assertEqual(quantile.call_args.args[1],[.05/56,1-.05/56])
        self.assertEqual(set(result['endpoints']),set(spec.ENDPOINTS))
        self.assertIn('Stage67_critic_route_HOLD_unchanged',result['native_trial_prerequisite'])
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False)


if __name__ == '__main__':
    unittest.main()
