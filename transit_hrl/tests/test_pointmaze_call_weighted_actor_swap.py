import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_actor_swap as experiment
from scripts import pointmaze_call_weighted_actor_swap_stage89_spec as spec
from scripts.submit_pointmaze_actor_swap_stage84_scheduleurm import task_specification, qualification_task
from test_pointmaze_feasible_credit import FeasibleCreditTest


class CallWeightedActorSwapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def donors(self):
        model = FeasibleCreditTest().source()
        original = experiment.native.joint.inference_weights(model)
        trained = {}
        for method in spec.CHECKPOINT_METHODS:
            changed = copy.deepcopy(model)
            with torch.no_grad():
                for actor in spec.source.METHODS[method]:
                    next(getattr(changed,actor+'_actor').net.parameters()).add_(.01 if method == 'joint_call' else .02)
            trained[method] = experiment.native.joint.inference_weights(changed)
        return model,original,trained

    def test_stage88_last_checkpoint_identity_and_frozen_parameters(self):
        _,original,trained = self.donors()
        for method in spec.CHECKPOINT_METHODS:
            payload = {'protocol':spec.source.EXPERIMENT_PROTOCOL,'root':310011,'period':50,
                'method':method,'updates':8,'weights':copy.deepcopy(trained[method])}
            experiment.check_checkpoint(payload,original,root=310011,period=50,method=method,protocol=spec)
            for field,value in (('protocol',spec.source.source.EXPERIMENT_PROTOCOL),('updates',2),('method','joint_level')):
                bad = copy.deepcopy(payload);bad[field] = value
                with self.assertRaises(ValueError):
                    experiment.check_checkpoint(bad,original,root=310011,period=50,method=method,protocol=spec)
            for name,key in (('upper_actor','log_std'),('lower_value',next(iter(original['lower_value'])))):
                bad = copy.deepcopy(payload);bad['weights'][name][key].add_(.1)
                with self.assertRaises(AssertionError):
                    experiment.check_checkpoint(bad,original,root=310011,period=50,method=method,protocol=spec)

    def test_seven_exact_compositions_use_only_two_stage88_donors(self):
        model,original,trained = self.donors()
        before,donor_snapshot = copy.deepcopy(model.state_dict()),copy.deepcopy(trained)
        composed = experiment.compose_weights(original,trained,protocol=spec)
        self.assertEqual(set(composed),set(spec.VARIANTS))
        self.assertEqual(set(trained),{'joint_call','lower_trained'})
        donors = {'source':original,**trained}
        for variant,(upper,lower) in spec.COMPOSITIONS.items():
            expected = {**original,'upper_actor':donors[upper]['upper_actor'],'lower_actor':donors[lower]['lower_actor']}
            torch.testing.assert_close(composed[variant],expected,atol=0,rtol=0)
        composed['joint_upper_lower_only_lower']['upper_actor']['log_std'].add_(1.)
        torch.testing.assert_close(trained,donor_snapshot,atol=0,rtol=0)
        experiment.native.curves.support.assert_frozen(model,before)

    def test_new_variant_names_preserve_pairing_and_interaction(self):
        seeds = spec.seed_roles(310011,preflight=True)['native_evaluation']
        offsets = dict(zip(spec.VARIANTS,(0.,0.,4.,1.,2.,5.,3.)))
        evaluation = {v:[{'seed':s,'policy_seed':s,'lower_seed':s,'decision_steps':[0,50],
            'upper_standard_noise':[[0.,1.],[1.,0.]],'episode_return':float(i)+offsets[v]}
            for i,s in enumerate(seeds)] for v in spec.VARIANTS}
        observed = experiment.paired_effects(50,evaluation,seeds,protocol=spec)
        self.assertEqual(set(observed),{k for k in spec.ENDPOINTS if k.startswith('50/')})
        self.assertEqual(observed['50/joint_call_minus_source_upper_joint_lower'],3.)
        self.assertEqual(observed['50/upper_by_lower_interaction'],0.)
        bad = copy.deepcopy(evaluation);bad['joint_call'][0]['lower_seed'] += 1
        with self.assertRaises(ValueError):experiment.paired_effects(50,bad,seeds,protocol=spec)

    def test_fresh_stage89_roster_exact_budget_and_dynamic_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['native_episodes']*8,b['native_steps']*8,b['checkpoint_loads']*8),(3584,4300800,32))
        self.assertEqual(b['actor_composition_checks']*8,112)
        self.assertTrue(all(b[k]==0 for k in ('credit_episodes','policy_updates','checkpoint_writes')))
        self.assertEqual(len(spec.ENDPOINTS),22)
        all_seeds = []
        for preflight in (True,False):
            for root in spec.roots(preflight=preflight):
                seeds = spec.seed_roles(root,preflight=preflight)['native_evaluation']
                all_seeds.extend(seeds)
                for old_preflight in (True,False):
                    old = spec.source.seed_roles(root,preflight=old_preflight)
                    old_seeds = old['native_evaluation']+[s['scenario_seed'] for r in old['training_rounds']
                        for name in ('A','B') for s in r['credit_'+name]]
                    old_seeds += [n for r in old['training_rounds'] for name in ('A','B') for s in r['credit_'+name] for n in s['noise_seeds']]
                    self.assertFalse(set(seeds)&set(old_seeds))
                task = task_specification('unit_stage89',root,preflight=preflight,protocol_spec=spec)
                self.assertIn(spec.RUNNER_SCRIPT,task['cmd'])
                self.assertEqual((task['cpu'],task['ram_mb']),(3,3072) if preflight else (9,8192))
                self.assertEqual(task['allowed_nodes'],[f'node{i:03}' for i in range(1,7)])
                self.assertFalse(task.get('require_node'))
                self.assertTrue(task['result_dir'].endswith('/completion'))
            q = qualification_task('unit_stage89',preflight=preflight,protocol_spec=spec)
            self.assertIn(spec.ANALYZER_SCRIPT,q['cmd'])
            self.assertIsNone(q['result_dir'])
            self.assertEqual(len(q['wait_for_files']),len(spec.roots(preflight=preflight)))
        self.assertEqual(len(all_seeds),len(set(all_seeds)))
        self.assertIn(spec.TRAINING_RUN,str(spec.training_result(310011)))

    def test_all22_endpoints_keep_one_equal_root_family(self):
        cells = [{'root':r,'groups':{'both':{'effects':dict.fromkeys(spec.ENDPOINTS,2.)}},
            'cost':spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment,'qualify',side_effect=lambda c,**kw:c) as qualify, \
                patch.object(spec,'BOOTSTRAP_DRAWS',128), \
                patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            result = experiment.aggregate(cells,preflight=False,protocol=spec)
        self.assertEqual(qualify.call_args.kwargs['protocol'],spec)
        self.assertEqual(quantile.call_args.args[1],[.05/44,1-.05/44])
        self.assertEqual(set(result['endpoints']),set(spec.ENDPOINTS))
        self.assertEqual(result['contract'],spec.contract())
        self.assertEqual(result['protocol'],spec.EXPERIMENT_PROTOCOL)
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1],preflight=False,protocol=spec)


if __name__ == '__main__':
    unittest.main()
