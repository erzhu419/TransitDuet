import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_joint_mean as experiment
from scripts import pointmaze_joint_mean_stage82_spec as spec
from scripts.submit_pointmaze_joint_mean_stage82_scheduleurm import task_specification, qualification_task
import test_pointmaze_feasible_credit as feasible_fixture
import test_pointmaze_scenario_credit as scenario_fixture


class JointMeanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_credit_candidates_have_fixed_budget_exact_half_step_composition_and_frozen_std(self):
        model = feasible_fixture.FeasibleCreditTest().source()
        before = copy.deepcopy(model.state_dict())
        args, roles = spec.arguments(310011, preflight=True), spec.seed_roles(310011, preflight=True)
        collector = scenario_fixture.ScenarioCreditTest()
        batches = {name: collector.collect(model, args, roles['credit_' + name]) for name in ('A', 'B')}
        cost = dict.fromkeys(spec.budget(preflight=True), 0)
        candidates, credit = experiment.credit_directions(model, batches, period=50, horizon=args.horizon, cost=cost)
        experiment.native.curves.support.assert_frozen(model, before)
        self.assertEqual(set(candidates), set(spec.VARIANTS) - {'base', 'zero'})
        expected = {'objective_checks': 8, 'mc_calls': 16, 'actor_score_forward_batches': 16,
            'actor_score_backward_batches': 48, 'fisher_jvp_batches': 8, 'exact_kl_forward_batches': 16,
            'actor_parameter_perturbations': 8, 'parameter_part_checks': 8, 'joint_composition_checks': 4}
        for key, value in expected.items():self.assertEqual(cost[key], value)
        for a in ('upper', 'lower'):
            row = credit['actors'][a]
            full, half = [row['allocations'][b]['geometry'] for b in ('full', 'half')]
            self.assertAlmostEqual(half['step'] / full['step'], 1 / np.sqrt(2))
            self.assertEqual((full['nominal_fisher_kl'], half['nominal_fisher_kl']), (.001, .0005))
            for noise in row['scenario_group_noise'].values():self.assertEqual(noise['episodes'], 2)
            other = 'lower_actor' if a == 'upper' else 'upper_actor'
            for budget in spec.ALLOCATIONS:
                for sign in ('plus', 'minus'):
                    w = candidates[f'{a}_{budget}_{sign}']
                    torch.testing.assert_close(w[other], before[other], atol=0, rtol=0)
        for variant, signs in spec.JOINT_SIGNS.items():
            experiment.check_joint_composition(candidates[variant], candidates, signs)
            self.assertTrue(.0005 <= credit['joint_geometry'][variant]['exact_sum_kl'] <= .002)
        for w in candidates.values():
            for a in ('upper', 'lower'):torch.testing.assert_close(w[a+'_actor']['log_std'], before[a+'_actor']['log_std'], atol=0, rtol=0)
            for key in ('upper_value', 'lower_value'):torch.testing.assert_close(w[key], before[key], atol=0, rtol=0)
        bad = copy.deepcopy(candidates['joint_plus'])
        bad['upper_actor'] = candidates['upper_full_plus']['upper_actor']
        with self.assertRaises(AssertionError):experiment.check_joint_composition(bad, candidates, ('plus', 'plus'))

    def test_joint_interaction_uses_paired_native_rewards_and_common_noise(self):
        seeds = [11, 12]
        gains = dict.fromkeys(spec.VARIANTS, 0.)
        gains.update(upper_half_plus=1., lower_half_plus=2., joint_plus=4.)
        evaluation = {v: [{'seed': s, 'policy_seed': 100+s, 'lower_seed': 200+s,
            'decision_steps': [50], 'upper_standard_noise': [[.2, -.1]],
            'episode_return': 1000*s + gains[v]} for s in seeds] for v in spec.VARIANTS}
        effects = experiment.paired_effects(50, evaluation, seeds)
        self.assertEqual(effects['50/joint_plus_interaction'], 1.)
        self.assertEqual(effects['50/joint_plus_minus_upper_half_plus'], 3.)
        self.assertEqual(effects['50/joint_plus_minus_lower_half_plus'], 2.)
        self.assertEqual(len(effects), 30)
        bad = copy.deepcopy(evaluation)
        bad['joint_upper_plus_lower_minus'][0]['upper_standard_noise'][0][0] += .1
        with self.assertRaises(AssertionError):experiment.paired_effects(50, bad, seeds)

    def test_fresh_roles_fixed_cost_and_dynamic_completion_only_scheduler(self):
        b = spec.budget(preflight=False)
        self.assertEqual((b['native_episodes']*8, b['native_steps']*8), (8192, 9830400))
        self.assertEqual((b['actor_score_forward_batches']*8, b['actor_score_backward_batches']*8), (3072, 9216))
        self.assertEqual((b['fisher_jvp_batches']*8, b['exact_kl_forward_batches']*8), (2448, 4896))
        self.assertEqual((b['parameter_part_checks']*8, b['joint_composition_checks']*8), (128, 64))
        self.assertEqual((len(spec.ENDPOINTS), len(set(spec.ENDPOINTS))), (60, 60))
        for preflight in (True, False):
            all_seeds = []
            for root in spec.roots(preflight=preflight):
                roles = spec.seed_roles(root, preflight=preflight)
                seeds = [s['scenario_seed'] for name in ('A','B') for s in roles['credit_'+name]]
                seeds += [n for name in ('A','B') for s in roles['credit_'+name] for n in s['noise_seeds']]
                seeds += roles['native_evaluation']; all_seeds.extend(seeds)
                old = spec.source.seed_roles(root, preflight=preflight)
                prior = [s['scenario_seed'] for name in ('A','B') for s in old['credit_'+name]]
                prior += [n for name in ('A','B') for s in old['credit_'+name] for n in s['noise_seeds']]
                prior += old['native_evaluation']
                self.assertFalse(set(seeds) & set(prior))
                t = task_specification('unit_stage82', root, preflight=preflight)
                self.assertEqual((t['cpu'], t['ram_mb']), (3,3072) if preflight else (9,8192))
                self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1,7)])
                self.assertFalse(t.get('require_node'))
                self.assertTrue(t['result_dir'].endswith('/completion'))
                self.assertIn(spec.RUNNER_SCRIPT, t['cmd'])
            self.assertEqual(len(all_seeds), len(set(all_seeds)))
            q = qualification_task('unit_stage82', preflight=preflight)
            self.assertEqual(len(q['wait_for_files']), len(spec.roots(preflight=preflight)))
            self.assertIsNone(q['result_dir'])

    def test_all60_endpoints_share_one_root_bootstrap_family_and_hold(self):
        cells = [{'root': r, 'groups': {'both': {'effects': dict.fromkeys(spec.ENDPOINTS,2.)}},
            'cost': spec.budget(preflight=False)} for r in spec.roots(preflight=False)]
        with patch.object(experiment, 'qualify', side_effect=lambda cell, **kw: cell), \
                patch.object(spec,'BOOTSTRAP_DRAWS',128), patch.object(experiment.native.np,'quantile',wraps=np.quantile) as quantile:
            summary = experiment.aggregate(cells, preflight=False)
        self.assertEqual(quantile.call_args.args[1], [.05/120,1-.05/120])
        self.assertEqual(set(summary['endpoints']), set(spec.ENDPOINTS))
        self.assertEqual(summary['native_trial_prerequisite'], 'hold_Stage67_credit_gate_unchanged')
        with self.assertRaises(ValueError):experiment.aggregate(cells[:-1], preflight=False)


if __name__ == '__main__':
    unittest.main()
