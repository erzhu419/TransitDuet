import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_refit_matrix as learning
from scripts import run_native_transit_refit_matrix_stage237 as spec
from scripts.submit_native_transit_refit_matrix_stage237_scheduleurm import task_specification
from tests import test_native_transit_stability_admission_stage236 as fixtures

joint, strength, diagnostic = learning.joint, learning.strength, learning.diagnostic


def small_fixture():
    old = fixtures.fixtures.fixtures.old.old.old.fixtures
    bank = old.native_fixture.bank_fixture()[0]['training_data']
    groups = sorted({(r['root'], r['scene_seed']) for r in bank})[:4]
    train = [r for r in bank if (r['root'], r['scene_seed']) in groups[:3]]
    held = [r for r in bank if (r['root'], r['scene_seed']) == groups[3]]
    panels = old.old.panels_for(train+held)
    reference, _ = joint.fit_reference(train, panels, (0, 1, 2, 3))
    batch = joint.binding.learned.neural.batches(train)
    oof = []
    for _, fit, val in joint.binding.policy.prefix_folds(train):
        rs, vs = [train[i] for i in fit], [train[i] for i in val]
        for a in ((0, 1), (2, 3)):
            ref, _ = joint.fit_reference(rs, panels, a, batch=tuple(x[fit] for x in batch))
            oof.append(joint.causal_data(vs, ref, batch=tuple(x[val] for x in batch)))
    reused = {'originals': np.concatenate([d['originals'] for d in oof]), 'roots': np.concatenate([d['roots'] for d in oof]),
        'costs': np.concatenate([d['heads'][..., 0] for d in oof]), 'state_conditioned_pair': np.concatenate([d['raw_logits'] for d in oof])}
    curves, ledger, _ = joint.collect_paths(train, panels, (0, 1, 2, 3), reused, path_fitter=strength.fit_path)
    selection = {a: diagnostic.select(curves[a], ledger, a) for a in joint.ANCHORS}
    stable = {'anchors': {a: diagnostic.diagnose(curves[a], ledger, a, selection[a]) for a in joint.ANCHORS}}
    data = joint.causal_data(train, reference)
    paths = {a: strength.lowrank.fit_path(train, data, joint.physical_values(train, panels, (0, 1, 2, 3)), a)[0] for a in joint.ANCHORS}
    source = {'training_draws': [0, 1, 2, 3], 'evaluation_draws': [4, 5, 6, 7], 'selection': selection, 'scopes': {}}
    for scope in ('train_same', 'train_other', 'held_same', 'held_other'):
        rows = train if scope.startswith('train') else held
        d = joint.causal_data(rows, reference)
        values = joint.physical_values(rows, panels, source['training_draws' if scope.endswith('same') else 'evaluation_draws'])
        decisions = {'raw': d['raw_choices']}
        for a in joint.ANCHORS:
            for mode, field in diagnostic.FIELDS.items():
                key = selection[a][field]
                if key in (None, 'zero'): chosen = d['raw_choices']
                else:
                    c = strength.CANDIDATES[key]
                    cost = strength.base.costs(d, paths[a], c['level'], c['alpha'])
                    chosen = joint.binding.policy.selected(cost, d['raw_logits'], joint.anchors(d, a))
                decisions[f'cost_strength_{a}_{mode}'] = chosen
        source['scopes'][scope] = [{**r, 'decisions': {name: {'selected': int(js[i]), 'cost': float(values[i, js[i], 0]),
            'wait': float(values[i, js[i], 1])} for name, js in decisions.items()}} for i, r in enumerate(rows)]
    return train, held, panels, reference, paths, source, stable, reused


class RefitMatrixTest(unittest.TestCase):
    def test_physical_mse_rebasing_regret_and_two_raw_baselines(self):
        values = np.array([[[0., 0.], [-2., 1.], [-1., -1.], [-3., 2.], [1., 0.], [2., 0.]]])
        data = {'originals': np.array([0]), 'roots': np.array([397]), 'raw_choices': np.array([2]),
            'raw_logits': np.array([[0., -1., -1., -1., 1., 1.]])}
        common = {'raw_choices': np.array([1])}
        p = {'raw': {'chosen': np.array([2]), 'cost': values[..., 0].copy()}}
        first = learning.metrics(data, common, values, p)['all']['variants']['raw']
        p['raw']['cost'] += 100.
        second = learning.metrics(data, common, values, p)['all']['variants']['raw']
        self.assertEqual(first, second)
        self.assertEqual(first['cost_mse'], 0.)
        self.assertEqual(first['gate_cost_regret'], 2.)
        self.assertEqual(first['delta_cost_common'], 1.)
        self.assertEqual(first['delta_cost_own'], 0.)
        self.assertEqual(first['delta_wait_min_common'], -2.)

    def test_matched_fits_endpoint_reproduction_and_coordinate_intervention(self):
        captures = []
        original = strength.lowrank.fit_path
        def fitter(rows, data, values, kind, **kwargs):
            captures.append((len(rows), kind, data['x'].copy(), data['heads'].copy()))
            return original(rows, data, values, kind, **kwargs)
        with patch.object(joint.binding.learned.neural, 'STEPS', 3):
            args = small_fixture()
            with patch.object(strength.lowrank, 'fit_path', side_effect=fitter):
                result, cache = learning.fit_matrix(*args)
        self.assertEqual(len(result['trials']), 12)
        self.assertEqual(len(result['blocks']), 6)
        self.assertEqual(len(cache['references']), 11)
        self.assertEqual(len(cache['paths']), 11)
        self.assertEqual(len(captures), 66)
        for trial in result['trials'][1:]:
            self.assertEqual(set(trial['arms']), set(learning.ARMS))
        for i in range(0, len(captures), 6):
            np.testing.assert_array_equal(captures[i][2], captures[i+2][2])
            np.testing.assert_array_equal(captures[i+2][3], captures[i+4][3])
        self.assertTrue(any(not np.array_equal(captures[i][2], captures[i+4][2]) for i in range(0, len(captures), 6)))
        r = result['reproduction']
        self.assertEqual(r['choice_comparisons'], 336)
        self.assertEqual(r['score_comparisons'], 672)
        self.assertEqual(r['changed_choices'], 0)
        self.assertEqual(r['oof_changed_choices'], 0)
        self.assertEqual(r['prefix_count_differences'], 0)
        self.assertLessEqual(r['prefix_max_abs_difference'], 1e-12)
        for arm in learning.ARMS:
            self.assertEqual(result['trials'][0]['arms'][arm]['scopes'], result['trials'][0]['arms']['fixed']['scopes'])

    def test_future_holdout_cannot_change_reference_or_residual_weights(self):
        with patch.object(joint.binding.learned.neural, 'STEPS', 3):
            args = small_fixture()
            first, cache = learning.fit_matrix(*args)
            train, held, panels, reference, paths, source, stable, reused = args
            altered, updated = copy.deepcopy(panels), copy.deepcopy(source)
            for rows, scope in ((train, 'train_other'), (held, 'held_other')):
                for row, record in zip(rows, updated['scopes'][scope]):
                    key = joint.binding.learned.state_key(row)
                    for j in range(6):
                        if j != row['original_choice']: altered[key]['values'][4:, j, 0] += 100.
                    values = altered[key]['values'][4:].mean(axis=0)
                    for d in record['decisions'].values(): d['cost'] = float(values[d['selected'], 0])
            second, other = learning.fit_matrix(train, held, altered, reference, paths, updated, stable, reused)
        self.assertEqual(cache, other)
        self.assertEqual(first['blocks'], second['blocks'])
        self.assertNotEqual(first['trials'][0]['arms']['fixed']['scopes']['held_other'], second['trials'][0]['arms']['fixed']['scopes']['held_other'])

    def test_equal_block_averaging_not_case_weighting(self):
        pieces = [{'cases': n, 'variants': {name: {'cost_mse': v, 'delta_cost_common': v} for name in learning.VARIANTS}}
            for n, v in ((24, 1.), (30, -1.))]
        out = learning.averaged(pieces)
        self.assertEqual(out['matched_blocks'], 2)
        self.assertEqual(out['mean_cases'], 27.)
        self.assertEqual(out['variants']['raw']['delta_cost_common'], 0.)

    def test_complete_matrix_report_and_matched_alias_validation(self):
        old = fixtures.fixtures.fixtures.old.old.old.fixtures
        rows, registered = old.cohort.fixture()
        views = spec.training.cohort.source.matched_views(rows, registered)
        results = {}
        for key in spec.KEYS:
            train, _ = spec.training.cohort.split_rows(views[key[0]], key[1])
            groups = sorted({(r['root'], r['scene_seed']) for r in train})
            a = spec.PARTITIONS[key[2]][key[3]]
            trials, lookup = [], {}
            for group in (None, *groups):
                for half in (None, 0, 1):
                    gs, n = set(groups)-{group}, 54 if group else 60
                    draws = list(a if half is None else a[2*half:2*half+2])
                    fit_rows = [r for r in train if (r['root'], r['scene_seed']) != group]
                    def stats(scope):
                        rs = fit_rows if scope == 'fit_training' else train
                        def one(sub): return {'cases': len(sub), 'variants': {v: {'cost_mse': float(n), 'delta_cost_common': float(len(draws))} for v in learning.VARIANTS}}
                        return {'all': one(rs), 'by_root': {str(root): one([r for r in rs if r['root'] == root]) for root in {r['root'] for r in rs}}}
                    full = group is None and half is None
                    fits = {kind: {'anchor': kind, 'training_states': n, 'training_groups': list(map(list, gs)), 'steps': 0, 'solver': 'training_pair_TSVD'} for kind in joint.ANCHORS}
                    lookup[group, half] = len(trials)
                    trials.append({'removed_group': list(group) if group else None, 'half': half, 'training_states': n,
                        'training_draws': draws, 'reference_fit': None if full else {'draws': draws, 'training_states': n,
                            'training_groups': list(map(list, gs)), 'wait_training_groups': list(map(list, gs)), 'raw_steps': 600, 'raw_parameters': 3393},
                        'arms': {arm: {'fits': {} if full else fits, 'scopes': {s: stats(s) for s in learning.SCOPES}} for arm in learning.ARMS}})
            blocks = [{'removed_group': list(g), 'half': h, 'trials': {f'{n}/{k}': lookup[q, d]
                for n, q in ((54, g), (60, None)) for k, d in ((2, h), (4, None))}} for g in groups for h in (0, 1)]
            repro = {'choice_comparisons': 1680, 'score_comparisons': 3360, 'changed_choices': 0, 'max_abs_difference': 0.,
                'oof_max_abs_difference': 0., 'oof_changed_choices': 0, 'prefix_max_abs_difference': 0., 'prefix_count_differences': 0}
            results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key),
                'cell': {'trials': trials, 'blocks': blocks, 'reproduction': repro}}
        report = spec.summarize(results, views)
        self.assertEqual(report['reproduction']['choice_comparisons'], 53760)
        self.assertEqual(report['reproduction']['score_comparisons'], 107520)
        for v in report['views'].values():
            for cfg in spec.CONFIGS:
                self.assertEqual(v[cfg]['fixed']['train_same']['all']['matched_blocks'], 160)
                self.assertEqual(v[cfg]['fixed']['fit_training']['all']['mean_cases'], 54. if cfg.startswith('54') else 60.)
        self.assertEqual(report['mixed_pooled']['60/4']['fixed']['train_same']['all']['matched_blocks'], 320)
        key = spec.KEYS[0]
        results[key]['cell']['blocks'][0]['trials']['54/2'] = results[key]['cell']['blocks'][0]['trials']['60/4']
        with self.assertRaisesRegex(ValueError, 'Matched refit comparison changed'): spec.summarize(results, views)

    def test_protocol_budget_and_unbound_cpu_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 1024)
        self.assertEqual(c['budget']['residual_SVDs'], 6144)
        self.assertEqual(c['budget']['new_native_episodes'], 0)
        with patch.object(spec.previous, 'contract', wraps=spec.previous.contract) as parent:
            spec.contract()
            self.assertEqual(parent.call_count, 1)
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification('unit_refit_matrix', *key)
            self.assertEqual((task['cpu'], task['ram_mb']), (1, 3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', task['result_dir'])


if __name__ == '__main__': unittest.main()
