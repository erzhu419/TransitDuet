import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_risk_alignment as learning
from freq_hrl.domains.transit import native_local_risk as previous
from freq_hrl.domains.transit import native_held_removal as removal
from scripts import run_native_transit_risk_alignment_stage274 as spec
from scripts import submit_native_transit_risk_alignment_stage274_scheduleurm as submit


class RiskAlignmentTest(unittest.TestCase):
    def test_exact_decomposition_and_beneficial_alignment_sign(self):
        raw = np.array([[2., 20.], [5., 7.]])
        prediction = raw+np.array([[-1., 100.], [2., -2.]])
        weights = np.array([[1., 0.], [.5, .5]])
        s, error = learning.components(prediction, raw, np.zeros_like(raw), weights, 2)
        self.assertAlmostEqual(s['correction_squared'], 5.)
        self.assertAlmostEqual(s['alignment'], -8.)
        self.assertAlmostEqual(s['delta_MSE'], -3.)
        self.assertAlmostEqual(s['model_MSE']-s['raw_MSE'], -3.)
        self.assertLess(error, 1e-12)

    def test_future_changes_alignment_not_frozen_correction_energy(self):
        raw = np.array([[1., 2.]]);prediction = raw-.5;weights = np.array([[.5, .5]])
        a, _ = learning.components(prediction, raw, np.zeros_like(raw), weights, 1)
        b, _ = learning.components(prediction, raw, np.ones_like(raw)*3, weights, 1)
        self.assertEqual(a['correction_squared'], b['correction_squared'])
        self.assertLess(a['alignment'], 0.);self.assertGreater(b['alignment'], 0.)

    def test_common_actions_balance_probe_not_number_of_held_replicas(self):
        ex = [{'probe_index': 0}, {'probe_index': 1}]
        he = [{'probe_index': 0}, {'probe_index': 0}, {'probe_index': 1}]
        a = np.array([[False, True, True, False], [True, False, False, False]])
        b = np.array([[False, True, True, False], [False, True, False, False], [False, False, True, False]])
        ei, hi, w, c = learning.common_pairs(ex, he, a, b)
        np.testing.assert_array_equal(ei, [0, 0, 1]);np.testing.assert_array_equal(hi, [0, 1, 2])
        np.testing.assert_allclose(w, [[0., .25, .25, 0.], [0., .5, 0., 0.], [0., 0., 0., 0.]])
        self.assertEqual(c, {'units': 1, 'attempted_state_pairs': 3, 'retained_state_pairs': 2, 'shared_action_exposures': 3})
        with self.assertRaisesRegex(ValueError, 'Repeated'):learning.common_pairs(ex+ex[:1], he, a, b)

    def test_squaring_precedes_replica_averaging(self):
        raw = np.zeros((2, 1));pred = np.array([[1.], [-1.]])
        s, _ = learning.components(pred, raw, raw, np.ones((2, 1))*.5, 1)
        self.assertEqual(s['correction_squared'], 1.)
        self.assertEqual(s['delta_MSE'], 1.)
        self.assertEqual(float(pred.mean()**2), 0.)

    def test_summary_uses_actual_supported_units(self):
        x = {'units': 1, **{k: 2. for k in learning.FIELDS}}
        empty = {'units': 0, **{k: 0. for k in learning.FIELDS}}
        self.assertEqual(learning.summarize([x, empty])['model_MSE'], 2.)
        z = learning.summarize([empty]);self.assertIsNone(z['model_MSE'])
        self.assertIsNone(learning.contrast(z, z)['delta_MSE'])

    def test_complete_cached_cell_without_inference_fits_or_selection(self):
        rng = np.random.default_rng(274)
        def rows(offset):
            return [{'root': 397 if i//6 < 5 else 401, 'scene_seed': offset+i//6,
                'scenario': str(i//6%5), 'probe_index': i%6, 'original_choice': 0} for i in range(60)]
        train, held = rows(0), rows(100)
        folds = list(spec.cache.learning.joint.binding.policy.prefix_folds(train))
        groups = [tuple(g) for g, _, _ in folds]
        masks, support = removal.conditions(train, held, groups)
        eligible = {};raw = {};values = {}
        for label in ('train', 'held'):
            eligible[label] = np.ones((60, 6), bool);eligible[label][:, 0] = False;eligible[label][::7] = False
            raw[label] = rng.normal(size=(60, 6))*.01;raw[label][:, 0] = 0.
            values[label] = np.zeros((60, 32, 6, 4))
            values[label][..., 1] = rng.choice([0., -.4, .5], size=(60, 32, 6), p=[.8, .1, .1])
            values[label][:, :, 0] = 0.
        models, queries, saved, old_errors, held_costs = [], [], {}, [], {a: [] for a in spec.ARMS}
        for bank in (0, 1):
            for group, ix, val in [(None, np.arange(60), np.arange(60)), *folds]:
                for penalty in spec.PENALTIES:
                    i = len(models);models.append({'bank': bank, 'removed': list(group) if group else None,
                        'penalty': penalty, 'fit_indices': list(map(int, ix)), 'training_draws': list(range(bank*16, (bank+1)*16))})
                    costs = {}
                    for label in ('train', 'held'):
                        delta = rng.normal(size=(60, 6))*.01;delta[:, 0] = 0.
                        costs[label] = {a: raw[label]+(0. if a == 'raw_prior' else np.clip(delta, -.01, .01) if a.endswith('_clip') else delta) for a in spec.ARMS}
                    for a in spec.ARMS:held_costs[a].append(costs['held'][a])
                    scopes = [('train', 'train', ix), ('held', 'held', val)] if group is None else [('retained', 'train', ix), ('excluded', 'train', val)]
                    for scope, label, ids in scopes:
                        queries.append({'index': i, 'scope': scope, 'label': label, 'ids': list(map(int, ids)), 'eligible': eligible[label][ids].tolist()})
                        if scope in ('excluded', 'held'):
                            saved.update({f'{i}_{scope}_{a}': c[ids] for a, c in costs[label].items()})
                        if scope == 'excluded':
                            for future, b in (('same', bank), ('opposite', 1-bank)):
                                y = values[label][ids, b*16:(b+1)*16, :, 1]
                                old_errors.append({**models[-1], 'scope': scope+'_'+future, 'metrics': {a: previous.metrics(c[ids], costs[label]['raw_prior'][ids], y, eligible[label][ids], .01) for a, c in costs[label].items()}})
        held_costs = {a: np.asarray(v) for a, v in held_costs.items()}
        held_errors = []
        for bank in (0, 1):
            for penalty in spec.PENALTIES:
                full = next(i for i, m in enumerate(models) if m['bank'] == bank and m['penalty'] == penalty and m['removed'] is None)
                prefix = [next(i for i, m in enumerate(models) if m['bank'] == bank and m['penalty'] == penalty and m['removed'] == list(g)) for g in groups]
                for future, b in (('same', bank), ('opposite', 1-bank)):
                    y = values['held'][:, b*16:(b+1)*16, :, 1].mean(axis=1)
                    held_errors.append({'bank': bank, 'penalty': penalty, 'future': future, 'metrics': {a: removal.paired_stats(c[full], c[prefix], y, eligible['held'], masks)[0] for a, c in held_costs.items()}})
        held_queries = {'held_states': [[r['root'], r['scene_seed'], r['probe_index']] for r in held], 'removed_groups': groups,
            'masks': masks, 'support_prefixes_before': support}
        def file(path):
            v = {'models': models, 'queries': queries, 'errors': old_errors}[path.stem] if spec.PREDICTION_RUN in path.parts else {'queries': held_queries, 'errors': held_errors}[path.stem]
            return json.dumps(v, default=lambda a: a.tolist())
        class Archive(dict):
            def __enter__(self):return self
            def __exit__(self, *args):return False
        def draws(rs, panels, ds):
            label = 'held' if rs[0]['scene_seed'] >= 100 else 'train'
            ids = [(r['scene_seed']%100)*6+r['probe_index'] for r in rs]
            return values[label][ids][:, ds]
        with patch.object(spec.Path, 'read_text', autospec=True, side_effect=file), patch.object(spec.np, 'load', side_effect=[Archive(saved), Archive(held_costs)]), \
                patch.object(spec.Path, 'mkdir'), patch.object(spec, 'write_json'), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(train, held)), patch.object(spec.cache.learning.base, 'training_draws', side_effect=draws), \
                patch.object(spec.cache.learning.base, 'causal_data', side_effect=AssertionError('Reference inference')), patch.object(previous, 'predict', side_effect=AssertionError('Head inference')), \
                patch.object(previous, 'fit', side_effect=AssertionError('Refit')), patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')), \
                patch.object(np.linalg, 'solve', side_effect=AssertionError('Solve')):
            result = spec.analyze_cell(spec.Path('/tmp/unit_alignment'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(len(result['records']), 120);self.assertEqual(len(result['pairing']), 60)
        self.assertLess(result['metric_max_abs_error'], 1e-10);self.assertLess(result['identity_max_abs_error'], 1e-12)
        for r in result['records']:
            for pop in spec.POPULATIONS:
                self.assertEqual(r['metrics']['raw_prior'][pop]['delta_MSE'], 0.)

    def test_single_cpu_scheduler_requires_completed_parent(self):
        task = submit.task_specification('unit_alignment')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.SOURCE_PROTOCOL, 'contract': {'source': 'fixed'}, 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_alignment']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
