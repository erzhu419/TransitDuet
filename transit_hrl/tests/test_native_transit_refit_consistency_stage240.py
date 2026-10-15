import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_refit_consistency as learning
from scripts import run_native_transit_refit_consistency_stage240 as spec
from scripts.submit_native_transit_refit_consistency_stage240_scheduleurm import task_specification
from tests.test_native_transit_refit_matrix_stage237 import small_fixture


class RefitConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = learning.previous
        with patch.object(learning.joint.binding.learned.neural, 'STEPS', 3):
            args = small_fixture()
            refit_source, refits = previous.base.previous.fit_matrix(*args)
        train, held, panels, reference, paths, old, _, oof = args
        source, cached = previous.fit_cell(train, held, panels, reference, paths, refit_source, refits, oof, old['selection'])
        cls.args = train, held, panels, reference, refits, refit_source, source, cached

    def test_no_raw_training_or_reselection_and_cached_endpoints_match(self):
        calls, original = [], learning.previous.fit_paths
        def fit(data, values, cached):
            calls.append((data['heads'].copy(), data['x'].copy(), values.copy()))
            return original(data, values, cached)
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw retraining')), \
                patch.object(learning.previous, 'choose', side_effect=AssertionError('Reselection')), \
                patch.object(learning.previous, 'fit_paths', side_effect=fit):
            result, models = learning.fit_cell(*self.args)
        self.assertEqual(len(result['blocks']), 6)
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertEqual(models, json.loads(json.dumps(models)))
        self.assertEqual(len(models), 8)
        self.assertEqual(len(calls), 20)
        for i in (0, 4):
            np.testing.assert_array_equal(calls[i][0], calls[i+2][0])
            np.testing.assert_array_equal(calls[i][1], calls[i+2][1])
        self.assertEqual(result['reproduction']['probe_records'], 120)
        self.assertEqual(result['reproduction']['full_metric_records'], 168)
        for f in ('probe_max_abs_difference', 'full_max_abs_difference', 'subspace_max_abs_difference'):
            self.assertLessEqual(result['reproduction'][f], 1e-12)
        for block in result['blocks']:
            for e in (learning.ENDPOINTS[1], learning.ENDPOINTS[3]):
                for scope in learning.SCOPES:
                    self.assertEqual(block['endpoints'][e][scope]['all']['variants']['raw']['switch_preceding'], 0.)

    def test_held_states_and_other_future_truth_do_not_change_weights(self):
        first, models = learning.fit_cell(*self.args)
        train, held, panels, reference, refits, refit_source, source, cached = self.args
        changed = copy.deepcopy(panels)
        for r in train+held:
            values = changed[learning.joint.binding.learned.state_key(r)]['values']
            if r in held: values[:, 1:, (0, 3)] += 100.
            else: values[4:, 1:, (0, 3)] += 100.
        updated = copy.deepcopy(source)
        for scope in learning.previous.SCOPES:
            rows = train if scope.startswith('train') else held
            ds = (0, 1, 2, 3) if scope.endswith('same') else (4, 5, 6, 7)
            data = learning.joint.causal_data(rows, reference)
            vv = learning.joint.physical_values(rows, changed, ds)
            stats = learning.previous.base.metrics(data, vv, learning.previous.predict(data, cached['full'], source['selection']))
            for p in (stats['all'], *stats['by_root'].values()):
                for v in p['variants'].values():
                    v['score_mse_against_cost'] = v.pop('cost_mse')
                    v['feasible_score_mse_against_cost'] = v.pop('feasible_cost_mse')
            updated['scopes'][scope] = stats
        second, other = learning.fit_cell(train, held, changed, reference, refits, refit_source, updated, cached)
        self.assertEqual(models, other)
        self.assertEqual(first['prefix_diagnostics'], second['prefix_diagnostics'])
        self.assertNotEqual(first['blocks'][0]['endpoints']['60_4_native']['held_other'], second['blocks'][0]['endpoints']['60_4_native']['held_other'])

    def test_single_prefix_benefit_is_detected_without_new_selection(self):
        selection = {a: {o: {f: {'candidate': None, 'objective': None} for f in learning.previous.FILTERS}
            for o in learning.previous.FAMILIES} for a in learning.joint.ANCHORS}
        selection['raw']['joint']['half_safe'] = {'candidate': '1@1', 'objective': 'rank_regret'}
        blocks = []
        for group in range(3):
            for h in (0, 1):
                stats = {'changed_count': int(group == 0), 'by_root': {'397': {'cases': 6, 'cost_sum': -3. if group == 0 else 0.,
                    'wait_sum_min': -3. if group == 0 else 0., 'unsafe_delta_count': 0}}}
                blocks.append({'held_group': [397, group], 'half': h, 'scales': {a: [1., 1.] for a in learning.joint.ANCHORS},
                    'curves': {'raw': {'rank_regret': {'1@1': stats}}}})
        result = learning.prefix_diagnostics({'selection': selection, 'blocks': blocks})['joint_raw_half_safe']
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertEqual(sum(not d['eligible'] for d in result['deletions']), 1)
        self.assertEqual(result['concentration']['397']['cost_sum']['top1_benefit_share'], 1.)
        self.assertEqual(result['concentration']['397']['cost_sum']['improving_prefixes'], 1)

    def test_full_report_endpoint_coverage_and_probe_source_averaging(self):
        train = [{'root': 397, 'scene_seed': g} for g in range(10) for _ in range(6)]
        fits = {a: [{'objective': o, 'level': l, 'success': True, 'parameter_count': int(l), 'retained_rank': int(l),
            'optimality': 0., 'norm': 1., 'radius': 1., 'loss_before': 2., 'loss_after': 1.}
            for o in learning.previous.OBJECTIVES[1:] for l in learning.strength.lowrank.LEVELS[1:]] for a in learning.joint.ANCHORS}
        selection = {a: {o: {f: {'candidate': None, 'objective': None} for f in learning.previous.FILTERS}
            for o in learning.previous.FAMILIES} for a in learning.joint.ANCHORS}
        results = {}
        for key in spec.KEYS:
            blocks = []
            for g in range(10):
                for h in (0, 1):
                    endpoints = {}
                    for e in learning.ENDPOINTS:
                        endpoints[e] = {}
                        for scope in learning.SCOPES:
                            p = {'cases': 6 if scope == 'prefix_probe' else 60, 'variants': {v: {'delta_cost_own': 0.} for v in learning.VARIANTS}}
                            endpoints[e][scope] = {'all': p, 'by_root': {'397': p}}
                    blocks.append({'held_group': [397, g], 'half': h, 'expanded_fits': fits, 'endpoints': endpoints})
            cell = {'blocks': blocks, 'training_draws': list(spec.PARTITIONS[key[2]][key[3]]),
                'evaluation_draws': list(spec.PARTITIONS[key[2]][1-key[3]]), 'half_fits': {str(h): {e: fits for e in learning.ENDPOINTS[2:4]} for h in (0, 1)},
                'selection': selection, 'prefix_diagnostics': {v: {'admitted': False, 'deletions': [], 'concentration': {}} for v in learning.VARIANTS[1:]},
                'reproduction': {'probe_records': 400, 'full_metric_records': 168, 'probe_max_abs_difference': 0., 'full_max_abs_difference': 0., 'subspace_max_abs_difference': 0.}}
            results[key] = {'protocol': spec.EXPERIMENT_PROTOCOL, 'contract': spec.contract(), 'key': list(key), 'cell': cell}
        with patch.object(spec.training.cohort, 'split_rows', return_value=(train, train)):
            report = spec.summarize(results, dict.fromkeys(spec.VIEWS))
            self.assertEqual(report['reproduction']['probe_records'], 12800)
            self.assertEqual(report['mixed_pooled']['54_2_cached']['prefix_probe']['by_root']['397']['matched_blocks'], 320)
            first = results[spec.KEYS[0]]['cell']
            first['blocks'][0]['endpoints'].pop('60_2_native')
            with self.assertRaisesRegex(ValueError, 'Dropped refit endpoint'): spec.summarize(results, dict.fromkeys(spec.VIEWS))

    def test_protocol_budget_and_unbound_cpu_scheduler(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['objective_solves'], 18432)
        self.assertEqual(c['budget']['residual_SVDs'], 4480)
        for key in (spec.KEYS[0], (None, None, None, None)):
            task = task_specification('unit_refit_consistency', *key)
            self.assertEqual((task['cpu'], task['ram_mb']), (1, 3072))
            self.assertIsNone(task['require_node'])
            self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', task['result_dir'])


if __name__ == '__main__': unittest.main()
