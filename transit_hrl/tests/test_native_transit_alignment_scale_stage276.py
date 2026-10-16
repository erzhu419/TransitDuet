import json
from math import prod, sqrt
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_alignment_scale as learning
from freq_hrl.domains.transit import native_local_risk as local
from scripts import run_native_transit_alignment_scale_stage276 as spec
from scripts import submit_native_transit_alignment_scale_stage276_scheduleurm as submit


def sample(root, seed, probe, bank=0, index=0, delta=.002, residual=.03, held_residual=.05, he_units=1):
    metrics = {}
    for future in learning.source.FUTURES:
        metrics[future] = {}
        for mode, (ex, he) in learning.source.MODES.items():
            for pop, r, n in ((ex, residual, 1), (he, held_residual, he_units if mode == 'natural' else 1)):
                if future == 'opposite':r = .7*r-.001
                metrics[future][pop] = {'units': n, 'raw_MSE': n*r*r, 'model_MSE': n*(r+delta)**2,
                    'correction_squared': n*delta*delta, 'alignment': n*2*delta*r, 'delta_MSE': n*(delta*delta+2*delta*r)}
    return {'root': root, 'scene_seed': seed, 'probe_index': probe, 'bank': bank, 'index': index, 'scenario': 'shift', 'metrics': metrics}


class AlignmentScaleTest(unittest.TestCase):
    def test_uncentered_cosine_and_population_mass(self):
        s = {'units': 2, 'correction_squared': 1., 'raw_MSE': 10., 'alignment': 6., 'delta_MSE': 7.}
        p = learning.population(s, 10)
        self.assertAlmostEqual(p['normalized_alignment'], 3/sqrt(10))
        self.assertEqual(p['population_mass'], .2)
        self.assertAlmostEqual(2*prod(p[k] for k in learning.FACTORS), 1.2)

    def test_single_factor_scale_and_direction_changes(self):
        held = dict(zip(learning.FACTORS, (1., .1, 2., -.5)))
        excluded = {**held, 'normalized_alignment': .25}
        e = learning.product_effects(held, excluded)
        self.assertAlmostEqual(e['normalized_alignment'], .3)
        self.assertAlmostEqual(sum(abs(v) for k, v in e.items() if k != 'normalized_alignment'), 0.)
        e = learning.product_effects(held, {**held, 'raw_residual_RMS': 4.})
        self.assertAlmostEqual(e['raw_residual_RMS'], -.2)

    def test_symmetric_product_effects_reverse_and_add_exactly(self):
        a = dict(zip(learning.FACTORS, (.2, .01, .3, -.7)))
        b = dict(zip(learning.FACTORS, (.6, .02, .1, .4)))
        forward, backward = learning.product_effects(a, b), learning.product_effects(b, a)
        self.assertAlmostEqual(sum(forward.values()), 2*(prod(b.values())-prod(a.values())))
        for k in learning.FACTORS:self.assertAlmostEqual(forward[k], -backward[k])
        self.assertNotEqual(forward['population_mass'], 0.)

    def test_undefined_cosine_not_replaced_by_zero(self):
        s = {'units': 1, 'correction_squared': 1., 'raw_MSE': 0., 'alignment': 0., 'delta_MSE': 1.}
        p = learning.population(s, 1);self.assertIsNone(p['normalized_alignment'])
        c = learning.compare(s, {**s, 'raw_MSE': 1., 'alignment': -1., 'delta_MSE': 0.}, 1, 1)
        self.assertIsNone(c['effects']);self.assertEqual(c['global_contribution']['alignment'], 1.)
        empty = {'units': 0, **{k: None for k in ('raw_MSE', 'correction_squared', 'alignment', 'delta_MSE')}}
        c = learning.compare(empty, s, 1, 1)
        self.assertIsNone(c['effects']);self.assertEqual(c['excluded']['population_mass'], 0.)

    def test_moments_pooled_before_normalizing_and_partitions_add(self):
        rs = [sample(401, 914010401, 8, delta=1., residual=1., held_residual=2.),
            sample(401, 914010401, 22, delta=1., residual=100., held_residual=3., he_units=2),
            sample(397, 999, 9, residual=.02), sample(397, 999, 36, residual=.04)]
        s = learning.summarize(rs)
        p = s['comparisons']['same']['natural']['dominant_prefix']['excluded']
        self.assertAlmostEqual(p['normalized_alignment'], 50.5/sqrt(5000.5))
        self.assertNotAlmostEqual(p['normalized_alignment'], 1.)
        self.assertLess(s['partition_max_abs_error'], 1e-12)
        self.assertLess(s['factorization_max_abs_error'], 1e-12)
        for c in s['comparisons']['same']['paired'].values():
            if c['effects'] is not None:self.assertAlmostEqual(c['effects']['population_mass'], 0.)

    def test_scalar_ledger_only_complete_runner_and_parent_replay(self):
        cells = []
        for key in spec.KEYS:
            rows = [sample(learning.DOMINANT_PREFIXES[i%3][0] if i < 3 else 397,
                learning.DOMINANT_PREFIXES[i%3][1] if i < 3 else 900000+i, probe,
                bank=i//10, index=i, delta=.001*(1+i%3), residual=.02+.001*i,
                held_residual=.03+.002*(i%4), he_units=1+i%2) for i in range(20) for probe in (8, 9, 22, 23, 36, 37)]
            cells.append({'key': list(key), 'rows': rows})
        predicates = {'mixed_pooled': lambda k: k[0].startswith('mixed'), **{v: lambda k, v=v: k[0] == v for v in spec.VIEWS}}
        before = {'groups': {}}
        for g, predicate in predicates.items():
            rows = [r for c in cells if predicate(c['key']) for r in c['rows']]
            before['groups'][g] = {b: {'totals': learning.source.aggregate([r for r in rows if bank is None or r['bank'] == bank])}
                for b, bank in (('pooled', None), ('bank0', 0), ('bank1', 1))}
        by_path = {'_'.join(map(str, c['key'])): c['rows'] for c in cells};writes = []
        def read(p):
            if p.name == 'result.json':return json.dumps(before)
            self.assertEqual(p.name, 'probes.json')
            return json.dumps(by_path[p.parent.name])
        with patch('sys.argv', ['run', '--run-name', 'unit_scale']), patch.object(spec.Path, 'read_text', autospec=True, side_effect=read), \
                patch.object(spec, 'write_json', side_effect=lambda p, d: writes.append((p, d))), patch.object(np, 'load', side_effect=AssertionError('Array cache read')), \
                patch.object(spec.source.cache.sampling, 'read_panels', side_effect=AssertionError('Panel read')), patch.object(spec.source, 'analyze_cell', side_effect=AssertionError('Prediction analysis')), \
                patch.object(local, 'fit', side_effect=AssertionError('Fit')), patch.object(local, 'predict', side_effect=AssertionError('Inference')), patch.object(np.linalg, 'solve', side_effect=AssertionError('Solve')):
            spec.main()
        full, compact = writes[0][1], writes[1][1]
        self.assertTrue(full['coverage_qualified']);self.assertLess(full['reproduction']['group_metric_max_abs_error'], 1e-10)
        self.assertIn('dominant_both', compact['groups']['mixed_pooled']['bank0']['comparisons']['opposite']['paired'])
        self.assertEqual(set(compact['groups']['root_397']['bank0']['comparisons']['opposite']['paired']), {'all'})
        self.assertIn('dominant_both', compact['groups']['root_397']['pooled']['comparisons']['opposite']['paired'])
        before['groups']['mixed_pooled']['pooled']['totals']['same']['excluded']['units'] += 1
        with self.assertRaisesRegex(ValueError, 'replay'):spec.report(cells, before)

    def test_scheduler_parent_gate_and_frozen_dominance(self):
        task = submit.task_specification('unit_scale')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}, 'runtime_seconds': None}}
        findings = {'primary': {'top3_prefixes_both_ranking_directions': [list(p) for p in learning.DOMINANT_PREFIXES]}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_scale']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else findings if p.name == 'findings.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();findings['primary']['top3_prefixes_both_ranking_directions'] = []
            with self.assertRaisesRegex(SystemExit, 'prefixes changed'):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
