import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_risk_influence as learning
from scripts import run_native_transit_risk_influence_stage268 as spec
from scripts import submit_native_transit_risk_influence_stage268_scheduleurm as submit
from tests import test_native_transit_future_forcing_stage263 as fixtures


class RiskInfluenceTest(unittest.TestCase):
    def test_common_fleet_offset_and_ineligible_pair_are_not_events(self):
        paired = np.ones((2, 8, 6, 2));paired[0, :, 5, 1] = 3.
        eligible = np.ones((2, 6), bool);eligible[0, 5] = False
        np.testing.assert_array_equal(learning.fleet_events(paired, eligible), np.zeros((2, 8), bool))
        paired[1, 3, 2, 1] += .25
        e = learning.fleet_events(paired, eligible)
        self.assertEqual(np.argwhere(e).tolist(), [[1, 3]])

    def test_sample_anova_separates_bernoulli_occurrence_and_event_amplitude(self):
        scores = np.zeros((1, 8, 1));scores[0, 0, 0] = 2.;scores[0, 4, 0] = 4.
        events = scores[..., 0]!=0
        parts = learning.scatter_parts(scores, events)
        self.assertEqual(float(np.sum(parts['within_event']**2)), 0.)
        self.assertEqual(float(np.sum(parts['within_none']**2)), 0.)
        self.assertAlmostEqual(float(np.sum(parts['occurrence']**2)), float(np.sum(parts['total']**2)))
        scores[0, 1, 0] = 6.;events = scores[..., 0]!=0
        parts = learning.scatter_parts(scores, events)
        np.testing.assert_allclose(parts['total'], sum(parts[k] for k in learning.PARTS))
        self.assertGreater(np.sum(parts['within_event']**2), 0.)
        self.assertAlmostEqual(float(np.sum(parts['total']**2)), sum(float(np.sum(parts[k]**2)) for k in learning.PARTS))
        for constant in (np.ones((1, 8), bool), np.zeros((1, 8), bool)):
            self.assertEqual(float(np.sum(learning.scatter_parts(scores, constant)['occurrence']**2)), 0.)

    def test_query_loo_matches_literal_deletion_mean_and_variance_for_both_banks(self):
        rng = np.random.default_rng(268);sf = rng.normal(size=(3, 8, 1, 2));sp = rng.normal(size=sf.shape)
        scores = {t: s[:, :, 0] for t, s in learning.coupled.response_scores(sf, sp, np.eye(2), np.diag([.3, 2.])).items()}
        events = rng.uniform(size=(3, 8))>.6;retained = np.array([False, True, True]);design = rng.normal(size=(5, 2))
        parts = {t: learning.scatter_parts(s, events) for t, s in scores.items()}
        for anchor in (0, 1):
            stats, influence = learning.query_stats(design, scores, parts, events, retained, anchor)
            for pop, mask in (('all', np.ones(3, bool)), ('excluded', ~retained), ('retained', retained)):
                expected = stats['populations'][pop];s = scores['coupled'];oldvar = expected['variance']['coupled']['total'];oldobs = expected['observed_squared_sum']['coupled']
                for state in np.flatnonzero(mask):
                    for draw in range(8):
                        means, variance = [[], []], 0.
                        for n in np.flatnonzero(mask):
                            for b in (0, 1):
                                x = s[n, b*4:(b+1)*4]
                                if n==state and draw//4==b:x = np.delete(x, draw%4, axis=0)
                                means[b].append(x.mean(axis=0))
                                pred = x@design.T
                                variance += float(np.sum(pred.var(axis=0, ddof=1))/len(x))
                        gap = (np.sum(means[1], axis=0)-np.sum(means[0], axis=0))*(1 if anchor==0 else -1)
                        obs = float(np.sum((design@gap)**2))
                        self.assertAlmostEqual(influence[pop]['variance_delta'][state, draw], variance-oldvar)
                        self.assertAlmostEqual(influence[pop]['observed_squared_delta'][state, draw], obs-oldobs)

    def test_endpoint_shared_cancellation_and_non_event_centered_energy_remain(self):
        x = np.zeros((1, 8, 1));x[0, 0, 0] = 3.;x[0, 4, 0] = 2.
        scores = {'prefix': x, 'minus_full': -x, 'coupled': x*0, 'one_ended': x*0, 'operator_change': x*0}
        events = x[..., 0]!=0;parts = {t: learning.scatter_parts(v, events) for t, v in scores.items()}
        stats, _ = learning.query_stats(np.ones((1, 1)), scores, parts, events, np.ones(1, bool), 0)
        a = stats['populations']['all']
        self.assertEqual(a['variance']['coupled']['total'], 0.)
        self.assertAlmostEqual(a['cross']['endpoints']['total'], -a['variance']['prefix']['total']-a['variance']['minus_full']['total'])
        scores['minus_full'] = x*0;scores['coupled'] = x;scores['one_ended'] = x
        parts = {t: learning.scatter_parts(v, events) for t, v in scores.items()}
        stats, _ = learning.query_stats(np.ones((1, 1)), scores, parts, events, np.ones(1, bool), 0)
        a = stats['populations']['all']
        self.assertGreater(a['centered_energy_on_none_draws'], 0.)
        self.assertAlmostEqual(a['centered_energy_on_event_draws']+a['centered_energy_on_none_draws'], a['variance']['coupled']['total'])

    def test_cached_risk_scores_reproduce_parent_variance_without_new_fits(self):
        data, cs, prior, geometry, residuals, full, prefix, x = fixtures.FutureForcingTest().fixture()
        rng = np.random.default_rng(268);paired = {}
        for b in (0, 1):
            v = rng.normal(size=(12, 16, 6, 2));v -= v.mean(axis=1, keepdims=True)
            paired[b] = (residuals[b][:, None]+v)*prior['scale']
        with patch.object(learning.coupled.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    args = (x@basis, cs[0], data['raw_logits']<=0,
                        np.asarray(prior['weights'][level])@coordinates.T, np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T, np.arange(6, 12), model, .8)
                    scores = learning.risk_scores(*args[:2], np.concatenate([paired[b] for b in (0, 1)], axis=1), *args[2:], level, prior['scale'])
                    old = learning.coupled.context(*args[:2], paired, *args[2:], 0, level, prior['scale'])
                    draw = np.concatenate([paired[b] for b in (0, 1)], axis=1);events = learning.fleet_events(draw, data['raw_logits']<=0)
                    retained = np.arange(12)>=6;parts = {t: learning.scatter_parts(s, events) for t, s in scores.items()}
                    design = (x@basis).reshape(-1, basis.shape[1])*2.
                    stats, _ = learning.query_stats(design, scores, parts, events, retained, 0)
                    for t in learning.TERMS:
                        cov = old['terms'][t]['parameter_covariance']['risk']
                        self.assertAlmostEqual(stats['populations']['all']['variance'][t]['total'], float(np.trace(design.T@design@cov)))

    def test_unique_future_aggregation_deduplicates_repeated_model_exposures(self):
        events = np.array([[True, False]*16]);x = np.arange(32, dtype=float)[None]
        influence = {f: x for f in ('energy', 'variance_delta', 'observed_squared_delta')}
        p = {'_states': [[397, 11, 9]], '_events': events, '_retained': np.array([True]), '_influence': {'held': {'all': influence}}}
        summary = spec.future_summary([p, p], 'held', 'all', float(x.sum()*2), 1.)
        self.assertEqual(summary['unique_state_future_keys'], 32)
        self.assertEqual(summary['unique_keys_with_eligible_event'], 16)
        self.assertAlmostEqual(summary['top_energy_fraction']['1'], 31/x.sum())
        self.assertEqual(summary['top_futures'][0]['exposures'], 2)

    def test_cell_npz_retains_piece_region_population_state_draw_axes(self):
        events = np.ones((2, 32), bool)
        p = {'_states': [[397, 11, 9], [397, 11, 12]], '_events': events, '_retained': np.array([True, False]),
            '_influence': {r: {pop: {f: np.ones((2, 32))*i for f in ('energy', 'variance_delta', 'observed_squared_delta')}
                for i, pop in enumerate(learning.POPULATIONS)} for r in spec.REPORT_REGIONS}}
        with patch.object(spec.np, 'savez_compressed') as save:
            spec.save_cell_arrays(spec.Path('/tmp/unit_risk'), {'key': ['mixed_a', 0, 'even_odd', 0], 'pieces': [p, p]})
        self.assertEqual(save.call_args.kwargs['energy'].shape, (2, 3, 3, 2, 32))
        np.testing.assert_array_equal(save.call_args.kwargs['energy'][0, 0, 2], np.full((2, 32), 2.))
        np.testing.assert_array_equal(save.call_args.kwargs['state_keys'], p['_states'])

    def test_compact_report_retains_anova_and_primary_unique_future_leaders(self):
        leaders = [{'state': [397, 11, 9], 'draw': 7, 'energy': 1.}]
        s = {'regions': {'held': {'populations': {'all': {'coupled_scatter_fractions': {'occurrence': .8},
            'future_influence': {'top_futures': leaders, 'top_energy_fraction': {'1': .2}}}}}}}
        payload = {'groups': {'mixed_pooled': {'pooled': {'v': {'removed': s}}}}, 'focus_mixed_removed': {'v': s}}
        with patch.object(spec, 'PRIMARY', ('v',)):
            c = spec.compact_report(payload)
        a = c['groups']['mixed_pooled']['pooled']['v']['removed']['regions']['held']['populations']['all']
        self.assertNotIn('top_futures', a['future_influence'])
        self.assertEqual(c['mixed_removed_future_leaders']['v'], leaders)
        self.assertEqual(a['coupled_scatter_fractions']['occurrence'], .8)

    def test_main_reads_only_cached_panels_and_ledger_without_fits_or_selection(self):
        ledger = [{'key': list(k), 'pieces': []} for k in spec.KEYS]
        with patch('sys.argv', ['runner', '--run-name', 'unit_risk']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as panels, \
                patch.object(spec.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(ledger) if p.name=='cells.json' else '{}'), \
                patch.object(spec, 'analyze_cell', return_value={'pieces': []}) as analyze, patch.object(spec, 'save_cell_arrays'), \
                patch.object(spec, 'report', return_value={}), patch.object(spec, 'compact_report', return_value={}), patch.object(spec, 'write_json'), \
                patch.object(learning.coupled.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        panels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN);self.assertEqual(analyze.call_count, 32)

    def test_scheduler_single_unpinned_cpu_and_completed_parent_gate(self):
        task = submit.task_specification('unit_risk')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_risk']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
