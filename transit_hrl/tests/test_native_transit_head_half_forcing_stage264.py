import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_head_half_forcing as learning
from scripts import run_native_transit_head_half_forcing_stage264 as spec
from scripts import submit_native_transit_head_half_forcing_stage264_scheduleurm as submit
from tests import test_native_transit_future_forcing_stage263 as fixtures


class HeadHalfTest(unittest.TestCase):
    def test_target_gradient_matches_existing_profiled_loss(self):
        rng = np.random.default_rng(264)
        z = rng.normal(size=(12, 6, 3))
        target, change = rng.normal(size=(2, 12, 6, 2))
        precision = rng.uniform(.2, 2., size=(12, 15, 2))
        eligible = np.ones((12, 6), bool);eligible[:4, 5] = False
        total, difference = rng.normal(size=(2, 3))
        for model, penalty in (('free_old', 0.), ('shrunk', None), ('shrunk', 0.), ('shrunk', .7)):
            before = learning.calibration.objectives(z, target, precision, eligible, difference, penalty)[model][0]
            after = learning.calibration.objectives(z, target+change, precision, eligible, difference, penalty)[model][0]
            gradient = learning.label_gradient(z, precision, eligible, model, penalty)
            np.testing.assert_allclose(gradient(change), after(total)[1]-before(total)[1], atol=1e-13)

    def test_cached_cap_heads_populations_and_halves_without_fit(self):
        data, cs, prior, geometry, residuals, full, prefix, x = fixtures.FutureForcingTest().fixture()
        rng = np.random.default_rng(264)
        halves = {}
        for bank in (0, 1):
            noise = rng.normal(size=residuals[bank].shape)
            halves[bank, 0], halves[bank, 1] = residuals[bank]-noise, residuals[bank]+noise
        with patch.object(learning.calibration.source.source.optimizer, 'quadratic_fit', side_effect=AssertionError('Refit')):
            for level, (basis, coordinates) in geometry.items():
                for model in ('free_old', 'shrunk'):
                    c = learning.context(x@basis, cs[0], residuals, halves, data['raw_logits']<=0,
                        np.asarray(prior['weights'][level])@coordinates.T,
                        np.asarray(full[model]['weights'][level])@coordinates.T,
                        np.asarray(prefix[model]['weights'][level])@coordinates.T,
                        np.arange(6, 12), model, .8, 0, level)
                    v = c['responses']
                    np.testing.assert_allclose(v['cross16_joint'], sum(v[f'cross16_{p}'] for p in learning.PARTS), atol=1e-12)
                    np.testing.assert_allclose(v['cross16_service'], v['cross16_excluded_service']+v['cross16_retained_service'], atol=1e-12)
                    for h in learning.HEADS:
                        np.testing.assert_allclose(v[f'cross16_{h}'], (v[f'first8_{h}']+v[f'last8_{h}'])/2, atol=1e-12)
                        np.testing.assert_allclose(v[f'first8_{h}']-v[f'last8_{h}'], v[f'within_anchor_{h}']-v[f'within_other_{h}'], atol=1e-12)
                    self.assertLess(c['diagnostics']['reference_max_abs_difference'], 1e-12)

    def test_raw_risk_counts_use_draw_state_feasible_action_exposures(self):
        paired = np.zeros((2, 4, 6, 2))
        paired[0, 0, 1, 1] = .01;paired[0, 1, 2, 1] = -.02
        paired[1, :, :, 1] = 10.
        feasible = np.ones((2, 6), bool);feasible[:, 0] = False
        s = learning.raw_label_stats(paired, feasible, np.array([True, False]))
        self.assertEqual((s['draw_state_action_exposures'], s['risk_nonzero'], s['risk_positive'], s['risk_negative']), (20, 2, 1, 1))
        self.assertAlmostEqual(s['risk_mean'], -.01/20)
        empty = learning.raw_label_stats(paired, np.zeros_like(feasible), np.array([True, False]))
        self.assertEqual(empty['draw_state_action_exposures'], 0)
        self.assertIsNone(empty['risk_mean'])

    def cancelling_predictions(self, n):
        v = {name: np.zeros(n) for name in learning.NAMES}
        for c in ('cross16', 'first8', 'last8'):
            v[f'{c}_service'][:] = 1.;v[f'{c}_risk'][:] = -1.
        v['cross16_excluded_service'][:] = 1.;v['cross16_excluded_risk'][:] = -1.
        return v

    def test_cancellation_keeps_all_cross_terms_and_half_dot(self):
        s, error = learning.squares(self.cancelling_predictions(2))
        self.assertEqual(s['cross16_joint'], 0.)
        self.assertEqual(s['cross16_cross_excluded_service_excluded_risk'], -4.)
        self.assertEqual(s['cross16_cross_service_risk'], -4.)
        self.assertEqual(s['first_last_dot_service'], 2.)
        self.assertEqual(error, 0.)

    def test_summary_retains_inactive_denominator_and_undefined_joint_cosine(self):
        s, _ = learning.squares(self.cancelling_predictions(6))
        region = {'feasible_action_cases': 6, 'state_cases': 1, 'disagreements': 0, 'bands': {b: s for b in ('full', 'low', 'rest')}}
        a = {'cell': ['mixed_a', 0, 'even_odd', 0], 'bank': 0, 'active': True, 'regions': {'held': region}}
        zero = {**region, 'bands': {b: dict.fromkeys(learning.FIELDS, 0.) for b in ('full', 'low', 'rest')}}
        out = spec.summarize([a, {**a, 'active': False, 'regions': {'held': zero}}])
        self.assertEqual((out['prefix_model_exposures'], out['active_prefix_models']), (2, 1))
        self.assertEqual(out['regions']['held']['feasible_action_exposures'], 12)
        band = out['regions']['held']['bands']['full']
        self.assertAlmostEqual(band['RMS']['cross16_service'], np.sqrt(.5))
        self.assertIsNone(band['half_alignment']['joint']['cosine'])
        self.assertEqual(band['half_alignment']['service']['positive_dot_prefix_models'], 1)

    def test_main_reads_cached_archive_once_without_fits_or_selection(self):
        with patch('sys.argv', ['runner', '--run-name', 'unit_head_half']), \
                patch.object(spec.cache.cached.training.cohort.source.source, 'load_data', return_value=({'training_data': []}, None, [], None)), \
                patch.object(spec.cache.cached.training.cohort.source, 'matched_views', return_value={}), \
                patch.object(spec.cache.sampling, 'read_panels', return_value={}) as labels, \
                patch.object(spec, 'analyze_cell', return_value={}) as analyze, patch.object(spec, 'report', return_value={}), \
                patch.object(spec.Path, 'read_text', return_value='{}'), patch.object(spec, 'write_json'), \
                patch.object(learning.calibration, 'fit_fixed', side_effect=AssertionError('Refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')):
            spec.main()
        labels.assert_called_once_with(spec.cache.cached.previous.DATA_RUN)
        self.assertEqual(analyze.call_count, 32)
        self.assertEqual(spec.FOCUS_PREFIX, (397, 913970203))

    def test_scheduler_single_unpinned_task_and_parent_completion(self):
        task = submit.task_specification('unit_head_half')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        self.assertFalse(any('/results/' in p for p in task['stage_input_paths']))
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_head_half']), patch.object(submit, 'inventory', return_value=[]), \
                patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name=='completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
