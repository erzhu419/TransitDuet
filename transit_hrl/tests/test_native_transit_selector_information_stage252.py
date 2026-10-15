import copy
import json
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_selector_information as learning
from scripts import run_native_transit_selector_information_stage252 as spec
from scripts import submit_native_transit_selector_information_stage252_scheduleurm as submit
from tests import test_native_transit_balanced_critic_stage247 as fixtures


class SelectorInformationTest(unittest.TestCase):
    def test_new_validation_is_nested_and_excludes_fit_and_evaluation(self):
        for bank in (0, 1):
            halves = []
            for half in (0, 1):
                train, evaluation = learning.draw_assignment(bank, 'expanded', half)
                small = learning.validation_draws(bank, half, 'matched2')
                large = learning.validation_draws(bank, half, 'expanded8')
                self.assertEqual((len(train), len(small), len(large), len(evaluation)), (8, 2, 8, 16))
                self.assertEqual(small, large[:2])
                self.assertFalse(set(large) & set(train))
                self.assertFalse(set(large) & set(evaluation))
                self.assertEqual(set(train+large), set(range(16*bank, 16*(bank+1))))
                halves.append(large)
            self.assertFalse(set(halves[0]) & set(halves[1]))

    def test_inner_reconstruction_matches_both_original_models(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        before = copy.deepcopy(f.cached)
        with patch.object(learning.joint, 'fit_reference', side_effect=AssertionError('Raw reference retraining')):
            models, _ = learning.rebuild_inner(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        old_shrunk, _ = learning.previous.fit_paths(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        old_free, _ = learning.previous.previous.fit_paths(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        self.assertEqual(f.cached, before)
        for level in learning.strength.lowrank.LEVELS:
            np.testing.assert_allclose(models['shrunk']['weights'][level], old_shrunk['shrunk']['weights'][level], atol=1e-12)
            np.testing.assert_allclose(models['free_old']['weights'][level], old_free['free_old']['weights'][level], atol=1e-12)

    def test_equal_validation_labels_reproduce_keys_without_cross_arm_selection(self):
        f = fixtures.BalancedCriticTest()
        f.setUp()
        models, _ = learning.rebuild_inner(f.rows, f.data, f.draws, f.cached['all_mse'], f.fit)
        paths = {m: learning.base.candidates(f.data, model) for m, model in models.items()}
        curves = {m: {c: learning.base.curve(f.data, f.draws, p) for c, p in ps.items()} for m, ps in paths.items()}
        all_curves = {f'{m}_{s}': curves[m] for m in learning.MODELS for s in learning.SELECTORS}
        blocks = [{'held_group': [r, 0], 'half': h, 'scales': [.1, .1],
            'curves': {'all_mse': curves['free_old'], **all_curves}} for r in (397, 401) for h in (0, 1)]
        selected, summaries = learning.select(blocks)
        self.assertEqual(set(selected), set(learning.ARMS))
        for m in learning.MODELS:
            for s in learning.SELECTORS:
                self.assertEqual(selected[f'{m}_{s}'], selected[f'{m}_old_A2'])
                self.assertEqual(learning.summary_difference(summaries[f'{m}_{s}'], summaries[f'{m}_old_A2']), 0.)
        selected.update(old=dict.fromkeys(learning.FILTERS), frozen_old=dict.fromkeys(learning.FILTERS))
        paths.update(old=paths['free_old'], frozen_old=paths['free_old'])
        before = copy.deepcopy(models)
        pred = learning.predict(paths, selected)
        self.assertEqual(set(pred), set(learning.VARIANTS))
        self.assertEqual(len(pred), 17)
        self.assertEqual(models, before)
        matrix = learning.previous.balanced.metrics(f.data, f.draws, pred)
        diagnostic = spec.information.learning.information_error(f.draws, f.draws, f.data, pred)
        records = [{'banks': [{'bank': b, 'matrix': {'held60': matrix},
            'diagnostics': {'held60': diagnostic}, 'selection': selected} for b in (0, 1)]}]
        result = spec.compact(records, 'held60', None)
        self.assertEqual(result['model_bank_exposures'], 2)
        self.assertEqual(spec.compact(records, 'held60', 0)['model_bank_exposures'], 1)
        self.assertEqual(result['contrasts']['free_old_expanded8-matched2']['delta_cost'], 0.)
        self.assertEqual(sum(result['selection_counts']['shrunk_expanded8']['pooled_safe'].values()), 2)

    def test_nested_summary_comparison_detects_changed_curves(self):
        before = {'candidate': {'halves': [{'score': 0.}, {'score': 1.}], 'pooled': {'score': .5}}}
        after = copy.deepcopy(before)
        after['candidate']['halves'][1]['score'] += .25
        self.assertEqual(learning.summary_difference(after, before), .25)
        after['candidate']['halves'].pop()
        with self.assertRaises(ValueError): learning.summary_difference(after, before)

    def test_scheduler_and_registered_budget_keep_full_models_frozen(self):
        c = spec.contract()
        self.assertEqual(c, json.loads(json.dumps(c)))
        self.assertEqual(c['budget']['new_full_critic_fits'], 0)
        self.assertEqual(c['budget']['new_raw_reference_fits'], 0)
        self.assertEqual(c['budget']['new_native_episodes'], 0)
        self.assertEqual(c['budget']['inner_reconstruction_solves'], 32*2*20*4*2)
        for key in (spec.PREFLIGHT, None):
            t = submit.task_specification('unit_selector_info', key)
            self.assertEqual(t['cpu'], 1)
            self.assertIsNone(t['require_node'])
            self.assertEqual(t['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
            self.assertNotIn('_raw', t['result_dir'])
            self.assertFalse(any('/results/' in p for p in t['stage_input_paths']))

    def test_expansion_requires_successful_preflight_and_cannot_repeat(self):
        records = [{**submit.task_specification('unit_selector_info', spec.PREFLIGHT), 'id': 't0', 'status': 'done', 'exit_code': 0}]
        with patch.object(sys, 'argv', ['submit', '--run-name', 'unit_selector_info', '--phase', 'remaining']), \
                patch.object(submit, 'inventory', return_value=records), \
                patch.object(submit.Path, 'read_text', return_value=json.dumps({'contract': spec.contract()})), \
                patch.dict(sys.modules, {'scheduler': SimpleNamespace(load_state=lambda: {'tasks': records})}), \
                patch.object(submit, 'execute_bulk') as bulk:
            submit.main()
            tasks = bulk.call_args.args[0]
            self.assertEqual(len(tasks), 31)
            bulk.reset_mock()
            records[0]['exit_code'] = 1
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()
            records[0]['exit_code'] = 0
            records.append({**tasks[0], 'id': 't1', 'status': 'queued'})
            with self.assertRaises(SystemExit): submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
