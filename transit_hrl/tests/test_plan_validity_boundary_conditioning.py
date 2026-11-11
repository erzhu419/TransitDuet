from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_conditioning as conditioning
from freq_hrl.experiments import plan_validity_boundary_parent_scale as parent
from freq_hrl.experiments import plan_validity_boundary_wide_transfer as transfer
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from scripts import run_plan_validity_boundary_conditioning as worker
from test_plan_validity_boundary_wide_transfer import data_bank


class BoundaryConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, _, cls.data = data_bank()
        cls.labels = np.asarray(cls.data['training_replicas'])
        cls.groups = np.asarray(cls.data['window_keys'])[:, 0]
        cls.baseline, cls.saved = transfer.diagnose(cls.data, cls.cell['development_prefix_seeds'], total_df=3)

    def test_equal_DF_column_count_norm_preservation_and_potential_symmetry(self):
        train = self.groups != 0
        for method in conditioning.METHODS:
            scores, model = conditioning.fit_conditioned_targets(self.data['now'], self.data['wait_one_check'], self.labels[:, 0],
                train, total_df=3, gate_kind=method)
            self.assertEqual(len(model['weights']), 240)
            self.assertAlmostEqual(model['total_effective_df'], 3)
            for arms in (self.data['now'], self.data['wait_one_check']):
                phi = parent.scaled_basis(arms, model['parent_mean'], model['parent_scale'])
                lifted = conditioning.conditioned_basis(arms, model)
                np.testing.assert_allclose((lifted ** 2).sum(axis=1), (phi ** 2).sum(axis=1), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(conditioning.predict_conditioned(model, self.data['now'], self.data['wait_one_check']), scores)
            np.testing.assert_allclose(conditioning.predict_conditioned(model, self.data['wait_one_check'], self.data['now']), -scores)
            np.testing.assert_array_equal(conditioning.predict_conditioned(model, self.data['now'], self.data['now']), 0)

    def test_uniform_lift_is_exact_old_kernel_and_predictor_without_refit(self):
        model = self.saved[0]['folds'][0]['model']
        now = parent.scaled_basis(self.data['now'], model['parent_mean'], model['parent_scale'])
        wait = parent.scaled_basis(self.data['wait_one_check'], model['parent_mean'], model['parent_scale'])
        x = wait - now
        lifted = np.column_stack((x, x)) / np.sqrt(2)
        weights = np.r_[model['weights'], model['weights']] / np.sqrt(2)
        np.testing.assert_allclose(lifted @ lifted.T, x @ x.T, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(lifted @ weights, parent.predict_parent(model, self.data['now'], self.data['wait_one_check']), rtol=1e-12, atol=1e-12)

    def test_excluded_state_and_labels_do_not_fit_gate_or_weights(self):
        train = self.groups != 0
        now, wait, target = self.data['now'].copy(), self.data['wait_one_check'].copy(), self.labels[:, 0].copy()
        now[~train], wait[~train] = wait[~train].copy(), now[~train].copy()
        target[~train] += 1
        for method in conditioning.METHODS:
            _, a = conditioning.fit_conditioned_targets(self.data['now'], self.data['wait_one_check'], self.labels[:, 0], train, total_df=3, gate_kind=method)
            _, b = conditioning.fit_conditioned_targets(now, wait, target, train, total_df=3, gate_kind=method)
            self.assertEqual(a, b)

    def test_same_crossfit_logic_counts_only_64_new_models_per_method_root(self):
        ledger = ComputeLedger()
        with ledger.recording('test'):
            public, private = transfer.diagnose(self.data, self.cell['development_prefix_seeds'], total_df=3,
                fit_targets=partial(conditioning.fit_conditioned_targets, gate_kind='tracking_distance'))
        self.assertEqual(ledger.summary()['counts'], {'ridge_fit_calls': 64, 'diagnostic_model_window_predictions': 24576})
        self.assertTrue(all(len(f['model']['weights']) == 240 for v in private for f in v['folds']))
        result = conditioning.compare_views(private, self.saved, self.data)
        for r, view in enumerate(result):
            expected = public['views'][r]['overall']['held_prefix_MSE'] - self.baseline['views'][r]['overall']['held_prefix_MSE']
            self.assertAlmostEqual(view['overall']['held_MSE_delta'], expected)
        c = deepcopy(private[0])
        c['matched_prediction_mean'] = np.zeros(384)
        c['matched_prediction_square_mean'] = np.ones(384)
        ref = deepcopy(c)
        ref['matched_prediction_square_mean'] = np.zeros(384)
        score = conditioning.paired_readout(c, ref, self.labels[:, 1], np.ones(384, dtype=bool))
        self.assertEqual(score['matched_future_MSE_delta'], 1)

    def test_followup_rule_preserves_views_halves_and_matched_future_gain(self):
        public = {'tracking_distance': {'views': [{'overall': {'matched_minus_zero': -1.}} for _ in range(2)]}}
        contrast = {name: [{'overall': {'held_MSE_delta': -1., 'matched_future_MSE_delta': -1.},
            'window_halves': [{'held_MSE_delta': -1.}, {'held_MSE_delta': -1.}]} for _ in range(2)]
            for name in ('tracking_distance_minus_uniform', 'tracking_distance_minus_state_rms')}
        self.assertTrue(conditioning.followup_rule(public, contrast))
        bad = deepcopy(contrast)
        bad['tracking_distance_minus_state_rms'][1]['window_halves'][1]['held_MSE_delta'] = 1.
        self.assertFalse(conditioning.followup_rule(public, bad))
        bad = deepcopy(contrast)
        bad['tracking_distance_minus_uniform'][0]['overall']['matched_future_MSE_delta'] = 1.
        self.assertFalse(conditioning.followup_rule(public, bad))

    def test_worker_checks_all_sources_first_and_enforces_512_fit_budget(self):
        @contextmanager
        def runtime():
            yield {'fixture': True}

        prepared = [{'cell': {**self.cell, 'task': 'fixture', 'optimizer_seed': i}, 'data': self.data,
            'baseline_public': self.baseline, 'baseline_private': self.saved} for i in range(4)]
        def load(_):
            count_work('frozen_model_window_predictions', 1536)
            return {'run_name': 'source'}, prepared

        def diagnose(*_, **kwargs):
            count_work('ridge_fit_calls', 64)
            count_work('diagnostic_model_window_predictions', 24576)
            return self.baseline, self.saved

        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'design.json').write_text(json.dumps({'source_hardware_profile': {'fixture': True}}))
                path = root / 'manifest.json'
                path.write_text(json.dumps({'run_name': 'fixture', 'output_directory': 'out', 'design_manifest': 'design.json',
                    'total_effective_df': 3, 'ridge_fit_calls': 512, 'expected_counts': {'frozen_model_window_predictions': 1536,
                    'ridge_fit_calls': 512, 'diagnostic_model_window_predictions': 196608}}))
                loader = patch.object(worker, 'load_sources', side_effect=ValueError('fixture invalid final source')) if fail else patch.object(worker, 'load_sources', side_effect=load)
                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), loader, \
                     patch.object(worker, 'diagnose', side_effect=diagnose) as fitter:
                    args = ['--manifest', str(path)]
                    self.assertEqual(worker.main(args), int(fail))
                    self.assertEqual(fitter.call_count, 0 if fail else 8)
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                result = json.loads((root / 'out/public/result.json').read_text())
                self.assertEqual(result['measured_cost']['counts'].get('ridge_fit_calls', 0), 0 if fail else 512)
                self.assertEqual(result['new_environment_steps'], 0)
                if not fail:
                    self.assertFalse(result['evaluation_prefixes_accessed'] or result['frozen_models_modified'])


if __name__ == '__main__':
    unittest.main()
