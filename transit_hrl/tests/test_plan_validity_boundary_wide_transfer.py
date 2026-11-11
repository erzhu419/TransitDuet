from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from freq_hrl.experiments import plan_validity_boundary_information_fit as assembly
from freq_hrl.experiments import plan_validity_boundary_wide_transfer as transfer
from freq_hrl.experiments.plan_validity_boundary_parent_scale import fit_parent_targets, predict_parent
from scripts import run_plan_validity_boundary_wide_transfer as worker
from test_plan_validity_boundary_information_fit import banks


def data_bank():
    cell, design, cached, _, added = banks()
    return cell, design, assembly.assemble(cell, design, cached, added, 'wide')


class BoundaryWideTransferTest(unittest.TestCase):
    def test_fixed_64_fits_match_31_prefix_losses_without_ensemble_scoring(self):
        cell, _, data = data_bank()
        ledger = ComputeLedger()
        with ledger.recording('test'):
            public, private = transfer.diagnose(data, cell['development_prefix_seeds'], total_df=3)
        self.assertEqual(ledger.summary()['counts'], {'ridge_fit_calls': 64, 'diagnostic_model_window_predictions': 24576})
        labels = np.asarray(data['training_replicas'])
        groups = np.asarray(data['window_keys'])[:, 0]
        for view, saved in zip(public['views'], private):
            self.assertEqual(view['training_prefixes_per_fit'], 31)
            self.assertEqual(len(saved['folds']), 32)
            self.assertTrue(all(len(f['training_prefixes']) == 31 and f['excluded_prefix'] not in f['training_prefixes'] for f in saved['folds']))
            prefix = cell['development_prefix_seeds'][0]
            mask = groups == prefix
            target = labels[mask, view['evaluation_A_column']]
            scores = [predict_parent(f['model'], data['now'], data['wait_one_check'])[mask]
                for f in saved['folds'] if f['excluded_prefix'] != prefix]
            report = next(r for r in view['prefix_reports'] if r['seed'] == prefix)
            self.assertAlmostEqual(report['matched_future_MSE'], np.mean((np.asarray(scores) - target) ** 2))
            self.assertAlmostEqual(report['held_minus_matched'], report['held_prefix_MSE'] - report['matched_future_MSE'])
            self.assertAlmostEqual(view['overall']['held_minus_matched'], np.mean([r['held_minus_matched'] for r in view['prefix_reports']]))

    def test_moment_scoring_keeps_individual_model_variance(self):
        predictions = np.asarray([[0., 0.], [2., 4.]])
        mean, square = predictions.mean(axis=0), (predictions ** 2).mean(axis=0)
        target = np.asarray([0., 1.])
        scores = transfer.errors(mean, square, mean, target, target, np.ones(2, dtype=bool))
        self.assertAlmostEqual(scores['matched_future_MSE'], np.mean((predictions - target) ** 2))
        self.assertGreater(scores['matched_future_MSE'], np.mean((mean - target) ** 2))

    def test_future_labels_and_excluded_prefix_cannot_enter_their_predictions(self):
        cell, _, data = data_bank()
        _, original = transfer.diagnose(data, cell['development_prefix_seeds'], total_df=3)
        changed = deepcopy(data)
        held = np.asarray(data['window_keys'])[:, 0] == 0
        labels = np.asarray(changed['training_replicas'])
        labels[held, 1] += 1000
        changed['training_replicas'] = labels.tolist()
        _, other = transfer.diagnose(changed, cell['development_prefix_seeds'], total_df=3)
        for key in ('matched_prediction_mean', 'matched_prediction_square_mean', 'prefix_held_predictions'):
            np.testing.assert_array_equal(original[0][key], other[0][key])
        np.testing.assert_array_equal(original[1]['prefix_held_predictions'][held], other[1]['prefix_held_predictions'][held])
        for view in original:
            self.assertTrue(all(abs(f['model']['total_effective_df'] - 3) < 1e-10 for f in view['folds']))

    def test_loader_needs_no_E_or_Dense_files_and_reproduces_frozen_A_loss(self):
        cell, design, cached, _, added = banks()
        data = assembly.assemble(cell, design, cached, added, 'wide')
        pred, model = fit_parent_targets(data['now'], data['wait_one_check'], data['training_mean'], np.ones(384, dtype=bool), total_df=3)
        mse = float(np.mean((pred - data['training_mean']) ** 2))
        cells = [{**cell, 'task': 'fixture' + str(i)} for i in range(4)]
        design.update(cells=cells, source_manifest='source.json')
        frozen = {'run_name': design['run_name'], 'cells': [{'task': c['task'], 'optimizer_seed': c['optimizer_seed'],
            'models': {'wide': model}, 'training_rosters': {'wide': {'window_keys': data['window_keys'], 'label_counts': data['label_counts']}}} for c in cells]}
        report = {'run_name': design['run_name'], 'status': 'complete', 'stage': 'boundary_information_fit_complete',
            'all_models_frozen': True, 'new_ridge_fits': 8, 'new_environment_steps': 0,
            'cells': [{'task': c['task'], 'optimizer_seed': c['optimizer_seed'], 'training_readout': {
                'wide': {'MSE_to_own_training_A_mean': {'wide': mse}}}} for c in cells]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for part in ('public', 'private'):
                (root / 'fit' / part).mkdir(parents=True)
            (root / 'fit/private/frozen_models.json').write_text(json.dumps(frozen))
            (root / 'fit/public/result.json').write_text(json.dumps(report))
            (root / 'design.json').write_text(json.dumps(design))
            (root / 'fit.json').write_text(json.dumps({'output_directory': 'fit'}))
            (root / 'source.json').write_text('{}')
            shards = []
            for i in range(4):
                directory = root / f'wide{i}' / 'private'
                directory.mkdir(parents=True)
                (directory / 'rows.json').write_text(json.dumps(added))
                shards.append({'cell_index': i, 'method': 'wide', 'output_directory': f'wide{i}'})
            (root / 'sample.json').write_text(json.dumps({'shards': shards}))
            manifest = {'design_manifest': 'design.json', 'fit_manifest': 'fit.json', 'sample_manifest': 'sample.json'}
            with patch.object(worker, 'ROOT', root), patch.object(worker, 'load_cache', return_value=cached):
                _, prepared = worker.load_wide_inputs(manifest)
            self.assertEqual(len(prepared), 4)
            np.testing.assert_array_equal(prepared[0]['data']['training_replicas'], data['training_replicas'])

    def test_worker_verifies_all_banks_before_fitting_and_preserves_exact_budget(self):
        @contextmanager
        def runtime():
            yield {'fixture': True}

        cell, _, data = data_bank()
        prepared = [{'cell': {**cell, 'task': 'fixture', 'optimizer_seed': i}, 'data': data, 'frozen_mean_A_MSE': 0.} for i in range(4)]
        def load(_):
            count_work('frozen_model_window_predictions', 1536)
            return {'run_name': 'source'}, prepared

        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'design.json').write_text(json.dumps({'source_hardware_profile': {'fixture': True}}))
                manifest = {'run_name': 'diagnostic', 'design_manifest': 'design.json', 'output_directory': 'out', 'total_effective_df': 3,
                    'ridge_fit_calls': 256, 'expected_counts': {'ridge_fit_calls': 256, 'diagnostic_model_window_predictions': 98304, 'frozen_model_window_predictions': 1536}}
                path = root / 'manifest.json'
                path.write_text(json.dumps(manifest))
                loader = patch.object(worker, 'load_wide_inputs', side_effect=ValueError('fixture invalid last bank')) if fail else patch.object(worker, 'load_wide_inputs', side_effect=load)
                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), loader:
                    args = ['--manifest', str(path)]
                    self.assertEqual(worker.main(args), int(fail))
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                report = json.loads((root / 'out/public/result.json').read_text())
                self.assertEqual(report['measured_cost']['counts'].get('ridge_fit_calls', 0), 0 if fail else 256)
                self.assertEqual(report['new_environment_steps'], 0)
                if not fail:
                    self.assertFalse(report['evaluation_prefixes_accessed'] or report['frozen_models_modified'])


if __name__ == '__main__':
    unittest.main()
