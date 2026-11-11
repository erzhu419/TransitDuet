from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments import plan_validity_boundary_parent_scale as parent
from freq_hrl.experiments import plan_validity_boundary_wide_transfer as transfer
from freq_hrl.experiments import plan_validity_boundary_wide_influence as influence
from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from scripts import run_plan_validity_boundary_wide_influence as worker
from test_plan_validity_boundary_wide_transfer import data_bank


class BoundaryWideInfluenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell, _, cls.data = data_bank()
        cls.prior, cls.saved = transfer.diagnose(cls.data, cell['development_prefix_seeds'], total_df=3)
        _, cls.frozen = parent.fit_parent_targets(cls.data['now'], cls.data['wait_one_check'],
            cls.data['training_mean'], np.ones(384, dtype=bool), total_df=3)

    def test_raw_coordinates_preserve_centering_and_spent_interactions(self):
        rng = np.random.default_rng(11405)
        model = {'weights': rng.normal(size=120).tolist(), 'parent_mean': rng.normal(size=39).tolist(),
            'parent_scale': rng.uniform(.5, 2, size=39).tolist()}
        original = deepcopy(model)
        now, wait = self.data['now'].copy(), self.data['wait_one_check'].copy()
        wait[:, 41] = 1 - now[:, 41]
        wait[:, 81:120] = wait[:, :39] * wait[:, 41:42]
        scores = parent.predict_parent(model, now, wait)
        np.testing.assert_allclose((wait - now) @ influence.raw_weights(model), scores, rtol=1e-12, atol=1e-12)
        self.assertEqual(model, original)

    def test_fixed_model_attribution_closes_without_new_fits(self):
        before = deepcopy([[f['model'] for f in v['folds']] for v in self.saved])
        ledger = ComputeLedger()
        with patch.object(parent, 'fit_parent_targets', side_effect=AssertionError('no new fits')), \
             patch.object(transfer, 'fit_parent_targets', side_effect=AssertionError('no new fits')), ledger.recording('test'):
            public, private = influence.attribute(self.data, self.frozen, self.saved, self.prior)
        self.assertEqual(ledger.summary()['counts'], {'saved_fold_window_predictions': 24576, 'geometry_svd_calls': 32})
        self.assertEqual(before, [[f['model'] for f in v['folds']] for v in self.saved])
        self.assertEqual(len(public['prefix_geometry']), 32)
        for r, view in enumerate(public['views']):
            overall = self.prior['views'][r]['overall']
            self.assertAlmostEqual(sum(b['gap'] for b in view['block_gap_allocations']), overall['held_minus_matched'])
            self.assertAlmostEqual(sum(p['allocation'] for p in view['held_label_allocation_by_training_prefix']), overall['held_minus_zero'])
            self.assertAlmostEqual(view['matched_self_label_allocation'] + view['matched_other_label_allocation'], overall['matched_minus_zero'])
            np.testing.assert_allclose(private['held_loss_allocation_by_training_prefix'][r].sum(axis=1),
                [p['held_minus_zero'] for p in self.prior['views'][r]['prefix_reports']], atol=1e-10)

    def test_reciprocal_geometry_mismatch_is_not_misreported_as_label_noise(self):
        changed = deepcopy(self.saved)
        changed[1]['folds'][0]['model']['alpha'] *= 2
        with self.assertRaisesRegex(ValueError, 'share label-independent'):
            influence.attribute(self.data, self.frozen, changed, self.prior)

    def test_worker_enforces_zero_fit_budget_and_preserves_attempt(self):
        @contextmanager
        def runtime():
            yield {'fixture': True}

        for extra_fit in (False, True):
            with self.subTest(extra_fit=extra_fit), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'design.json').write_text(json.dumps({'source_hardware_profile': {'fixture': True}}))
                counts = {'frozen_model_window_predictions': 384, 'saved_fold_window_predictions': 24576, 'geometry_svd_calls': 32}
                path = root / 'manifest.json'
                path.write_text(json.dumps({'run_name': 'fixture', 'output_directory': 'out', 'design_manifest': 'design.json', 'expected_counts': counts}))

                def run(_):
                    for key, value in counts.items():
                        count_work(key, value)
                    if extra_fit:
                        count_work('ridge_fit_calls')
                    return {'task': 'fixture'}, {'fixed': True}

                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), \
                     patch.object(worker, 'run_attribution', side_effect=run):
                    args = ['--manifest', str(path)]
                    self.assertEqual(worker.main(args), int(extra_fit))
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                report = json.loads((root / 'out/public/result.json').read_text())
                if not extra_fit:
                    self.assertFalse(report['evaluation_prefixes_accessed'] or report['frozen_models_modified'])
                    self.assertEqual((report['new_fits'], report['new_environment_steps']), (0, 0))


if __name__ == '__main__':
    unittest.main()
