from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.compute_budget import count_work
from freq_hrl.experiments import plan_validity_boundary_conditioning as conditioning
from freq_hrl.experiments.plan_validity_boundary_parent_scale import fit_parent_targets
from scripts import run_plan_validity_boundary_conditioning_freeze as worker
from scripts import prepare_plan_validity_boundary_conditioning_confirmation as planner
from test_plan_validity_boundary_wide_transfer import data_bank


class BoundaryConditioningFreezeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell, _, cls.data = data_bank()
        _, model = fit_parent_targets(cls.data['now'], cls.data['wait_one_check'], cls.data['training_mean'], np.ones(384, dtype=bool), total_df=3)
        cls.controls = {'base': deepcopy(model), 'wide': deepcopy(model), 'constant': {'value': .01},
            'causal_t': {'feature_mean': [0.] * 120, 'feature_scale': [1.] * 120, 'weights': [.01] + [0.] * 120}}
        cls.prepared = [{'cell': {**cls.cell, 'task': 'fixture', 'optimizer_seed': i}, 'data': cls.data,
            'frozen_model': cls.controls['wide'], 'controls': deepcopy(cls.controls)} for i in range(4)]

    def test_eight_full_A_means_frozen_together_without_control_refit(self):
        @contextmanager
        def runtime():
            yield {'fixture': True}

        def load(_):
            count_work('frozen_model_window_predictions', 1536)
            return {'run_name': 'information'}, deepcopy(self.prepared)

        def fit(now, wait, target, train, **kwargs):
            np.testing.assert_array_equal(target, np.asarray(self.data['training_replicas']).mean(axis=1))
            self.assertTrue(np.all(train))
            return conditioning.fit_conditioned_targets(now, wait, target, train, **kwargs)

        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'design.json').write_text(json.dumps({'source_hardware_profile': {'fixture': True}}))
                path = root / 'freeze.json'
                path.write_text(json.dumps({'run_name': 'freeze', 'design_manifest': 'design.json', 'output_directory': 'out',
                    'total_effective_df': 3, 'ridge_fit_calls': 8, 'expected_counts': {'frozen_model_window_predictions': 1536,
                        'ridge_fit_calls': 8, 'full_A_gate_window_predictions': 3072}}))
                loader = patch.object(worker, 'load_freeze_inputs', side_effect=ValueError('fixture bad last source')) if fail else patch.object(worker, 'load_freeze_inputs', side_effect=load)
                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), loader, \
                     patch.object(worker, 'fit_conditioned_targets', side_effect=fit) as fitter:
                    args = ['--manifest', str(path)]
                    self.assertEqual(worker.main(args), int(fail))
                    self.assertEqual(fitter.call_count, 0 if fail else 8)
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                result = json.loads((root / 'out/public/result.json').read_text())
                self.assertEqual(result['new_environment_steps'], 0)
                if not fail:
                    self.assertTrue(result['all_models_frozen'])
                    self.assertFalse(result['evaluation_prefixes_accessed'] or result['training_metric_selection_performed'])
                    saved = json.loads((root / 'out/private/frozen_models.json').read_text())
                    self.assertEqual(len(saved['cells']), 4)
                    for c in saved['cells']:
                        self.assertEqual(set(c['models']), {'base', 'wide', 'constant', 'causal_t', *conditioning.METHODS})
                        for method in self.controls:
                            self.assertEqual(c['models'][method], self.controls[method])

    def test_bundle_prediction_does_not_read_target_labels(self):
        models = deepcopy(self.controls)
        for method in conditioning.METHODS:
            _, models[method] = conditioning.fit_conditioned_targets(self.data['now'], self.data['wait_one_check'],
                self.data['training_mean'], np.ones(384, dtype=bool), total_df=3, gate_kind=method)
        scores = conditioning.predict_frozen_bundle(models, self.data)
        changed = deepcopy(self.data)
        changed['training_mean'] += 100
        changed['training_replicas'] = (np.asarray(changed['training_replicas']) + 100).tolist()
        changed['tail'] = np.full((384, 8), -100.)
        other = conditioning.predict_frozen_bundle(models, changed)
        self.assertEqual(set(scores), {'zero', *models})
        for method in scores:
            np.testing.assert_array_equal(scores[method], other[method])

    def test_loader_requires_primary_rule_and_preserves_old_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for p in ('comparison/public', 'original/private'):
                (root / p).mkdir(parents=True)
            (root / 'comparison.json').write_text(json.dumps({'run_name': 'comparison', 'output_directory': 'comparison',
                'primary_root': {'task': 'fixture', 'optimizer_seed': 2}, 'source_manifest': 'wide.json'}))
            (root / 'wide.json').write_text(json.dumps({'fit_manifest': 'old.json'}))
            (root / 'old.json').write_text(json.dumps({'output_directory': 'original'}))
            report = {'run_name': 'comparison', 'status': 'complete', 'stage': 'boundary_conditioning_A_complete',
                'new_ridge_fits': 512, 'baseline_refits': 0, 'new_environment_steps': 0, 'evaluation_prefixes_accessed': False,
                'frozen_models_modified': False, 'cells': [{'task': 'fixture', 'optimizer_seed': 2, 'training_A_followup_rule_passed': True}]}
            report_path = root / 'comparison/public/result.json'
            report_path.write_text(json.dumps(report))
            bundle = {'run_name': 'information', 'cells': [{'task': 'fixture', 'optimizer_seed': i, 'models': self.controls} for i in range(4)]}
            (root / 'original/private/frozen_models.json').write_text(json.dumps(bundle))
            with patch.object(worker, 'ROOT', root), patch.object(worker, 'load_wide_inputs', return_value=({'run_name': 'information'}, deepcopy(self.prepared))):
                _, prepared = worker.load_freeze_inputs({'source_manifest': 'comparison.json'})
                self.assertEqual(prepared[0]['controls'], self.controls)
                report['cells'][0]['training_A_followup_rule_passed'] = False
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'did not pass'):
                    worker.load_freeze_inputs({'source_manifest': 'comparison.json'})

    def test_fresh_plan_has_distinct_roles_exact_budget_and_no_jobs(self):
        design = json.loads((worker.ROOT / 'protocols/plan_validity_boundary_information_2026-11-11.json').read_text())
        result = planner.prepare(design, freeze_manifest=Path('freeze.json'), forbidden=set())
        self.assertEqual((result['proposed_environment_steps'], result['new_paired_futures'], result['factual_episodes']), (14822400, 6144, 64))
        self.assertEqual(result['suffix_indices'], list(range(40, 48)))
        self.assertEqual(result['jobs'], [])
        self.assertFalse(result['native_sampling_released'])
        roles = [s for c in result['cells'] for s in [*c['readout_prefix_seeds'], c['bootstrap_seed']]]
        self.assertEqual((len(roles), len(set(roles))), (68, 68))
        self.assertEqual(result['primary_comparators'], ['wide', 'state_rms', 'base', 'zero'])
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            planner.prepare(design, freeze_manifest=Path('freeze.json'), forbidden={roles[0]})

    def test_historical_exclusions_include_effective_controller_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'plan_validity_old.json').write_text(json.dumps({'readout_prefix_seeds': [123], 'budget': 111}))
            cell = {'optimizer_seed': 1000, 'phase': 'preflight', 'seed_roles': {'train_seeds': [5]}}
            with patch.object(planner, 'seed_cells', return_value=[cell]), patch.object(planner, '_training_seed', side_effect=lambda **kw: 2000 + kw['iteration']):
                seeds = planner.historical_seeds(root, {'contract': {'primary_iterations': 9}})
            self.assertTrue({123, 1000, 5, 2000, 2001}.issubset(seeds))
            self.assertNotIn(2002, seeds)
            self.assertNotIn(111, seeds)


if __name__ == '__main__':
    unittest.main()
