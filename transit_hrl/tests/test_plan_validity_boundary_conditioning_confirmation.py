from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.experiments.compute_budget import ComputeLedger, count_work
from freq_hrl.experiments.plan_validity_boundary_conditioning import fit_conditioned_targets
from freq_hrl.experiments.plan_validity_boundary_conditioning_confirmation import COMPARISONS, bootstrap_root
from freq_hrl.experiments.plan_validity_boundary_information import draw_seed
from freq_hrl.experiments.plan_validity_boundary_information_readout import summarize
from scripts import run_plan_validity_boundary_conditioning_confirmation as worker
from scripts import submit_plan_validity_boundary_conditioning_confirmation_scheduleurm as submitter
from scripts import analyze_plan_validity_boundary_conditioning_confirmation as analyzer
from scripts.analyze_plan_validity_boundary_information_readout import merge_root
from test_plan_validity_boundary_information_readout import fixture
from test_plan_validity_boundary_wide_transfer import data_bank


def moments(cell, plan):
    groups = np.repeat(cell['readout_prefix_seeds'], 12)
    windows = np.tile(np.arange(4), len(groups) // 4)
    truth = np.repeat(np.linspace(.8, 1.2, 16), 12)
    tail = truth[:, None] + np.random.default_rng(11407).normal(0, .02, size=(192, 8))
    predictions = {'zero': np.zeros(192), 'constant': np.full(192, .25), 'causal_t': np.full(192, .35),
        'base': .5 * truth, 'wide': .6 * truth, 'state_rms': .75 * truth, 'tracking_distance': .9 * truth}
    data = {'groups': groups, 'windows': windows, 'tail': tail}
    panels = []
    for j in range(4):
        mask = np.isin(groups, cell['readout_prefix_seeds'][j * 4:(j + 1) * 4])
        panels.append(summarize({k: v[mask] for k, v in data.items()}, {k: v[mask] for k, v in predictions.items()}, comparisons=COMPARISONS))
    return panels, summarize(data, predictions, comparisons=COMPARISONS)


class BoundaryConditioningConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((worker.ROOT / 'protocols/plan_validity_boundary_conditioning_confirmation_2026-11-11.json').read_text())
        cls.design = json.loads((worker.ROOT / 'protocols/plan_validity_boundary_information_2026-11-11.json').read_text())
        cls.freeze = json.loads((worker.ROOT / cls.plan['freeze_manifest']).read_text())
        _, _, cls.rows, cls.models = fixture()
        del cls.models['dense']
        _, _, data = data_bank()
        for method in ('tracking_distance', 'state_rms'):
            _, cls.models[method] = fit_conditioned_targets(data['now'], data['wait_one_check'], data['training_mean'],
                np.ones(384, dtype=bool), total_df=3, gate_kind=method)
        for row in cls.rows:
            for j, window in enumerate(row['windows']):
                for k, replica in enumerate(window['A_replicas'], 40):
                    replica['suffix_index'] = k
                    replica['suffix_seed'] = draw_seed(17, row['seed'], row['opportunity_step'], j, k, 11402)

    def test_release_retains_all_reserved_roles_and_exact_resource_budget(self):
        release = submitter.prepare(self.plan, self.freeze, self.design, Path('plan.json'), Path('release.json'))
        self.assertEqual((len(release['shards']), release['released_environment_steps']), (16, 14822400))
        self.assertEqual(sum(s['new_pairs'] for s in release['shards']), 6144)
        for i, cell in enumerate(self.plan['cells']):
            self.assertEqual([s for shard in release['shards'] if shard['cell_index'] == i for s in shard['prefix_seeds']], cell['readout_prefix_seeds'])
        for job in release['jobs']:
            self.assertEqual((job['cpu'], job['ram_mb'], job['vram'], job['require_node'], job['preferred_node']), (1, 2048, 0, None, None))
            self.assertEqual(job['allowed_nodes'], [f'node{i:03d}' for i in range(1, 7)])
            self.assertTrue(job['allow_cpu_training'] and job['protect_from_auto_eviction'])
        bad = deepcopy(self.plan)
        bad['cells'][0]['readout_prefix_seeds'][0] = self.design['cells'][0]['readout_prefix_seeds'][0]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            submitter.prepare(bad, self.freeze, self.design, Path('plan.json'), Path('release.json'))

    def test_public_merge_and_whole_prefix_bootstrap_match_direct_readout(self):
        cell = self.plan['cells'][2]
        panels, direct = moments(cell, self.plan)
        merged = merge_root(panels, comparisons=COMPARISONS)
        for name, result in direct['comparisons'].items():
            self.assertAlmostEqual(result['MSE_delta'], merged['comparisons'][name]['MSE_delta'])
            self.assertAlmostEqual(result['conditional_suffix_MC_SE'], merged['comparisons'][name]['conditional_suffix_MC_SE'])
        values = np.asarray([p['MSE_delta'] for p in merged['comparisons']['tracking_distance_minus_wide']['prefix_reports']])
        indices = np.random.default_rng(cell['bootstrap_seed']).integers(0, 16, (10000, 16))
        expected = np.quantile(values[indices].mean(axis=1), .95, method='linear')
        result = bootstrap_root(merged, cell, self.plan)
        actual = result['comparisons']['tracking_distance_minus_wide']['prefix_bootstrap']['nominal_one_sided_95_upper']
        self.assertAlmostEqual(expected, actual)
        self.assertTrue(result['prediction_confirmation_rule_passed'])
        changed = deepcopy(result)
        for part in changed['comparisons'].values():
            part['conditional_suffix_MC_SE'] = 1000.
            for p in part['prefix_reports']:
                p['conditional_suffix_MC_SE'] = 1000.
        other = bootstrap_root(changed, cell, self.plan)
        self.assertEqual(other['comparisons']['tracking_distance_minus_wide']['prefix_bootstrap'], result['comparisons']['tracking_distance_minus_wide']['prefix_bootstrap'])
        bad = deepcopy(result)
        bad['comparisons']['tracking_distance_minus_base']['window_halves'][1]['MSE_delta'] = 0.
        self.assertFalse(bootstrap_root(bad, cell, self.plan)['prediction_confirmation_rule_passed'])

    def test_all_sixteen_shards_audited_before_any_bootstrap(self):
        release = submitter.prepare(self.plan, self.freeze, self.design, Path('plan.json'), Path('release.json'))
        panels = [moments(cell, self.plan)[0] for cell in self.plan['cells']]
        records = []
        for shard in release['shards']:
            i, j = shard['cell_index'], shard['shard_index']
            cell = self.plan['cells'][i]
            records.append({'run_name': self.plan['run_name'], 'task': cell['task'], 'optimizer_seed': cell['optimizer_seed'],
                'shard_index': j, 'status': 'complete', 'stage': 'boundary_conditioning_confirmation_shard_complete',
                'sampled_prefix_seeds': shard['prefix_seeds'], 'completed_states': 12, 'completed_pairs': 384,
                'new_environment_steps': 926400, 'new_fits': 0, 'all_models_frozen': True, 'evaluation_prefixes_accessed': True,
                'readout': panels[i][j], 'measured_cost': {'cpu_seconds': 1., 'wall_seconds': 2.,
                    'counts': {'environment_steps': 926400, 'frozen_model_window_predictions': 336}}})
        for changed in (records[:-1], [*records[:-1], {**records[-1], 'status': 'failed'}]):
            ledger = ComputeLedger()
            with ledger.recording('test'), self.assertRaises(ValueError):
                analyzer.aggregate_records(self.plan, release, changed)
            self.assertEqual(ledger.summary()['counts'], {})
        ledger = ComputeLedger()
        with ledger.recording('test'):
            result = analyzer.aggregate_records(self.plan, release, records)
        self.assertEqual(ledger.summary()['counts'], {'bootstrap_prefix_resamples': 40000})
        self.assertEqual((result['completed_prefixes'], result['completed_states'], result['E_pairs'], result['evaluation_environment_steps']), (64, 192, 6144, 14822400))
        self.assertTrue(result['primary_prediction_rule_passed'])

    def test_global_snapshot_checks_all_models_before_new_prefix_access(self):
        design = {'cells': [{'task': 'fixture', 'optimizer_seed': 17 + i} for i in range(4)]}
        plan = {'freeze_manifest': 'freeze.json', 'methods': self.plan['methods'],
            'cells': [{'task': 'fixture', 'optimizer_seed': 17 + i, 'readout_prefix_seeds': list(range(200 + i * 1000, 216 + i * 1000))} for i in range(4)]}
        roster = {'window_keys': [[g, step, j] for g in range(32) for step in (50, 105, 150) for j in range(4)], 'label_counts': [2] * 384}
        bundle = {'run_name': 'frozen', 'evaluation_prefixes_accessed': False, 'cells': [
            {'task': 'fixture', 'optimizer_seed': 17 + i, 'models': deepcopy(self.models), 'training_roster': roster} for i in range(4)]}
        counts = {'frozen_model_window_predictions': 1536, 'ridge_fit_calls': 8, 'full_A_gate_window_predictions': 3072}
        report = {'run_name': 'frozen', 'status': 'complete', 'stage': 'boundary_conditioning_full_A_freeze_complete',
            'all_models_frozen': True, 'new_ridge_fits': 8, 'baseline_refits': 0, 'new_environment_steps': 0,
            'evaluation_prefixes_accessed': False, 'original_models_modified': False, 'measured_cost': {'counts': counts}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for part in ('public', 'private'):
                (root / 'fit' / part).mkdir(parents=True)
            (root / 'freeze.json').write_text(json.dumps({'run_name': 'frozen', 'output_directory': 'fit', 'expected_counts': counts}))
            (root / 'fit/public/result.json').write_text(json.dumps(report))
            path = root / 'fit/private/frozen_models.json'
            path.write_text(json.dumps(bundle))
            with patch.object(worker, 'ROOT', root):
                self.assertEqual(len(worker.load_snapshot(plan, design)), 4)
                del bundle['cells'][-1]['models']['tracking_distance']
                path.write_text(json.dumps(bundle))
                with self.assertRaises(ValueError):
                    worker.load_snapshot(plan, design)

    def test_worker_freeze_gate_before_native_work_and_no_fit_scoring_budget(self):
        @contextmanager
        def runtime():
            yield {'fixture': True}

        rows = {(r['seed'], r['opportunity_step']): r for r in self.rows}
        cell = {'task': 'fixture', 'optimizer_seed': 17}
        plan = {**self.plan, 'run_name': 'fixture', 'cells': [cell]}
        design = {'cells': [cell], 'source_hardware_profile': {'fixture': True}, 'horizon': 1200, 'period': 50, 'windows_per_state': 4}
        shard = {'cell_index': 0, 'shard_index': 0, 'prefix_seeds': [200, 201, 202, 203], 'environment_steps': 926400,
            'new_pairs': 384, 'output_directory': 'out'}
        for fail in (True, False):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / 'plan.json').write_text(json.dumps(plan))
                (root / 'design.json').write_text(json.dumps(design))
                path = root / 'release.json'
                path.write_text(json.dumps({'plan_manifest': 'plan.json', 'design_manifest': 'design.json', 'shards': [shard]}))

                def observe(*_, seed, **kwargs):
                    count_work('environment_steps', 1200)
                    return {'seed': seed, 'check_steps': [50, 105, 150]}

                def acquire(*args, **kwargs):
                    count_work('environment_steps', 76800)
                    return deepcopy(args[2])

                gate = patch.object(worker, 'load_snapshot', side_effect=ValueError('fixture invalid final frozen model')) if fail else patch.object(
                    worker, 'load_snapshot', return_value={('fixture', 17): self.models})
                with patch.object(worker, 'ROOT', root), patch.object(worker, 'single_thread_runtime', runtime), gate, \
                     patch.object(worker, 'load_frozen_C0', return_value=(None, None, None, None)), \
                     patch.object(worker, 'observe', side_effect=observe) as observer, \
                     patch.object(worker, 'fresh_reference', side_effect=lambda o, step, **kw: rows[o['seed'], step]), \
                     patch.object(worker, 'acquire_windows', side_effect=acquire):
                    args = ['--manifest', str(path), '--shard-index', '0']
                    self.assertEqual(worker.main(args), int(fail))
                    self.assertEqual(observer.call_count, 0 if fail else 4)
                    with self.assertRaises(FileExistsError):
                        worker.main(args)
                result = json.loads((root / 'out/public/result.json').read_text())
                self.assertEqual(result['new_environment_steps'], 0 if fail else 926400)
                self.assertEqual(result['measured_cost']['counts'].get('ridge_fit_calls', 0), 0)
                if not fail:
                    self.assertEqual(result['measured_cost']['counts']['frozen_model_window_predictions'], 336)


if __name__ == '__main__':
    unittest.main()
