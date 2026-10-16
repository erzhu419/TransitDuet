import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_held_removal as learning
from scripts import run_native_transit_held_removal_stage273 as spec
from scripts import submit_native_transit_held_removal_stage273_scheduleurm as submit


class HeldRemovalTest(unittest.TestCase):
    def test_conditions_distinguish_zero_remaining_support_from_single_root_redundancy(self):
        mixed = [{'root': root, 'scene_seed': root*10+i, 'scenario': s} for root in (397, 401) for i, s in enumerate(('burst', 'shift'))]
        held = [{**r, 'scene_seed': r['scene_seed']+1000} for r in mixed]
        groups = [(r['root'], r['scene_seed']) for r in mixed]
        masks, support = learning.conditions(mixed, held, groups)
        np.testing.assert_array_equal(support, 1)
        np.testing.assert_array_equal(masks['matched'].sum(axis=0), 1)
        self.assertEqual(int(masks['other_root_same_regime'].sum()), 4)
        single = [{**r, 'root': 397} for r in mixed]
        held = [{**r, 'scene_seed': r['scene_seed']+1000} for r in single]
        masks, support = learning.conditions(single, held, [(r['root'], r['scene_seed']) for r in single])
        np.testing.assert_array_equal(support, 2)
        self.assertEqual(int(masks['other_root_same_regime'].sum()), 0)
        with self.assertRaisesRegex(ValueError, 'Incomplete'):learning.conditions(mixed, held, groups[:-1])

    def test_paired_loss_averages_models_per_state_not_ensemble_predictions(self):
        full = np.zeros((2, 2));target = np.zeros_like(full)
        deleted = np.array([[[1., 1.], [2., 2.]], [[-1., -1.], [-2., -2.]]])
        mask = np.eye(2, dtype=bool)
        eligible = np.ones((2, 2), bool);eligible[0, 1] = False
        s, error = learning.paired_stats(full, deleted, target, eligible, {'all': np.ones((2, 2), bool), 'matched': mask, 'other': ~mask})
        self.assertAlmostEqual(s['all']['prefix54_squared_error_sum'], 5.)
        self.assertEqual(s['all']['state_exposures'], 2)
        self.assertEqual(s['all']['prefix_state_exposures'], 4)
        self.assertAlmostEqual(error, 0.)
        self.assertEqual(float(np.mean(deleted, axis=0).sum()), 0.)

    def test_same_query_equal_size_contrast_cancels_heterogeneous_full_losses(self):
        target = np.array([[1.], [10.]])
        full = np.zeros_like(target)
        deleted = np.array([[[2.], [10.]], [[1.], [12.]]])
        masks = {'all': np.ones((2, 2), bool), 'matched': np.eye(2, dtype=bool), 'other': ~np.eye(2, dtype=bool)}
        s, _ = learning.paired_stats(full, deleted, target, np.ones((2, 1), bool), masks)
        summary = learning.summarize([s, s])
        self.assertAlmostEqual(summary['matched_minus_other_MSE'], 2.5)
        self.assertAlmostEqual(summary['matched']['prefix54_minus_full60_MSE'], -48.)
        self.assertAlmostEqual(summary['other']['prefix54_minus_full60_MSE'], -50.5)

    def test_zero_eligible_queries_do_not_inflate_state_denominator(self):
        full = np.zeros((3, 2));deleted = np.ones((2, 3, 2));target = np.zeros_like(full)
        eligible = np.array([[False, False], [True, False], [True, True]])
        masks = {'all': np.ones((2, 3), bool), 'matched': np.array([[True, False, True], [False, True, False]])}
        masks['other'] = ~masks['matched']
        s, _ = learning.paired_stats(full, deleted, target, eligible, masks)
        self.assertEqual(s['matched']['state_exposures'], 2)
        self.assertAlmostEqual(s['matched']['prefix54_squared_error_sum'], 2.)

    def test_inference_preserves_original_and_bounds_without_solves(self):
        rng = np.random.default_rng(273);z = rng.normal(size=(7, 6, 9));z[..., 0] = 1.
        priors = np.zeros((7, 6, 3));eligible = np.ones((7, 6), bool);eligible[:, 0] = False
        m = {'radius': .01, 'maps': {k: {'center': np.zeros(8), 'scale': np.ones(8), 'weights': rng.normal(size=(9, 3))} for k in ('linear', 'interaction')}}
        r = {'priors': {p: {'unbounded': np.ones((9, 1)), 'bounded': np.ones((9, 1))*.1} for p in spec.source.learning.PRIORS}}
        c = {'direct_mse': np.ones((9, 1))}
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('Fit')):
            cost, _ = spec.infer(m, r, c, priors, z, np.zeros(7, int), eligible)
        for a in spec.ARMS:
            np.testing.assert_array_equal(cost[a][:, 0], 0.)
            if a.endswith('_clip'):self.assertLessEqual(np.max(np.abs(cost[a])), .01+1e-12)

    def test_complete_cached_cell_never_reads_training_labels_or_fits(self):
        rng = np.random.default_rng(273)
        def rows(offset):
            return [{'root': 397 if i//6 < 5 else 401, 'scene_seed': offset+i//6, 'scenario': str(i//6%5), 'probe_index': i%6} for i in range(60)]
        train, held = rows(0), rows(100)
        data = {}
        for name in ('train', 'held'):
            x = rng.normal(size=(60, 6, 198));x[:, 0] = 0.
            components = rng.normal(size=(60, 6, 2))*.001;components[:, 0] = 0.
            data[name] = {'x': x, 'components': components, 'originals': np.zeros(60, int), 'raw_logits': -np.ones((60, 6))}
        basis = np.eye(198)[:, :8]
        zs = {s: np.concatenate((np.ones((60, 6, 1)), d['x']@basis), axis=-1) for s, d in data.items()}
        eligible = np.ones((60, 6), bool);eligible[:, 0] = False
        full = {str(b): {p: {'scale': 1., 'weights': {'8': np.zeros((2, 198)).tolist()}} for p in ('free_old', 'shrunk')} for b in (0, 1)}
        folds = list(spec.cache.learning.joint.binding.policy.prefix_folds(train))
        prefixes = [{'bank': b, 'held_group': list(g), 'training_draws': list(range(b*16, (b+1)*16)), 'models': full[str(b)]} for b in (0, 1) for g, _, _ in folds]
        contexts = [(None, np.arange(60)), *[(list(g), np.asarray(ix)) for g, ix, _ in folds]]
        local, residual, cold, queries, saved, errors = [], [], [], [], {}, []
        values = np.zeros((60, 32, 6, 4));values[..., 1] = rng.choice([0., .4, -.5], size=(60, 32, 6), p=[.8, .1, .1]);values[:, :, 0, 1] = 0.
        for bank in (0, 1):
            for group, ids in contexts:
                for penalty in spec.PENALTIES:
                    i = len(local);meta = {'bank': bank, 'removed': group, 'penalty': penalty, 'fit_indices': ids.tolist(), 'training_draws': list(range(bank*16, (bank+1)*16))}
                    m = {**meta, 'radius': .01, 'maps': {k: {'center': np.zeros(8), 'scale': np.ones(8), 'weights': rng.normal(size=(9, 3))*.001} for k in ('linear', 'interaction')}}
                    r = {**meta, 'priors': {p: {'radius': .01, 'unbounded': rng.normal(size=(9, 1))*.001, 'bounded': rng.normal(size=(9, 1))*.0005} for p in spec.source.learning.PRIORS}}
                    c = {**meta, 'direct_mse': rng.normal(size=(9, 1))*.001}
                    local.append(m);residual.append(r);cold.append(c)
                    scope, label, query_ids = ('held', 'held', np.arange(60)) if group is None else ('retained', 'train', ids)
                    priors = np.repeat(data[label]['components'][query_ids, :, 1, None], 3, axis=-1)
                    cost, _ = spec.infer(m, r, c, priors, zs[label][query_ids], np.zeros(len(query_ids), int), eligible[query_ids])
                    queries.append({'index': i, 'scope': scope})
                    saved.update({f'{i}_{scope}_{a}': value for a, value in cost.items()})
                    if group is None:
                        for which, b in (('same', bank), ('opposite', 1-bank)):
                            y = values[:, b*16:(b+1)*16, :, 1]
                            errors.append({**meta, 'scope': 'held_'+which, 'metrics': {a: spec.source.learning.metrics(v, cost['raw_prior'], y, eligible, .01) for a, v in cost.items()}})
        class Archive(dict):
            def __enter__(self):return self
            def __exit__(self, *args):return False
        writes = []
        def file(p):
            if p.name == 'models.json':return json.dumps(local if spec.SOURCE_RUN in p.parts else residual if spec.source.SOURCE_RUN in p.parts else cold, default=lambda a: a.tolist())
            return json.dumps(queries if p.name == 'queries.json' else errors)
        def read(run, key, name):
            if run == spec.source.source.PREFIX_RUN:return prefixes
            if run == spec.cache.cached.SOURCE_RUN:return full
            return {'reference': {}}
        def draws(rs, panels, ds):
            self.assertTrue(all(r['scene_seed'] >= 100 for r in rs))
            self.assertIn('predictions', writes)
            return values[:, ds]
        with patch.object(spec.Path, 'read_text', autospec=True, side_effect=file), patch.object(spec, 'read_cell', side_effect=read), \
                patch.object(spec.np, 'load', side_effect=[Archive(basis=basis), Archive(saved)]), patch.object(spec.Path, 'mkdir'), \
                patch.object(spec, 'write_json'), patch.object(spec.np, 'savez_compressed', side_effect=lambda *a, **k: writes.append('predictions')), \
                patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(train, held)), \
                patch.object(spec.cache.learning.base, 'causal_data', side_effect=lambda rs, ref: data['held' if rs[0]['scene_seed'] >= 100 else 'train']), \
                patch.object(spec.cache.learning.base, 'training_draws', side_effect=draws), \
                patch.object(spec.source.learning, 'fit', side_effect=AssertionError('Local refit')), patch.object(spec.source.source.learning, 'fit', side_effect=AssertionError('Residual refit')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')), patch.object(np.linalg, 'solve', side_effect=AssertionError('Solve')):
            result = spec.analyze_cell(spec.Path('/tmp/unit_held'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(len(result['records']), 12)
        self.assertLess(result['prediction_max_abs_error'], 1e-12)
        self.assertLess(result['metric_max_abs_error'], 1e-10)
        self.assertEqual(result['support_before'], {'1': 60})

    def test_compact_summary_preserves_same_query_contrasts(self):
        s = {'all': {'state_exposures': 60, 'full60_MSE': 1., 'prefix54_MSE': 2.}, 'matched': {'prefix54_MSE': 3.}, 'other': {'prefix54_MSE': 1.}, 'matched_minus_other_MSE': 2.}
        payload = {'groups': {'mixed': {'support_prefixes_before': {'1': 60}, 'banks': {'pooled': {'.01': {'opposite': {'arm': s}}}}}}}
        c = spec.compact_report(payload)['groups']['mixed']['banks']['pooled']['.01']['opposite']['arm']
        self.assertEqual(c['prefix54_MSE_by_removal']['matched'], 3.)
        self.assertEqual(c['full60_MSE'], 1.);self.assertEqual(c['matched_minus_other_MSE'], 2.)

    def test_scheduler_single_unpinned_cpu_and_completed_parent_gate(self):
        task = submit.task_specification('unit_held')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_held']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
