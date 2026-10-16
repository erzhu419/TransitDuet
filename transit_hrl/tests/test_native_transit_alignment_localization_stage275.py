import json
import unittest
from unittest.mock import patch

import numpy as np

from freq_hrl.domains.transit import native_alignment_localization as learning
from freq_hrl.domains.transit import native_local_risk as local
from scripts import run_native_transit_alignment_localization_stage275 as spec
from scripts import submit_native_transit_alignment_localization_stage275_scheduleurm as submit


def row(seed, ea, ha, eu=1, hu=1):
    metrics = {}
    for j, future in enumerate(learning.FUTURES):
        metrics[future] = {}
        for mode, (ex, he) in learning.MODES.items():
            for pop, n, a in ((ex, eu, ea[j]), (he, hu, ha[j])):
                units = min(eu, hu) if mode == 'paired' else n
                metrics[future][pop] = {'units': units, 'raw_MSE': 10.*units, 'model_MSE': 10.5*units+a,
                    'correction_squared': .5*units, 'alignment': a, 'delta_MSE': .5*units+a}
    return {'root': 397, 'scene_seed': seed, 'scenario': 'shift', 'probe_index': seed%2, 'bank': 0, 'metrics': metrics}


class AlignmentLocalizationTest(unittest.TestCase):
    def test_global_population_denominators_make_bins_additive(self):
        rs = [row(0, [1., 1.], [1., 1.]), row(1, [4., 4.], [12., 12.], hu=2)]
        total = learning.aggregate(rs)
        bins = learning.contributions(rs, ('scene_seed',), total)
        self.assertAlmostEqual(bins[0]['contributions']['opposite']['natural']['alignment'], 1/2-1/3)
        self.assertAlmostEqual(bins[1]['contributions']['opposite']['natural']['alignment'], 4/2-12/3)
        self.assertAlmostEqual(sum(b['contributions']['opposite']['natural']['alignment'] for b in bins), 5/2-13/3)

    def test_concentration_ranks_one_bank_then_renormalizes_other_bank(self):
        rs = [row(i, [a, b], [0., 0.]) for i, (a, b) in enumerate(zip([10., 9., 8., 7., 6., 5.], [1., 30., -2., 4., 3., 2.]))]
        total = learning.aggregate(rs);bins = learning.contributions(rs, ('root', 'scene_seed'), total)
        c = learning.concentration(rs, bins, total, 'paired', 'same')
        self.assertEqual(c[0]['prefixes'], [[397, 0]])
        self.assertAlmostEqual(c[0]['share_of_net_gap'], 1/38)
        self.assertAlmostEqual(c[0]['share_of_absolute_mass'], 1/42)
        self.assertAlmostEqual(c[0]['remaining_alignment_gap'], 37/5)
        self.assertEqual(c[0]['remaining_units']['paired_excluded'], 5)
        self.assertEqual(learning.concentration(rs, bins, total, 'paired', 'opposite')[0]['prefixes'], [[397, 1]])

    def test_stability_is_for_the_same_row_with_two_futures(self):
        rs = [row(i, a, [0., 0.]) for i, a in enumerate(([1., 2.], [-2., -3.], [5., -5.], [0., 4.]))]
        s = learning.stability(rs, learning.aggregate(rs), 'paired')
        self.assertEqual([v['rows'] for v in s['buckets'].values()], [1, 1, 1, 1])
        self.assertAlmostEqual(s['same_sign_fraction'], 2/3)
        self.assertAlmostEqual(sum(b['opposite_alignment_contribution'] for b in s['buckets'].values()), -.5)

    def test_probe_sums_preserve_natural_and_replica_balanced_weights(self):
        rng = np.random.default_rng(275)
        ex, he = [{'probe_index': i} for i in range(2)], [{'probe_index': i} for i in (0, 0, 1, 1)]
        ee = np.array([[False, True, True], [False, True, False]])
        eh = np.array([[False, True, True], [False, True, False], [False, False, True], [False, True, True]])
        re, rh = rng.normal(size=(2, 3)), rng.normal(size=(4, 3))
        pe, ph = re+.1, rh-.2
        ey = {f: rng.normal(size=re.shape) for f in learning.FUTURES};hy = {f: rng.normal(size=rh.shape) for f in learning.FUTURES}
        with patch.object(np.linalg, 'solve', side_effect=AssertionError('Fit')):
            rs, error = learning.probe_records(ex, he, pe, ph, re, rh, ee, eh, ey, hy)
        ei, hi, w, counts = learning.source.common_pairs(ex, he, ee, eh)
        for f in learning.FUTURES:
            args = ((pe, re, ey[f], learning.source.state_weights(ee), 2), (ph, rh, hy[f], learning.source.state_weights(eh), 4),
                (pe[ei], re[ei], ey[f][ei], w, counts['units']), (ph[hi], rh[hi], hy[f][hi], w, counts['units']))
            for pop, values in zip(spec.source.POPULATIONS, args):
                expected, _ = learning.source.components(*values)
                for k, v in expected.items():self.assertAlmostEqual(sum(r['metrics'][f][pop][k] for r in rs), v)
        self.assertLess(error, 1e-12)

    def test_all_partitions_and_compact_output(self):
        rs = [row(i, [float(i), float(i+1)], [0., 0.]) for i in range(6)]
        s = learning.summarize(rs)
        self.assertLess(s['partition_max_abs_error'], 1e-12)
        c = spec.compact_report({'groups': {'g': {'pooled': s, 'bank0': s}}})
        self.assertIn('breakdown', c['groups']['g']['pooled']);self.assertNotIn('breakdown', c['groups']['g']['bank0'])
        self.assertIn('concentration', c['groups']['g']['bank0'])

    def test_complete_cached_cell_has_no_inference_fits_or_selection(self):
        rng = np.random.default_rng(275)
        def rows(offset):
            return [{'root': 397 if i//6 < 5 else 401, 'scene_seed': offset+i//6, 'scenario': str(i//6%5), 'probe_index': i%6, 'original_choice': 0} for i in range(60)]
        train, held = rows(0), rows(100);folds = list(spec.cache.learning.joint.binding.policy.prefix_folds(train))
        eligibility = np.ones((60, 6), bool);eligibility[:, 0] = False;eligibility[::7] = False
        raw = {s: rng.normal(size=(60, 6))*.01 for s in ('train', 'held')}
        values = {s: rng.normal(size=(60, 32, 6, 4))*.01 for s in raw}
        for s in raw:raw[s][:, 0] = 0.;values[s][:, :, 0] = 0.
        models, queries, saved, predictions, old = [], [], {}, {a: [] for a in (spec.ARM, 'raw_prior')}, []
        for bank in (0, 1):
            for group, ix, val in [(None, np.arange(60), np.arange(60)), *folds]:
                for penalty in spec.source.PENALTIES:
                    i = len(models);models.append({'bank': bank, 'removed': list(group) if group else None, 'penalty': penalty})
                    ep = raw['train']+rng.uniform(-.003, .003, (60, 6));hp = raw['held']+rng.uniform(-.003, .003, (60, 6))
                    predictions[spec.ARM].append(hp);predictions['raw_prior'].append(raw['held'])
                    scopes = [('train', 'train', ix), ('held', 'held', val)] if group is None else [('retained', 'train', ix), ('excluded', 'train', val)]
                    for scope, label, ids in scopes:queries.append({'index': i, 'scope': scope, 'label': label, 'ids': list(map(int, ids)), 'eligible': eligibility[ids].tolist()})
                    if group is None or penalty != spec.PENALTY:continue
                    saved[f'{i}_excluded_{spec.ARM}'] = ep[val];saved[f'{i}_excluded_raw_prior'] = raw['train'][val]
                    condition = train[val[0]]['root'], train[val[0]]['scenario']
                    he = np.array([j for j, r in enumerate(held) if (r['root'], r['scenario']) == condition])
                    er, hr = [train[j] for j in val], [held[j] for j in he]
                    ei, hi, w, counts = learning.source.common_pairs(er, hr, eligibility[val], eligibility[he])
                    for f, b in (('same', bank), ('opposite', 1-bank)):
                        ey = values['train'][val, b*16:(b+1)*16, :, 1].mean(axis=1);hy = values['held'][he, b*16:(b+1)*16, :, 1].mean(axis=1)
                        args = ((ep[val], raw['train'][val], ey, learning.source.state_weights(eligibility[val]), int(eligibility[val].any(axis=1).sum())),
                            (hp[he], raw['held'][he], hy, learning.source.state_weights(eligibility[he]), int(eligibility[he].any(axis=1).sum())),
                            (ep[val][ei], raw['train'][val][ei], ey[ei], w, counts['units']), (hp[he][hi], raw['held'][he][hi], hy[hi], w, counts['units']))
                        old.append({'index': i, 'future': f, 'metrics': {spec.ARM: {pop: learning.source.components(*a)[0] for pop, a in zip(spec.source.POPULATIONS, args)}}})
        predictions = {a: np.array(v) for a, v in predictions.items()}
        class Archive(dict):
            def __enter__(self):return self
            def __exit__(self, *args):return False
        def file(path):
            v = models if path.name == 'models.json' else old if path.name == 'errors.json' else queries if spec.source.PREDICTION_RUN in path.parts else {'held_states': [[r['root'], r['scene_seed'], r['probe_index']] for r in held]}
            return json.dumps(v)
        def draws(rs, panels, ds):
            label = 'held' if rs[0]['scene_seed'] >= 100 else 'train'
            return values[label][[(r['scene_seed']%100)*6+r['probe_index'] for r in rs]][:, ds]
        with patch.object(spec.Path, 'read_text', autospec=True, side_effect=file), patch.object(spec.np, 'load', side_effect=[Archive(saved), Archive(predictions)]), \
                patch.object(spec.Path, 'mkdir'), patch.object(spec, 'write_json'), patch.object(spec.cache.cached.training.cohort, 'split_rows', return_value=(train, held)), \
                patch.object(spec.cache.learning.base, 'training_draws', side_effect=draws), patch.object(spec.cache.learning.base, 'causal_data', side_effect=AssertionError('Reference inference')), \
                patch.object(local, 'fit', side_effect=AssertionError('Fit')), patch.object(local, 'predict', side_effect=AssertionError('Inference')), \
                patch.object(spec.cache.learning, 'select', side_effect=AssertionError('Selection')), patch.object(np.linalg, 'solve', side_effect=AssertionError('Solve')):
            result = spec.analyze_cell(spec.Path('/tmp/unit_localize'), ('mixed_a', 0, 'even_odd', 0), {'mixed_a': []}, {})
        self.assertEqual(len(result['rows']), 120);self.assertLess(result['metric_max_abs_error'], 1e-10)

    def test_scheduler_single_unpinned_cpu_and_parent_gate(self):
        task = submit.task_specification('unit_localize')
        self.assertEqual(task['cpu'], 1);self.assertIsNone(task['require_node'])
        self.assertEqual(task['allowed_nodes'], [f'node{i:03}' for i in range(1, 7)])
        parent = {'protocol': spec.source.EXPERIMENT_PROTOCOL, 'contract': spec.source.contract(), 'coverage_qualified': True}
        receipt = {'scheduler': {'status_counts': {'done': 1}, 'exit_code_counts': {'0': 1}}}
        with patch('sys.argv', ['submit', '--run-name', 'unit_localize']), patch.object(submit, 'inventory', return_value=[]), patch.object(submit.Path, 'exists', return_value=False), \
                patch.object(submit.Path, 'read_text', autospec=True, side_effect=lambda p: json.dumps(receipt if p.name == 'completion_receipt.json' else parent)), \
                patch.object(submit.subprocess, 'check_output', return_value='revision'), patch.object(submit, 'write_json'), patch.object(submit, 'execute_bulk') as bulk:
            submit.main();self.assertEqual(len(bulk.call_args.args[0]), 1)
            bulk.reset_mock();receipt['scheduler']['exit_code_counts'] = {'1': 1}
            with self.assertRaises(SystemExit):submit.main()
            bulk.assert_not_called()


if __name__ == '__main__': unittest.main()
