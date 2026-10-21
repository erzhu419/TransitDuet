import ast
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v173_confirmation as confirm


class TemporalConfirmationTests(unittest.TestCase):
    def inputs(self, time=7200.):
        q = dict(actor=1, mode='old', query_key=[1, 2, True], query_time_s=time, episode=[2, 4, 2, True],
            candidate_key=[1, 'old', 'active'], point_s=100., denominator_eligible=True, native_supported=True,
            native_stage=dict(reason='not_applicable', kind='moving-launch'),
            condition_values=dict(actor=1, mode='old', query_phase='active'), observable=dict(hour=int(time//3600)))
        entry = dict(status='finite', day_count=64, qlow_s=-10., qhigh_s=10.)
        candidate = {(1, 'old', 'active', 'hour2'): entry, (1, 'old', 'active', 'hour13'): entry}
        reference = {(1, 'old', 'active'): dict(entry, qlow_s=-100., qhigh_s=100.)}
        secondary = {(1, 'old', 'active'): dict(entry, qlow_s=-20., qhigh_s=20.)}
        protocol = dict(comparator=dict(key=['actor', 'mode', 'query_phase']))
        return q, candidate, reference, secondary, protocol

    def test_new_key_uses_only_query_clock_and_lookup_does_not_change_maps(self):
        args = self.inputs()
        before = deepcopy(args)
        p = confirm.prediction(*args)
        self.assertEqual(p['candidate_key'][-1], 'hour2')
        self.assertEqual(p['bounds'], [90., 110.])
        self.assertEqual(args, before)
        args = self.inputs(46800.)
        self.assertEqual(confirm.prediction(*args)['candidate_key'][-1], 'hour13')
        args = self.inputs(10800.)
        self.assertFalse(confirm.prediction(*args)['finite'])

    def test_low_unseen_and_native_abstentions_do_not_pool(self):
        q, candidate, ref, second, protocol = self.inputs()
        for table in ({}, {(*q['candidate_key'], 'hour2'): dict(status='abstain_less_than_19_days', day_count=18)}):
            p = confirm.prediction(q, table, ref, second, protocol)
            self.assertFalse(p['finite'])
            self.assertTrue(p['reference_finite'])
        q.update(native_supported=False, candidate_key=None)
        self.assertFalse(confirm.prediction(q, candidate, ref, second, protocol)['finite'])

    def test_primary_secondary_scoring_uses_identical_candidate_mask(self):
        p = confirm.prediction(*self.inputs())
        v = dict(zip(confirm.METRICS, confirm.values(p, dict(status='resolved', full_eta_s=100.))))
        self.assertEqual((v['score_s'], v['reference_score_s'], v['secondary_score_s']), (20., 200., 40.))
        self.assertEqual((v['score_delta_s'], v['secondary_score_delta_s']), (-180., -20.))
        p['secondary_finite'] = False
        self.assertIsNone(confirm.values(p, dict(status='resolved', full_eta_s=100.)))

    def test_moving_scope_contains_unsupported_requests_and_episode_denominator(self):
        q, *tables = self.inputs(46800.)
        p = confirm.prediction(q, *tables)
        other = deepcopy(q)
        other.update(native_supported=False, candidate_key=None, episode=[3, 5, 2, True])
        rows = [(q, p, dict(status='resolved', full_eta_s=100.)) for _ in range(9)]
        rows.append((other, confirm.prediction(other, *tables), dict(status='resolved', full_eta_s=1000.)))
        cells, _ = confirm.reduce_rows(rows)
        self.assertEqual(cells['hour13_moving_launch']['finite_fraction'], .5)
        self.assertEqual(cells['hour13_moving_launch']['eligible_queries'], 10)
        self.assertEqual(cells['hour13_moving_launch']['coverage'], 1.)

    def test_missing_labels_and_comparators_remain_inconclusive(self):
        q, *tables = self.inputs()
        p = confirm.prediction(q, *tables)
        cells, _ = confirm.reduce_rows([(q, p, dict(status='unresolved', full_eta_s=None))])
        self.assertEqual(cells['hour2']['finite_unresolved_n'], 1)
        self.assertIsNone(cells['hour2']['coverage'])
        p['secondary_finite'] = False
        cells, _ = confirm.reduce_rows([(q, p, dict(status='resolved', full_eta_s=100.))])
        self.assertEqual(cells['hour2']['missing_reference_n'], 1)
        self.assertIsNone(cells['hour2']['score_delta_s'])

    def test_crossed_bootstrap_preserves_shared_axes_and_rejects_partial_grid(self):
        cells = [dict(actor=a, day=d, eligible_episodes=1, finite_episodes=1, finite_fraction=.9,
            finite_unresolved_n=0, missing_reference_n=0, **{m: float(a+d) for m in confirm.METRICS})
            for a in (1, 2) for d in (10, 11, 12)]
        expected = confirm.prior.crossed(cells, [1, 2], [10, 11, 12])
        actual = confirm.crossed(cells, [1, 2], [10, 11, 12])
        self.assertEqual(actual['estimates']['coverage'], expected['estimates']['coverage'])
        self.assertEqual(actual['estimates']['secondary_coverage'], actual['estimates']['coverage'])
        with self.assertRaises(AssertionError):
            confirm.crossed(cells[:-1], [1, 2], [10, 11, 12])

    def test_migration_has_shared_actor_axis_and_independent_day_axes(self):
        def rows(days, shift):
            return [dict(actor=a, day=d, finite_fraction=shift+a,
                **{m: shift+a for m in confirm.METRICS}) for a in (1, 2) for d in days]
        result = confirm.migration(rows([10, 11], 0.), rows([20, 21, 22], 3.), [1, 2], [10, 11], [20, 21, 22])
        np.testing.assert_allclose(result['coverage']['ci95'], [3., 3.])
        self.assertEqual(result['coverage']['delta'], 3.)

    def test_calibration_moving_reproduces_saved_abstention_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q = np.array([[10, 0, 17, 100., 46800., 1, 2, 1], [10, 0, 17, 100., 46801., 2, 2, 1],
                [10, 1, 17, 100., 46802., 3, 2, 1], [11, 2, 17, 100., 46800., 4, 2, 1]])
            finite = np.array([True, True, False, True])
            bounds = np.tile(np.array([[90., 110.], [0., 200.]]), (4, 1, 1))
            bounds[0, 0] = [100., 100.]
            bounds[2, 0] = np.nan
            residuals = np.zeros((4, 3))
            residuals[0, 0] = 1e-12
            np.savez_compressed(root/'atoms.npz', requests=q, atom_request=np.arange(4),
                candidate_key=np.array([0, 0, 1, 0]), finite_candidate=finite, finite_reference=np.ones(4, dtype=bool),
                finite_frozen_v167=np.ones(4, dtype=bool), bounds=bounds,
                frozen_v167_bounds=np.tile([80., 120.], (4, 1)), residual_rsd=residuals)
            (root/'calibration/residuals').mkdir(parents=True)
            for day in (10, 11):
                with gzip.open(root/'calibration/residuals'/f'1_{day}_old.jsonl.gz', 'wt') as handle:
                    for row in q[q[:, 0] == day]:
                        handle.write(json.dumps(dict(query_key=row[5:].astype(int).tolist(), query_time_s=row[4],
                            label=dict(status='resolved', full_eta_s=100.)))+'\n')
            confirm.write(root/'map.json', dict(candidate=[dict(key=['moving-launch'])]*2,
                cells=[dict(day=d, scope='hour13_moving_launch', eligible_queries=n, finite_queries=f,
                    finite_fraction=ratio) for d, n, f, ratio in ((10, 3, 2, .5), (11, 1, 1, 1.))]))
            model = dict(calibration_days=[10, 11], capture_root=str(root),
                frozen_protocol=dict(conditioning=dict(candidate_key=['native_stage_kind'])),
                chunks=[dict(actor=1, mode='old', map_file='map.json', atom_file='atoms.npz')])
            rows = confirm.calibration_moving(SimpleNamespace(v172_root=root), model)
            self.assertEqual([r['finite_fraction'] for r in rows], [.5, 1.])
            self.assertEqual([r['coverage'] for r in rows], [1., 1.])
            self.assertEqual([r['score_delta_s'] for r in rows], [-190., -180.])
            self.assertEqual([r['secondary_score_delta_s'] for r in rows], [-30., -20.])

    def test_predictions_precede_labels_and_no_update_entry(self):
        tree = ast.parse(Path(confirm.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'fit_key', 'build_bucket', 'train', 'registration'}))
        lines = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id: n.lineno
            for n in ast.walk(functions['collect_path']) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertLess(lines['save_queries'], lines['save_predictions'])
        self.assertLess(lines['save_predictions'], lines['attach_labels'])
        self.assertEqual(len(confirm.cases(dict(reservation=dict(actors=list(range(12)),
            confirmation_days=list(confirm.plan.CONFIRMATION))))), 384)


if __name__ == '__main__':
    unittest.main()
