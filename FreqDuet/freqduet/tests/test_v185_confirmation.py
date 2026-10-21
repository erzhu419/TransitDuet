import ast
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v185_confirmation as stage


class ConfirmationTests(unittest.TestCase):
    def inputs(self, active=False):
        now = 7200. if active else 46800.
        phase = 'active' if active else 'clearance'
        q = dict(query_key=[1, 2, True], query_time_s=now, episode=[2, 4, 2, True],
            candidate_key=[1, 'old', phase, 'moving-launch'], point_s=100., denominator_eligible=True,
            native_supported=True, native_stage=dict(reason='not_applicable', kind='moving-launch'),
            condition_values=dict(query_phase=phase), observable=dict(hour=int(now//3600)))
        period = 'hour2' if active else 'hour13'
        baseline = {(*q['candidate_key'], period): dict(status='finite', day_count=64, qhigh_s=20., qlow_s=-10.)}
        params = [dict(sign='nonpositive', status='finite', day_count=44, delta_s=-2.),
            dict(sign='positive', status='finite', day_count=28, delta_s=10.)]
        history = dict(n=9, latest=now-1., mean=np.array([0., 1., 0.]))
        return q, params, baseline, history

    def test_lookup_uses_frozen_signed_delta_and_leaves_upper_unchanged(self):
        q, params, baseline, h = self.inputs()
        before = deepcopy((q, params, baseline))
        p = stage.prediction(q, params, baseline, h)
        self.assertEqual((q, params, baseline), before)
        self.assertEqual(p['bounds'], [70., 110.])
        self.assertEqual(p['reference_bounds'], [80., 110.])
        h['mean'][1] = 0.
        self.assertEqual(stage.prediction(q, params, baseline, h)['bounds'], [82., 110.])

    def test_non_target_and_low_support_have_no_fallback(self):
        q, params, baseline, h = self.inputs(True)
        h.update(n=0, latest=None, mean=None)
        p = stage.prediction(q, params, baseline, h)
        self.assertEqual(p['bounds'], p['reference_bounds'])
        q, params, baseline, h = self.inputs()
        h.update(n=8, mean=None)
        self.assertFalse(stage.prediction(q, params, baseline, h)['finite'])
        h.update(n=9, mean=np.array([0., 1., 0.]))
        params[1].update(status='abstain_less_than_19_days', day_count=18, delta_s=None)
        p = stage.prediction(q, params, baseline, h)
        self.assertFalse(p['finite'])
        self.assertTrue(p['reference_finite'])
        self.assertFalse(stage.prediction(q, params, {}, h)['finite'])
        q.update(native_supported=False, candidate_key=None)
        self.assertFalse(stage.prediction(q, params, baseline, h)['finite'])

    def test_prefix_deduplicates_physical_episode_and_never_evaluates_future_label(self):
        def q(now, coarse=True, trip=1):
            return dict(query_time_s=now, clocks=dict(query_ready_order=int(now)),
                denominator_eligible=True, prediction_coarse={} if coarse else None, episode=[trip, 10, 2, True])
        calls = []
        def residual(row, now):
            self.assertGreater(now, 100. if row['episode'][0] == 1 else 200.)
            calls.append(row['query_time_s'])
            return [0., 2., 0.]
        ready = {(1, 2, True): dict(bus_id=10, ready_time_s=100.),
            (2, 2, True): dict(bus_id=10, ready_time_s=200.)}
        prefix = stage.CompletedPrefix([q(0., False), q(2.), q(1.), q(3., trip=2)], ready, residual)
        self.assertEqual(prefix.query(100.)['n'], 0)
        self.assertEqual(calls, [])
        self.assertEqual(prefix.query(101.)['n'], 1)
        self.assertEqual(calls, [1.])
        self.assertEqual(prefix.query(200.)['n'], 1)
        self.assertEqual(prefix.query(201.)['n'], 2)
        self.assertEqual(prefix.query(3801.)['n'], 0)

    def test_original_episode_denominator_and_v172_own_mask_are_reported(self):
        q, *tables = self.inputs()
        p = stage.prediction(q, *tables)
        low = deepcopy(q)
        low['episode'] = [3, 5, 2, True]
        abstention = stage.prediction(low, *tables[:2], dict(n=0, latest=None, mean=None))
        rows = [(q, p, dict(status='resolved', full_eta_s=85.)) for _ in range(9)]
        rows.append((low, abstention, dict(status='resolved', full_eta_s=85.)))
        cell = stage.reduce_rows(rows)[0]['hour13_moving_launch']
        self.assertEqual(cell['finite_fraction'], .5)
        self.assertEqual(cell['v172_own_finite_fraction'], 1.)
        self.assertEqual(cell['coverage'], 1.)
        unresolved = stage.reduce_rows([(q, p, dict(status='unresolved'))])[0]['clearance']
        self.assertIsNone(unresolved['score_delta_s'])
        self.assertEqual(unresolved['finite_unresolved_n'], 1)

    def test_bootstrap_matches_frozen_shared_actor_day_sampling_and_rejects_missing_cells(self):
        metrics = ('finite_fraction', *stage.METRICS, 'v172_own_finite_fraction', *('v172_own_'+m for m in stage.CORE))
        rows = [dict(actor=a, day=d, eligible_episodes=1, finite_episodes=1,
            finite_unresolved_n=0, missing_reference_n=0, **{m: float(a+d) for m in metrics})
            for a in (1, 2) for d in (10, 11, 12)]
        actual = stage.crossed(rows, [2, 1], [12, 10, 11])
        frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], metric=r['coverage']) for r in rows])
        expected = stage.saved.saved.prior.hierarchical_bootstrap(frame, 'metric', draws=5000)
        np.testing.assert_array_equal(actual['estimates']['coverage']['ci95'], np.percentile(expected, [2.5, 97.5]))
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(AssertionError):
                stage.crossed(bad, [1, 2], [10, 11, 12])

    def test_fixed_cohort_and_no_fitting_or_label_pass_before_predictions(self):
        self.assertEqual(len(stage.cases(dict(actors=list(range(12)), confirmation_days=list(stage.DAYS)))), 384)
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'shared_fit', 'fit_key', 'envelope', 'train'}))
        collect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_path')
        attached = next(n.lineno for n in ast.walk(collect) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Attribute) and n.func.attr == 'attach_labels')
        opened = next(n.lineno for n in ast.walk(collect) if isinstance(n, ast.Call) and
            isinstance(n.func, ast.Attribute) and n.func.attr == 'open')
        self.assertLess(opened, attached)


if __name__ == '__main__':
    unittest.main()
