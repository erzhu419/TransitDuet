import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v177_confirmation as stage


class EnvelopeConfirmationTests(unittest.TestCase):
    def inputs(self, phase='clearance', time=46800.):
        fields = ['actor', 'mode', 'query_phase', 'native_stage_kind', 'query_time_stratum']
        q = dict(actor=1, mode='old', query_key=[1, 2, True], query_time_s=time, episode=[2, 4, 2, True],
            candidate_key=[1, 'old', phase, 'moving-launch'], point_s=100., denominator_eligible=True,
            native_supported=True, native_stage=dict(reason='not_applicable', kind='moving-launch'),
            condition_values=dict(actor=1, mode='old', query_phase=phase), observable=dict(hour=int(time//3600)))
        period = stage.saved.plan.STRATA[int(stage.saved.plan.strata(np.array([time]))[0])]
        key = (*q['candidate_key'], period)
        entry = dict(status='finite', day_count=64, qlow_s=-10., qhigh_s=20.)
        temporal = {key: entry}
        candidate = {key: dict(entry, qlow_s=-30.)}
        coarse = {(1, 'old', phase): dict(entry, qlow_s=-100., qhigh_s=100.)}
        old = {tuple(q['candidate_key']): dict(entry, qlow_s=-20.)}
        protocol = dict(comparator=dict(key=['actor', 'mode', 'query_phase']), conditioning=dict(candidate_key=fields))
        return q, candidate, coarse, old, temporal, protocol

    def test_frozen_lookup_and_three_references_do_not_mutate_tables(self):
        inputs = self.inputs()
        before = deepcopy(inputs)
        p = stage.prediction(*inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(p['bounds'], [80., 130.])
        self.assertEqual(p['temporal_bounds'], [80., 110.])
        self.assertEqual(p['secondary_bounds'], [80., 120.])
        self.assertEqual(p['reference_bounds'], [0., 200.])
        self.assertEqual(p['map_source'], '36108d4f49')

    def test_envelope_geometry_and_non_target_changes_are_blocked(self):
        q, candidate, coarse, old, temporal, protocol = self.inputs()
        for entry in (dict(status='finite', day_count=64, qlow_s=-5., qhigh_s=20.),
                dict(status='finite', day_count=64, qlow_s=-30., qhigh_s=21.)):
            with self.assertRaises(AssertionError):
                stage.prediction(q, {next(iter(candidate)): entry}, coarse, old, temporal, protocol)
        with self.assertRaises(AssertionError):
            stage.prediction(*self.inputs('active', 7200.))

    def test_low_unseen_native_abstentions_do_not_use_reference_fallback(self):
        q, candidate, coarse, old, temporal, protocol = self.inputs()
        key = next(iter(candidate))
        for table in ({}, {key: dict(status='abstain_less_than_19_days', day_count=18)}):
            p = stage.prediction(q, table, coarse, old, table, protocol)
            self.assertFalse(p['finite'])
            self.assertTrue(p['reference_finite'])
        q.update(native_supported=False, candidate_key=None)
        self.assertFalse(stage.prediction(q, candidate, coarse, old, temporal, protocol)['finite'])

    def test_same_mask_scores_width_and_new_repaired_misses(self):
        p = stage.prediction(*self.inputs())
        for truth in (100., 120., 130., 140.):
            result = dict(zip(stage.METRICS, stage.values(p, dict(status='resolved', full_eta_s=truth))))
            self.assertEqual(result['temporal_width_delta_s'], 20.)
            self.assertEqual(result['v172_new_above'], 0.)
            self.assertEqual(result['v172_lower_penalty_delta_s'], 0.)
            for c, delta, width in (('coarse', 'score_delta_s', 'width_delta_s'),
                    ('v167', 'secondary_score_delta_s', 'secondary_width_delta_s'),
                    ('v172', 'temporal_score_delta_s', 'temporal_width_delta_s')):
                self.assertAlmostEqual(result[delta], result[width]+result[c+'_lower_penalty_delta_s']+
                    result[c+'_upper_penalty_delta_s'])
        result = dict(zip(stage.METRICS, stage.values(p, dict(status='resolved', full_eta_s=120.))))
        self.assertEqual(result['temporal_score_delta_s'], -80.)
        self.assertEqual(result['v172_repaired_above'], 1.)

    def test_scope_and_episode_denominators_keep_unsupported_queries(self):
        q, *tables = self.inputs()
        p = stage.prediction(q, *tables)
        unsupported = deepcopy(q)
        unsupported.update(native_supported=False, candidate_key=None, episode=[3, 5, 2, True])
        rows = [(q, p, dict(status='resolved', full_eta_s=120.)) for _ in range(9)]
        rows.append((unsupported, stage.prediction(unsupported, *tables), dict(status='unresolved')))
        cells, counts = stage.reduce_rows(rows)
        self.assertEqual(cells['hour13_moving_launch']['finite_fraction'], .5)
        self.assertEqual(cells['hour13_moving_launch']['coverage'], 1.)
        self.assertEqual(cells['hour13_moving_launch']['finite_unresolved_n'], 0)
        self.assertEqual(counts['requests'], 10)

    def test_finite_missing_labels_and_references_are_inconclusive(self):
        q, *tables = self.inputs()
        p = stage.prediction(q, *tables)
        cells, _ = stage.reduce_rows([(q, p, dict(status='unresolved'))])
        self.assertIsNone(cells['clearance']['coverage'])
        self.assertEqual(cells['clearance']['finite_unresolved_n'], 1)
        for field in ('reference_finite', 'secondary_finite', 'temporal_finite'):
            missing = dict(p, **{field: False})
            cells, _ = stage.reduce_rows([(q, missing, dict(status='resolved', full_eta_s=120.))])
            self.assertEqual(cells['clearance']['missing_reference_n'], 1)
            self.assertIsNone(cells['clearance']['temporal_score_delta_s'])

    def test_crossed_bootstrap_matches_existing_shared_day_draws(self):
        rows = [dict(actor=a, day=d, eligible_episodes=1, finite_episodes=1, finite_fraction=.9,
            finite_unresolved_n=0, missing_reference_n=0, **{m: float(a+d) for m in stage.METRICS})
            for a in (1, 2) for d in (10, 11, 12)]
        result = stage.crossed(rows, [2, 1], [12, 10, 11])
        frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], value=r['coverage']) for r in rows])
        draws = stage.saved.prior.hierarchical_bootstrap(frame, 'value', draws=5000)
        np.testing.assert_array_equal(result['estimates']['coverage']['ci95'], np.percentile(draws, [2.5, 97.5]))
        for bad in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(AssertionError):
                stage.crossed(bad, [1, 2], [10, 11, 12])

    def test_freshness_blocks_existing_reserved_artifact_without_reading_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root/'capture/residuals'/f'1_{stage.plan.CONFIRMATION[0]}_old.jsonl.gz'
            artifact.parent.mkdir(parents=True)
            artifact.touch()
            args = SimpleNamespace(parent=[root], v176_root=root/'base/seal', out_root=root/'new/confirm')
            with patch.object(stage.plan.ledger.archives, 'result_roots', return_value=[root]):
                with self.assertRaisesRegex(AssertionError, 'already consumed'):
                    stage.unconsumed(args)

    def test_full_reserved_budget_and_predictions_precede_labels_no_fit_entry(self):
        self.assertEqual(len(stage.cases(dict(reservation=dict(actors=list(range(12)),
            confirmation_days=list(stage.plan.CONFIRMATION))))), 768)
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'fit_key', 'envelope', 'block_order_statistic', 'train'}))
        collect = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect_path')
        lines = {n.func.attr: n.lineno for n in ast.walk(collect) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertLess(lines['save_queries'], lines['open'])
        self.assertLess(lines['open'], lines['attach_labels'])


if __name__ == '__main__':
    unittest.main()
