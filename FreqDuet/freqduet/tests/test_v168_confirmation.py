import ast
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_freqduet_v168_confirmation as confirmation


class ConfirmationTests(unittest.TestCase):
    def inputs(self):
        q = dict(actor=1, mode='old', query_key=[1, 2, True], query_time_s=7200.,
            episode=[2, 4, 2, True], candidate_key=[1, 'old', 'active'], point_s=100.,
            denominator_eligible=True, native_supported=True, native_stage=dict(reason='not_applicable'),
            condition_values=dict(actor=1, mode='old', query_phase='active'), observable=dict(hour=2))
        candidate = {(1, 'old', 'active'): dict(status='finite', day_count=64, qlow_s=-10., qhigh_s=10.)}
        reference = {(1, 'old', 'active'): dict(status='finite', day_count=64, qlow_s=-100., qhigh_s=100.)}
        sealed = dict(comparator=dict(key=['actor', 'mode', 'query_phase']))
        return q, candidate, reference, sealed

    def test_sealed_lookup_does_not_update_and_never_pools_unseen_or_low_keys(self):
        args = self.inputs()
        before = deepcopy(args)
        predicted = confirmation.prediction(*args)
        self.assertEqual(predicted['bounds'], [90., 110.])
        self.assertEqual(args, before)
        q, candidate, reference, sealed = args
        for table in ({}, {(1, 'old', 'active'): dict(status='abstain_less_than_19_days', day_count=18)}):
            value = confirmation.prediction(q, table, reference, sealed)
            self.assertFalse(value['finite'])
            self.assertIsNone(value['bounds'])
            self.assertTrue(value['reference_finite'])
        q['native_supported'] = False
        value = confirmation.prediction(q, candidate, reference, sealed)
        self.assertFalse(value['finite'])
        self.assertFalse(value['reference_finite'])

    def test_zero_physical_bounds_are_finite_not_missing(self):
        q, candidate, reference, sealed = self.inputs()
        q['point_s'] = 0.
        candidate[(1, 'old', 'active')].update(qlow_s=5., qhigh_s=10.)
        value = confirmation.prediction(q, candidate, reference, sealed)
        self.assertTrue(value['finite'])
        self.assertEqual(value['bounds'], [0., 0.])

    def test_nominal_score_sign_and_inclusive_coverage(self):
        pred = confirmation.prediction(*self.inputs())
        for truth, expected in ((100., (1., 0., 0., 20., 20., -180.)),
                (80., (0., 1., 0., 20., 120., -80.)), (200., (0., 0., 1., 20., 920., 720.))):
            value = confirmation.values(pred, dict(status='resolved', full_eta_s=truth))
            np.testing.assert_allclose(value[[0, 1, 2, 3, 4, 10]], expected)
        self.assertEqual(confirmation.values(pred, dict(status='resolved', full_eta_s=90.))[0], 1.)
        self.assertIsNone(confirmation.values(pred, dict(status='unresolved', full_eta_s=None)))

    def test_finite_fraction_is_episode_equal_and_scores_are_candidate_matched(self):
        q, candidate, reference, sealed = self.inputs()
        pred = confirmation.prediction(q, candidate, reference, sealed)
        rows = [(deepcopy(q), deepcopy(pred), dict(status='resolved', full_eta_s=100.)) for _ in range(9)]
        other = deepcopy(q)
        other['episode'] = [3, 5, 2, True]
        other['candidate_key'] = [1, 'old', 'unseen']
        rows.append((other, confirmation.prediction(other, candidate, reference, sealed),
            dict(status='resolved', full_eta_s=1000.)))
        scopes, keys, counts = confirmation.reduce_rows(iter(rows))
        self.assertEqual(scopes['all']['finite_fraction'], .5)
        self.assertEqual(scopes['all']['coverage'], 1.)
        self.assertEqual(scopes['all']['score_delta_s'], -180.)
        self.assertEqual(scopes['all']['eligible_queries'], 10)
        self.assertEqual(scopes['all']['finite_queries'], 9)
        self.assertIsNone(scopes['clearance']['coverage'])
        self.assertEqual(counts['support/unseen_key'], 1)

    def test_unresolved_and_missing_reference_are_not_zero_filled_or_dropped(self):
        q, *rest = self.inputs()
        pred = confirmation.prediction(q, *rest)
        scopes, _, _ = confirmation.reduce_rows([(q, pred, dict(status='unresolved', full_eta_s=None))])
        self.assertEqual(scopes['all']['finite_unresolved_n'], 1)
        self.assertEqual(scopes['all']['finite_fraction'], 1.)
        self.assertIsNone(scopes['all']['coverage'])
        pred.update(reference_finite=False, reference_bounds=None)
        scopes, _, _ = confirmation.reduce_rows([(q, pred, dict(status='resolved', full_eta_s=100.))])
        self.assertEqual(scopes['all']['missing_reference_n'], 1)
        self.assertIsNone(scopes['all']['score_delta_s'])

    def test_crossed_bounds_reproduce_existing_bootstrap_and_shared_axes(self):
        cells = []
        for actor in (1, 2):
            for day in (10, 11, 12):
                value = actor+day/10.
                cells.append(dict(actor=actor, day=day, eligible_episodes=1, finite_episodes=1,
                    finite_unresolved_n=0, missing_reference_n=0, finite_fraction=.9,
                    **{m: 2.*value if m == 'reference_score_s' else value for m in confirmation.METRICS}))
        result = confirmation.crossed(cells, [1, 2], [10, 11, 12])
        frame = pd.DataFrame([dict(train_seed=r['actor'], eval_seed=r['day'], coverage=r['coverage']) for r in cells])
        draws = confirmation.hierarchical_bootstrap(frame, 'coverage', draws=5000)
        np.testing.assert_array_equal(result['estimates']['coverage']['ci95'], np.percentile(draws, [2.5, 97.5]))
        self.assertEqual(result['coverage_lower95'], float(np.percentile(draws, 5.)))
        np.testing.assert_allclose(result['estimates']['reference_score_s']['ci95'], 2.*np.array(result['estimates']['score_s']['ci95']))
        with self.assertRaisesRegex(AssertionError, 'grid differs'):
            confirmation.crossed(cells[:-1], [1, 2], [10, 11, 12])
        with self.assertRaisesRegex(AssertionError, 'grid differs'):
            confirmation.crossed(cells[:-1]+cells[:1], [1, 2], [10, 11, 12])

    def test_reserved_days_only_and_source_isolation(self):
        r = dict(actors=[1, 2], confirmation_days=list(range(2852001, 2852033)),
            calibration_days=list(range(2951001, 2951065)), excluded_development_days=list(range(2851001, 2851033)),
            sampler_seeds={str(d): 1600000032+i for i, d in enumerate(range(2852001, 2852033))})
        plan = dict(reservation=r)
        model = dict(status='sealed', execution_source=confirmation.MAP_SOURCE, frozen_protocol=deepcopy(plan),
            calibration_days=r['calibration_days'], confirmation_release_eligible=True,
            support_admission=dict(decision='pass'), confirmation_started=False, confirmation_outcomes_read=False)
        args = type('Args', (), {'v167_root': Path('/unused')})()
        with patch.object(confirmation.entry, 'parents', return_value=(plan, {})), patch.object(confirmation, 'read', return_value=model):
            confirmation.parents(args)
            self.assertEqual(len(confirmation.cases(plan)), 64)
            self.assertFalse(set(d for _, d in confirmation.cases(plan)) & set(r['calibration_days']))
            model['confirmation_release_eligible'] = False
            with self.assertRaisesRegex(AssertionError, 'support release'):
                confirmation.parents(args)

    def test_prediction_artifact_precedes_labels_without_fit_or_training_calls(self):
        tree = ast.parse(Path(confirmation.__file__).read_text())
        functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertTrue(called.isdisjoint({'fit_key', 'build_bucket', 'train'}))
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id): n.lineno
            for n in ast.walk(functions['collect_path']) if isinstance(n, ast.Call)
            and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertLess(calls['save_queries'], calls['save_predictions'])
        self.assertLess(calls['save_predictions'], calls['attach_labels'])
        for name in ('prediction', 'save_predictions'):
            names = {n.attr for n in ast.walk(functions[name]) if isinstance(n, ast.Attribute)}
            self.assertTrue(names.isdisjoint({'attach_labels', 'label_record', 'actual_parts'}))


if __name__ == '__main__':
    unittest.main()
