import ast
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import preregister_freqduet_v171_temporal_tails as stage


def model():
    cal, confirm = list(range(101, 133)), list(range(201, 233))
    parent = dict(status='v160_metadata_and_preregistration_complete', execution_source='14d930ee25',
        new_days_collected=0, interval_fit=False, registration=dict(status='preregistered_not_collected',
            collector_submitted=False, calibration_days=cal, confirmation_days=confirm, actors=list(range(12)),
            modes=['old', 'handoff'], checkpoint_ep=199,
            sampler_seeds={str(d): 1600000000+i for i, d in enumerate(cal+confirm)},
            service_point_source='1b783ea8f3', running_shadow_source='5be4cea534', v146_extensions_s=[3., 0.],
            window_s=3600., tails=[.1, .1], freshness=dict(overlap_days=[], overlap_samplers=[])))
    plan = stage.original.protocol.registration(parent)
    days = list(range(2951001, 2951065))
    plan['reservation'].update(calibration_days=days, confirmation_days=list(range(2852001, 2852033)),
        sampler_seeds={str(d): 1650000000+i for i, d in enumerate(days)})
    return dict(status='sealed', execution_source='e6fc851ec1', calibration_days=days,
        actors=list(range(12)), frozen_protocol=plan)


class TemporalTailTests(unittest.TestCase):
    def test_fixed_clock_boundaries(self):
        np.testing.assert_array_equal(stage.strata(np.array([7199., 7200., 10799., 10800., 46799., 46800., 50400.])),
            [0, 1, 1, 0, 0, 2, 0])

    def test_distinct_days_not_queries_and_no_cross_stratum_support(self):
        days = np.r_[np.arange(1, 20), np.arange(1, 19), np.ones(40, dtype=int)]
        periods = np.r_[np.ones(19, dtype=int), np.zeros(18, dtype=int), np.full(40, 2)]
        finite, keys = stage.supported(np.zeros(len(days), dtype=int), days, periods, np.arange(len(days)), len(days)+1)
        self.assertTrue(finite[:19].all())
        self.assertFalse(finite[19:].any())
        self.assertEqual({r['stratum']: r['days'] for r in keys}, dict(other=18, hour2=19, hour13=1))

    def test_episode_support_includes_native_abstentions(self):
        r = stage.support_cell(10, 'hour2', np.ones(3, dtype=bool), np.array([10]*3), np.array([1, 1, 2]),
            np.array([True, True, False]), np.array([True, False, False]), np.array([True, True, False]))
        self.assertEqual(r['finite_fraction'], .25)
        self.assertEqual(r['baseline_fraction'], .5)
        self.assertEqual((r['native_abstentions'], r['key_abstentions']), (1, 1))

    def test_unsupported_moving_query_remains_in_scope(self):
        q = np.array([[10, 1, 17, 100, 46800., 1, 2, 1], [10, 2, 17, 100, 46801., 3, 4, 1],
            [10, 3, 17, 100, 46802., 5, 6, 0]])
        records = [dict(actor=1, day=10, mode='old', query_key=q[i, 5:].astype(int).tolist(),
            query_time_s=q[i, 4], denominator_eligible=True, native_supported=False,
            native_stage=dict(kind=kind)) for i, kind in ((1, 'moving-launch'), (2, 'service'))]
        args = type('Args', (), dict(capture_root=Path('/unused')))()
        with patch.object(stage.original, 'stream', return_value=iter(records)):
            mask, counts = stage.moving_mask(args, 1, 'old', q, np.array([True, False, False]), np.array([0]),
                np.array([0]), [dict(key=['moving-launch'])], 0)
        np.testing.assert_array_equal(mask, [True, True, False])
        self.assertEqual(counts['unsupported_hour13_moving'], 1)

    def test_registration_preserves_freezes_and_reserves_only_fresh_confirmation(self):
        source = model()
        before = deepcopy(source)
        with patch.dict('os.environ', FREQDUET_DIAGNOSTIC_COMMIT='test-source'):
            plan = stage.registration(source, dict(collisions=[]))
        self.assertEqual(source, before)
        for field in ('frozen', 'target', 'support'):
            self.assertEqual(plan[field], before['frozen_protocol'][field])
        self.assertEqual(plan['conditioning']['candidate_key'][-1], 'query_time_stratum')
        self.assertEqual(plan['inference']['scopes'], list(stage.SCOPES))
        self.assertEqual(len(plan['reservation']['confirmation_days']), 32)
        self.assertFalse(set(plan['reservation']['confirmation_days']) & set(range(2852001, 2852033)))
        self.assertFalse(set(plan['reservation']['confirmation_days']) & set(source['calibration_days']))
        self.assertFalse(any(plan['execution'].values()))
        self.assertEqual(len(set(stage.SAMPLERS.values())), 32)

    def test_collision_blocks_protocol(self):
        with self.assertRaisesRegex(AssertionError, 'not fresh'):
            stage.registration(model(), dict(collisions=[3152001]))

    def test_all_old_and_new_scope_gates_are_required(self):
        rows = [dict(mode=m, scope=s, complete_actor_day_grid=True, finite_unresolved_n=0, missing_reference_n=0,
            coverage_lower95=.82, below_upper95=.09, above_upper95=.09, finite_fraction_lower95=.85,
            score_delta_upper95=-1.) for m in ('old', 'handoff') for s in stage.SCOPES]
        self.assertEqual(stage.gate(rows)['decision'], 'pass')
        self.assertFalse(stage.gate(rows)['controller_admitted'])
        rows[-1]['above_upper95'] = .101
        self.assertIn('handoff/hour13_moving_launch/above', stage.gate(rows)['failures'])
        rows[0]['coverage_lower95'] = .799
        self.assertIn('old/all/coverage', stage.gate(rows)['failures'])
        self.assertEqual(stage.gate(rows[:-1])['decision'], 'inconclusive')

    def test_no_interval_fit_replay_or_confirmation_data_entry(self):
        tree = ast.parse(Path(stage.__file__).read_text())
        calls = {(n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id)
            for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
        self.assertTrue(calls.isdisjoint({'fit_key', 'inverse_day_mass', 'quantile', 'percentile',
            'rollout', 'collect', 'train', 'attach_labels'}))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr in ('v168_root', 'v169_root', 'v170_root')
            for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
