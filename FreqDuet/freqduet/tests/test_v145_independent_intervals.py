from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from freqduet_interval_validation import describe, crossed
from freqduet_rank_interval import rank_interval
from run_freqduet_v145_independent_intervals import ACTORS, DAYS, SAMPLERS, ITEMS, stage_request, duration_label


class IndependentIntervalTests(unittest.TestCase):
    def row(self, covered=True, episode=1, supported=True, resolved=True):
        return dict(rank=rank_interval(np.arange(9 if supported else 8, dtype=float)),
            label_status='resolved' if resolved else 'missing_departure_tail',
            episode=[episode], support_reason='finite' if supported else 'prefix_less_than_nine',
            covered=covered if supported and resolved else None,
            below=False if supported and resolved else None, above=not covered if supported and resolved else None)

    def test_registered_grid_and_shared_day_sampler(self):
        self.assertEqual((len(ACTORS), len(DAYS), len(ITEMS)), (12, 32, 768))
        self.assertEqual(len(set(SAMPLERS.values())), 32)
        self.assertEqual(len(set(ITEMS)), 768)

    def test_unsupported_and_unresolved_requests_remain_in_denominator(self):
        s = describe([self.row(), self.row(supported=False), self.row(resolved=False)])
        self.assertEqual((s['requests'], s['finite_n'], s['finite_resolved_n']), (3, 2, 1))
        self.assertEqual(s['query_coverage'], 1.)
        self.assertEqual(s['finite_unresolved_n'], 1)
        self.assertIsNone(describe([self.row(supported=False)])['episode_coverage'])

    def test_episode_average_precedes_actor_day_average(self):
        s = describe([self.row(True, 1) for _ in range(9)]+[self.row(False, 2)])
        self.assertEqual(s['query_coverage'], .9)
        self.assertEqual(s['episode_coverage'], .5)

    def test_incomplete_support_is_not_zero_filled_for_bootstrap(self):
        rows = [dict(seed=a, day=d, stats=describe([self.row(supported=(a, d) != (2, 20))]))
            for a in (1, 2) for d in (10, 20)]
        s = crossed(rows, (1, 2), (10, 20))
        self.assertEqual(s['contributing_cells'], 3)
        self.assertIsNone(s['coverage_ci95'])
        self.assertFalse(s['coverage_confirmed'])

    def test_unresolved_finite_labels_cannot_pass_coverage(self):
        rows = [dict(seed=a, day=d, stats=describe([self.row(), self.row(resolved=False)]))
            for a in (1, 2) for d in (10, 20)]
        self.assertEqual(crossed(rows, (1, 2), (10, 20))['decision'], 'inconclusive_unresolved_labels')

    def test_stage_eligibility_uses_only_observed_prefix(self):
        k = (2, 1, False)
        path = dict(arrivals={k: dict(time_s=80.)}, ready={k: dict(ready_time_s=150.)}, trace={})
        forecast, row = dict(current_stop_key=list(k)), dict(station_id=2, evidence=dict(ready_time_s=100.))
        geometry = {False: dict(index={1: 0, 2: 1})}
        self.assertEqual(stage_request(path, forecast, row, geometry), ('service', k, 80.))
        path['ready'][k]['ready_time_s'] = 999.
        self.assertEqual(stage_request(path, forecast, row, geometry), ('service', k, 80.))
        path['ready'][k]['ready_time_s'] = 90.
        path['trace'][k] = dict(policy_time_s=100.)
        self.assertEqual(stage_request(path, forecast, row, geometry), ('ready', k, 90.))
        path['trace'][k]['policy_time_s'] = 99.
        self.assertIsNone(stage_request(path, forecast, row, geometry))

    def test_wrong_vehicle_outcome_is_missing_not_zero(self):
        k = (2, 1, False)
        status, truth = duration_label(dict(ready={k: dict(bus_id=3, ready_time_s=110.)}), k,
            dict(evidence=dict(predicted_follower_bus_id=4)), 'service', 100.)
        self.assertEqual(status, 'same_vehicle_identity_mismatch')
        self.assertIsNone(truth)


if __name__ == '__main__':
    unittest.main()
