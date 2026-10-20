from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import freqduet_historical_service_eta as model


def prior(days):
    return [dict(direction=True, station=1, clearance=True, days=days)]


class HistoricalServiceTests(unittest.TestCase):
    def test_days_and_modes_not_raw_event_counts_set_weight(self):
        cells = prior([dict(day=1, modes=[[10.]*100, [20.]]), dict(day=2, modes=[[30.]])])
        law = model.ServiceLaw(cells, {}, {}, 100.)
        self.assertAlmostEqual(law.estimate(True, 1, 150., 140.)['mean'], 22.5)

    def test_condition_mixture_before_normalizing_survivors(self):
        cells = prior([dict(day=1, modes=[[10., 30.]]), dict(day=2, modes=[[40.]])])
        law = model.ServiceLaw(cells, {}, {}, 100.)
        result = law.estimate(True, 1, 150., 140., 20.)
        self.assertAlmostEqual(result['mean'], (10.*.5+20.)/1.5)
        self.assertEqual(result['prior_survivor_days'], 2)

    def test_current_day_is_one_extra_empirical_day(self):
        cells = prior([dict(day=1, modes=[[10.]]), dict(day=2, modes=[[30.]])])
        k = (1, 1, True)
        law = model.ServiceLaw(cells, {k: dict(time_s=110.)}, {k: dict(ready_time_s=130.)}, 100.)
        result = law.estimate(True, 1, 150., 140.)
        self.assertEqual(result['mean'], 20.)
        self.assertEqual(result['local_n'], 1)

    def test_equal_time_and_future_labels_are_excluded(self):
        a = {(t, 1, True): dict(time_s=110.) for t in (1, 2)}
        r = {(1, 1, True): dict(ready_time_s=150.), (2, 1, True): dict(ready_time_s=200.)}
        law = model.ServiceLaw([], a, r, 100.)
        self.assertIsNone(law.estimate(True, 1, 150., 140.))

    def test_unsupported_tail_abstains_instead_of_extrapolating(self):
        law = model.ServiceLaw(prior([dict(day=1, modes=[[10., 20.]])]), {}, {}, 100.)
        self.assertIsNone(law.estimate(True, 1, 150., 140., 21.))
        self.assertEqual(law.estimate(True, 1, 150., 140., 20.)['mean'], 0.)

    def test_phase_is_arrival_phase_not_query_phase(self):
        k = (1, 1, True)
        law = model.ServiceLaw([], {k: dict(time_s=90.)}, {k: dict(ready_time_s=110.)}, 100.)
        self.assertIsNone(law.estimate(True, 1, 120., 120.))
        self.assertEqual(law.estimate(True, 1, 120., 90.)['mean'], 20.)

    def test_current_window_is_unchanged_with_original_all_past_fallback(self):
        a = {(1, 1, True): dict(time_s=110.)}
        r = {(1, 1, True): dict(ready_time_s=130.)}
        law = model.ServiceLaw([], a, r, 100.)
        result = law.estimate(True, 1, 5000., 5000.)
        self.assertTrue(result['historical_fallback'])
        self.assertEqual(result['latest_current_s'], 130.)

    def test_same_clock_ready_requires_prior_callback(self):
        a, b = (1, 1, True), (2, 2, True)
        view = model.ObservedReady({a: dict(ready_time_s=100.), b: dict(ready_time_s=100.)})
        self.assertTrue(view.same_clock_before(a, b, 100.))
        self.assertFalse(view.same_clock_before(b, a, 100.))
        self.assertFalse(view.same_clock_before(a, b, 99.))

    def test_observed_launch_allows_nonzero_current_progress(self):
        old = model.frozen.state
        model.frozen.state = lambda *a: (None, 'no_strict_physical_segment')
        self.addCleanup(setattr, model.frozen, 'state', old)
        route = dict(ids=[0, 1], index={0: 0, 1: 1}, positions=[0., 100.])
        row = dict(trip_id=1, station_id=1, direction=True, prediction_time_s=100.,
            evidence=dict(predicted_follower_trip_id=2, follower_gap_prediction_s=10., spatial_gap_m=90.))
        path = dict(dispatch={2: dict(actual_s=100.)})
        index = {2: SimpleNamespace(motion=SimpleNamespace(at=[], dt=[]))}
        p, status, _ = model.position(path, row, index, {True: route}, (None, None, None, index), events=True)
        self.assertEqual((p['origin'], p['progress'], status), (0, 10., 'observed_same_clock_launch'))
        p, _, _ = model.position(path, row, index, {True: route}, (None, None, None, index), events=False)
        self.assertIsNone(p)

    def pending_case(self, prior_ready_callback=True):
        old = model.frozen.state
        model.frozen.state = lambda *a: (None, 'no_completed_service_survivor')
        self.addCleanup(setattr, model.frozen, 'state', old)
        current, own = (4, 1, True), (2, 2, True)
        route = dict(ids=[0, 1, 2], index={0: 0, 1: 1, 2: 2}, positions=[0., 100., 200.])
        ready = {current: dict(ready_time_s=100., bus_id=7), own: dict(ready_time_s=100., bus_id=5)}
        if not prior_ready_callback:
            ready = dict(reversed(list(ready.items())))
        path = dict(ready=ready, _observed_ready=model.ObservedReady(ready),
            trace={current: dict(policy_time_s=100., chosen_s=9999.)})
        row = dict(trip_id=2, station_id=2, direction=True, prediction_time_s=100.,
            evidence=dict(predicted_follower_trip_id=4, predicted_follower_bus_id=7,
                follower_gap_prediction_s=10., spatial_gap_m=100.))
        arrival = dict(trip_id=4, station_id=1, direction=True, time_s=90.)
        index = {4: SimpleNamespace(motion=SimpleNamespace(at=[90.], dt=[0.], arrivals=[arrival]))}
        history = SimpleNamespace(estimate=lambda *a: dict(mean=2., latest=80.))
        durations = SimpleNamespace(estimate=lambda *a: dict(mean=8.))
        law = SimpleNamespace(estimate=lambda *a: dict(mean=5.))
        return path, row, index, {True: route}, (history, None, durations, index), law

    def test_observed_ready_drops_service_but_keeps_unissued_residence(self):
        args = self.pending_case(True)
        p, status, _ = model.position(*args, events=True)
        self.assertEqual((status, p['initial_s']), ('observed_same_clock_ready', 8.))

    def test_later_same_clock_ready_and_future_command_do_not_become_inputs(self):
        args = self.pending_case(False)
        p, status, _ = model.position(*args, events=True)
        self.assertEqual((status, p['initial_s']), ('historical_current_service', 9.))
        args[0]['trace'][(4, 1, True)]['chosen_s'] = 0.
        again, _, _ = model.position(*args, events=True)
        self.assertEqual(again, p)

    def test_observed_ready_rejects_different_physical_follower(self):
        args = self.pending_case(True)
        args[0]['ready'][(4, 1, True)]['bus_id'] = 8
        with self.assertRaises(AssertionError):
            model.position(*args, events=True)


if __name__ == '__main__':
    unittest.main()
