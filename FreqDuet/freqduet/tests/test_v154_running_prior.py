from types import SimpleNamespace
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import freqduet_phase_running_prior as candidate


class RunningPriorTests(unittest.TestCase):
    def law(self, cells, held=1):
        prefix = SimpleNamespace(times={(True, 2): [10., 30.]},
            rows={(True, 2): [(10., 0., 10.), (30., 0., 30.)]})
        return candidate.RunningLaw(cells, held, prefix)

    def cell(self, days):
        return dict(direction=True, station=2, hour=2, days=days)

    def test_whole_held_day_excluded_from_service_and_running(self):
        days = [dict(day=1, mean=9999., second=9999.**2), dict(day=2, mean=40., second=1600.)]
        cells = [self.cell(days)]
        law = self.law(cells)
        value = law.estimate(True, 2, 20., 7200., dict(mean=10.))
        self.assertEqual(value['mean'], 25.)
        self.assertEqual(value['prior_days'], 1)
        self.assertEqual(candidate.exclude_day(cells, 1)[0]['days'], [days[1]])
        self.assertEqual(len(cells[0]['days']), 2)

    def test_missing_hour_retains_native_exactly(self):
        value = self.law([]).estimate(True, 2, 20., 7200., dict(mean=10.))
        self.assertEqual(value['mean'], 10.)
        self.assertEqual(value['historical_weight'], 0.)

    def test_simultaneous_completion_is_not_in_anchor(self):
        value = self.law([]).estimate(True, 2, 30., 7200., dict(mean=10.))
        self.assertEqual(value['latest_current_s'], 10.)

    def test_hour_uses_predicted_entry_not_query_clock(self):
        cells = [self.cell([dict(day=2, mean=40., second=1600.)])]
        law = self.law(cells)
        self.assertEqual(law.estimate(True, 2, 20., 7199., dict(mean=10.))['mean'], 10.)
        self.assertEqual(law.estimate(True, 2, 20., 7200., dict(mean=10.))['mean'], 25.)

    def test_mixture_dispersion_includes_between_day_difference(self):
        cells = [self.cell([dict(day=2, mean=40., second=1600.)])]
        value = self.law(cells).estimate(True, 2, 20., 7200., dict(mean=10.))
        self.assertEqual(value['dispersion_s'], 15.)

    def test_partial_leg_retained_and_future_entry_clock_propagated(self):
        route = dict(ids=[1, 2, 3], positions=[0., 100., 200.])
        position = dict(route=route, direction=True, origin=0, target=2,
            progress=50., now=20., initial_s=0.)
        history = SimpleNamespace(estimate=lambda k, now: dict(latest=0.,
            mean={'running': 100., 'service': 10., 'holding': 0., 'ticks': 0.}[k[0]]))
        calls = []
        def running(direction, station, now, entry, native):
            calls.append((station, entry))
            return dict(mean=40., prior_days=1, dispersion_s=15.)
        with patch.object(candidate.frozen, 'position', return_value=(position, 'structured', None)):
            value, detail = candidate.forecast({}, {}, {}, (history, None, None, None), None,
                SimpleNamespace(estimate=lambda *a: None), SimpleNamespace(estimate=running))
        self.assertEqual(value, 110.)
        self.assertEqual(detail['running_s'], 90.)
        self.assertEqual(detail['partial_legs_retained'], 1)
        self.assertEqual(calls, [(3, 80.)])


if __name__ == '__main__':
    unittest.main()
