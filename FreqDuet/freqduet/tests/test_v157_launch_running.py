from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_freqduet_v157_launch_running as audit


class RecordedRunningTests(unittest.TestCase):
    def test_recording_preserves_frozen_value_and_strict_prefix(self):
        prefix = SimpleNamespace(times={(True, 2): [10., 30.]},
            rows={(True, 2): [(10., 0., 10.), (30., 0., 30.)]})
        cells = [dict(direction=True, station=2, hour=2,
            days=[dict(day=2, mean=40., second=1600.)])]
        recorded = audit.RecordedRunning(cells, 1, prefix)
        frozen = audit.audit.candidate.RunningLaw(cells, 1, prefix)
        self.assertEqual(recorded.estimate(True, 2, 30., 7200., dict(mean=10.)),
            frozen.estimate(True, 2, 30., 7200., dict(mean=10.)))
        call = recorded.calls[True, 2]
        self.assertEqual(call['selected']['latest_current_s'], 10.)
        self.assertEqual(call['prior_day_moments'], cells[0]['days'])

    def test_unsupported_hour_records_zero_day_native_fallback(self):
        prefix = SimpleNamespace(times={(True, 2): [10.]}, rows={(True, 2): [(10., 0., 10.)]})
        law = audit.RecordedRunning([], 1, prefix)
        self.assertEqual(law.estimate(True, 2, 20., 7200., dict(mean=10.))['mean'], 10.)
        self.assertEqual(law.calls[True, 2]['prior_day_moments'], [])


if __name__ == '__main__':
    unittest.main()
