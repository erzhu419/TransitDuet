import json
from pathlib import Path
import tempfile
import unittest

import recover_freqduet_v216_passive as v


class RecoveryTests(unittest.TestCase):
    def test_complete_common_prefix_survives_reachable_initial_inventory_write_failure(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'failed.json.gz'
            data = dict(result=dict(journey=1.), trace=dict(actions=[0, 5], times=[10., 20.]),
                        capture=dict(queries=[dict(last_observed_return_s=float('-inf'))]))
            with self.assertRaises(ValueError):
                v.v.trace(path, data)
            common, tail = v.written_common(path)
            self.assertEqual(common, dict(result=data['result'], trace=data['trace']))
            self.assertTrue(tail.endswith('"last_observed_return_s":'))

    def test_incomplete_common_trace_cannot_be_claimed_recovered(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'failed.json.gz'
            with self.assertRaises(ValueError):
                v.v.trace(path, dict(result={}, trace=dict(bad=float('-inf')), capture={}))
            with self.assertRaises(json.JSONDecodeError):
                v.written_common(path)


if __name__ == '__main__':
    unittest.main()
