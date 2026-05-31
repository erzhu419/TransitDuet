import unittest

from freq_hrl.core import BinnedExogenousStreamAdapter


class StreamAdapterTest(unittest.TestCase):
    def test_get_bin_uses_only_observed_events(self):
        adapter = BinnedExogenousStreamAdapter(bin_sec=60, rate_per_sec=1.0)
        adapter.observe({"timestamp": 10, "value": 2.0}, t=10)
        first = adapter.get_bin(30)
        self.assertEqual(float(first["x_raw"][0]), 2.0)

        adapter.observe({"timestamp": 50, "value": 3.0}, t=50)
        second = adapter.get_bin(59)
        self.assertEqual(float(second["x_raw"][0]), 5.0)

        # Future bin is not present until its event is explicitly observed.
        third = adapter.get_bin(70)
        self.assertEqual(float(third["x_raw"][0]), 0.0)

    def test_rejects_future_observe(self):
        adapter = BinnedExogenousStreamAdapter()
        with self.assertRaises(ValueError):
            adapter.observe({"timestamp": 20, "value": 1.0}, t=10)


if __name__ == "__main__":
    unittest.main()
