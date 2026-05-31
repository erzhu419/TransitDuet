import unittest

import numpy as np

from freq_hrl.core import FrequencyRouter


class RouterMaskTest(unittest.TestCase):
    def test_upper_cannot_see_raw_high_sequence(self):
        router = FrequencyRouter(upper_extra_keys=("x_high_sequence", "custom_ok"))
        features = {
            "x_low": np.array([1.0]),
            "x_low_slope": np.array([0.1]),
            "x_low_forecast": np.ones((3, 1)),
            "x_high": np.array([5.0]),
            "x_high_energy": np.array([2.0]),
            "x_high_sequence": np.arange(10),
            "custom_ok": np.array([4.0]),
        }
        state = router.upper_view(features, {"fleet": 10})
        self.assertNotIn("x_high", state)
        self.assertNotIn("x_high_sequence", state)
        self.assertIn("x_high_energy", state)
        self.assertIn("custom_ok", state)

    def test_lower_cannot_see_full_low_forecast(self):
        router = FrequencyRouter(lower_extra_keys=("x_low_forecast_full", "custom_ok"))
        features = {
            "x_low": np.array([1.0]),
            "x_low_forecast": np.ones((4, 1)),
            "x_low_forecast_full": np.ones((16, 1)),
            "x_high": np.array([2.0]),
            "x_mid": np.array([0.5]),
            "custom_ok": np.array([7.0]),
        }
        state = router.lower_view(features, {"load": 0.2}, {"target": 1.0})
        self.assertNotIn("x_low_forecast", state)
        self.assertNotIn("x_low_forecast_full", state)
        self.assertIn("x_high", state)
        self.assertIn("current_plan", state)
        self.assertIn("custom_ok", state)


if __name__ == "__main__":
    unittest.main()
