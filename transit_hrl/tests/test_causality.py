import unittest

import numpy as np

from freq_hrl.encoders import (
    CausalEMAEncoder,
    CausalFourierEncoder,
    CausalHaarWaveletEncoder,
    CausalStateSpaceEncoder,
)


class CausalityTest(unittest.TestCase):
    def _capture(self, encoder, values):
        encoder.reset()
        out = []
        for t, value in enumerate(values):
            encoder.update({"timestamp": t, "x_raw": [float(value)]})
            feats = encoder.features()
            out.append({
                "low": feats["x_low"].copy(),
                "high": feats["x_high"].copy(),
                "forecast": feats["x_low_forecast"].copy(),
            })
        return out

    def _assert_encoder_causal(self, encoder):
        rng = np.random.default_rng(7)
        prefix = rng.normal(size=20)
        future = rng.normal(loc=100.0, scale=20.0, size=10)
        a = self._capture(encoder, prefix)
        b = self._capture(encoder, np.concatenate([prefix, future]))
        np.testing.assert_allclose(a[-1]["low"], b[len(prefix) - 1]["low"])
        np.testing.assert_allclose(a[-1]["high"], b[len(prefix) - 1]["high"])
        np.testing.assert_allclose(a[-1]["forecast"], b[len(prefix) - 1]["forecast"])

    def test_ema_encoder_is_causal(self):
        self._assert_encoder_causal(
            CausalEMAEncoder(
                update_interval_s=60,
                low_period_s=600,
                fast_period_s=120,
                forecast_horizon_s=300,
            )
        )

    def test_fourier_encoder_is_causal(self):
        self._assert_encoder_causal(
            CausalFourierEncoder(
                update_interval_s=60,
                period_s=3600,
                fourier_k=2,
                forecast_horizon_s=300,
            )
        )

    def test_state_space_encoder_is_causal(self):
        self._assert_encoder_causal(
            CausalStateSpaceEncoder(
                update_interval_s=60,
                process_var=1e-3,
                measurement_var=1e-2,
                forecast_horizon_s=300,
            )
        )

    def test_haar_wavelet_encoder_is_causal(self):
        self._assert_encoder_causal(
            CausalHaarWaveletEncoder(
                update_interval_s=60,
                low_window_s=600,
                short_window_s=180,
                forecast_horizon_s=300,
            )
        )

    def test_state_space_uncertainty_is_nonnegative(self):
        encoder = CausalStateSpaceEncoder(
            update_interval_s=60,
            process_var=1e-3,
            measurement_var=1e-2,
            forecast_horizon_s=300,
        )
        for t, value in enumerate([0.0, 1.0, 1.0, 2.0]):
            encoder.update({"timestamp": t, "x_raw": [value]})
        feats = encoder.features()
        self.assertTrue(np.all(feats["x_low_uncertainty"] >= 0.0))
        self.assertEqual(feats["encoder"], "causal_state_space")


if __name__ == "__main__":
    unittest.main()
