import unittest

import numpy as np

from lower.state_encoder import PhysicalLowerStateEncoder


class PhysicalLowerStateEncoderTest(unittest.TestCase):
    def setUp(self):
        self.encoder = PhysicalLowerStateEncoder(
            base_state_dim=10,
            max_station_id=22,
            service_duration_h=14.0,
            action_range_s=45.0,
        )

    def test_replaces_identity_and_normalizes_physical_units(self):
        raw = np.asarray([
            91.0, 5.0, 7.0, 1.0, 450.0, 300.0, 150.0, 0.5,
            15.0, 7.5, 1.25, -0.5,
        ], dtype=np.float32)
        encoded = self.encoder.encode(raw)

        self.assertEqual(encoded.shape, raw.shape)
        self.assertAlmostEqual(encoded[0], 0.5)
        self.assertAlmostEqual(encoded[1], 5.0 / 22.0)
        self.assertAlmostEqual(encoded[2], 0.5)
        self.assertAlmostEqual(encoded[3], 1.0)
        self.assertAlmostEqual(encoded[4], 1.5)
        self.assertAlmostEqual(encoded[5], 1.0)
        self.assertAlmostEqual(encoded[6], 0.5)
        self.assertAlmostEqual(encoded[7], 0.5)
        self.assertAlmostEqual(encoded[8], 1.0)
        self.assertAlmostEqual(encoded[9], 0.5)
        np.testing.assert_allclose(encoded[10:], [1.25, -0.5])
        self.assertNotEqual(encoded[0], raw[0])

    def test_station_progress_follows_direction(self):
        up = np.asarray(
            [1, 4, 0, 1, 360, 360, 0, 0, 10, 10], dtype=np.float32)
        down = np.asarray(
            [2, 18, 0, 0, 360, 360, 0, 0, 10, 10], dtype=np.float32)
        self.assertAlmostEqual(
            self.encoder.encode(up)[1], self.encoder.encode(down)[1])
        self.assertEqual(self.encoder.encode(up)[3], 1.0)
        self.assertEqual(self.encoder.encode(down)[3], -1.0)

    def test_action_is_scaled_and_clipped(self):
        self.assertAlmostEqual(self.encoder.encode_action(22.5), 0.5)
        self.assertAlmostEqual(self.encoder.encode_action(90.0), 1.0)
        self.assertAlmostEqual(self.encoder.encode_action(-5.0), 0.0)

    def test_rejects_incompatible_schema(self):
        with self.assertRaisesRegex(ValueError, "shorter"):
            self.encoder.encode(np.zeros(9, dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
