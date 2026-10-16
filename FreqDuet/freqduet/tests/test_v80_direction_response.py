from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v80_direction_response as response


class DirectionResponseTest(unittest.TestCase):
    def test_symmetric_slope_and_full_scale_curvature_are_separate(self):
        values = {label: np.array([10. + 2. * scale + 3. * scale ** 2])
                  for label, scale in response.SCALES.items()}
        result = response.responses(values)
        np.testing.assert_allclose(result["centered"], [2.])
        np.testing.assert_allclose(result["positive"], [2.3])
        np.testing.assert_allclose(result["full_minus_centered"], [3.])

    def test_paired_day_is_the_unit_not_sampler_or_actor(self):
        rows = []
        for day in (1, 2):
            for seed in response.pilot.SEEDS:
                for rep in response.REPEATS:
                    rows.append(dict(scenario_seed=day, train_seed=seed,
                        results={label: {m: 10. + day * scale for m in response.COSTS}
                                 for label, scale in response.SCALES.items()}))
        with patch.object(response, "DAYS", (1, 2)):
            paired = response.paired_responses(rows, response.pilot.SEEDS)
        self.assertEqual(paired["centered"].shape, (2, 3))
        np.testing.assert_allclose(paired["centered"][:, 0], [1., 2.])

    def test_cells_are_unique_and_do_not_reuse_v78_days(self):
        self.assertEqual(len(response.ITEMS), 128)
        self.assertEqual(len(set(response.ITEMS)), 128)
        self.assertFalse(set(response.DAYS) & set(response.pilot.TRAIN_DAYS + response.pilot.VALID_DAYS))


if __name__ == "__main__":
    unittest.main()
