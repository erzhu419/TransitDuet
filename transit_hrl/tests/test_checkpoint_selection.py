import unittest

from freq_hrl.rl import RobustValidationCheckpointSelector


class RobustValidationCheckpointSelectorTest(unittest.TestCase):
    def test_trailing_window_rejects_an_isolated_validation_spike(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
            smoothing_window=3,
            min_delta=0.05,
        )

        first = selector.consider(
            score=0.6, state={"step": 0}, iteration=0
        )
        second = selector.consider(
            score=-0.6, state={"step": 1}, iteration=1
        )
        third = selector.consider(
            score=0.2, state={"step": 2}, iteration=2
        )

        self.assertFalse(first["checkpoint_selection_eligible"])
        self.assertFalse(first["checkpoint_selected"])
        self.assertFalse(second["checkpoint_selected"])
        self.assertTrue(third["checkpoint_selected"])
        self.assertEqual(selector.selected_iteration, 2)
        self.assertEqual(selector.best_state, {"step": 2})

        metadata = selector.metadata(total_iterations=5)
        self.assertEqual(
            metadata["checkpoint_selection_protocol"],
            "trailing_mean_material_improvement_v1",
        )
        self.assertEqual(metadata["checkpoint_smoothing_window"], 3)
        self.assertEqual(metadata["checkpoint_plateau_tail_iterations"], 2)

    def test_window_one_preserves_strict_best_score_selection(self):
        selector = RobustValidationCheckpointSelector(
            initial_score=0.0,
            initial_state={"step": -1},
        )
        selector.consider(score=0.1, state={"step": 0}, iteration=0)
        selector.consider(score=0.05, state={"step": 1}, iteration=1)

        self.assertAlmostEqual(selector.best_score, 0.1)
        self.assertEqual(selector.selected_iteration, 0)
        self.assertEqual(selector.best_state, {"step": 0})
        self.assertEqual(
            selector.metadata(total_iterations=2)[
                "checkpoint_selection_protocol"
            ],
            "disjoint_validation_paths",
        )

    def test_invalid_selection_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                smoothing_window=0,
            )
        with self.assertRaises(ValueError):
            RobustValidationCheckpointSelector(
                initial_score=0.0,
                initial_state={},
                min_delta=-1e-3,
            )


if __name__ == "__main__":
    unittest.main()
