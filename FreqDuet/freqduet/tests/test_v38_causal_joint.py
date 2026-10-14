from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fit_freqduet_v38_causal_joint import (
    ACTIONS, PRIMARY, CausalJointScorer, Ridge, feasible_choice, fit_joint,
    first_basis, second_basis, state_vector,
)


def snapshot():
    return dict(clock_hours=2, target_headway_s=360, forward_gap_s=390,
                follower_gap_s=0, follower_valid=0, load_ratio=.22, ready_return=1,
                overdue_return_count=0, return_margin_proxy_s=-1500,
                past_mean_hold_s=10, actor_hold_s=0)


class CausalJointTest(unittest.TestCase):
    def test_first_stage_ignores_future_payloads(self):
        observed = snapshot()
        model = Ridge().fit([first_basis(observed, a) for a in ACTIONS],
                            [[-a, -a, -a] for a in ACTIONS], intercept=True)
        scorer = CausalJointScorer(model, None)
        before = scorer.choose_first(observed)
        observed.update(second_snapshot=object(), future_outcomes=object(), seed=999999)
        after = scorer.choose_first(observed)
        self.assertEqual(before[:2], after[:2])
        np.testing.assert_array_equal(before[2], after[2])

    def test_second_stage_incumbent_is_exact_zero_for_any_first_prefix(self):
        observed = snapshot()
        observed["actor_hold_s"] = 15
        np.testing.assert_array_equal(second_basis(snapshot(), 30, observed, 15), np.zeros(40))

    def test_invalid_follower_zero_is_not_a_measured_zero_gap(self):
        observed = snapshot()
        self.assertEqual(state_vector(observed)[2], 0)
        observed["follower_valid"] = 1
        self.assertEqual(state_vector(observed)[2], -1)

    def test_predicted_journey_alone_cannot_override(self):
        predictions = np.tile([-1, .001, -10], (7, 1))
        self.assertEqual(feasible_choice(predictions, 15), (3, False))
        predictions[4] = [-.1, 0, 0]
        self.assertEqual(feasible_choice(predictions, 15), (4, True))

    def test_joint_target_includes_second_stage_recourse(self):
        contexts = []
        for index in range(4):
            observed = snapshot()
            observed["load_ratio"] += index * .01
            rows = []
            for first in ACTIONS:
                for second in ACTIONS:
                    response = .1 * (first + second) / 45 - first * second / (45 * 45)
                    rows.append(dict(first_action_s=first, second_action_s=second,
                                     **dict(zip(PRIMARY, [response, response / 100, response * 10]))))
            contexts.append(dict(key=(index, 1, 7200), first=observed,
                                 second={a: dict(observed) for a in ACTIONS},
                                 pairs=pd.DataFrame(rows), reference=np.zeros(3)))
        scorer = fit_joint(contexts)
        first, active, _ = scorer.choose_first(snapshot())
        second, _ = scorer.choose_second(snapshot(), first, snapshot())
        self.assertTrue(active)
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)
        self.assertLess(.1 * (first + second) / 45 - first * second / (45 * 45), 0)


if __name__ == "__main__":
    unittest.main()
