import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v36_lower_credit import (
    KEYS, OUTCOMES, RETURNS, action_differences, action_values, check_unchanged,
    preference_pairs, same_prefix,
)


class LowerCreditTest(unittest.TestCase):
    def test_action_terms_reproduce_actor_objective_without_rng_consumption(self):
        probability = torch.tensor([[0.25, 0.75]])
        q = torch.tensor([[2.0, 5.0]])
        std = torch.tensor([[0.5, 1.0]])
        cost = torch.tensor([[0.2, 0.4]])
        trainer = SimpleNamespace(
            device="cpu", alpha=0.1, lambda_param=2.0,
            policy_net=SimpleNamespace(
                dist_info=lambda state: (probability, probability.log(), None),
                action_bins=torch.tensor([[0.0], [15.0]])),
            q_net=None, cost_q_net=None,
            _discrete_q_values=lambda net, state: q,
            _policy_q_value=lambda values: (q - 2 * std, q, std),
            _discrete_cost_values=lambda net, state: cost)
        before = torch.random.get_rng_state().clone()
        rows = action_values(trainer, np.array([0.3, 0.7]), torch)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        for index, row in enumerate(rows):
            self.assertAlmostEqual(row["energy"], float(-q[0, index] + 2 * std[0, index] + 2 * cost[0, index]), places=6)
        expected = float((probability * (0.1 * probability.log() - q + 2 * std + 2 * cost)).sum())
        actual = sum(row["probability"] * row["actor_loss_term"] for row in rows)
        self.assertAlmostEqual(expected, actual, places=6)

    def prefix(self):
        return dict(identity={"bus_id": 3}, raw_state=np.array([1, 2]),
                    encoded_state=np.array([0.1, 0.2]), previous_action_s=0.0,
                    actor_action_s=15.0, departure_context={"valid": True},
                    lower_prefix=[{"action": 0.0}], upper_prefix=[{"action": 2.0}],
                    rng={"state": np.array([11, 13])}, trip_reward_return=1.0)

    def test_prefix_allows_changed_consequences_but_rejects_state_or_rng_changes(self):
        left = self.prefix()
        right = copy.deepcopy(left)
        right["trip_reward_return"] = -100.0
        same_prefix(left, right)
        for field in ("raw_state", "encoded_state"):
            changed = copy.deepcopy(right)
            changed[field][0] += 1
            with self.assertRaises(AssertionError):
                same_prefix(left, changed)
        right["rng"]["state"][0] += 1
        with self.assertRaises(AssertionError):
            same_prefix(left, right)

    def test_noop_checks_ignore_only_instrumentation_not_operational_outcomes(self):
        left = dict(headway_cv=0.2, wall_env_s=1.0, policy_digest="a")
        right = dict(headway_cv=0.2, wall_env_s=9.0, policy_digest="b")
        check_unchanged(left, right)
        right["headway_cv"] = 0.3
        with self.assertRaises(AssertionError):
            check_unchanged(left, right)

    def test_preference_contrasts_use_common_states_and_correct_direction(self):
        key = dict(zip(KEYS, ("main", 11, 21, 199, 7200)))
        values, rollouts = [], []
        for action in (0.0, 15.0, 30.0):
            rollouts.append(dict(key, action_s=action,
                                 **{metric: action ** 2 for metric in (*OUTCOMES, *RETURNS)}))
            for ep, preferred, energy_preferred in ((39, 15.0, 0.0), (199, 30.0, 15.0)):
                values.append(dict(key, model_ep=ep, action_s=action,
                                   probability=0.8 if action == preferred else 0.1,
                                   energy=(action - energy_preferred) ** 2))
        pairs = preference_pairs(pd.DataFrame(values), pd.DataFrame(rollouts)).set_index("contrast")
        self.assertEqual(pairs.loc["policy200_minus_policy40", "action_delta_s"], 15.0)
        self.assertEqual(pairs.loc["policy200_minus_policy40", "trip_reward_return"], 675.0)
        self.assertEqual(pairs.loc["energy200_minus_energy40", "trip_cost_return"], 225.0)
        self.assertEqual(pairs.loc["max_hold_minus_zero", "action_delta_s"], 30.0)
        self.assertEqual(pairs.loc["max_hold_minus_zero", "headway_cv"], 900.0)

    def test_action_differences_keep_contexts_and_subtract_zero_in_correct_direction(self):
        rows = pd.DataFrame([dict(context=context, action_s=action, value=context + 2 * action)
                             for context in (11, 13) for action in (0, 15)])
        paired = action_differences(rows, ["context"], ["value"])
        self.assertEqual(len(paired), 4)
        self.assertEqual(set(paired.loc[paired.action_s == 15, "value_delta_vs_zero"]), {30})
        with self.assertRaisesRegex(ValueError, "missing zero-hold"):
            action_differences(rows.loc[~((rows.context == 13) & (rows.action_s == 0))],
                               ["context"], ["value"])


if __name__ == "__main__":
    unittest.main()
