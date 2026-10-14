from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_freqduet_v36_actor_alignment import OUTCOMES, audit_state, summarize


class ActorAlignmentTest(unittest.TestCase):
    def records(self, probability, energy=(0.0, 2.0), alpha=0.5):
        actions = [0.0, 15.0]
        values = pd.DataFrame(dict(action_s=actions, probability=probability,
                                   energy=energy, alpha=alpha, lambda_value=1.0,
                                   q_mean=-np.asarray(energy), pessimism=0.0, cost_penalty=0.0))
        rollouts = pd.DataFrame([dict(action_s=a, **{m: 2 * a for m in OUTCOMES}) for a in actions])
        return values, rollouts

    def test_boltzmann_solution_has_zero_objective_gap_and_same_mode(self):
        probability = np.exp([0.0, -4.0])
        probability /= probability.sum()
        result = audit_state(*self.records(probability))
        for name in ("energy_regret", "action_mismatch", "kl_actor_to_boltzmann",
                     "total_variation", "soft_objective_gap"):
            self.assertAlmostEqual(result[name], 0.0, places=12)

    def test_regret_components_and_counterfactual_direction(self):
        values, rollouts = self.records([0.1, 0.9])
        values["q_mean"] = [1.0, 0.0]
        values["pessimism"] = [0.2, 0.6]
        values["cost_penalty"] = [0.3, 0.9]
        values["energy"] = -values.q_mean + values.pessimism + values.cost_penalty
        result = audit_state(values.iloc[::-1], rollouts.iloc[::-1])
        self.assertAlmostEqual(result["energy_regret"], 2.0)
        self.assertAlmostEqual(result["reward_q_regret_component"], 1.0)
        self.assertAlmostEqual(result["pessimism_regret_component"], 0.4)
        self.assertAlmostEqual(result["cost_regret_component"], 0.6)
        self.assertEqual(result["actor_minus_energy_action_s"], 15.0)
        self.assertEqual(result["energy_minus_actor_trip_reward_return"], -30.0)
        self.assertAlmostEqual(result["soft_objective_gap"], 0.5 * result["kl_actor_to_boltzmann"])

    def test_temperature_changes_distribution_but_not_energy_mode(self):
        cold = audit_state(*self.records([0.3, 0.7], alpha=0.01))
        warm = audit_state(*self.records([0.3, 0.7], alpha=10.0))
        self.assertEqual(cold["energy_action_s"], warm["energy_action_s"])
        self.assertLess(cold["boltzmann_expected_action_s"], warm["boltzmann_expected_action_s"])

    def test_zero_probability_and_tied_energy_are_valid(self):
        result = audit_state(*self.records([0.0, 1.0], energy=(0.0, 0.0)))
        self.assertEqual(result["action_mismatch"], 1.0)
        self.assertEqual(result["positive_energy_regret"], 0.0)
        self.assertAlmostEqual(result["kl_actor_to_boltzmann"], np.log(2.0))

    def test_missing_action_cannot_be_silently_paired(self):
        values, rollouts = self.records([0.2, 0.8])
        with self.assertRaisesRegex(ValueError, "counterfactual actions"):
            audit_state(values, rollouts.iloc[:1])

    def test_contexts_are_averaged_before_crossed_seed_interval(self):
        frame = pd.DataFrame([dict(config="main", anchor_lower_ep=199,
                                   train_seed=t, eval_seed=e, value=c + t + e)
                              for t in (1, 2) for e in (3, 4) for c in (0, 10, 20)])
        def interval(paired, metric):
            self.assertEqual(len(paired), 4)
            self.assertFalse(paired.duplicated(["train_seed", "eval_seed"]).any())
            return float(paired[metric].min()), float(paired[metric].max())
        result = summarize(frame, ["config", "anchor_lower_ep"], ["value"], interval)
        self.assertEqual(float(result.iloc[0]["mean"]), 15.0)

    def test_single_group_column_preserves_the_complete_contrast_name(self):
        frame = pd.DataFrame([dict(contrast="joint_vs_actor", train_seed=1, eval_seed=2, value=3)])
        result = summarize(frame, ["contrast"], ["value"], lambda frame, metric: (3, 3))
        self.assertEqual(result.iloc[0].contrast, "joint_vs_actor")


if __name__ == "__main__":
    unittest.main()
