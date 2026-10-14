import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v35_actor_cross import (
    BASE_EP, METRICS, assert_same_state, factorial_effects, job_name,
    non_actor_state, transplant_actors,
)


class Policy(torch.nn.Linear):
    def sampling_state(self):
        return {"state": np.array([1, 2, 3], dtype=np.uint32)}


def trainer(lower=False):
    result = SimpleNamespace(policy_net=Policy(2, 1), q_net=torch.nn.Linear(2, 1),
                             target_q_net=torch.nn.Linear(2, 1), alpha=0.1,
                             log_alpha=torch.tensor(-2.3))
    if lower:
        result.cost_q_net = torch.nn.Linear(2, 1)
        result.target_cost_q_net = torch.nn.Linear(2, 1)
        result.log_lambda = torch.tensor(0.4)
    return result


def runner():
    return SimpleNamespace(lower_trainer=trainer(True), upper_trainer=trainer(),
                           _deployment_state_dict=lambda ep: dict(
                               episode=ep, belief=np.array([0.25, 0.75]),
                               cached={"unknown_cap": float("nan")}))


class ActorCrossTest(unittest.TestCase):
    def test_transplant_changes_only_execution_actor_state(self):
        model = runner()
        before = non_actor_state(model)
        lower = {"policy": {key: torch.full_like(value, 2.0)
                            for key, value in model.lower_trainer.policy_net.state_dict().items()}}
        upper = {"policy": {key: torch.full_like(value, 3.0)
                            for key, value in model.upper_trainer.policy_net.state_dict().items()}}
        transplant_actors(model, lower, upper)
        assert_same_state(before, non_actor_state(model))
        assert_same_state(model.lower_trainer.policy_net.state_dict(), lower["policy"])
        assert_same_state(model.upper_trainer.policy_net.state_dict(), upper["policy"])
        self.assertEqual(model._deployment_state_dict(BASE_EP)["episode"], 199)

    def test_transplant_rejects_critic_mutation(self):
        model = runner()
        lower = {"policy": copy.deepcopy(model.lower_trainer.policy_net.state_dict())}
        upper = {"policy": copy.deepcopy(model.upper_trainer.policy_net.state_dict())}
        load = model.lower_trainer.policy_net.load_state_dict

        def corrupt(state):
            with torch.no_grad():
                model.lower_trainer.q_net.weight.add_(1.0)
            return load(state)

        with patch.object(model.lower_trainer.policy_net, "load_state_dict", side_effect=corrupt):
            with self.assertRaisesRegex(AssertionError, "lower_q"):
                transplant_actors(model, lower, upper)

    def test_missing_diagnostic_values_compare_equal_but_real_changes_do_not(self):
        assert_same_state({"cap": float("nan"), "value": 1.0},
                          {"cap": np.float64("nan"), "value": 1.0})
        with self.assertRaisesRegex(AssertionError, "state.value"):
            assert_same_state({"cap": float("nan"), "value": 1.0},
                              {"cap": float("nan"), "value": 2.0})

    def frames(self):
        rows, old = [], []
        for seed in (11, 13):
            for scenario in (21, 23):
                key = dict(config="main", train_seed=seed, eval_seed=scenario)
                old.append(dict(key, **{metric: 8.0 for metric in METRICS}))
                for upper, u in ((39, 0), (199, 1)):
                    for lower, l in ((39, 0), (199, 1)):
                        value = 10.0 + 2.0 * u + 3.0 * l + 4.0 * u * l
                        rows.append(dict(key, upper_actor_ep=upper, lower_actor_ep=lower,
                                         **{metric: value for metric in METRICS}))
        return pd.DataFrame(rows), pd.DataFrame(old)

    def test_factorial_effects_and_residual_have_correct_signs_and_accounting(self):
        frame, early = self.frames()
        effects = factorial_effects(frame.sample(frac=1, random_state=4), early)
        values = effects.groupby("effect").headway_cv.mean()
        expected = dict(lower_with_upper40=3.0, lower_with_upper200=7.0,
                        upper_with_lower40=2.0, upper_with_lower200=6.0,
                        lower_averaged_effect=5.0, upper_averaged_effect=4.0,
                        upper_lower_interaction=4.0, actor_age_total=9.0,
                        non_actor_context_and_stage_residual=2.0, observed_40_to_200=11.0)
        for name, value in expected.items():
            self.assertEqual(values[name], value)
        self.assertEqual(values["lower_averaged_effect"] + values["upper_averaged_effect"]
                         + values["non_actor_context_and_stage_residual"],
                         values["observed_40_to_200"])

    def test_missing_actor_context_cannot_be_silently_dropped(self):
        frame, early = self.frames()
        with self.assertRaisesRegex(ValueError, "unmatched or duplicated"):
            factorial_effects(frame.iloc[:-1], early)

    def test_job_labels_use_episode_counts_not_zero_based_indices(self):
        self.assertEqual(job_name("main", 11, 39, 199), "main_seed11_U40_L200")


if __name__ == "__main__":
    unittest.main()
