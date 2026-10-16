import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v68_repeated_continuations as repeated
from env.scenario import ScenarioTape
from lower.resac_lagrangian import CategoricalPolicy


def example_runner():
    passenger = SimpleNamespace(appear_time=12.5, destination_station=SimpleNamespace(station_id=3))
    env = SimpleNamespace(current_time=10, stations=[SimpleNamespace(_pending_passengers=[passenger])],
                          routes=[SimpleNamespace(speed_limit=7)], _demand_multipliers={6: 1.2},
                          _od_multipliers={(1, True, "stop"): .8}, _peak_shift=1, scenario_tape=ScenarioTape(67))
    env.scenario_tape.normal_stream(0, 1, "route_speed", 1)
    upper = CategoricalPolicy(3, repeated.joint.ACTIONS, hidden_dim=8, sample_seed=42)
    streams = {"fleet": np.random.RandomState(1), "selector": np.random.RandomState(2)}
    runner = SimpleNamespace(env=env, upper_trainer=SimpleNamespace(policy_net=upper),
                             _runtime_numpy_streams=lambda: streams)
    policy = CategoricalPolicy(3, repeated.joint.ACTIONS, hidden_dim=8, sample_seed=43)
    return runner, policy


class RepeatedContinuationTest(unittest.TestCase):
    def test_fixed_design_repeats_all_prefixes_not_only_oracle_winners(self):
        self.assertEqual(len(repeated.ITEMS), 768)
        self.assertEqual(len(repeated.ITEMS) * len(repeated.joint.ACTIONS), 5376)
        seeds = {repeated.future_seed(c, f) for c, f in repeated.ITEMS}
        self.assertEqual(len(seeds), len(repeated.ITEMS))
        for context in range(96):
            self.assertEqual([f for c, f in repeated.ITEMS if c == context], list(range(8)))

    def test_fork_preserves_pending_arrivals_profiles_current_speed_and_policy_weights(self):
        runner, policy = example_runner()
        before = repeated.physical_prefix(runner.env)
        weights = copy.deepcopy(policy.state_dict())
        pending = runner.env.stations[0]._pending_passengers[0]
        audit = repeated.fork_future(runner, policy, repeated.future_seed(0, 0))
        repeated.joint.assert_same_state(before, repeated.physical_prefix(runner.env))
        repeated.joint.assert_same_state(weights, policy.state_dict())
        self.assertIs(pending, runner.env.stations[0]._pending_passengers[0])
        self.assertEqual(audit["scenario_streams"], {})
        self.assertEqual(audit["scenario_seed"], repeated.future_seed(0, 0))

    def test_matched_future_resets_every_stream_but_distinct_futures_do_not_match(self):
        first, left = example_runner()
        second, right = example_runner()
        seed = repeated.future_seed(3, 2)
        a = repeated.fork_future(first, left, seed)
        b = repeated.fork_future(second, right, seed)
        repeated.joint.assert_same_state(a, b)
        left.get_action(np.zeros(3, dtype=np.float32))
        x = first.env.scenario_tape.normal_stream(0, 1, "route_speed", 1)
        y = second.env.scenario_tape.normal_stream(0, 1, "route_speed", 1)
        self.assertEqual(x, y)
        fresh = repeated.fork_future(second, right, repeated.future_seed(3, 3))
        self.assertFalse(torch.equal(a["policy_sampling"], fresh["policy_sampling"]))
        self.assertNotEqual(x, second.env.scenario_tape.normal_stream(0, 1, "route_speed", 1))

    def test_current_actor_draw_is_preserved_before_future_policy_sampling_is_reset(self):
        runner, policy = example_runner()
        state = np.zeros(3, dtype=np.float32)
        actor, executed = repeated.labels.draw_action(policy, state, 45)
        before = policy.sampling_state().clone()
        repeated.fork_future(runner, policy, repeated.future_seed(5, 1))
        self.assertIn(actor, repeated.joint.ACTIONS)
        self.assertEqual(executed, 45)
        self.assertFalse(torch.equal(before, policy.sampling_state()))

    def test_selection_uses_only_training_futures_and_evaluation_keeps_harmful_sign(self):
        arrays = {name: np.zeros((8, 7)) for name in repeated.RESPONSE_METRICS}
        arrays["passenger_person_s"][:4, 1] = -20
        arrays["passenger_person_s"][4:, 1] = 60
        outcome = repeated.evaluate_selection(arrays, 0, np.arange(4), np.arange(4, 8))
        self.assertEqual(outcome["action_s"], 5)
        self.assertEqual(outcome["passenger_person_s"], 60)
        reverse = repeated.evaluate_selection(arrays, 0, np.arange(4, 8), np.arange(4))
        self.assertEqual(reverse["action_s"], 0)

    def test_new_ties_preserve_native_but_v67_replication_keeps_original_grid_tie_rule(self):
        means = {name: np.zeros(7) for name in repeated.RESPONSE_METRICS}
        self.assertEqual(repeated.constrained_action(means, 3), 3)
        rows = [{name: means[name][i] for name in means} for i in range(7)]
        self.assertEqual(repeated.original_oracle(rows, 3), 0)
        means["passenger_person_s"][1] = -10
        means["readiness_delay_s"][1] = 1
        self.assertEqual(repeated.constrained_action(means, 3), 3)

    def test_noise_correction_remains_signed_and_reference_ties_are_not_signal(self):
        zero_mean = np.zeros((8, 7))
        zero_mean[:, 1:] = np.tile([1, -1], 4)[:, None]
        result = repeated.signal_noise(zero_mean)
        self.assertLess(result["debiased_signal_squared"], 0)
        constant = np.zeros((8, 7))
        constant[:, 1:] = 3
        stable = repeated.signal_noise(constant)
        self.assertEqual(stable["single_future_noise_variance"], 0)
        self.assertEqual(stable["debiased_signal_squared"], 9)
        self.assertEqual(repeated.signal_noise(np.zeros((8, 7)))["half_sign_pairs"], 0)

    def test_intervals_cluster_original_days_not_futures_actors_or_prefixes(self):
        contexts = [dict(scenario_seed=day, value=float(i))
                    for i, day in enumerate(repeated.labels.DEMAND_SEEDS) for _ in range(12)]
        interval = repeated.day_intervals(contexts, lambda c: c["value"], 68)
        self.assertEqual(interval["n_demand_days"], 8)
        self.assertEqual(interval["mean"], 3.5)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
