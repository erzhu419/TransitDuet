import unittest

import numpy as np

from env.scenario import ScenarioTape


class ScenarioTapeTest(unittest.TestCase):
    def test_scalar_draws_exactly_match_fresh_legacy_generators(self):
        tape = ScenarioTape(45007)
        for index in range(100):
            key = (index, True, "stop", float(index * 20))
            seed = tape._seed_for("legacy", *key)
            for rate in (0.0, 0.01, 2.5, 50.0):
                self.assertEqual(tape.poisson(rate, "legacy", *key),
                                 np.random.RandomState(seed).poisson(rate))
            self.assertEqual(tape.normal(3.0, 2.0, "legacy", *key),
                             np.random.RandomState(seed).normal(3.0, 2.0))
            self.assertEqual(tape.uniform(-2.0, 7.0, "legacy", *key),
                             np.random.RandomState(seed).uniform(-2.0, 7.0))
            self.assertEqual(tape.lognormal(0.2, 0.7, "legacy", *key),
                             np.random.RandomState(seed).lognormal(0.2, 0.7))
            self.assertEqual(tape.choice(["a", "b", "c"], "legacy", *key),
                             ["a", "b", "c"][np.random.RandomState(seed).choice(3)])
            self.assertEqual(
                tape.choice(["a", "b", "c"], "legacy", *key,
                            probabilities=[1.0, 2.0, 7.0]),
                ["a", "b", "c"][np.random.RandomState(seed).choice(3, p=[0.1, 0.2, 0.7])])

    def test_scalar_draws_do_not_reseed_or_alias_persistent_streams(self):
        tape = ScenarioTape(45007)
        seed_a = tape._seed_for("a:stream", 3)
        seed_b = tape._seed_for("b:stream", 4)
        a, b = np.random.RandomState(seed_a), np.random.RandomState(seed_b)
        for index in range(100):
            self.assertEqual(tape.normal_stream(0.0, 1.0, "a", 3), a.normal())
            tape.poisson(3.0, "interleaved", index)
            self.assertEqual(tape.poisson_stream(2.5, "b", 4), b.poisson(2.5))
            tape.normal(0.0, 1.0, "interleaved", index)
            self.assertEqual(tape.normal_stream(0.0, 1.0, "a", 3), a.normal())

    def test_draws_are_keyed_instead_of_consumption_ordered(self):
        first = ScenarioTape(123)
        expected = first.poisson(4.2, "arrival", 60, 3, "X9")
        first.normal(0.0, 1.0, "unrelated", 99)
        actual = first.poisson(4.2, "arrival", 60, 3, "X9")
        self.assertEqual(expected, actual)

    def test_seed_and_key_change_realisation(self):
        draws_a = [
            ScenarioTape(123).normal(0.0, 1.0, "speed", key)
            for key in range(8)
        ]
        draws_b = [
            ScenarioTape(124).normal(0.0, 1.0, "speed", key)
            for key in range(8)
        ]
        self.assertNotEqual(draws_a, draws_b)

    def test_identifier_is_stable(self):
        self.assertEqual(
            ScenarioTape(456).identifier,
            ScenarioTape(456).identifier,
        )

    def test_independent_streams_are_reproducible(self):
        left = ScenarioTape(789)
        right = ScenarioTape(789)
        expected = [
            left.poisson_stream(2.5, "arrival", 3, True, "X9")
            for _ in range(10)
        ]
        right.normal_stream(0.0, 1.0, "speed", 8)
        actual = [
            right.poisson_stream(2.5, "arrival", 3, True, "X9")
            for _ in range(10)
        ]
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
