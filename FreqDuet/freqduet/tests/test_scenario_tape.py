import unittest

from env.scenario import ScenarioTape


class ScenarioTapeTest(unittest.TestCase):
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
