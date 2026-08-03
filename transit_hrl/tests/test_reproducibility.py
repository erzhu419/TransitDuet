import unittest

from freq_hrl.experiments.reproducibility import (
    derive_seed,
    training_rollout_seed,
    validate_evaluation_seed_roles,
)


class ReproducibilityProtocolTest(unittest.TestCase):
    def test_structured_seed_derivation_is_stable_and_namespaced(self):
        first = derive_seed("alpha", 7, 42, 0)
        self.assertEqual(first, derive_seed("alpha", 7, 42, 0))
        self.assertNotEqual(first, derive_seed("beta", 7, 42, 0))

    def test_training_paths_change_by_replicate_root_and_iteration(self):
        seeds = {
            training_rollout_seed(rep, root, iteration, domain="trading")
            for rep in (7, 11)
            for root in (42, 123)
            for iteration in (0, 1)
        }
        self.assertEqual(len(seeds), 8)

    def test_validation_and_heldout_test_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            validate_evaluation_seed_roles([1, 2], [2, 3])


if __name__ == "__main__":
    unittest.main()
