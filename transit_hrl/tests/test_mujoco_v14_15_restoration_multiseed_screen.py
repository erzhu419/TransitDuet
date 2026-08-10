import unittest

import numpy as np

from scripts import mujoco_v14_15_closed_loop_restoration_filter_screen_spec as base
from scripts import mujoco_v14_15_restoration_multiseed_screen_spec as spec
from scripts.analyze_mujoco_v14_15_restoration_multiseed_screen import (
    _condition_effects,
    _simultaneous_basic_lower_bounds,
    _wilson_lower,
)
from scripts.submit_mujoco_v14_15_closed_loop_restoration_filter_screen_scheduleurm import (
    build_parser,
    normalize_args,
)


class MujocoV1415RestorationMultiseedScreenTest(unittest.TestCase):
    def test_candidate_and_seed_namespace_are_frozen_after_preflight(self):
        self.assertEqual(
            spec.PRESELECTED_CANDIDATE_ARM,
            "group_replay1_trust1_outer1_restore1_eps5e3_bt8_f3",
        )
        self.assertIn(spec.PRESELECTED_CANDIDATE_ARM, base.AUTHORIZING_ARMS)
        self.assertEqual(spec.OPTIMIZER_SEEDS, base.OPTIMIZER_SEEDS[1:])
        self.assertNotIn(base.OPTIMIZER_SEEDS[0], spec.OPTIMIZER_SEEDS)
        self.assertEqual(len(spec.OPTIMIZER_SEEDS), 15)
        self.assertEqual(len(spec.PRIMARY_CONTRAST_ORDER), 18)

    def test_launcher_profile_rejects_seed_or_environment_subsets(self):
        seed_csv = ",".join(map(str, spec.OPTIMIZER_SEEDS))
        args = normalize_args(build_parser().parse_args([
            "--run-name", "mujoco_v14_15_multiseed_unit",
            "--fixed-candidate-multiseed",
            "--optimizer-seeds", seed_csv,
            "--python-executable", "python3",
        ]))
        self.assertEqual(args.environments, list(spec.ENVIRONMENTS))
        self.assertEqual(args.optimizer_seeds, list(spec.OPTIMIZER_SEEDS))
        with self.assertRaisesRegex(SystemExit, "15 fresh seeds"):
            normalize_args(build_parser().parse_args([
                "--run-name", "mujoco_v14_15_multiseed_unit",
                "--fixed-candidate-multiseed",
                "--optimizer-seeds", str(spec.OPTIMIZER_SEEDS[0]),
                "--python-executable", "python3",
            ]))
        with self.assertRaisesRegex(SystemExit, "all environments"):
            normalize_args(build_parser().parse_args([
                "--run-name", "mujoco_v14_15_multiseed_unit",
                "--fixed-candidate-multiseed",
                "--optimizer-seeds", seed_csv,
                "--environments", spec.ENVIRONMENTS[0],
                "--python-executable", "python3",
            ]))

    def test_simultaneous_bootstrap_is_optimizer_level_and_deterministic(self):
        values = np.asarray([
            [0.20 + index * 0.002, 0.30 + index * 0.001]
            for index in range(15)
        ])
        first = _simultaneous_basic_lower_bounds(
            values, confidence=0.95, draws=1_000, seed=42
        )
        second = _simultaneous_basic_lower_bounds(
            values, confidence=0.95, draws=1_000, seed=42
        )
        np.testing.assert_allclose(
            first["simultaneous_lower"], second["simultaneous_lower"]
        )
        self.assertEqual(first["observed"].shape, (2,))
        self.assertTrue(np.all(
            first["simultaneous_lower"] <= first["observed"]
        ))
        with self.assertRaisesRegex(ValueError, "optimizer-level"):
            _simultaneous_basic_lower_bounds(
                np.ones((1, 40)), confidence=0.95, draws=1_000, seed=42
            )

    def test_condition_effects_use_paired_normalization_and_log_ratios(self):
        status = {"conditions": [
            {
                "disturbance_mode": mode,
                "reward_difference": 5.0,
                "reward_noninferiority_margin": 2.0,
                "frequency_reduction_fraction": {
                    metric: 0.10 for metric in spec.FREQUENCY_METRICS
                },
            }
            for mode in spec.EVALUATION_DISTURBANCE_MODES
        ]}
        pooled, per_mode = _condition_effects(status)
        self.assertEqual(len(per_mode), 5)
        self.assertAlmostEqual(pooled["normalized_episode_return"], 0.05)
        self.assertAlmostEqual(
            pooled[spec.FREQUENCY_METRICS[0]], -np.log(0.90)
        )

    def test_wilson_gate_is_not_a_raw_success_fraction(self):
        lower = _wilson_lower(45, 45, spec.WILSON_ONE_SIDED_Z)
        self.assertGreater(lower, 0.90)
        self.assertLess(lower, 1.0)
        self.assertLess(
            _wilson_lower(36, 45, spec.WILSON_ONE_SIDED_Z), 0.80
        )


if __name__ == "__main__":
    unittest.main()
