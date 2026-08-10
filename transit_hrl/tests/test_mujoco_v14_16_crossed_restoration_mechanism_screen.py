import math
import unittest
from pathlib import Path

from scripts import (
    mujoco_v14_15_closed_loop_restoration_filter_screen_spec as predecessor,
)
from scripts import (
    mujoco_v14_16_crossed_restoration_mechanism_screen_spec as spec,
)
from scripts.analyze_mujoco_v14_16_crossed_restoration_mechanism_screen import (
    EXPECTED_MERGED_MANIFEST_STATUS,
    FREQUENCY_METRICS,
    _effect_gate,
    _paired_path_effects,
    _pooled_effects,
)
from scripts.submit_mujoco_v14_16_crossed_restoration_mechanism_screen_scheduleurm import (
    build_parser,
    build_scheduler_spec,
    build_training_command,
    frozen_execution_identity,
    normalize_args,
    selected_experiment_cells,
)


class MujocoV1416MechanismScreenTest(unittest.TestCase):
    def _args(self):
        return normalize_args(build_parser().parse_args([
            "--run-name", "mujoco_v14_16_mechanism_unit",
            "--python-executable", "python3",
        ]))

    def test_frozen_identity_and_seed_namespace_are_new(self):
        args = self._args()
        self.assertEqual(
            frozen_execution_identity(args),
            (
                spec.FROZEN_ALGORITHM_REVISION,
                spec.FROZEN_SOURCE_MANIFEST_SHA256,
            ),
        )
        current_roles = (
            spec.OPTIMIZER_SEEDS,
            spec.PRETRAIN_SEEDS,
            spec.PRETRAIN_SELECTION_SEEDS,
            spec.CONTINUATION_TRAIN_SEEDS,
            spec.CONTINUATION_SELECTION_SEEDS,
            spec.DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS,
            spec.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
            spec.DEVELOPMENT_EVALUATION_SEEDS,
        )
        current = {int(seed) for role in current_roles for seed in role}
        predecessor_roles = (
            predecessor.OPTIMIZER_SEEDS,
            predecessor.PRETRAIN_SEEDS,
            predecessor.PRETRAIN_SELECTION_SEEDS,
            predecessor.CONTINUATION_TRAIN_SEEDS,
            predecessor.CONTINUATION_SELECTION_SEEDS,
            predecessor.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
            predecessor.DEVELOPMENT_EVALUATION_SEEDS,
        )
        previous = {
            int(seed) for role in predecessor_roles for seed in role
        }
        self.assertFalse(current & previous)

    def test_analyzer_requires_the_full_screen_merge_status(self):
        self.assertEqual(
            EXPECTED_MERGED_MANIFEST_STATUS,
            "development_screen_complete_unanalyzed",
        )

    def test_cumulative_ablation_chain_is_complete(self):
        observed = [
            (
                spec.ARMS[arm][
                    "deployment_frequency_projection_objective"
                ],
                spec.ARMS[arm]["deployment_frequency_pathwise_robust"],
                spec.ARMS[arm][
                    "deployment_frequency_restoration_freeze_reward_actor"
                ],
                bool(spec.ARMS[arm][
                    "deployment_frequency_anchor_state_replay_seed_roots"
                ]),
            )
            for arm in spec.LEARNED_ARMS
        ]
        self.assertEqual(observed, [
            ("worst_group", False, False, False),
            ("violation_l2", False, False, False),
            ("violation_l2", True, False, False),
            ("violation_l2", True, True, False),
            ("violation_l2", True, True, True),
        ])
        self.assertEqual(
            spec.expected_anchor_replay_path_count(
                spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM
            ),
            16,
        )
        self.assertEqual(
            spec.expected_closed_loop_guard_constraint_count(
                spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM
            ),
            96,
        )
        self.assertTrue(
            spec.expected_closed_loop_guard_contract(
                spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM
            ).endswith("restoration_merit_v3")
        )

    def test_launcher_builds_dynamic_unpinned_one_core_matrix(self):
        args = self._args()
        cells = selected_experiment_cells(args)
        self.assertEqual(len(cells), 81)
        scheduler_spec = build_scheduler_spec(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(set(scheduler_spec["allowed_nodes"]), set(args.nodes))
        self.assertTrue(scheduler_spec["reroute_on_node_down"])

    def test_commands_share_protocol_but_isolate_new_mechanisms(self):
        args = self._args()
        common = dict(
            args=args,
            environment="HalfCheetah-v5",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
        )
        anchor = build_training_command(
            phase="anchor", arm="closed_loop_guard_pretrain_anchor", **common
        )
        control = build_training_command(
            phase="continuation", arm=spec.WORST_MODE_TRAIN_REPLAY_ARM,
            **common,
        )
        candidate = build_training_command(
            phase="continuation",
            arm=spec.L2_PATH_FREEZE_CROSSED_REPLAY_ARM,
            **common,
        )
        for command in (anchor, control, candidate):
            self.assertIn(
                f"--control-protocol-version {spec.FROZEN_CORE_PROTOCOL_VERSION}",
                command,
            )
        self.assertIn(
            "--deployment-frequency-projection-objective worst_group",
            control,
        )
        self.assertNotIn(
            "--deployment-frequency-pathwise-robust", control
        )
        self.assertIn(
            "--deployment-frequency-projection-objective violation_l2",
            candidate,
        )
        self.assertIn(
            "--deployment-frequency-pathwise-robust", candidate
        )
        self.assertIn(
            "--deployment-frequency-restoration-freeze-reward-actor",
            candidate,
        )
        self.assertIn(
            "--deployment-frequency-anchor-state-replay-seeds",
            candidate,
        )
        for seed in spec.DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS:
            self.assertIn(str(seed), candidate)

    def test_path_effects_keep_paths_paired_and_use_log_reductions(self):
        def rows(first: float, second: float):
            return [
                {
                    "disturbance_mode": "standard",
                    "seed": seed,
                    "episode_return": 100.0 + index,
                    **{metric: value for metric in FREQUENCY_METRICS},
                }
                for index, (seed, value) in enumerate(
                    ((11, first), (13, second))
                )
            ]

        effects = _paired_path_effects(rows(0.5, 1.3), rows(1.0, 1.0))
        pooled = _pooled_effects(effects)
        self.assertAlmostEqual(
            pooled[FREQUENCY_METRICS[0]],
            0.5 * (math.log(2.0) + math.log(1.0 / 1.3)),
            places=9,
        )
        self.assertEqual(_effect_gate(pooled)["pass_count"], 6)
        with self.assertRaisesRegex(ValueError, "do not align"):
            _paired_path_effects(rows(0.5, 1.3)[:-1], rows(1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
