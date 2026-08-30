import math
import unittest
from pathlib import Path

from freq_hrl.experiments.reproducibility import git_source_manifest_sha256
from scripts import (
    mujoco_v14_16_crossed_restoration_mechanism_screen_spec as predecessor,
)
from scripts import mujoco_v14_17_native_pd_cvar_screen_spec as spec
from scripts.analyze_mujoco_v14_17_native_pd_cvar_screen import (
    EXPECTED_MERGED_MANIFEST_STATUS,
    _engineering_gate,
)
from scripts.analyze_mujoco_v14_16_crossed_restoration_mechanism_screen import (
    FREQUENCY_METRICS,
    _effect_gate,
    _paired_path_effects,
    _pooled_effects,
)
from scripts.submit_mujoco_v14_17_native_pd_cvar_screen_scheduleurm import (
    build_parser,
    build_scheduler_spec,
    build_training_command,
    frozen_execution_identity,
    normalize_args,
    selected_experiment_cells,
)


class MujocoV1417NativePDCVaRScreenTest(unittest.TestCase):
    def _args(self):
        return normalize_args(build_parser().parse_args([
            "--run-name", "mujoco_v14_17_native_pd_cvar_unit",
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
            predecessor.DEPLOYMENT_FREQUENCY_ANCHOR_STATE_REPLAY_SEEDS,
            predecessor.DEPLOYMENT_FREQUENCY_CLOSED_LOOP_GUARD_SEEDS,
            predecessor.DEVELOPMENT_EVALUATION_SEEDS,
            predecessor.RETIRED_ENGINEERING_SEEDS,
        )
        previous = {
            int(seed) for role in predecessor_roles for seed in role
        }
        self.assertEqual(len(current), sum(map(len, current_roles)))
        self.assertFalse(current & previous)

    def test_frozen_manifest_matches_registered_algorithm_revision(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            git_source_manifest_sha256(
                root,
                Path("freq_hrl"),
                revision=spec.FROZEN_ALGORITHM_REVISION,
            ),
            spec.FROZEN_SOURCE_MANIFEST_SHA256,
        )

    def test_matrix_is_capacity_matched_and_factorial(self):
        self.assertEqual(len(spec.ARMS), 7)
        self.assertEqual(set(spec.NATIVE_PD_ARMS), {
            spec.NATIVE_PD_ARM,
            spec.HYBRID_ARM,
        })
        self.assertEqual(set(spec.CVAR_ARMS), {
            spec.CVAR_PROJECTION_ARM,
            spec.HYBRID_ARM,
        })
        self.assertTrue(all(
            arm["upper_constraint_mode"] == "primal_dual"
            for arm in (spec.ANCHOR_SPEC, *spec.ARMS.values())
        ))
        self.assertTrue(all(
            not arm["deployment_frequency_restoration_freeze_reward_actor"]
            for arm in spec.ARMS.values()
        ))
        self.assertEqual(
            spec.expected_closed_loop_guard_constraint_count(spec.HYBRID_ARM),
            24,
        )
        self.assertEqual(
            spec.expected_closed_loop_guard_constraint_count(
                spec.V14_16_COMPARATOR_ARM
            ),
            96,
        )
        self.assertIn(
            "mode_cvar",
            spec.expected_closed_loop_guard_contract(spec.HYBRID_ARM),
        )

    def test_launcher_builds_dynamic_unpinned_one_core_matrix(self):
        args = self._args()
        self.assertEqual(len(selected_experiment_cells(args)), 72)
        scheduler_spec = build_scheduler_spec(
            args,
            phase="continuation",
            environment="HalfCheetah-v5",
            arm=spec.HYBRID_ARM,
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
        )
        self.assertIsNone(scheduler_spec["require_node"])
        self.assertEqual(scheduler_spec["cpu"], 1)
        self.assertEqual(set(scheduler_spec["allowed_nodes"]), set(args.nodes))
        self.assertTrue(scheduler_spec["reroute_on_node_down"])

    def test_commands_isolate_native_pd_cvar_and_hybrid(self):
        args = self._args()
        common = dict(
            args=args,
            environment="HalfCheetah-v5",
            optimizer_seed=spec.OPTIMIZER_SEEDS[0],
            output_dir=Path("results/unit"),
            phase="continuation",
        )
        commands = {
            arm: build_training_command(arm=arm, **common)
            for arm in (
                spec.MATCHED_COMPARATOR_ARM,
                spec.V14_16_COMPARATOR_ARM,
                spec.NATIVE_PD_ARM,
                spec.CVAR_PROJECTION_ARM,
                spec.HYBRID_ARM,
            )
        }
        for command in commands.values():
            self.assertIn(
                f"--control-protocol-version {spec.FROZEN_CORE_PROTOCOL_VERSION}",
                command,
            )
            self.assertNotIn(
                "--deployment-frequency-restoration-freeze-reward-actor",
                command,
            )
        native = commands[spec.NATIVE_PD_ARM]
        self.assertIn("--constraint-dual-normalization ema_abs", native)
        self.assertIn(f"--upper-dual-lr {spec.NATIVE_DUAL_LR}", native)
        self.assertNotIn("--deployment-frequency-groupwise-robust", native)

        cvar = commands[spec.CVAR_PROJECTION_ARM]
        self.assertIn(
            "--deployment-frequency-projection-objective violation_cvar",
            cvar,
        )
        self.assertIn(
            f"--deployment-frequency-projection-cvar-alpha {spec.PROJECTION_CVAR_ALPHA}",
            cvar,
        )
        self.assertIn(
            "--deployment-frequency-closed-loop-risk-mode mode_cvar", cvar
        )
        self.assertIn("--constraint-dual-normalization none", cvar)

        hybrid = commands[spec.HYBRID_ARM]
        self.assertIn("--constraint-dual-normalization ema_abs", hybrid)
        self.assertIn(
            "--deployment-frequency-projection-objective violation_cvar",
            hybrid,
        )
        legacy = commands[spec.V14_16_COMPARATOR_ARM]
        self.assertIn("--deployment-frequency-pathwise-robust", legacy)
        self.assertIn(
            "--deployment-frequency-projection-objective violation_l2",
            legacy,
        )

    def test_engineering_gate_is_arm_specific(self):
        common = {
            "protocol_version": spec.FROZEN_CORE_PROTOCOL_VERSION,
            "protocol_version_selection": spec.FROZEN_CORE_PROTOCOL_VERSION,
            "selected_checkpoint_iteration": (
                spec.ANALYSIS_TRAINED_CHECKPOINT_MINIMUM_ITERATION
            ),
            "deployment_frequency_restoration_freeze_reward_actor": False,
        }
        native = {
            **common,
            "constraint_dual_normalization": "ema_abs",
            "upper_constraint_violation_scale_final": 0.7,
            "lower_constraint_violation_scale_final": 0.01,
            "upper_constraint_lambda_final": 0.5,
            "lower_constraint_lambda_final": 0.4,
        }
        self.assertTrue(_engineering_gate(
            native, arm=spec.NATIVE_PD_ARM
        )["pass"])

        hybrid = {
            **native,
            "deployment_frequency_projection_objective": "violation_cvar",
            "deployment_frequency_closed_loop_risk_mode": "mode_cvar",
            "deployment_frequency_closed_loop_cvar_alpha": (
                spec.CLOSED_LOOP_CVAR_ALPHA
            ),
            "deployment_frequency_closed_loop_guard_contract": (
                spec.expected_closed_loop_guard_contract(spec.HYBRID_ARM)
            ),
            "deployment_frequency_closed_loop_guard_effective_update_count": 1,
            "deployment_frequency_closed_loop_guard_selected_reward_"
            "violation_count": 0,
            "deployment_frequency_closed_loop_guard_selected_frequency_"
            "violation_count": 0,
        }
        self.assertTrue(_engineering_gate(
            hybrid, arm=spec.HYBRID_ARM
        )["pass"])
        hybrid["lower_constraint_violation_scale_final"] = 0.0
        self.assertFalse(_engineering_gate(
            hybrid, arm=spec.HYBRID_ARM
        )["pass"])

    def test_analyzer_uses_paired_log_reductions(self):
        self.assertEqual(
            EXPECTED_MERGED_MANIFEST_STATUS,
            "development_screen_complete_unanalyzed",
        )

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


if __name__ == "__main__":
    unittest.main()
