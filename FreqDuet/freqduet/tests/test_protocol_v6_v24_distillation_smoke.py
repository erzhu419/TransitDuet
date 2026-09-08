import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_v24_distillation_smoke import (
    CANDIDATES,
    CANDIDATE_SPECS,
    EVAL_SEEDS,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
    evaluate_v24_distillation_smoke,
)


class ProtocolV6V24DistillationSmokeTest(unittest.TestCase):
    @staticmethod
    def _training_row(
        candidate: str,
        ep: int,
        *,
        post_kl: float = 0.20,
        target_cost: float = 0.036,
    ) -> dict[str, object]:
        distillation, steps = CANDIDATE_SPECS[candidate]
        return {
            "ep": ep,
            "lower_discrete_critic": "zero_hold_advantage",
            "lower_regularity_policy_mode": (
                "analytic_two_sided_hf_aggregate_gain_projection_v11"),
            "lower_regularity_projection_mode": (
                "joint_kl_soft_policy_distillation_v2"),
            "lower_regularity_projection_distillation": distillation,
            "lower_regularity_projection_distillation_steps": steps,
            "lower_regularity_projection_applied": 1.0,
            "lower_regularity_projection_converged": 1.0,
            "lower_regularity_projection_iterations": 5.0,
            "lower_regularity_projection_valid_count": 128.0,
            "lower_regularity_projection_regularity_target": 0.036,
            "lower_regularity_projection_passenger_target": 0.075,
            "lower_regularity_projection_target_regularity_cost": target_cost,
            "lower_regularity_projection_target_passenger_cost": 0.075,
            "lower_regularity_projection_max_constraint_violation": 0.0,
            "lower_regularity_projection_actor_distillation_steps": steps,
            "lower_regularity_projection_actor_reverse_kl_episode_mean": 0.40,
            "lower_regularity_projection_actor_forward_kl_episode_mean": 0.40,
            "lower_regularity_projection_actor_post_reverse_kl_episode_mean": (
                post_kl),
            "lower_regularity_projection_actor_post_forward_kl_episode_mean": (
                post_kl),
            "lower_regularity_projection_actor_post_regularity_cost_episode_mean": (
                0.049),
            "lower_regularity_projection_actor_post_passenger_cost_episode_mean": (
                0.079),
            "lower_regularity_projection_target_action_change_abs_mean_s_episode_mean": (
                1.0),
            "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean": (
                0.5),
            "lower_regularity_lambda": 0.0,
            "lower_regularity_passenger_lambda": 0.0,
            "lower_regularity_policy_penalty": 0.0,
            "lower_regularity_policy_augmented_penalty": 0.0,
            "lower_regularity_passenger_actor_penalty": 0.0,
            "lower_regularity_passenger_actor_augmented_penalty": 0.0,
            "lower_causal_guard_adjustment_mean_s": 0.0,
        }

    def _write_fixture(
        self,
        root: Path,
        *,
        bad_kl_candidate: str | None = None,
        target_cost: float = 0.036,
    ) -> tuple[Path, Path]:
        aggregate = root / "aggregate"
        logs = root / "logs"
        aggregate.mkdir()
        logs.mkdir()
        git_record = {
            "commit": "a" * 40,
            "branch": "codex/freqduet-v6-causal-protocol",
            "tracked_dirty": False,
        }
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": CANDIDATES,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": TRAIN_EPISODES - 1,
            "reference": CANDIDATES[0],
            "run_git_provenance": git_record,
            "git": git_record,
            "expected_rollouts": len(CANDIDATES),
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))
        pd.DataFrame([{
            "config": candidate,
            "train_seed": TRAIN_SEEDS[0],
            "eval_seed": EVAL_SEEDS[0],
            "lower_policy_frozen": 1.0,
            "lower_critic_frozen": 1.0,
            "upper_policy_frozen": 1.0,
            "lower_regularity_projection_applied": 0.0,
            "lower_causal_guard_adjustment_mean_s": 0.0,
            "lower_regularity_policy_evidence_valid_mean": 0.70,
        } for candidate in CANDIDATES]).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)

        for candidate in CANDIDATES:
            run = logs / f"{candidate}_seed{TRAIN_SEEDS[0]}"
            run.mkdir()
            rows = []
            for ep in range(TRAIN_EPISODES):
                post_kl = (
                    0.50
                    if candidate == bad_kl_candidate and ep == 1
                    else 0.20)
                rows.append(self._training_row(
                    candidate,
                    ep,
                    post_kl=post_kl,
                    target_cost=target_cost,
                ))
            pd.DataFrame(rows).to_csv(
                run / "diagnostics.csv", index=False)
        return aggregate, logs

    def test_passes_only_as_non_effect_mechanical_evidence(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            result = evaluate_v24_distillation_smoke(aggregate, logs)

        self.assertEqual(result["status"], "mechanical_pass")
        self.assertTrue(result["formal_screen_authorized"])
        self.assertFalse(result["effect_evidence"])
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertTrue(all(result["evaluation_checks"].values()))
        self.assertTrue(all(
            all(checks.values())
            for checks in result["candidate_checks"].values()))

    def test_post_update_kl_regression_blocks_formal_screen(self):
        candidate = CANDIDATES[2]
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), bad_kl_candidate=candidate)
            result = evaluate_v24_distillation_smoke(aggregate, logs)

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["formal_screen_authorized"])
        self.assertFalse(result["candidate_checks"][candidate][
            "configured_kl_descends_each_episode"])

    def test_teacher_target_violation_blocks_formal_screen(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), target_cost=0.03600002)
            result = evaluate_v24_distillation_smoke(aggregate, logs)

        self.assertFalse(result["formal_screen_authorized"])
        self.assertTrue(all(
            not checks["every_teacher_projection_meets_targets"]
            for checks in result["candidate_checks"].values()))

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            manifest_path = aggregate / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["eval_seeds"] = [1]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v24_distillation_smoke(aggregate, logs)


if __name__ == "__main__":
    unittest.main()
