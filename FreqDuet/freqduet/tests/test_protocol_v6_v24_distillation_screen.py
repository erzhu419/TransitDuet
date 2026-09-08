import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_v24_distillation_screen import (
    CANDIDATES,
    CANDIDATE_SPECS,
    CONFIGS,
    EVAL_SEEDS,
    TRAIN_EPISODES,
    TRAIN_SEEDS,
    V13_ANCHOR,
    V23_CANDIDATE,
    evaluate_v24_distillation_screen,
)
from scripts.audit_protocol_v6_aggregate_gain_screen import (
    V13_ZERO_HOLD_ADVANTAGE,
    V20_QADV_B080,
)
from scripts.audit_protocol_v6_capacity_gain_screen import (
    CURRENT_MAIN,
    HARD_MAIN,
    REFERENCE,
)


class ProtocolV6V24DistillationScreenTest(unittest.TestCase):
    @staticmethod
    def _evaluation_row(config: str) -> dict[str, object]:
        candidate = config in CANDIDATES
        row = {
            "lower_discrete_critic": (
                "zero_hold_advantage" if candidate else "continuous_action"),
            "lower_policy_frozen": 1.0,
            "lower_critic_frozen": 1.0,
            "upper_policy_frozen": 1.0,
            "lower_causal_guard_adjustment_mean_s": 0.0,
            "lower_regularity_policy_evidence_valid_mean": (
                0.70 if candidate else 0.0),
            "lower_regularity_policy_mode": (
                "analytic_two_sided_hf_aggregate_gain_projection_v11"
                if candidate else "disabled"),
            "lower_regularity_projection_enabled": float(candidate),
            "lower_regularity_projection_mode": (
                "joint_kl_soft_policy_distillation_v2"
                if candidate else "disabled"),
            "lower_regularity_projection_applied": 0.0,
            "lower_regularity_projection_regularity_target": (
                0.036 if candidate else 0.0),
            "lower_regularity_projection_passenger_target": (
                0.075 if candidate else 0.0),
            "lower_regularity_gain_floor_enabled": float(candidate),
            "lower_regularity_gain_floor_mode": (
                "causal_hf_aggregate_gain_floor_v2"
                if candidate else "disabled"),
            "lower_regularity_gain_floor_base_fraction": (
                0.30 if candidate else 0.0),
            "lower_regularity_gain_floor_hf_increment": (
                0.30 if candidate else 0.0),
            "lower_regularity_gain_floor_required_gain_mean": (
                0.002 if candidate else 0.0),
            "lower_regularity_gain_floor_expected_absolute_shortfall_mean": (
                0.00008 if candidate else 0.0),
            "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio": (
                0.04 if candidate else 0.0),
            "lower_regularity_passenger_holding_enabled": float(candidate),
            "lower_regularity_passenger_holding_mode": (
                "causal_apc_person_delay_dual_v1"
                if candidate else "disabled"),
            "lower_regularity_passenger_cost_limit": (
                0.08 if candidate else 0.0),
            "lower_regularity_passenger_expected_cost_mean": (
                0.07 if candidate else 0.0),
            "lower_regularity_lambda": 0.0,
            "lower_regularity_passenger_lambda": 0.0,
            "lower_regularity_policy_penalty": 0.0,
            "lower_regularity_policy_augmented_penalty": 0.0,
            "lower_regularity_passenger_actor_penalty": 0.0,
            "lower_regularity_passenger_actor_augmented_penalty": 0.0,
        }
        return row

    @staticmethod
    def _training_row(
        config: str,
        ep: int,
        *,
        post_kl: float,
        action_gap: float,
    ) -> dict[str, object]:
        if config == V23_CANDIDATE:
            distillation, steps = "reverse_kl_v1", 1
            policy_mode = (
                "analytic_two_sided_hf_aggregate_gain_projection_v10")
            projection_mode = "joint_kl_soft_policy_target_v1"
        else:
            distillation, steps = CANDIDATE_SPECS[config]
            policy_mode = (
                "analytic_two_sided_hf_aggregate_gain_projection_v11")
            projection_mode = "joint_kl_soft_policy_distillation_v2"
        return {
            "ep": ep,
            "lower_regularity_policy_mode": policy_mode,
            "lower_regularity_projection_mode": projection_mode,
            "lower_regularity_projection_distillation": distillation,
            "lower_regularity_projection_distillation_steps": steps,
            "lower_regularity_projection_applied": 1.0,
            "lower_regularity_projection_converged": 1.0,
            "lower_regularity_projection_iterations": 5.0,
            "lower_regularity_projection_valid_count": 128.0,
            "lower_regularity_projection_regularity_target": 0.036,
            "lower_regularity_projection_passenger_target": 0.075,
            "lower_regularity_projection_target_regularity_cost": 0.036,
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
            "lower_regularity_projection_actor_post_target_action_change_abs_mean_s_episode_mean": (
                action_gap),
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
        bad_kl_candidates: set[str] | None = None,
        action_gap: float = 0.50,
    ) -> tuple[Path, Path]:
        bad_kl_candidates = bad_kl_candidates or set()
        aggregate = root / "aggregate"
        logs = root / "logs"
        aggregate.mkdir()
        logs.mkdir()
        git_record = {
            "commit": "a" * 40,
            "branch": "codex/freqduet-v6-causal-protocol",
            "tracked_dirty": False,
        }
        expected_pairs = len(TRAIN_SEEDS) * len(EVAL_SEEDS)
        manifest = {
            "strict_complete": True,
            "run_manifests_verified": True,
            "common_random_numbers_verified": True,
            "stage": "exploratory",
            "independent_confirmation": False,
            "configs": CONFIGS,
            "train_seeds": TRAIN_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "train_episodes": TRAIN_EPISODES,
            "checkpoint_ep": TRAIN_EPISODES - 1,
            "reference": V13_ANCHOR,
            "run_git_provenance": git_record,
            "git": git_record,
            "expected_rollouts": len(CONFIGS) * expected_pairs,
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))
        (aggregate / "frozen_summary.csv").write_text("config\n")

        values = {
            HARD_MAIN: (20.2, 0.210, 11.0, 115.0, 11.0),
            CURRENT_MAIN: (20.5, 0.202, 11.0, 115.0, 11.0),
            REFERENCE: (21.0, 0.250, 12.0, 120.0, 12.0),
            V13_ANCHOR: (20.0, 0.205, 10.0, 110.0, 10.0),
            V13_ZERO_HOLD_ADVANTAGE: (20.0, 0.203, 10.0, 110.0, 10.0),
            V20_QADV_B080: (19.65, 0.230, 8.0, 95.0, 8.0),
            V23_CANDIDATE: (19.55, 0.185, 9.0, 100.0, 9.0),
        }
        values.update({
            candidate: (19.30, 0.170, 9.0, 100.0, 9.0)
            for candidate in CANDIDATES
        })
        evaluation_rows = []
        for config in CONFIGS:
            journey, cv, action, holding, denied = values[config]
            for train_seed in TRAIN_SEEDS:
                for eval_seed in EVAL_SEEDS:
                    row = self._evaluation_row(config)
                    row.update({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "restricted_total_journey_horizon_min": journey,
                        "headway_cv": cv,
                        "lower_action_mean": action,
                        "holding_vehicle_seconds": holding,
                        "fleet_denied_dispatch_events": denied,
                    })
                    evaluation_rows.append(row)
        pd.DataFrame(evaluation_rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)
        pd.DataFrame([{
            "candidate": candidate,
            "reference": V13_ANCHOR,
            "n_pairs": expected_pairs,
        } for candidate in CANDIDATES]).to_csv(
            aggregate / "frozen_paired_deltas.csv", index=False)

        for config in [V23_CANDIDATE, *CANDIDATES]:
            for train_seed in TRAIN_SEEDS:
                run = logs / f"{config}_seed{train_seed}"
                run.mkdir()
                post_kl = (
                    0.32 if config == V23_CANDIDATE
                    else 0.35 if config in bad_kl_candidates
                    else 0.20)
                gap = 1.0 if config == V23_CANDIDATE else action_gap
                pd.DataFrame([
                    self._training_row(
                        config, ep, post_kl=post_kl, action_gap=gap)
                    for ep in range(TRAIN_EPISODES)
                ]).to_csv(run / "diagnostics.csv", index=False)
        return aggregate, logs

    def test_selects_first_complete_pass_by_registered_priority(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            result = evaluate_v24_distillation_screen(aggregate, [logs])

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], CANDIDATES[0])
        self.assertFalse(result["claim_eligible"])
        self.assertTrue(all(result["strict_checks"].values()))
        self.assertTrue(all(result["v23_control_checks"].values()))

    def test_failed_weaker_candidate_does_not_override_next_pass(self):
        first = CANDIDATES[0]
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), bad_kl_candidates={first})
            result = evaluate_v24_distillation_screen(aggregate, [logs])

        self.assertFalse(result["candidate_results"][first]["passes"])
        self.assertEqual(result["selected_for_confirmation"], CANDIDATES[1])

    def test_action_gap_failure_rejects_entire_factorial(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp), action_gap=0.80)
            result = evaluate_v24_distillation_screen(aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertIsNone(result["selected_for_confirmation"])
        self.assertTrue(all(
            not candidate["training_checks"][
                "late_action_gap_improves_v23"]
            for candidate in result["candidate_results"].values()))

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(Path(tmp))
            manifest_path = aggregate / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["train_seeds"] = [1, 2, 3, 4]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v24_distillation_screen(aggregate, [logs])


if __name__ == "__main__":
    unittest.main()
