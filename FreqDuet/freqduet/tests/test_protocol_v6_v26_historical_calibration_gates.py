import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_v26_historical_calibration_common import (
    CANDIDATES,
    CANDIDATE_SPECS,
    PRIORITY,
    V13_ANCHOR,
)
from scripts.audit_protocol_v6_v26_historical_calibration_screen import (
    CONFIGS as SCREEN_CONFIGS,
    EVAL_SEEDS as SCREEN_EVAL_SEEDS,
    TRAIN_EPISODES as SCREEN_TRAIN_EPISODES,
    TRAIN_SEEDS as SCREEN_TRAIN_SEEDS,
    evaluate_v26_historical_calibration_screen,
)
from scripts.audit_protocol_v6_v26_historical_calibration_smoke import (
    CONFIGS as SMOKE_CONFIGS,
    EVAL_SEEDS as SMOKE_EVAL_SEEDS,
    TRAIN_EPISODES as SMOKE_TRAIN_EPISODES,
    TRAIN_SEEDS as SMOKE_TRAIN_SEEDS,
    evaluate_v26_historical_calibration_smoke,
)


class ProtocolV6V26HistoricalCalibrationGateTest(unittest.TestCase):
    @staticmethod
    def _training_row(candidate: str, ep: int) -> dict[str, object]:
        spec = CANDIDATE_SPECS[candidate]
        active = ep >= 5
        return {
            "ep": ep,
            "follower_target_calibration_enabled": 1.0,
            "follower_target_calibration_mode": spec["mode"],
            "follower_target_calibration_post_update_active": float(ep >= 4),
            "follower_target_calibration_post_update_history_episodes": ep + 1,
            "follower_target_calibration_post_update_history_samples": (
                (ep + 1) * 200),
            "follower_target_calibration_post_update_coefficient_norm": 1.0,
            "follower_target_calibration_episode_samples": 200,
            "follower_target_calibration_episode_updated": 1.0,
            "follower_target_calibration_update_source": (
                "completed_learned_training_days_v1"),
            "follower_forecast_registered_count": 200,
            "follower_forecast_resolved_count": 200,
            "follower_forecast_resolution_rate": 1.0,
            "follower_forecast_calibration_active_mean": float(active),
            "follower_forecast_calibration_history_episodes_mean": float(ep),
            "follower_forecast_calibration_requested_adjustment_abs_mean_s": (
                2.0 if active else 0.0),
            "follower_forecast_calibration_requested_adjustment_abs_max_s": (
                min(5.0, spec["adjustment_cap_s"]) if active else 0.0),
            "follower_forecast_target_action_prediction_mae_s": 7.0,
            "follower_forecast_base_target_action_prediction_mae_s": 8.0,
        }

    @staticmethod
    def _evaluation_row(config: str, train_episodes: int) -> dict[str, object]:
        candidate = config in CANDIDATES
        spec = CANDIDATE_SPECS.get(config, {
            "mode": "disabled", "adjustment_cap_s": 0.0})
        target_mae = 7.0 if candidate else 8.0
        base_target_mae = 8.0
        predicted_gap = 101.0 if candidate else 100.0
        base_gap = 100.0
        error = 1.0 if candidate else 0.0
        base_error = 0.0
        return {
            "checkpoint_ep": train_episodes - 1,
            "lower_policy_frozen": 1.0,
            "lower_critic_frozen": 1.0,
            "upper_policy_frozen": 1.0,
            "follower_target_calibration_enabled": float(candidate),
            "follower_target_calibration_mode": spec["mode"],
            "follower_target_calibration_post_update_active": float(candidate),
            "follower_target_calibration_post_update_history_episodes": (
                train_episodes if candidate else 0),
            "follower_target_calibration_post_update_history_samples": (
                train_episodes * 200 if candidate else 0),
            "follower_target_calibration_post_update_coefficient_norm": (
                1.0 if candidate else 0.0),
            "follower_target_calibration_post_update_intercept_s": (
                1.0 if candidate else 0.0),
            "follower_target_calibration_episode_samples": 0,
            "follower_target_calibration_episode_updated": 0.0,
            "follower_target_calibration_update_source": (
                "completed_learned_training_days_v1"),
            "follower_forecast_registered_count": 200,
            "follower_forecast_resolved_count": 200,
            "follower_forecast_resolution_rate": 1.0,
            "follower_forecast_calibration_active_mean": float(candidate),
            "follower_forecast_calibration_history_episodes_mean": (
                float(train_episodes) if candidate else 0.0),
            "follower_forecast_calibration_requested_adjustment_abs_mean_s": (
                2.0 if candidate else 0.0),
            "follower_forecast_calibration_requested_adjustment_abs_max_s": (
                min(5.0, spec["adjustment_cap_s"]) if candidate else 0.0),
            "follower_forecast_calibration_target_adjustment_abs_mean_s": (
                1.0 if candidate else 0.0),
            "follower_forecast_target_action_prediction_mae_s": target_mae,
            "follower_forecast_base_target_action_prediction_mae_s": (
                base_target_mae),
            "follower_forecast_hold_need_false_positive_mean": 0.05,
            "follower_forecast_hold_need_false_negative_mean": 0.05,
            "follower_forecast_base_hold_need_false_positive_mean": (
                0.055 if candidate else 0.05),
            "follower_forecast_base_hold_need_false_negative_mean": (
                0.055 if candidate else 0.05),
            "follower_forecast_predicted_follower_gap_s_mean": predicted_gap,
            "follower_forecast_base_predicted_follower_gap_s_mean": base_gap,
            "follower_forecast_raw_gap_prediction_error_s_mean": error,
            "follower_forecast_base_gap_prediction_error_s_mean": base_error,
        }

    def _write_fixture(
        self,
        root: Path,
        *,
        configs: list[str],
        train_seeds: list[int],
        eval_seeds: list[int],
        train_episodes: int,
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
            "configs": configs,
            "train_seeds": train_seeds,
            "eval_seeds": eval_seeds,
            "train_episodes": train_episodes,
            "checkpoint_ep": train_episodes - 1,
            "reference": V13_ANCHOR,
            "run_git_provenance": git_record,
            "git": git_record,
            "expected_rollouts": (
                len(configs) * len(train_seeds) * len(eval_seeds)),
        }
        (aggregate / "matrix_manifest.json").write_text(json.dumps(manifest))
        (aggregate / "frozen_summary.csv").write_text("config\n")
        (aggregate / "frozen_paired_deltas.csv").write_text(
            "candidate,reference\n")

        evaluation_rows = []
        for config in configs:
            for train_seed in train_seeds:
                for eval_seed in eval_seeds:
                    row = self._evaluation_row(config, train_episodes)
                    candidate = config in CANDIDATES
                    row.update({
                        "config": config,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        "headway_cv": 0.195 if candidate else 0.200,
                        "restricted_total_journey_horizon_min": (
                            20.05 if candidate else 20.0),
                        "service_cost": 0.501 if candidate else 0.500,
                        "passenger_unserved_rate": (
                            0.011 if candidate else 0.010),
                        "lower_action_mean": 10.5 if candidate else 10.0,
                    })
                    evaluation_rows.append(row)
        pd.DataFrame(evaluation_rows).to_csv(
            aggregate / "frozen_per_eval.csv", index=False)

        for candidate in CANDIDATES:
            for train_seed in train_seeds:
                run = logs / f"{candidate}_seed{train_seed}"
                run.mkdir()
                pd.DataFrame([
                    self._training_row(candidate, ep)
                    for ep in range(train_episodes)
                ]).to_csv(run / "diagnostics.csv", index=False)
        return aggregate, logs

    def test_smoke_passes_only_as_mechanical_evidence(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SMOKE_CONFIGS,
                train_seeds=SMOKE_TRAIN_SEEDS,
                eval_seeds=SMOKE_EVAL_SEEDS,
                train_episodes=SMOKE_TRAIN_EPISODES,
            )
            result = evaluate_v26_historical_calibration_smoke(
                aggregate, [logs])

        self.assertEqual(result["status"], "mechanical_pass")
        self.assertTrue(result["formal_screen_authorized"])
        self.assertFalse(result["effect_evidence"])

    def test_same_day_activation_blocks_the_smoke(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SMOKE_CONFIGS,
                train_seeds=SMOKE_TRAIN_SEEDS,
                eval_seeds=SMOKE_EVAL_SEEDS,
                train_episodes=SMOKE_TRAIN_EPISODES,
            )
            candidate = CANDIDATES[0]
            path = logs / (
                f"{candidate}_seed{SMOKE_TRAIN_SEEDS[0]}") / "diagnostics.csv"
            frame = pd.read_csv(path)
            frame.loc[frame["ep"] == 4,
                      "follower_forecast_calibration_active_mean"] = 1.0
            frame.to_csv(path, index=False)
            result = evaluate_v26_historical_calibration_smoke(
                aggregate, [logs])

        self.assertEqual(result["status"], "no_pass")
        self.assertFalse(result["candidate_results"][candidate][
            "training_checks"][
                "used_generation_has_no_same_day_activation"])

    def test_screen_selects_registered_priority_candidate(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            result = evaluate_v26_historical_calibration_screen(
                aggregate, [logs])

        self.assertEqual(result["status"], "exploratory_candidate_selected")
        self.assertEqual(result["selected_for_confirmation"], PRIORITY[0])
        self.assertFalse(result["claim_eligible"])

    def test_forecast_failure_falls_through_to_next_priority(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            frame_path = aggregate / "frozen_per_eval.csv"
            frame = pd.read_csv(frame_path)
            failed = PRIORITY[0]
            mask = frame["config"] == failed
            frame.loc[mask,
                      "follower_forecast_target_action_prediction_mae_s"] = 8.0
            frame.to_csv(frame_path, index=False)
            result = evaluate_v26_historical_calibration_screen(
                aggregate, [logs])

        self.assertFalse(result["candidate_results"][failed]["passes"])
        self.assertEqual(result["selected_for_confirmation"], PRIORITY[1])

    def test_manifest_seed_drift_fails_closed(self):
        with TemporaryDirectory() as tmp:
            aggregate, logs = self._write_fixture(
                Path(tmp),
                configs=SCREEN_CONFIGS,
                train_seeds=SCREEN_TRAIN_SEEDS,
                eval_seeds=SCREEN_EVAL_SEEDS,
                train_episodes=SCREEN_TRAIN_EPISODES,
            )
            manifest_path = aggregate / "matrix_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["eval_seeds"] = [1, 2, 3, 4]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "strict checks failed"):
                evaluate_v26_historical_calibration_screen(
                    aggregate, [logs])


if __name__ == "__main__":
    unittest.main()
