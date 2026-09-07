from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from scripts.audit_protocol_v6_v23_projection_screen import (
    CANDIDATE,
    EVAL_SEEDS,
    TRAIN_SEEDS,
)
from scripts.diagnose_protocol_v6_v23_projection_transfer import (
    analyze_v23_projection_transfer,
)


class ProtocolV6V23ProjectionTransferTest(unittest.TestCase):
    def _write_fixture(
        self,
        root: Path,
        *,
        actor_regularity: float,
        frozen_regularity: float,
    ) -> tuple[Path, Path]:
        logs = root / "logs"
        logs.mkdir()
        for train_seed in TRAIN_SEEDS:
            run = logs / f"{CANDIDATE}_seed{train_seed}"
            run.mkdir()
            rows = []
            for ep in range(40):
                rows.append({
                    "ep": ep,
                    "lower_regularity_policy_cost_mean": actor_regularity,
                    "lower_regularity_passenger_actor_cost_mean": 0.070,
                    "lower_regularity_projection_applied": 1.0,
                    "lower_regularity_projection_target_regularity_cost": 0.036,
                    "lower_regularity_projection_target_passenger_cost": 0.075,
                    "lower_regularity_projection_actor_reverse_kl": 0.20,
                    "lower_regularity_projection_base_action_mean_s": 10.0,
                    "lower_regularity_projection_target_action_mean_s": 7.0,
                    "lower_regularity_projection_target_action_change_mean_s": 1.5,
                })
            pd.DataFrame(rows).to_csv(run / "diagnostics.csv", index=False)

        frozen_path = root / "frozen_per_eval.csv"
        rows = []
        for train_seed in TRAIN_SEEDS:
            for eval_seed in EVAL_SEEDS:
                rows.append({
                    "config": CANDIDATE,
                    "train_seed": train_seed,
                    "eval_seed": eval_seed,
                    "headway_cv": 0.27,
                    "restricted_total_journey_horizon_min": 16.7,
                    "lower_action_mean": 5.7,
                    "lower_regularity_gain_floor_expected_aggregate_shortfall_ratio": frozen_regularity,
                    "lower_regularity_passenger_expected_cost_mean": 0.075,
                })
        pd.DataFrame(rows).to_csv(frozen_path, index=False)
        return logs, frozen_path

    def test_identifies_actor_teacher_distillation_gap(self):
        with TemporaryDirectory() as tmp:
            logs, frozen = self._write_fixture(
                Path(tmp), actor_regularity=0.060,
                frozen_regularity=0.080)
            result = analyze_v23_projection_transfer(logs, frozen)

        aggregate = result["aggregate"]
        self.assertTrue(aggregate["teacher_target_exact"])
        self.assertEqual(aggregate["actor_gap_seed_count"], 4)
        self.assertEqual(
            aggregate["primary_diagnosis"],
            "actor_teacher_distillation_gap_observed",
        )
        self.assertAlmostEqual(
            aggregate["actor_regularity_excess_last10_mean"], 0.024)

    def test_separates_frozen_distribution_transfer_gap(self):
        with TemporaryDirectory() as tmp:
            logs, frozen = self._write_fixture(
                Path(tmp), actor_regularity=0.036,
                frozen_regularity=0.080)
            result = analyze_v23_projection_transfer(logs, frozen)

        aggregate = result["aggregate"]
        self.assertEqual(aggregate["actor_gap_seed_count"], 0)
        self.assertEqual(aggregate["rollout_gap_seed_count"], 4)
        self.assertEqual(
            aggregate["primary_diagnosis"],
            "frozen_distribution_transfer_gap_observed",
        )

    def test_rejects_incomplete_training_history(self):
        with TemporaryDirectory() as tmp:
            logs, frozen = self._write_fixture(
                Path(tmp), actor_regularity=0.060,
                frozen_regularity=0.080)
            path = logs / f"{CANDIDATE}_seed{TRAIN_SEEDS[0]}" / "diagnostics.csv"
            frame = pd.read_csv(path).iloc[:-1]
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "episodes 0--39"):
                analyze_v23_projection_transfer(logs, frozen)


if __name__ == "__main__":
    unittest.main()
