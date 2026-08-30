from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from scripts.audit_lower_replay_allocation import audit_lower_replay_allocation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs_freqduet"
    / "F_freqduet_protocol_v6_w2adregret_l001_e25_r0010_hiro.yaml"
)
FEATURES = [
    "load",
    "capacity",
    "queue",
    "speed_residual",
    "shock_age",
    "schedule_slack",
    "regularity_hold_target_norm",
    "regularity_hold_target_valid",
]


class LowerReplayAllocationAuditTest(unittest.TestCase):
    def _checkpoint(self, path: Path, *, valid_index: int = 16) -> Path:
        base_dim = 9
        rows = []
        specs = [
            (0.20, 0.80, 0.25, 1.0 / 3.0, 1.0, 10.0),
            (0.50, 0.50, 0.50, 2.0 / 3.0, 1.0, 20.0),
            (0.80, 0.20, 0.75, 1.0, 1.0, 45.0),
            (0.20, 0.80, 0.10, 1.0, 0.0, 45.0),
        ]
        for load, capacity, queue, target_norm, valid, action in specs:
            state = np.zeros(base_dim + len(FEATURES), dtype=np.float32)
            state[0] = 1.0
            context = {
                "load": load,
                "capacity": capacity,
                "queue": queue,
                "regularity_hold_target_norm": target_norm,
                "regularity_hold_target_valid": valid,
            }
            for name, value in context.items():
                state[base_dim + FEATURES.index(name)] = value
            rows.append(
                (
                    state,
                    np.asarray([action], dtype=np.float32),
                    0.0,
                    0.0,
                    state.copy(),
                    0.0,
                    1,
                )
            )
        payload = {
            "format": "freqduet-exact-training-state-v4",
            "episode": 39,
            "lower_trainer": {
                "regularity_policy_contract": {
                    "enabled": True,
                    "mode": "analytic_two_sided_zero_hold_regret_dual_v2",
                    "target_feature_index": base_dim + FEATURES.index(
                        "regularity_hold_target_norm"
                    ),
                    "valid_feature_index": valid_index,
                    "target_headway_feature_index": 0,
                    "action_target_scale_s": 45.0,
                    "target_headway_scale_s": 600.0,
                    "cost_cap": 0.25,
                }
            },
            "lower_replay_buffer": {"buffer": rows},
        }
        torch.save(payload, path)
        return path

    def test_reports_valid_joint_action_allocation(self):
        with TemporaryDirectory() as tmp:
            result = audit_lower_replay_allocation(
                self._checkpoint(Path(tmp) / "training.pt"), CONFIG
            )

        self.assertEqual(result["replay_transitions"], 4)
        self.assertEqual(result["valid_transitions"], 3)
        self.assertEqual(result["base_state_dim"], 9)
        self.assertAlmostEqual(result["valid_overall"]["action_mean_s"], 25.0)
        self.assertAlmostEqual(
            result["valid_by_load"]["low_0_033"]["action_mean_s"], 10.0
        )
        high_target_high_load = next(
            row
            for row in result["valid_by_target_and_load"]
            if row["target_band"] == "high_30_plus"
            and row["load_band"] == "high_067_plus"
        )
        self.assertEqual(high_target_high_load["count"], 1)
        self.assertAlmostEqual(high_target_high_load["zero_hold_regret_mean"], 0.0)

    def test_rejects_config_checkpoint_context_mismatch(self):
        with TemporaryDirectory() as tmp:
            checkpoint = self._checkpoint(
                Path(tmp) / "training.pt", valid_index=15
            )
            with self.assertRaisesRegex(ValueError, "context order"):
                audit_lower_replay_allocation(checkpoint, CONFIG)


if __name__ == "__main__":
    unittest.main()
