import unittest

import pandas as pd

from scripts.calibrate_protocol_v6_v23_projection_margin import (
    calibrate_projection_margin,
)


AGGREGATE = "aggregate_candidate"
RELATIVE = "relative_candidate"
SPECS = [
    (AGGREGATE, "aggregate", "projected_violation_v1", 0.0, True),
    (RELATIVE, "relative", "log_adam_v1", 0.0, False),
]


def _projection_summary():
    rows = []
    for config, allocation, *_ in SPECS:
        rows.append({
            "config": config,
            "train_seed": 11,
            "constraint_cost_mode": (
                "hf_aggregate_gain_shortfall_v4"
                if allocation == "aggregate"
                else "hf_relative_gain_shortfall_v3"),
            "learned_regularity_cost": 0.02,
            "learned_passenger_cost": 0.04,
        })
    return {
        "schema": "freqduet-replay-joint-kl-projection-aggregate-v1",
        "all_joint_frontiers_feasible": True,
        "all_projections_converged": True,
        "batch_projection_supported": True,
        "rows": rows,
    }


def _frozen_rows():
    rows = []
    values = {
        AGGREGATE: [(0.0312, 0.0431), (0.0321, 0.0441)],
        RELATIVE: [(0.0250, 0.0410), (0.0260, 0.0420)],
    }
    for config, pairs in values.items():
        for eval_seed, (regularity, passenger) in zip((21, 22), pairs):
            rows.append({
                "config": config,
                "train_seed": 11,
                "eval_seed": eval_seed,
                "ep": 39,
                "checkpoint_ep": 39,
                "lower_policy_frozen": 1.0,
                "lower_critic_frozen": 1.0,
                "lower_regularity_gain_floor_expected_shortfall_mean": (
                    regularity),
                "lower_regularity_gain_floor_"
                "expected_aggregate_shortfall_ratio": regularity,
                "lower_regularity_passenger_expected_cost_mean": passenger,
            })
    return pd.DataFrame(rows)


class ProtocolV6V23ProjectionMarginTest(unittest.TestCase):
    def test_locks_selected_allocation_with_upward_rounding(self):
        result = calibrate_projection_margin(
            _projection_summary(),
            _frozen_rows(),
            specs=SPECS,
            train_seeds=[11],
            eval_seeds=[21, 22],
            selected_allocation="aggregate",
        )

        self.assertEqual(result["factorial_checkpoint_count"], 2)
        self.assertEqual(result["frozen_rollout_count"], 4)
        self.assertEqual(result["locked_regularity_replay_target"], 0.037)
        self.assertEqual(result["locked_passenger_replay_target"], 0.075)

    def test_rejects_missing_frozen_rollout(self):
        frozen = _frozen_rows().iloc[:-1].copy()
        with self.assertRaisesRegex(ValueError, "frozen inventory mismatch"):
            calibrate_projection_margin(
                _projection_summary(), frozen, specs=SPECS,
                train_seeds=[11], eval_seeds=[21, 22])

    def test_rejects_nonfrozen_policy(self):
        frozen = _frozen_rows()
        frozen.loc[0, "lower_policy_frozen"] = 0.0
        with self.assertRaisesRegex(ValueError, "lower_policy_frozen"):
            calibrate_projection_margin(
                _projection_summary(), frozen, specs=SPECS,
                train_seeds=[11], eval_seeds=[21, 22])


if __name__ == "__main__":
    unittest.main()
