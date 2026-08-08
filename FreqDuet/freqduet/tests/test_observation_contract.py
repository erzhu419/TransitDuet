import unittest

from lower.observation_contract import LowerObservationContract


class LowerObservationContractTest(unittest.TestCase):
    def _deployable(self, **overrides):
        values = {
            "mode": "deployable_apc_avl_v4",
            "input_schema": "causal_forward_v4",
            "reward_mode": "forward_event_only",
            "unobserved_action_mode": "zero",
            "frequency_enabled": True,
            "frequency_source": "apc_boardings",
            "context_features": [
                "load", "capacity", "queue", "speed_residual",
                "shock_age", "schedule_slack", "causal_hold_limit",
            ],
        }
        values.update(overrides)
        return LowerObservationContract.create(**values)

    def test_deployable_ledger_contains_no_latent_feature(self):
        contract = self._deployable()

        rows = contract.ledger()

        self.assertTrue(rows)
        self.assertTrue(all(row["deployable"] for row in rows))
        self.assertTrue(all("latent" not in row["source"] for row in rows))
        self.assertEqual(len(contract.fingerprint), 64)
        limit_row = next(
            row for row in rows if row["feature"] == "causal_hold_limit")
        self.assertIn("predecessor departure", limit_row["source"])

    def test_rejects_stale_follower_and_true_neighbor_queues(self):
        with self.assertRaisesRegex(ValueError, "rejects context"):
            self._deployable(context_features=[
                "load", "bwd_headway_norm", "next_queue"])

    def test_rejects_latent_frequency_observation(self):
        with self.assertRaisesRegex(ValueError, "APC frequency"):
            self._deployable(frequency_source="latent_arrivals")


if __name__ == "__main__":
    unittest.main()
