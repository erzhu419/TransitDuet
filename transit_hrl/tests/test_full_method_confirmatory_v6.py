import unittest

from freq_hrl.experiments.trading import full_method_confirmatory_v6 as confirm
from freq_hrl.experiments.trading import full_method_hpo_v6 as hpo


class FullMethodConfirmatoryV6Test(unittest.TestCase):
    def test_exact_paired_randomization_uses_training_replicates(self):
        values = [1.0] * 12
        self.assertAlmostEqual(
            confirm.paired_randomization_p(values),
            2.0 / (2 ** 12),
        )
        self.assertEqual(confirm.paired_randomization_p([0.0] * 12), 1.0)

    def test_holm_adjustment_is_monotone_within_metric_family(self):
        rows = [
            {"metric": "return", "p_value_raw": value}
            for value in (0.001, 0.01, 0.04)
        ]
        confirm.holm_adjust(rows)
        adjusted = [row["p_value_holm"] for row in rows]
        self.assertEqual(adjusted, [0.003, 0.02, 0.04])

    def test_paired_effects_average_paths_before_training_replicate_inference(self):
        replicates = list(confirm.DEFAULT_CONFIRMATORY_REPLICATES[:12])
        heldout = [31415, 27182]
        rows = []
        for variant_id in hpo.ALL_VARIANT_IDS:
            is_full = variant_id == hpo.ABLATION_PARENT_VARIANT
            for replicate in replicates:
                for scenario in confirm.EVALUATION_SCENARIOS:
                    for seed in heldout:
                        rows.append({
                            "variant_id": variant_id,
                            "training_replicate_seed": str(replicate),
                            "scenario": scenario,
                            "seed": str(seed),
                            "total_return": "2.0" if is_full else "1.0",
                            "sharpe": "2.0" if is_full else "1.0",
                            "max_drawdown": "0.1" if is_full else "0.2",
                            "turnover": "1.0" if is_full else "2.0",
                            "LowerLFDriftAbs": "0.1" if is_full else "0.2",
                        })
        effects = confirm._paired_effect_rows(
            rows,
            training_replicates=replicates,
            heldout_seeds=heldout,
        )
        self.assertEqual(
            len(effects),
            (len(hpo.ALL_VARIANT_IDS) - 1)
            * len(confirm.EVALUATION_SCENARIOS)
            * len(confirm.PRIMARY_METRICS),
        )
        for row in effects:
            self.assertEqual(row["inferential_unit"], "training_replicate")
            self.assertEqual(row["heldout_paths_per_replicate"], 2)
            self.assertGreater(row["directional_improvement_mean"], 0.0)
            self.assertEqual(row["claim_status"], "supported_improvement")

    def test_confirmatory_roles_reject_hpo_replicate_and_development_seed(self):
        frozen = {
            "training_replicate_seeds": [2026, 2039, 2053],
            "rollout_seed_roots": [42],
            "checkpoint_validation_seeds": [57721],
            "tuning_validation_seeds": [68207],
        }
        with self.assertRaisesRegex(ValueError, "overlaps HPO"):
            confirm._validate_confirmatory_roles(
                frozen,
                training_replicate_seed=2026,
                heldout_seeds=[31415],
            )
        with self.assertRaisesRegex(ValueError, "overlap development"):
            confirm._validate_confirmatory_roles(
                frozen,
                training_replicate_seed=7001,
                heldout_seeds=[42],
            )


if __name__ == "__main__":
    unittest.main()
