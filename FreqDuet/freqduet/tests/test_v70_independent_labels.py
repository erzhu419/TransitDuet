from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v70_independent_labels as audit


def groups():
    panel = []
    for future in audit.FUTURES:
        rows = []
        for action in audit.joint.ACTIONS:
            row = {m: 0. for m in audit.repeated.RESPONSE_METRICS}
            row.update(action_s=action, conservation_max_error=0.,
                       alignment=None if action == 0 else dict(common_events=1, uniform_mismatches=0))
            rows.append(row)
        panel.append(dict(future_index=future, native_action_s=0, rows=rows, exact_prefix=True,
                          common_future_start=True, forced_native_verified=future == 0))
    return panel


class IndependentLabelsTest(unittest.TestCase):
    def test_new_days_balanced_windows_and_private_samplers(self):
        old = set(audit.labels.DEMAND_SEEDS) | set(audit.labels.expanded.TRAIN_SEEDS)
        old |= set(audit.labels.expanded.VALIDATION_SEEDS) | set(audit.labels.residual.VALIDATION_SEEDS)
        old |= set(audit.labels.shrink.VALIDATION_SEEDS)
        self.assertTrue(set(audit.DEMAND_SEEDS).isdisjoint(old))
        self.assertEqual((len(audit.REFERENCES), len(audit.CONTEXTS), len(audit.ITEMS)), (48, 144, 4608))
        self.assertEqual(len(audit.ITEMS) * 7, 32256)
        for reference in range(48):
            self.assertEqual([window for r, window in audit.CONTEXTS if r == reference], [0, 1, 2])
        self.assertEqual(len({audit.reference_spec(i)["sampler_seed"] for i in range(48)}), 48)
        self.assertEqual(audit.labels.DEMAND_SEEDS, tuple(range(670001, 670009)))

    def test_future_seeds_are_unique_and_independent_of_v68_v69(self):
        seeds = {audit.future_seed(c, f) for c, f in audit.ITEMS}
        old = {audit.repeated.future_seed(c, f) for c, f in audit.repeated.ITEMS}
        self.assertEqual(len(seeds), 4608)
        self.assertTrue(seeds.isdisjoint(old))

    def test_legacy_entrypoint_passes_unchanged_reference_to_shared_replay(self):
        with patch.object(audit.labels, "run_spec", return_value="replay") as replay:
            result = audit.labels.run_branch(Path("out"), Path("actors"), 3, (7200,), "example", 15, "continuation")
        self.assertEqual(result, "replay")
        replay.assert_called_once_with(Path("out"), Path("actors"), audit.labels.reference_spec(3),
                                       (7200,), "example", 15, "continuation")

    def test_smoke_spans_all_three_windows_and_two_original_training_seeds(self):
        specs = [audit.context_spec(audit.ITEMS[i][0]) for i in audit.SMOKE_ITEMS]
        self.assertEqual({s["target_time_s"] for s in specs}, set(audit.TARGETS))
        self.assertEqual(len({s["train_seed"] for s in specs}), 2)
        self.assertEqual(len({s["reference_index"] for s in specs}), 2)

    def test_selection_trains_on_16_futures_and_preserves_harmful_heldout_sign(self):
        panel = groups()
        for future, group in enumerate(panel):
            group["rows"][1]["passenger_person_s"] = -20. if future < 16 else 60.
        result = audit.analyze_context(audit.context_spec(0), panel)
        self.assertEqual(result["halves"][0]["action_s"], 5)
        self.assertEqual(result["halves"][0]["passenger_person_s"], 60)
        self.assertEqual(result["halves"][1]["action_s"], 0)
        self.assertEqual(result["cross_future"]["passenger_person_s"], 30)
        self.assertEqual(result["alignment"]["common_events"], 32 * 6)

    def test_all_four_constraints_prevent_selection_and_ties_keep_native(self):
        for metric in audit.repeated.CONSTRAINT_METRICS:
            panel = groups()
            for group in panel:
                group["rows"][1]["passenger_person_s"] = -20.
                group["rows"][1][metric] = 1.
            result = audit.analyze_context(audit.context_spec(0), panel)
            self.assertTrue(all(h["action_s"] == 0 for h in result["halves"]))
        result = audit.analyze_context(audit.context_spec(0), groups())
        self.assertTrue(all(h["action_s"] == 0 for h in result["halves"]))

    def test_signal_sign_uses_16_16_not_legacy_4_28(self):
        differences = np.zeros((32, 7))
        differences[:4, 1:] = -1.
        differences[4:, 1:] = 2.
        self.assertEqual(audit.repeated.signal_noise(differences)["half_sign_agreement"], 1.)
        with self.assertRaises(ValueError):
            audit.analyze_context(audit.context_spec(0), groups()[:-1])

    def test_intervals_cluster_12_days_and_mean32_noise_is_scaled_correctly(self):
        contexts = [dict(scenario_seed=day, value=float(i), signal_noise=dict(metric=dict(
            debiased_signal_squared=2., single_future_noise_variance=4., half_sign_agreement=.75)))
            for i, day in enumerate(audit.DEMAND_SEEDS) for _ in range(12)]
        interval = audit.interval(contexts, lambda c: c["value"])
        self.assertEqual((interval["n_demand_days"], interval["mean"]), (12, 5.5))
        signal = audit.signal_summary(contexts, "metric")
        self.assertEqual(signal["single_future_snr"], .5)
        self.assertEqual(signal["mean32_snr"], 16.)
        self.assertEqual(signal["mean32_snr_ci95"], [16., 16.])

    def test_positive_signal_does_not_waive_readiness_or_noise_uncertainty(self):
        signal = {m: dict(corrected_signal_squared=dict(ci95=[1., 2.]), mean32_snr_ci95=[1.1, 3.])
                  for m in ("passenger_person_s", "cv_squared")}
        effects = {m: dict(ci95=[-1., 0.]) for m in audit.repeated.CONSTRAINT_METRICS}
        effects["passenger_person_s"] = dict(ci95=[-20., -1.])
        self.assertTrue(all(audit.gate_decision(signal, effects).values()))
        effects["readiness_delay_s"]["ci95"][1] = .0001
        self.assertFalse(audit.gate_decision(signal, effects)["joint_opportunity_gate_passed"])
        signal["passenger_person_s"]["mean32_snr_ci95"][0] = .9
        self.assertFalse(audit.gate_decision(signal, effects)["independent_label_gate_passed"])


if __name__ == "__main__":
    unittest.main()
