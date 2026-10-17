from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_freqduet_v96_frozen_compensation as study


class FrozenCompensationTest(unittest.TestCase):
    def test_one_fixed_fresh_grid_and_unchanged_contrasts(self):
        self.assertEqual(len(study.DAYS), 32)
        self.assertFalse(set(study.DAYS) & set(study.confirmation.DAYS))
        self.assertFalse(set(study.DAYS) & set(study.frozen.DAYS))
        self.assertEqual(len(study.ITEMS) * len(study.MODES), 2880)
        self.assertEqual(study.CONTRASTS, study.prior.CONTRASTS)

    def test_collector_delegates_unchanged_runtime_root_and_isolates_artifacts(self):
        root, artifacts = Path("/results/v96"), Path("/results/v96/smoke_runs")
        contexts = {g: dict(expected_holding_reduction_s=2., count=1) for g in study.frozen.GROUPS}
        rule = SimpleNamespace(save=Mock(), group_summary=contexts)
        rows = dict.fromkeys(study.frozen.METRICS, 0.)
        replay = (rows, {}, [], rule)
        with patch.object(study.frozen, "replay", return_value=replay) as native, patch.object(
                study.prior, "replay", return_value=replay) as uniform, patch.object(
                study.confirmation, "timing_effect", return_value={}), patch.object(study.pilot, "write_json") as write:
            study.collect(root, 12037, 1941001, 0, 7, .5, artifact_root=artifacts)
        self.assertEqual(native.call_args_list[0].args, (root, 12037, 1941001, 0, "late"))
        self.assertEqual(native.call_args_list[1].args, (root, 12037, 1941001, 0, "preserve"))
        uniform.assert_called_once_with(root, 12037, 1941001, 0, .5)
        self.assertEqual(write.call_args.args[0], artifacts / "pairs/p7.json")
        self.assertTrue(all(c.args[0].parent == artifacts / "records" for c in rule.save.call_args_list))

    def test_axis_review_averages_samplers_before_seed_and_day_means(self):
        rows = [dict(seed=s, day=d, effects=dict(comparison=dict(metric=v)))
            for s, d, values in ((1, 10, [-3., -1.]), (1, 20, [4.]), (2, 10, [0.]), (2, 20, [2.]))
            for v in values]
        stats = study.axis_summary(rows, "effects", ("metric",))["comparison"]
        self.assertEqual(stats["seed"]["means"], {"1": dict(metric=1.), "2": dict(metric=1.)})
        self.assertEqual(stats["day"]["means"], {"10": dict(metric=-1.), "20": dict(metric=3.)})
        self.assertEqual(stats["day"]["negative_counts"]["metric"], 1)

    def test_cv_only_gain_does_not_pass_direct_joint_gate(self):
        metrics = {m: dict(ci95=[-1., 0.]) for m in
            ("passenger_person_s", "cv_squared", "readiness_delay_s", "trips_unlaunched", "trips_incomplete")}
        metrics["cv_squared"]["ci95"] = [-.002, -.001]
        self.assertFalse(study.confirmation.joint_gate(dict(preserve_vs_late=metrics)))
        metrics["passenger_person_s"]["ci95"][1] = -.1
        metrics["readiness_delay_s"]["ci95"][1] = .01
        self.assertFalse(study.confirmation.joint_gate(dict(preserve_vs_late=metrics)))


if __name__ == "__main__":
    unittest.main()
