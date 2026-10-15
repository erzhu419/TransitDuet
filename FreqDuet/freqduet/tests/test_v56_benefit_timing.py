from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_freqduet_v37_joint_lower import OUTCOMES
from run_freqduet_v56_benefit_timing import common_cuts, exposure_at, reversal_rows, timing_rows, unique_branches, validate_snapshot


def trace(second_return=3000, change=0):
    decisions = {"first": dict(identity=dict(time_s=1000, trip_id=1)),
                 "second": dict(identity=dict(time_s=1500, trip_id=2))}
    timeline = np.array([[1000, 10, 20, 0], [1900, 20 + change, 40, 0],
                         [2800, 40 + change, 80, 0], [3000, 50 + change, 100, 0],
                         [4000, 80 + change, 160, 0]])
    return dict(snapshot=dict(decisions=decisions, episode=dict(done_reason="service_cleared",
                trips_unlaunched=0, trips_incomplete=0, passengers_unserved=0)), timeline=timeline,
                terminals=pd.DataFrame([dict(trip_id=1, time_s=2900), dict(trip_id=2, time_s=second_return)]),
                holds=pd.DataFrame([dict(time_s=1000, action_s=15)]),
                launches=pd.DataFrame([dict(time_s=500, readiness_s=0)]))


class BenefitTimingTest(unittest.TestCase):
    def test_duplicate_arms_do_not_require_duplicate_rollouts(self):
        branches = {a: NS(first_action_s=x, second_action_s=y) for a, x, y in
                    [("reference", 0, 0), ("first_only", 0, 0), ("second_only", 0, 15), ("joint", 0, 15)]}
        self.assertEqual(unique_branches(branches), dict(reference="reference", first_only="reference",
                                                       second_only="second_only", joint="second_only"))

    def test_returns_are_reference_clock_cuts_not_candidate_returns(self):
        ref, candidate = trace(), trace(3500, change=10)
        context = dict(name="case", seed=1, scenario=2, target_time_s=900)
        rows = timing_rows(context, dict(reference=ref, joint=candidate), 4000)
        cut = next(r for r in rows if r["arm"] == "joint" and r["cut"] == "both_reference_returns")
        self.assertEqual(cut["time_s"], 3000)
        self.assertEqual(cut["delta_passenger_exposure_s"], 10)

    def test_command_boundary_excludes_first_action_at_before_first_cut(self):
        ref, candidate = trace(), trace()
        candidate["holds"].loc[0, "action_s"] = 45
        rows = timing_rows(dict(name="case", seed=1, scenario=2, target_time_s=900),
                           dict(reference=ref, joint=candidate), 4000)
        commands = {r["cut"]: r["delta_commanded_holding_s"] for r in rows if r["arm"] == "joint"}
        self.assertEqual(commands["before_first"], 0)
        self.assertEqual(commands["15min"], 30)

    def test_preintervention_exposure_mismatch_blocks_accounting(self):
        ref, candidate = trace(), trace()
        candidate["timeline"][0, 1] += 1
        with self.assertRaisesRegex(ValueError, "before the first"):
            timing_rows(dict(name="case", seed=1, scenario=2, target_time_s=900), dict(reference=ref, joint=candidate), 4000)

    def test_missing_return_is_not_silently_treated_as_day_end(self):
        ref = trace()
        ref["terminals"] = ref["terminals"].iloc[:1]
        with self.assertRaisesRegex(ValueError, "unique terminal return"):
            common_cuts(ref, 4000)

    def test_different_clearance_times_share_one_fixed_day_end(self):
        ref, candidate = trace(), trace(change=10)
        candidate["timeline"][-1, 0] = 4200
        rows = timing_rows(dict(name="case", seed=1, scenario=2, target_time_s=900),
                           dict(reference=ref, joint=candidate), 5000)
        final = [r for r in rows if r["cut"] == "day_end"]
        self.assertEqual([r["time_s"] for r in final], [5000, 5000])
        self.assertEqual(final[1]["delta_passenger_exposure_s"], 10)

    def test_unserved_passengers_keep_accumulating_wait_after_clearance(self):
        data = trace()
        data["snapshot"]["episode"]["passengers_unserved"] = 3
        np.testing.assert_array_equal(exposure_at(data, 5000), [3080, 160, 0])

    def test_incomplete_trace_cannot_be_extended_as_a_cleared_tail(self):
        data = trace()
        data["snapshot"]["episode"]["done_reason"] = "evaluation_horizon"
        with self.assertRaisesRegex(ValueError, "truncated"):
            exposure_at(data, 5000)

    def test_trace_beyond_fixed_horizon_still_fails(self):
        data = trace()
        with self.assertRaisesRegex(ValueError, "beyond fixed"):
            timing_rows(dict(name="case", seed=1, scenario=2, target_time_s=900), dict(reference=data), 3999)

    def test_endpoint_change_blocks_reuse(self):
        item = dict(seed=1, scenario=2, target_time_s=900, first_action_s=0, second_action_s=15,
                    expected=dict.fromkeys(OUTCOMES, 0),
                    identities=dict(first={}, second={}))
        snapshot = dict(item, matches_v37=True, episode=item["expected"].copy(),
                        decisions=dict(first=dict(identity={}), second=dict(identity={})))
        validate_snapshot(snapshot, item)
        snapshot["episode"]["headway_cv"] = 1
        with self.assertRaises(AssertionError):
            validate_snapshot(snapshot, item)

    def test_sign_reversal_is_distinct_from_metric_disagreement(self):
        rows, endpoints = [], []
        for arm in ("first_only", "second_only", "joint"):
            for cut, value in [("15min", 20), ("30min", 0), ("both_reference_returns", -10), ("day_end", -100)]:
                rows.append(dict(context="case", arm=arm, cut=cut, delta_passenger_exposure_s=value))
            endpoints.append(dict(context="case", arm=arm, delta_restricted_total_journey_horizon_min=-.1))
        result = reversal_rows(pd.DataFrame(rows), pd.DataFrame(endpoints)).set_index(["arm", "cut"])
        self.assertTrue(result.loc[("joint", "15min"), "exposure_sign_reversal"])
        self.assertFalse(result.loc[("joint", "30min"), "exposure_sign_reversal"])
        self.assertEqual(result.loc[("joint", "15min"), "tail_delta_person_s"], -120)
        self.assertFalse(result.exposure_vs_journey_sign_disagreement.any())


if __name__ == "__main__":
    unittest.main()
