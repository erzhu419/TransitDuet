import unittest

from freq_hrl.rl.restoration_portfolio import (
    paired_trace_invariance_diagnostics,
    select_guarded_restoration_portfolio,
)


def _snapshot(merit, worst=0.05, reward_violations=0):
    return {
        "reward_violation_count": reward_violations,
        "frequency_violation_merit": merit,
        "worst_frequency_violation": worst,
    }


def _row(seed, suffix="same"):
    return {
        "disturbance_mode": "mixed",
        "seed": seed,
        "ExecutedActionTraceSHA256": f"action-{suffix}-{seed}",
        "RewardTraceSHA256": f"reward-{suffix}-{seed}",
        "LatentPolicyTraceSHA256": f"latent-{suffix}-{seed}",
        "reward_mean": float(seed),
        "episode_return": float(seed * 10),
    }


class RestorationPortfolioTest(unittest.TestCase):
    def test_exact_paired_traces_are_invariant(self):
        baseline = [_row(1), _row(2)]
        diagnostics = paired_trace_invariance_diagnostics(
            [dict(row) for row in baseline], baseline
        )
        self.assertTrue(diagnostics["all_traces_invariant"])
        self.assertEqual(diagnostics["executed_action_trace_match_count"], 2)
        self.assertEqual(diagnostics["maximum_episode_return_absolute_delta"], 0.0)

    def test_one_trace_mismatch_rejects_invariance(self):
        baseline = [_row(1), _row(2)]
        candidate = [dict(row) for row in baseline]
        candidate[1]["ExecutedActionTraceSHA256"] = "changed"
        diagnostics = paired_trace_invariance_diagnostics(candidate, baseline)
        self.assertFalse(diagnostics["all_traces_invariant"])
        self.assertEqual(diagnostics["executed_action_trace_match_count"], 1)

    def test_selector_rejects_better_noninvariant_transaction(self):
        baseline = _snapshot(1.0)
        candidates = [
            {
                "snapshot": _snapshot(0.1),
                "fold_snapshots": [_snapshot(0.1), _snapshot(0.1)],
                "requires_trace_invariance": True,
                "trace_invariance": {"all_traces_invariant": False},
                "selection_priority": [0.0],
            },
            {
                "snapshot": _snapshot(0.4),
                "fold_snapshots": [_snapshot(0.4), _snapshot(0.4)],
                "requires_trace_invariance": False,
                "selection_priority": [1.0],
            },
        ]
        decision = select_guarded_restoration_portfolio(
            candidates,
            baseline=baseline,
            fold_baselines=[baseline, baseline],
            minimum_reduction=1e-4,
            funnel_multiplier=3.0,
        )
        self.assertEqual(decision.selected_index, 1)
        self.assertEqual(decision.design_eligibility, (False, True))

    def test_selector_requires_every_design_fold(self):
        baseline = _snapshot(1.0)
        candidate = {
            "snapshot": _snapshot(0.5),
            "fold_snapshots": [_snapshot(0.5), _snapshot(1.1)],
            "requires_trace_invariance": False,
        }
        decision = select_guarded_restoration_portfolio(
            [candidate],
            baseline=baseline,
            fold_baselines=[baseline, baseline],
            minimum_reduction=1e-4,
            funnel_multiplier=3.0,
        )
        self.assertIsNone(decision.selected_index)
        self.assertEqual(decision.fold_eligibility, ((True, False),))


if __name__ == "__main__":
    unittest.main()
