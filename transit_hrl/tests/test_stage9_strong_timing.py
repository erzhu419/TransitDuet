from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.analyze_reacher_stage9_tail_decisions import top_at_equal_counts
from scripts.run_stage9_strong_timing import ISE, METHODS, panels, readout, same_controller


def fixture():
    cells = [{"optimizer_seed": i, "audit_seeds": [i * 10 + 1, i * 10 + 2]} for i in (1, 2)]
    plan = {"development_cells": cells, "minimum_ise_gain": .02}
    reports = []
    for cell in cells:
        rows = {method: [{"seed": s, ISE: .96 if method == "history" else 1.,
                          "episode_return": 1., "trigger_score_checks": 2,
                          "decision_steps": [0, 55]} for s in cell["audit_seeds"]] for method in METHODS}
        ready = {"eligible_for_fresh_confirmation": True}
        reports.append({**cell, "status": "complete", "phase": "audit", "panels": panels(rows),
            "budgets": {m: {"admitted": True} for m in METHODS},
            "termination_selection_readiness": dict(ready), "termination_audit_readiness": dict(ready)})
    return plan, reports


class Stage9StrongTimingTest(unittest.TestCase):
    def test_requires_the_full_ordered_cohort(self):
        plan, records = fixture()
        for rows in (records[:1], records[::-1]):
            with self.assertRaises(ValueError):
                readout(plan, rows)

    def test_pooled_gain_alone_cannot_release(self):
        plan, records = fixture()
        self.assertTrue(readout(plan, records)["confirmation_builds_allowed"])
        for change in ("bad_half", "small_gain", "ineligible", "budget"):
            with self.subTest(change=change):
                rows = deepcopy(records)
                if change == "bad_half":
                    rows[0]["panels"]["last"]["threshold"]["mean_ise"] = .9
                elif change == "small_gain":
                    for row in rows:
                        row["panels"]["all"]["threshold"]["mean_ise"] = .97
                elif change == "ineligible":
                    rows[0]["termination_audit_readiness"]["eligible_for_fresh_confirmation"] = False
                else:
                    rows[1]["budgets"]["termination_history"]["admitted"] = False
                self.assertFalse(readout(plan, rows)["confirmation_builds_allowed"])

    def test_compares_actual_controller_parameters(self):
        params = {"config": {}, **{name: torch.nn.Linear(2, 2).state_dict() for name in
                  ("upper_actor", "lower_actor", "upper_value", "lower_value")}}
        copy = deepcopy(params)
        model = SimpleNamespace(state_dict=lambda: params)
        other = SimpleNamespace(state_dict=lambda: copy)
        self.assertTrue(same_controller(model, other))
        copy["upper_actor"]["bias"][0] += 1
        self.assertFalse(same_controller(model, other))

    def test_rank_diagnostic_cannot_increase_the_renewal_rate_or_move_it_to_a_new_clock(self):
        factual = np.array([True, False, True, False, False])
        clocks = np.array([0, 0, 5, 5, 10])
        selected = top_at_equal_counts(np.array([0., 1., 0., 2., 100.]), factual, clocks)
        np.testing.assert_array_equal(selected, [False, True, False, True, False])
