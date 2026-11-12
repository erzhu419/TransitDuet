from copy import deepcopy
import unittest

from scripts.run_stage9_timing_portfolio import METHODS, readout, winner, within_budget


class PortfolioTest(unittest.TestCase):
    def test_selection_does_not_consider_stage9_or_threshold(self):
        ise = {"termination_history": 1., "best_clock": 1.01, "history": 0., "threshold": 0.}
        self.assertEqual(winner(ise), "termination_history")
        self.assertEqual(winner({**ise, "best_clock": 1.}), "best_clock")
        self.assertEqual(winner({**ise, "best_clock": .99}), "best_clock")
        with self.assertRaises(ValueError):
            winner({**ise, "best_clock": float("nan")})

    def test_budget_charges_offline_and_deployment_cost(self):
        budget = {"cpu_seconds": 100, "offline_steps": 500, "p99_seconds": .01, "deployments": 100}
        charge = {"offline_cpu_seconds": 90, "offline_steps": 500}
        latency = {"cpu_seconds_per_episode": .1, "wall_p99_seconds": .009}
        self.assertTrue(within_budget(charge, latency, budget)["admitted"])
        for over in ({"offline_cpu_seconds": 91}, {"offline_steps": 501}):
            self.assertFalse(within_budget({**charge, **over}, latency, budget)["admitted"])

    def test_joint_gate_does_not_drop_clock_root_or_ignore_second_half(self):
        cells, reports = [], []
        for i in range(8):
            cell = {"optimizer_seed": i, "evaluation_seeds": [100 + 2 * i, 101 + 2 * i],
                    "portfolio_choice": "best_clock" if i == 0 else "termination_history"}
            panel = {m: {"mean_ise": .9 if m == "history" else 1.} for m in METHODS}
            cells.append(cell)
            reports.append({**cell, "status": "complete", "new_fits": 0,
                "measured_cost": {"counts": {"environment_steps": 7200}},
                "panels": {p: deepcopy(panel) for p in ("all", "first", "last")},
                "evaluation_budget_admission": {m: {"admitted": True} for m in METHODS}})
        plan = {"cells": cells, "steps_per_root": 7200, "scope": "fixed", "minimum_ise_gain": .02}
        result = readout(plan, reports)
        self.assertEqual(result["learned_choices"], 7)
        self.assertEqual(result["comparisons"]["portfolio"]["paired_ise_gain"]["n"], 8)
        self.assertTrue(result["joint_gate_passed"])
        self.assertFalse(result["original_confirmation_released"])
        for row in reports:
            row["panels"]["last"]["portfolio"]["mean_ise"] = .8
        self.assertFalse(readout(plan, reports)["joint_gate_passed"])
        with self.assertRaises(ValueError):
            readout(plan, reports[1:])
