import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.core.prefix_cost_credit import PrefixCostCredit
from scripts import run_native_transit_mc_upper_stage160 as spec
from scripts.analyze_native_transit_mc_upper_stage160 import summarize, validate_training
from scripts.submit_native_transit_mc_upper_stage160_scheduleurm import task_specification
from tests import test_native_transit_reference_credit as reference_tests


def fake_episode(root, scenario, scene, raw, checkpoint, action_fn, *, preflight=False, **kwargs):
    count, clock = (4, 5400) if preflight else (44, 61380)
    credit, decisions, cost = PrefixCostCredit(), [], 30.
    for i in range(count):
        state = np.full(34, i / count, dtype=np.float32)
        action = action_fn(state)
        credit.begin(state, action, cost, i * 1200)
        decisions.append({"state": state, "action": action})
        cost -= .1 - .02 * float(action.sum())
    summary = credit.finish(cost, clock)
    row = {k: 1. for k in spec.source.source.source_spec.authority.routing.METRICS}
    row.update(service_cost_restricted=round(cost, 6), ep=300, N_fleet=12,
        simulation_end_time_s=clock, done_reason="evaluation_horizon", passengers_generated=100,
        passengers_unserved=0, trips_completed=24, ep_steps=100)
    return row, SimpleNamespace(credit=credit, decisions=decisions), summary, {"endpoint_error_max_s": 0}


class NativeMonteCarloUpperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with patch.object(spec.source, "episode", side_effect=fake_episode):
            cls.agent, cls.full = spec.train(397, Path("unused"), Path("unused"), preflight=False)
            _, cls.short = spec.train(397, Path("unused"), Path("unused"), preflight=True)

    def test_mc_returns_cancel_values_and_stop_at_episode_boundaries(self):
        agent = spec.learner(397, True)
        reward = np.array([2., -3., 5., 7.])
        done = np.array([0., 1., 0., 1.])
        duration = np.array([1800, 57780, 1800, 57780])
        for values in (np.zeros(4), np.array([9., -12., 17., 80.])):
            advantage, returns = agent._gae(reward, done, duration, values)
            np.testing.assert_allclose(returns, [-1., -3., 12., 7.], atol=1e-5)
            np.testing.assert_allclose(advantage + values, returns, atol=1e-5)
        self.assertEqual(agent.config.entropy_coef, 0)

    def test_full_and_short_existing_ppo_learning_budgets(self):
        validate_training(self.full, 397)
        validate_training(self.short, 397, preflight=True)
        self.assertEqual(self.full["updates"], 768)
        self.assertEqual(self.full["transitions"], 5280)
        self.assertGreater(self.full["actor_change_max_abs"], 0)
        self.assertEqual(len(self.full["update_curve"]), 24)

    def test_stored_latent_likelihood_matches_bounded_execution(self):
        agent = spec.learner(397, True)
        draws = []

        def sample(state):
            d = agent.act_upper(state, sample=True)
            draws.append({"state": state.copy(), **d})
            return np.tanh(d["action"]).astype(np.float32)

        _, plan, _, _ = fake_episode(397, "low_noise", 1, None, None, sample, preflight=True)
        batch = spec.episode_batch(plan, draws, plan.credit.transitions)
        with torch.no_grad():
            logp, _ = agent.upper_actor.log_prob_entropy(torch.as_tensor(batch.state), torch.as_tensor(batch.action))
        np.testing.assert_allclose(np.exp(logp.numpy() - batch.old_logp), 1, atol=1e-6)
        changed = copy.deepcopy(draws)
        changed[0]["action"] += .1
        with self.assertRaises(RuntimeError): spec.episode_batch(plan, changed, plan.credit.transitions)
        transitions = copy.deepcopy(plan.credit.transitions)
        transitions[-1]["done"] = False
        with self.assertRaises(RuntimeError): spec.episode_batch(plan, draws, transitions)

    def test_analysis_rejects_changed_mc_credit_and_on_policy_budget(self):
        for failure in ("updates", "pair", "target", "batch", "tail"):
            run = copy.deepcopy(self.full)
            if failure == "updates": run["updates"] += 1
            elif failure == "pair": run["credit_pairs"][0]["paired_reward_sum"] += 1
            elif failure == "target": run["credit_pairs"][0]["mc_returns_qualified"] = False
            elif failure == "batch": run["update_curve"][0]["last_episode"] -= 1
            else: run["credit_pairs"][0]["terminal_transitions"] = 0
            with self.subTest(failure=failure), self.assertRaises(ValueError): validate_training(run, 397)

    def cells(self):
        sac, _ = reference_tests.ReferenceCreditTest().cells()
        cells = {}
        for root in spec.ROOTS:
            full, short = copy.deepcopy(self.full), copy.deepcopy(self.short)
            for run, preflight in ((full, False), (short, True)):
                seeds = [(600000000 if preflight else 700000000) + root * 1000 + ep
                         for ep in range(len(run["credit_pairs"]))]
                run["training_scene_seeds"] = seeds
                for pair, scene in zip(run["credit_pairs"], seeds): pair["scene_seed"] = scene
            rows = copy.deepcopy(sac[root]["evaluation"])
            for row in rows:
                if row["condition"] == "constant_residual": row["residual_action_mean"] = full["constant_action"]
            cells[root] = {"protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "seed": root,
                "software_qualified": True, "native_training_updates": 0, "native_steps": 320 * 61380,
                "worker_preflight_native_steps": 61380 + 4 * 5400, "forecast_source_reproduced": True,
                "short_learning": short, "training": full, "evaluation": rows}
        return cells, sac

    def test_summary_rejects_unpaired_or_changed_evaluation(self):
        cells, sac = self.cells()
        self.assertEqual(summarize(cells, sac)["native_steps"], 640 * 61380)
        for failure in ("duplicate", "constant", "control"):
            broken = copy.deepcopy(cells)
            if failure == "duplicate": broken[397]["evaluation"].append(copy.deepcopy(broken[397]["evaluation"][0]))
            else:
                row = next(r for r in broken[397]["evaluation"]
                           if r["condition"] == ("constant_residual" if failure == "constant" else "forecast"))
                if failure == "constant": row["residual_action_mean"] = [1., 1.]
                else: row["service_cost_restricted"] += .1
            with self.subTest(failure=failure), self.assertRaises((ValueError, RuntimeError)): summarize(broken, sac)

    def test_scheduler_is_code_only_unpinned_and_uses_mc_result_path(self):
        task = task_specification("test_mc", 397)
        self.assertIsNone(task["require_node"])
        self.assertEqual(task["allowed_nodes"], [f"node{i:03}" for i in range(1, 7)])
        self.assertEqual(task["cpu"], 1)
        self.assertTrue(task["result_dir"].endswith("/cells/mc_ppo/seed_397"))
        self.assertEqual(task["local_result_dir"], task["result_dir"])
        self.assertEqual([Path(p).name for p in task["stage_input_paths"]], ["scripts", "freq_hrl", "native_freqduet"])


if __name__ == "__main__":
    unittest.main()
