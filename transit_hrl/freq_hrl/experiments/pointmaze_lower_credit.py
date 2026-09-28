"""Audit the actual lower reward and credit boundary used by native PPO."""

import numpy as np

from freq_hrl.domains.mujoco import RelativeSubgoalAdapter
from .pointmaze_goal_validation import POINTMAZE_LOWER_ACTION_COST


def audit_credit_batch(batch, row, raw, *, args, mode):
    if mode.startswith("task_"):
        expected = np.asarray(raw["reward"], dtype=np.float32)
    else:
        adapter = RelativeSubgoalAdapter(
            maximum_delta=np.full(2, args.maximum_subgoal_delta, dtype=np.float32),
            action_cost=POINTMAZE_LOWER_ACTION_COST)
        expected = np.asarray([
            adapter.intrinsic_reward(achieved_before=before, achieved_after=after, subgoal=goal, action=action)
            for before, after, goal, action in zip(raw["achieved_before"], raw["achieved_after"],
                                                  raw["subgoal"], raw["action"])], dtype=np.float32)
    np.testing.assert_array_equal(batch.lower.reward, expected)
    done = np.zeros(args.horizon, dtype=np.float32)
    done[-1] = 1.
    if mode.endswith("_option"):
        done[np.asarray(row["decision_steps"][1:], dtype=int) - 1] = 1.
    np.testing.assert_array_equal(batch.lower.done, done)
    np.testing.assert_array_equal(batch.lower.duration, np.ones(args.horizon, dtype=np.int64))
    credit = row["lower_training_credit"]
    if credit["mode"] != mode or credit["primitive_steps"] != args.horizon:
        raise ValueError("lower credit mode or primitive count changed")
    np.testing.assert_allclose(credit["reward_sum"], np.sum(expected, dtype=np.float64), atol=1e-10, rtol=0)
    if credit["done_count"] != int(done.sum()) or credit["option_count"] != len(row["decision_steps"]):
        raise ValueError("lower training credit boundary accounting changed")


def audit_training_credit(result, *, specification):
    spec = specification
    mode = spec.LOWER_CREDIT[result["method"]]
    opt = spec.options(preflight=result["preflight"])
    horizon = spec.source.arguments(result["root"], preflight=result["preflight"]).horizon
    history = result["training_credit"]
    if (result["lower_credit"] != mode
            or [r["iteration"] for r in history] != list(range(1, opt["iterations"] + 1))):
        raise ValueError("lower training credit mode or iteration roster changed")
    for row in history:
        if row["primitive_steps"] != horizon * opt["rollouts_per_iteration"]:
            raise ValueError("lower credit primitive accounting changed")
        expected_cuts = row["option_count"] if mode.endswith("_option") else opt["rollouts_per_iteration"]
        if row["done_count"] != expected_cuts:
            raise ValueError("lower training credit boundary accounting changed")
        if mode.startswith("task_"):
            np.testing.assert_allclose(row["reward_sum"], row["task_reward_sum"], atol=1e-8, rtol=0)
    probe = result["initial_credit_probe"]
    if probe["seed"] != result["seed_roles"]["training"][0] or probe["mode"] != mode:
        raise ValueError("initial credit probe identity changed")
