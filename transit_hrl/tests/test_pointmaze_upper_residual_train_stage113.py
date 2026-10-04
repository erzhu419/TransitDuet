import numpy as np
import torch

from freq_hrl.experiments import pointmaze_upper_residual_train as experiment
from scripts import pointmaze_upper_residual_train_stage113_spec as spec
from scripts.submit_pointmaze_upper_residual_train_stage113_scheduleurm import (
    qualification_task, task_specification,
)
from freq_hrl.rl.dual_actor_critic import GaussianActor


def test_stage113_budget_has_independent_a_b_score_batches():
    budget = spec.budget(preflight=True)
    options = spec.options(preflight=True)
    horizon = spec.arguments(spec.roots(preflight=True)[0], preflight=True).horizon
    expected = sum(options["updates"] * 2 * (int(np.ceil(
        options["credit_scenarios_per_batch"] * options["rollouts_per_scenario"] * (horizon // period)
        / spec.CHUNK_SIZE))) for period in spec.PERIODS)
    assert budget["actor_score_forward_batches"] == expected
    assert budget["actor_score_backward_batches"] == 2 * expected


def test_stage113_check_row_has_explicit_forecast_and_learned_schedules():
    horizon = 1200
    learned = {"episode_length": horizon, "lower_calls": horizon, "upper_calls": 24,
        "decision_steps": list(range(0, horizon, 50)), "network_check": "passed",
        "plan_renewals": 24, "variant": "learned", "state_arm": "learned"}
    forecast = dict(learned, upper_calls=0, decision_steps=[], variant="forecast", state_arm="forecast")
    experiment.check_row(learned, period=50, horizon=horizon, variant="learned")
    experiment.check_row(forecast, period=50, horizon=horizon, variant="forecast")


def test_stage113_residual_is_zero_initialized_and_preserves_base():
    base = GaussianActor(3, 2, 8, -0.5)
    branch = experiment.OptionalActionResidual(base, feedback_dim=3, advice_dim=0)
    state = torch.randn(5, 3)
    with torch.inference_mode():
        before = branch.base.distribution(state).mean.clone()
        after = branch.distribution(state).mean.clone()
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert all(not value.requires_grad for value in branch.base.parameters())


def test_stage113_scheduler_is_dynamic_and_has_qualification_dependency():
    task = task_specification("stage113_test", spec.roots(preflight=True)[0], preflight=True)
    qualification = qualification_task("stage113_test", preflight=True)
    assert task["require_node"] is None
    assert set(task["allowed_nodes"]) == {f"node00{i}" for i in range(1, 7)}
    assert len(qualification["wait_for_files"]) == 1
