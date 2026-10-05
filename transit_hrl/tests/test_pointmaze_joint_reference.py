import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from freq_hrl.experiments import pointmaze_joint_reference as experiment
from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from scripts import pointmaze_joint_reference_stage121_spec as spec
from scripts.submit_pointmaze_joint_reference_stage121_scheduleurm import task_specification, qualification_task
import test_pointmaze_optional_plan as fixture
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


class TrackingDonor(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((2,), -.5))

    def distribution(self, state):
        return torch.distributions.Normal(state[..., 4:6] + 2 * state[..., 390:392], self.log_std.exp())


@pytest.fixture
def source_data():
    torch.set_num_threads(1)
    models, predictor, _, calibrations = fixture.OptionalPlanTest().sources()
    args = spec.arguments(410011, preflight=True)
    args.horizon = 100
    teachers = {p: experiment.base.lower_training.branch(model) for p, model in models.items()}
    with torch.no_grad():
        for teacher in teachers.values():
            teacher.readout.weight[0, 392] = .2
            teacher.readout.weight[1, 393] = .3
    return models, predictor, calibrations, args, {p: copy.deepcopy(t.state_dict()) for p, t in teachers.items()}


def test_new_distribution_preserves_teacher_and_trains_only_bounded_residual():
    teacher = OptionalActionResidual(TrackingDonor(), feedback_dim=392, advice_dim=4)
    actor = experiment.ReferenceResidualActor(teacher)
    state = torch.full((3, 402), .125)
    state[:, 396:400] = 0.
    original = teacher.distribution(state[:, :396])
    zero, reference, residual = actor.components(state)
    torch.testing.assert_close(zero.mean, original.mean, atol=0, rtol=0)
    torch.testing.assert_close(zero.stddev, original.stddev, atol=0, rtol=0)
    assert not reference.any() and not residual.any()
    state[:, 396:400] = torch.tensor([.01, -.02, .005, .01])
    before = state.clone()
    distribution, reference, residual = actor.components(state)
    expected = .05 * torch.tanh(torch.tensor([.02, 0.]) / .05)
    torch.testing.assert_close(reference, expected.expand(3, -1), atol=1e-7, rtol=0)
    (-distribution.log_prob(distribution.mean.detach() + .1).sum()).backward()
    assert actor.readout.weight.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in actor.teacher.parameters())
    with torch.no_grad():
        actor.readout.bias.fill_(100.)
    state[:, 396:400] = 100.
    _, reference, residual = actor.components(state)
    assert reference.abs().max() <= spec.REFERENCE_LIMIT + 1e-8
    assert residual.abs().max() <= spec.RESIDUAL_LIMIT + 1e-8
    torch.testing.assert_close(state[:, :396], before[:, :396], atol=0, rtol=0)


def test_native_zero_joint_is_forecast_and_credit_is_full_suffix_with_true_durations(source_data):
    models, pred, cal, args, teachers = source_data
    trainer = experiment.make_trainer(models["50"], teachers["50"], args)
    call = dict(args=args, seed=121001, noise_seed=121002, period=50, predictor=pred, envelope=cal["50"]["envelope"])
    with patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        _, forecast, forecast_audit = experiment.native_episode(trainer, arm="forecast", collect=False, **call)
        _, joint, joint_audit = experiment.native_episode(trainer, arm="joint", collect=False, **call)
        batch, row, audit = experiment.native_episode(trainer, arm="joint", collect=True, **call)
    assert joint["episode_return"] == forecast["episode_return"]
    assert joint["reference_correction_peak"] == joint["plan_delta_rms"] == 0.
    np.testing.assert_array_equal(joint_audit["innovations"], forecast_audit["innovations"])
    np.testing.assert_array_equal(batch.upper.duration, [50, 50])
    np.testing.assert_array_equal(np.flatnonzero(batch.upper.done), [1])
    np.testing.assert_array_equal(np.flatnonzero(batch.lower.done), [99])
    _, upper_returns = trainer._gae(batch.upper.reward, batch.upper.done, batch.upper.duration, batch.upper.old_value)
    _, lower_returns = trainer._gae(batch.lower.reward, batch.lower.done, batch.lower.duration, batch.lower.old_value)
    np.testing.assert_allclose(upper_returns, lower_returns[::50], atol=2e-5, rtol=0)
    assert batch.lower.state.shape == (100, 402)
    assert batch.upper.state.shape == (2, 392)
    assert row["reference_correction_peak"] > 0
    with patch.object(experiment.source.native.joint, "_make_task", return_value=Float32Task(1.)), \
            patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        future, _, _ = experiment.native_episode(trainer, arm="joint", collect=True, **call)
    np.testing.assert_array_equal(future.lower.state[:50], batch.lower.state[:50])
    np.testing.assert_array_equal(future.upper.state[:1], batch.upper.state[:1])


def test_native_joint_ppo_changes_both_actors_and_critics_but_not_teacher_or_std(source_data):
    models, pred, cal, args, teachers = source_data
    trainer = experiment.make_trainer(models["50"], teachers["50"], args)
    initial = experiment.weights(trainer)
    source_weights = experiment.weights(models["50"])
    job = (source_weights, teachers["50"], {m: initial for m in spec.METHODS}, 121001, [121002, 121003],
           50, pred, cal["50"]["envelope"], True)
    with patch.object(experiment.source.native, "_WORKER", (models["50"], args)), \
            patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        outputs = experiment.worker_group(job)["outputs"]
    batches = [batch for method, batch, _ in outputs if method == "joint"]
    joined = concat_hierarchical_batches(batches)
    _, returns = trainer._gae(joined.lower.reward, joined.lower.done, joined.lower.duration, joined.lower.old_value)
    np.testing.assert_allclose(returns[[0, 100]], [row["episode_return"] for m, _, row in outputs if m == "joint"], atol=.002, rtol=0)
    report = experiment.update(trainer, batches, optimizer_seed=121004)
    assert all(report["parameter_delta_rms"][name] > 0 for name in experiment.NETWORKS)
    assert report["old_logp_replay_max_error"] < 3e-5
    assert report["optimizer_steps"]["upper_actor_optimizer_steps"] == spec.EPOCHS
    assert report["optimizer_steps"]["lower_actor_optimizer_steps"] == spec.EPOCHS
    torch.testing.assert_close(trainer.lower_actor.teacher.state_dict(), teachers["50"], atol=0, rtol=0)
    torch.testing.assert_close(trainer.upper_actor.log_std, initial["upper_actor"]["log_std"], atol=0, rtol=0)
    twin = experiment.make_trainer(models["50"], teachers["50"], args)
    assert experiment.update(twin, batches, optimizer_seed=121004) == report
    torch.testing.assert_close(experiment.weights(twin), experiment.weights(trainer), atol=0, rtol=0)
    bad = copy.deepcopy(batches)
    bad[0].lower.old_logp += .1
    with pytest.raises(ValueError, match="likelihood disagree"):
        experiment.update(twin, bad, optimizer_seed=121004)
    rounded = copy.deepcopy(batches)
    for batch in rounded:
        batch.lower.old_logp += 1e-4
    tolerates_roundoff = experiment.make_trainer(models["50"], teachers["50"], args)
    roundoff_report = experiment.update(tolerates_roundoff, rounded, optimizer_seed=121004)
    assert roundoff_report["old_logp_replay_max_error"] < spec.LOGP_REPLAY_TOLERANCE


def test_reduced_runner_updates_all_methods_with_exact_native_optimizer_budget(source_data, tmp_path):
    models, pred, cal, args, teachers = source_data
    original_args = spec.arguments
    original_options = spec.options
    short = lambda root, **kw: SimpleNamespace(**{**vars(original_args(root, **kw)), "horizon": 100})
    small = lambda **kw: {**original_options(**kw), "scenarios_per_update": 2, "workers": 1}
    experiment.write_json(tmp_path / "source.json", {})
    with patch.object(spec, "arguments", side_effect=short), patch.object(spec, "options", side_effect=small), \
            patch.object(spec, "source_result", return_value=tmp_path / "source.json"), \
            patch.object(experiment.source, "qualify", side_effect=lambda cell, **kw: cell), \
            patch.object(experiment.source, "load_source", return_value=(models, pred, {}, cal)), \
            patch.object(experiment.base, "load_lower_state", side_effect=lambda root, period, **kw: teachers[str(period)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(experiment.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        cell = experiment.run(410011, preflight=True, output=tmp_path / "result.json")
        assert cell["cost"] == spec.budget(preflight=True)
        assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.npz"))
        assert (tmp_path / "completion" / "ready.json").is_file()
        assert experiment.aggregate([cell], preflight=True)["joint_gain_gate"] == "mechanical_only"
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["history"]["joint"][0]["old_logp_replay_max_error"] = .1
        with pytest.raises(ValueError, match="likelihood"):
            experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["evaluation"]["joint_blinded"][0]["upper_calls"] = 1
        with pytest.raises(ValueError, match="schedule"):
            experiment.qualify(bad, preflight=True)


def test_gain_gate_keeps_both_periods_and_practical_threshold():
    endpoints = {k: {"ci": [.1, .2]} for k in spec.ENDPOINTS}
    with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
        assert experiment.aggregate([], preflight=False)["joint_gain_gate"] == "not_closed"
        for p in spec.PERIODS:
            for b in ("flat", "forecast"):
                endpoints[f"{p}/joint_minus_{b}"]["ci"] = [.6, .8]
        assert experiment.aggregate([], preflight=False)["joint_gain_gate"] == "supported_both_periods"
        endpoints["50/joint_minus_joint_blinded"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["joint_gain_gate"] == "not_closed"


def test_budget_rosters_and_dynamic_scheduler():
    assert spec.budget(preflight=False)["native_episodes"] * 8 == 9728
    assert spec.budget(preflight=False)["native_steps"] * 8 == 11673600
    assert len(spec.ENDPOINTS) == 10
    assert spec.LOGP_REPLAY_TOLERANCE < .2 / 100
    assert spec.options(preflight=True)["scenarios_per_update"] == spec.options(preflight=False)["scenarios_per_update"]
    assert spec.arguments(410011, preflight=True).horizon == spec.arguments(410011, preflight=False).horizon
    seen = set()
    for preflight in (True, False):
        for root in spec.roots(preflight=preflight):
            roles = spec.seed_roles(root, preflight=preflight)
            seeds = [s for round_ in roles["training_rounds"] for r in round_ for s in [r["scenario_seed"], *r["noise_seeds"]]] + roles["native_evaluation"]
            assert len(set(seeds)) == len(seeds) and not seen.intersection(seeds)
            seen.update(seeds)
            task = task_specification("stage121_unit", root, preflight=preflight)
            assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]
            assert task["require_node"] is None and task["cpu"] == spec.options(preflight=preflight)["workers"] + 1
            assert task["ram_mb"] == (4096 if preflight else 8192)
            assert task["result_dir"].endswith("/completion")
    assert len(qualification_task("stage121_unit", preflight=False)["wait_for_files"]) == 8
