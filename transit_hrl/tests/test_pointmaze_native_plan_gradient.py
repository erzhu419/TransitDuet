import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_native_plan_gradient as experiment
from freq_hrl.rl.dual_actor_critic import GaussianActor
from scripts import pointmaze_native_plan_gradient_stage119_spec as spec
from scripts.submit_pointmaze_native_plan_gradient_stage119_scheduleurm import task_specification, qualification_task
import test_pointmaze_optional_plan as fixture
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def test_native_action_derivative_pullback_has_known_weight_and_bias_solution():
    torch.set_num_threads(1)
    actor = experiment.full.FullPlanCoordinateActor(GaussianActor(3, 4, 8, -.5))
    state = np.asarray([[1, 2, 3], [-2, 1, 4]], dtype=np.float32)
    native = np.asarray([np.arange(1, 9), np.arange(-4, 4)], dtype=np.float64)
    queries = [{"state": s, "gradients": {"A": a, "B": 2*a}} for s, a in zip(state, native)]
    before = copy.deepcopy(actor.state_dict())
    cost = {"actor_pullback_forward_batches": 0, "actor_pullback_backward_batches": 0}
    score = experiment.pullback(actor, queries, cost)
    expected = np.r_[(-native.T @ state / len(state)).ravel(), -native.mean(0)]
    np.testing.assert_allclose(score["gradients"]["A"], expected, atol=1e-8)
    np.testing.assert_allclose(score["gradients"]["B"], 2*expected, atol=1e-8)
    assert score["native_gradient_cosine"] == pytest.approx(1.)
    assert score["readout_gradient_cosine"] == pytest.approx(1.)
    assert cost == {"actor_pullback_forward_batches": 2, "actor_pullback_backward_batches": 2}
    torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)


def sources():
    torch.set_num_threads(1)
    models, predictor, _, calibrations = fixture.OptionalPlanTest().sources()
    states = {}
    for period, model in models.items():
        with torch.no_grad():
            for parameter in model.upper_actor.net.parameters():
                parameter.zero_()
        lower = experiment.base.lower_training.branch(model)
        with torch.no_grad():
            lower.readout.weight[0, 392] = .8
            lower.readout.weight[1, 393] = .6
        states[period] = copy.deepcopy(lower.state_dict())
    return models, predictor, calibrations, states


@pytest.mark.parametrize("start", [0, 150])
def test_current_policy_intervention_first_and_last_options_keep_common_prefix(start):
    models, predictor, calibration, states = sources()
    model, args = models["50"], spec.arguments(410011, preflight=True)
    args.horizon = 200
    actor = experiment.full.upper_branch(model)
    with torch.no_grad():
        actor.readout.bias.fill_(.1)
    upper_state = copy.deepcopy(actor.state_dict())
    weights = experiment.base.source.native.joint.inference_weights(model)
    query = {**spec.seed_roles(410011, preflight=True)["training_rounds"][0]["50"][0], "start": start}
    calls = []
    decode = experiment.probe.wide.WideBernsteinPlan.decode

    def inspect(plan, **kwargs):
        calls.append((kwargs["step"], kwargs["action"].copy()))
        return decode(plan, **kwargs)

    with patch.object(experiment.base.source.native, "_WORKER", (model, args)), \
            patch.object(experiment.base.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.base.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
            patch.object(experiment.probe.wide.WideBernsteinPlan, "decode", inspect):
        result = experiment.worker_query((weights, states["50"], upper_state, query, 50, predictor, calibration["50"]))
    assert result["pairing"] == result["policy_freeze"] == "passed"
    assert result["state"].shape == (390,)
    changed = [(step, value-.1) for step, value in calls if not np.allclose(value, .1, atol=1e-8, rtol=0)]
    assert len(changed) == len(spec.PANELS) * len(spec.DIRECTIONS)
    assert {step for step, value in changed} == {start}
    assert all(np.count_nonzero(np.abs(value) > 1e-8) == 1 for step, value in changed)
    for panel in result["panels"].values():
        for variant, row in panel.items():
            assert row["upper_actor_calls"] == 4
            assert row["lower_calls"] == row["episode_length"] == 200
            assert row["intervention_decisions"] == int(variant != "zero")
            assert row["suffix_return"] == pytest.approx(row["option_return"] + row["tail_return"], abs=1e-10)
            if start == 0:
                assert row["prefix_return"] == 0.
            else:
                assert row["tail_return"] == 0.
    torch.testing.assert_close(actor.state_dict(), upper_state, atol=0, rtol=0)


@pytest.mark.parametrize("preflight", [True, False])
def test_reduced_native_learning_budget_final_checkpoints_and_reference_freeze(tmp_path, preflight):
    models, predictor, calibration, states = sources()
    before = {p: copy.deepcopy(model.state_dict()) for p, model in models.items()}
    lower_before = copy.deepcopy(states)
    reference = {p: copy.deepcopy(experiment.full.upper_branch(model).state_dict()) for p, model in models.items()}
    reference_before = copy.deepcopy(reference)
    options, arguments = spec.options, spec.arguments
    small = lambda **kw: {**options(**kw), "updates": 1, "queries_per_update": 2, "evaluation_episodes": 1, "workers": 1}
    short = lambda root, **kw: SimpleNamespace(**{**vars(arguments(root, **kw)), "horizon": 100})
    source = experiment.base.source
    experiment.write_json(tmp_path / "source.json", {})
    with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=short), \
            patch.object(experiment.previous, "check_prerequisite"), patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(spec, "source_result", return_value=tmp_path / "source.json"), \
            patch.object(source, "qualify", side_effect=lambda cell, **kw: cell), \
            patch.object(source, "load_source", return_value=(models, predictor, {}, calibration)), \
            patch.object(experiment.base, "load_lower_state", side_effect=lambda root, period, **kw: states[str(period)]), \
            patch.object(experiment, "load_reference", side_effect=lambda root, period, model: reference[str(period)]), \
            patch.object(source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        cell = experiment.run(410011, preflight=preflight, output=tmp_path / "result.json")
        assert cell["cost"] == spec.budget(preflight=preflight)
        assert cell["cost"]["training_queries"] == 4
        assert cell["cost"]["native_episodes"] == 144
        assert cell["cost"]["residual_parameter_updates"] == 2
        assert len(list(tmp_path.rglob("*.pt"))) == (0 if preflight else 2)
        assert not list(tmp_path.rglob("*.npz"))
        for p, group in cell["groups"].items():
            source.native.curves.support.assert_frozen(models[p], before[p])
            assert group["history"][0]["geometry"]["gradient_norm"] > 0
            assert "state" not in group["history"][0]["queries"][0]
            assert group["evaluation"]["native_fd"][0]["upper_calls"] == 100 // int(p)
            assert group["evaluation"]["native_fd_blinded"][0]["upper_calls"] == 0
            assert group["evaluation"]["forecast"][0]["episode_return"] == group["evaluation"]["native_fd_blinded"][0]["episode_return"]
        torch.testing.assert_close(states, lower_before, atol=0, rtol=0)
        torch.testing.assert_close(reference, reference_before, atol=0, rtol=0)
        if preflight:
            assert experiment.aggregate([cell], preflight=True)["native_plan_gradient_gain_gate"] == "mechanical_only"
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["history"][0]["queries"][0]["panels"]["A"]["axis0_plus"]["tail_return"] += .01
        with pytest.raises(AssertionError):
            experiment.qualify(bad, preflight=preflight)
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["history"][0]["queries"][0]["panels"]["B"]["zero"]["suffix_seed"] += 1
        with pytest.raises(ValueError, match="noise roles"):
            experiment.qualify(bad, preflight=preflight)


def test_rosters_cover_all_decisions_and_budget_matches_dynamic_worker_count():
    seen = set()
    for preflight in (True, False):
        for root in spec.roots(preflight=preflight):
            roles = spec.seed_roles(root, preflight=preflight)
            h = spec.arguments(root, preflight=preflight).horizon
            current = list(roles["native_evaluation"])
            for period in spec.PERIODS:
                queries = [q for r in roles["training_rounds"] for q in r[str(period)]]
                steps = [q["start"] for q in queries]
                if preflight:
                    assert set(steps) == {0, h-period}
                else:
                    assert set(steps) == set(range(0, h, period))
                    assert len(set(steps.count(step) for step in set(steps))) == 1
                current += [seed for q in queries for seed in
                    [q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values()]]
            assert len(current) == len(set(current))
            assert not seen.intersection(current)
            seen.update(current)
            task = task_specification("stage119_unit", root, preflight=preflight)
            assert task["require_node"] is None
            assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]
            assert task["cpu"] == spec.options(preflight=preflight)["workers"] + 1
            assert task["ram_mb"] == (4096 if preflight else 8192)
            assert task["result_dir"].endswith("/completion")
    assert spec.budget(preflight=False)["native_episodes"] * 8 == 54272
    assert spec.budget(preflight=False)["native_steps"] * 8 == 65126400
    assert spec.budget(preflight=True)["native_steps"] == 91200
    assert len(qualification_task("stage119_unit", preflight=False)["wait_for_files"]) == 8
    assert len(spec.ENDPOINTS) == 4
    with pytest.raises(ValueError, match="all frozen roots"):
        experiment.aggregate([], preflight=False)


def test_gate_requires_forecast_gain_and_reference_gain_both_periods():
    endpoints = {key: {"ci": [.1, .2]} for key in spec.ENDPOINTS}
    with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
        assert experiment.aggregate([], preflight=False)["native_plan_gradient_gain_gate"] == "supported_both_periods"
        endpoints["50/native_fd_minus_forecast"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["native_plan_gradient_gain_gate"] == "partial"
        endpoints["100/native_fd_minus_stage118_suffix"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["native_plan_gradient_gain_gate"] == "not_supported"
