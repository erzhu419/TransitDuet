import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from freq_hrl.experiments import pointmaze_plan_authority as experiment
from freq_hrl.rl.optional_action_residual import OptionalActionResidual
from scripts import pointmaze_plan_authority_stage120_spec as spec
from scripts.submit_pointmaze_plan_authority_stage120_scheduleurm import task_specification, qualification_task
from test_pointmaze_control_response import Float32Task
from test_pointmaze_optional_plan import OptionalPlanTest
from test_pointmaze_update_isolation import ImmediatePool


class TrackingDonor(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((2,), -.5))

    def distribution(self, state):
        return torch.distributions.Normal(state[..., 4:6] + 2 * state[..., 390:392], self.log_std.exp())


def test_reference_response_uses_plan_error_and_velocity_not_a_free_bias():
    torch.set_num_threads(1)
    lower = OptionalActionResidual(TrackingDonor(), feedback_dim=392, advice_dim=4)
    with torch.no_grad():
        lower.readout.bias.fill_(.2)
    state = torch.full((3, 396), .125)
    before, weights = state.clone(), copy.deepcopy(lower.state_dict())
    original = lower.distribution(state)
    result, correction = experiment.probe.reference_tracking_distribution(lower, state, [0., 0.], [0., 0.], .05)
    torch.testing.assert_close(result.mean, original.mean, atol=0, rtol=0)
    assert not torch.count_nonzero(correction)
    result, correction = experiment.probe.reference_tracking_distribution(lower, state, [.01, -.02], [.005, .01], .05)
    expected = .05 * torch.tanh(torch.tensor([.02, 0.]) / .05)
    torch.testing.assert_close(correction, expected.expand(3, -1), atol=1e-7, rtol=0)
    torch.testing.assert_close(result.stddev, original.stddev, atol=0, rtol=0)
    _, bounded = experiment.probe.reference_tracking_distribution(lower, state, [1., -1.], [2., -2.], .05)
    assert bounded.abs().max() <= .05 + 1e-8
    torch.testing.assert_close(state, before, atol=0, rtol=0)
    torch.testing.assert_close(lower.state_dict(), weights, atol=0, rtol=0)


def test_native_pairing_exact_zero_two_channels_and_amplitude_execution():
    models, predictor, _, calibration = OptionalPlanTest().sources()
    model, args = models["50"], spec.arguments(410011, preflight=True)
    args.horizon = 100
    lower = experiment.probe.base.lower_training.branch(model)
    with torch.no_grad():
        lower.readout.weight[0, 392] = .2
        lower.readout.weight[1, 393] = .3
    state = copy.deepcopy(lower.state_dict())
    weights = experiment.probe.base.source.native.joint.inference_weights(model)
    query = {**spec.seed_roles(410011, preflight=True)["queries"][0], "start": 50}
    calls, decode = [], experiment.probe.wide.WideBernsteinPlan.decode

    def inspect(plan, **kwargs):
        calls.append((kwargs["step"], kwargs["action"].copy()))
        return decode(plan, **kwargs)

    with patch.object(spec, "DIRECTIONS", spec.DIRECTIONS[:2]), \
            patch.object(experiment.probe.base.source.native, "_WORKER", (model, args)), \
            patch.object(experiment.probe.base.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.probe.base.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
            patch.object(experiment.probe.wide.WideBernsteinPlan, "decode", inspect):
        result = experiment.worker_query((weights, state, query, 50, predictor, calibration["50"]))
    assert result["pairing"] == result["source_freeze"] == "passed"
    nonzero = [(step, a) for step, a in calls if a.any()]
    assert {step for step, a in nonzero} == {50}
    assert {round(float(np.abs(a).max()), 6) for step, a in nonzero} == {.05, 1.}
    assert all(np.count_nonzero(a) == 1 for step, a in nonzero)
    for rows in result["panels"].values():
        for channel in spec.CHANNELS:
            assert rows[f"{channel}/zero"]["episode_return"] == rows["forecast"]["episode_return"]
            assert rows[f"{channel}/zero"]["option_command_delta_rms"] == 0
            for size in spec.AMPLITUDES:
                row = rows[f"{channel}/{size}/axis0_plus"]
                assert row["prefix_return"] == rows["forecast"]["prefix_return"]
                assert row["tail_return"] == 0
                assert row["tail_command_delta_rms"] == 0
                assert row["reference_donor_calls"] == (200 if channel == "reference" else 0)
                assert row["reference_correction_peak"] <= .05 + 1e-8
        assert rows["reference/large/axis0_plus"]["reference_correction_peak"] > 0


def test_crossfit_never_selects_from_the_scored_panel():
    rows = {p: {name: {"suffix_return": 10.} for name, *_ in experiment.variants()} for p in spec.PANELS}
    rows["A"]["reference/large/axis0_plus"]["suffix_return"] += 3
    rows["B"]["reference/large/axis0_plus"]["suffix_return"] -= 2
    rows["B"]["reference/large/axis1_plus"]["suffix_return"] += 8
    rows["A"]["reference/large/axis1_plus"]["suffix_return"] -= .5
    gain, choices = experiment.crossfit_gain(rows, "reference", "large")
    assert gain == -1.25
    assert choices == {"A": "reference/large/axis0_plus", "B": "reference/large/axis1_plus"}
    effect = experiment.effects(50, [{"panels": rows}])
    assert effect["50/reference_large_minus_advice_large"] == -1.25


def test_reduced_run_measures_extra_donor_cost_and_writes_no_checkpoints(tmp_path):
    models, predictor, _, calibration = OptionalPlanTest().sources()
    states = {p: copy.deepcopy(experiment.probe.base.lower_training.branch(model).state_dict()) for p, model in models.items()}
    arguments, source = spec.arguments, experiment.probe.base.source
    short = lambda root, **kw: SimpleNamespace(**{**vars(arguments(root, **kw)), "horizon": 100})
    experiment.write_json(tmp_path / "source.json", {})
    with patch.object(spec, "arguments", side_effect=short), patch.object(spec, "DIRECTIONS", spec.DIRECTIONS[:2]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(spec, "source_result", return_value=tmp_path / "source.json"), \
            patch.object(source, "qualify", side_effect=lambda cell, **kw: cell), \
            patch.object(source, "load_source", return_value=(models, predictor, {}, calibration)), \
            patch.object(experiment.probe.base, "load_lower_state", side_effect=lambda root, period, **kw: states[str(period)]), \
            patch.object(source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        cell = experiment.run(410011, preflight=True, output=tmp_path / "result.json")
        assert cell["cost"] == spec.budget(preflight=True)
        assert cell["cost"]["reference_donor_calls"] == 4000
        assert experiment.aggregate([cell], preflight=True)["plan_authority_decision"] == "mechanical_only"
        assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.npz"))
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["queries"][0]["panels"]["A"]["reference/large/axis0_plus"]["reference_correction_peak"] = .1
        with pytest.raises(ValueError, match="correction exceeded"):
            experiment.qualify(bad, preflight=True)


def test_decision_requires_material_gain_not_just_positive_significance():
    endpoints = {k: {"ci": [.001, .002]} for k in spec.ENDPOINTS}
    with patch.object(experiment.probe.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
        assert experiment.aggregate([], preflight=False)["plan_authority_decision"] == "stop_tested_fixed_lower_branch"
        for p in spec.PERIODS:
            endpoints[f"{p}/reference_large_gain"]["ci"] = [.6, 1.]
        assert experiment.aggregate([], preflight=False)["plan_authority_decision"] == "learn_bounded_reference_channel"
        endpoints["100/reference_large_minus_advice_large"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["plan_authority_decision"] == "inconclusive_no_automatic_seed_extension"
        for p in spec.PERIODS:
            endpoints[f"{p}/advice_large_gain"]["ci"] = [.6, 1.]
        assert experiment.aggregate([], preflight=False)["plan_authority_decision"] == "learn_larger_existing_plan_channel"


def test_full_budget_fresh_seeds_and_dynamic_scheduler():
    assert len(spec.roots(preflight=False)) == 8
    assert len(spec.ENDPOINTS) == 10
    assert spec.budget(preflight=False)["native_episodes"] * 8 == 8576
    assert spec.budget(preflight=False)["native_steps"] * 8 == 10291200
    assert spec.budget(preflight=True)["native_steps"] == 80400
    seen = set()
    for pref in (True, False):
        for root in spec.roots(preflight=pref):
            for q in spec.seed_roles(root, preflight=pref)["queries"]:
                seeds = [q["scenario_seed"], q["prefix_noise_seed"], *q["suffix_noise_seeds"].values()]
                assert not seen.intersection(seeds)
                seen.update(seeds)
                assert all(q["start"] % p == 0 for p in spec.PERIODS)
            task = task_specification("stage120_unit", root, preflight=pref)
            assert task["require_node"] is None
            assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]
            assert task["cpu"] == spec.options(preflight=pref)["workers"] + 1
            assert task["result_dir"].endswith("/completion")
    assert len(qualification_task("stage120_unit", preflight=False)["wait_for_files"]) == 8
    with pytest.raises(ValueError, match="all frozen roots"):
        experiment.aggregate([], preflight=False)
