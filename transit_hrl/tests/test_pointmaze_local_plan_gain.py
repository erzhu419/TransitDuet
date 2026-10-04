import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_local_plan_gain as experiment
from scripts import pointmaze_local_plan_gain_stage117_spec as spec
from scripts.submit_pointmaze_local_plan_gain_stage117_scheduleurm import task_specification, qualification_task
from test_pointmaze_control_response import Float32Task
from test_pointmaze_optional_plan import OptionalPlanTest
from test_pointmaze_update_isolation import ImmediatePool


@pytest.fixture
def sources():
    torch.set_num_threads(1)
    models, predictor, _, calibration = OptionalPlanTest().sources()
    states = {}
    for key, model in models.items():
        lower = experiment.base.lower_training.branch(model)
        with torch.no_grad():
            lower.readout.weight[0, 392] = .8
            lower.readout.weight[0, 394] = .15
            lower.readout.weight[1, 393] = .6
            lower.readout.weight[1, 395] = .1
        states[key] = copy.deepcopy(lower.state_dict())
    return models, predictor, calibration, states


def test_all_eight_plan_axes_are_executed_not_lower_biases(sources):
    _, predictor, calibration, _ = sources
    plan = experiment.wide.WideBernsteinPlan(predictor, 50, .75, calibration["50"]["envelope"])
    kwargs = dict(observation=None, history=SimpleNamespace(history=np.zeros(384)), step=100,
                  world_low=-2 * np.ones(2), world_high=2 * np.ones(2))
    for coordinate in range(spec.ACTION_DIM):
        action = experiment.intervention_action(f"axis{coordinate}_plus")
        assert np.flatnonzero(action).tolist() == [coordinate]
        plan.decode(action=action, **kwargs)
        assert np.any(plan.points != plan.base_points)
        np.testing.assert_array_equal(plan.points[0], plan.base_points[0])
        positive = plan.points.copy()
        plan.decode(action=experiment.intervention_action(f"axis{coordinate}_minus"), **kwargs)
        np.testing.assert_allclose(plan.points, -positive, atol=1e-8)
    plan.decode(action=experiment.intervention_action("zero"), **kwargs)
    np.testing.assert_array_equal(plan.points, plan.base_points)


def test_native_probe_has_exact_prefix_noise_pairing_zero_identity_and_continuation(sources):
    models, predictor, calibration, states = sources
    model = models["50"]
    args = spec.arguments(410011, preflight=True)
    args.horizon = 300
    weights = experiment.base.source.native.joint.inference_weights(model)
    lower_before = copy.deepcopy(states["50"])
    query = spec.seed_roles(410011, preflight=True)["queries"][0]
    original_decode = experiment.wide.WideBernsteinPlan.decode
    calls = []

    def record_decode(plan, **kwargs):
        calls.append((kwargs["step"], kwargs["action"].copy()))
        return original_decode(plan, **kwargs)

    with patch.object(experiment.base.source.native, "_WORKER", (model, args)), \
            patch.object(experiment.base.source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(experiment.base.source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))), \
            patch.object(experiment.wide.WideBernsteinPlan, "decode", record_decode):
        result = experiment.worker_query((weights, states["50"], query, 50, predictor, calibration["50"]))
    assert result["pairing"] == result["source_freeze"] == "passed"
    assert result["innovation_max_error"] <= 3e-5
    nonzero = [(step, action) for step, action in calls if np.any(action)]
    assert len(nonzero) == len(spec.PANELS) * len(spec.DIRECTIONS)
    assert {step for step, action in nonzero} == {query["start"]}
    assert all(np.count_nonzero(action) == 1 for step, action in nonzero)
    torch.testing.assert_close(states["50"], lower_before, atol=0, rtol=0)
    for panel, rows in result["panels"].items():
        for variant, row in rows.items():
            assert row["lower_calls"] == row["episode_length"] == 300
            assert row["plan_renewals"] == 6
            assert row["upper_actor_calls"] == 0
            assert row["intervention_decisions"] == int(variant in spec.DIRECTIONS)
            assert row["prefix_return"] == rows["forecast"]["prefix_return"]
            assert row["suffix_return"] == pytest.approx(row["option_return"] + row["tail_return"], abs=1e-10)
        assert rows["zero"]["episode_return"] == rows["forecast"]["episode_return"]
        assert rows["zero"]["option_command_delta_rms"] == 0.
        assert any(rows[v]["option_command_delta_rms"] > 0 for v in spec.DIRECTIONS)
        assert any(rows[v]["tail_command_delta_rms"] > 0 for v in spec.DIRECTIONS)
    a, b = (result["panels"][p]["zero"] for p in spec.PANELS)
    assert a["prefix_seed"] == b["prefix_seed"]
    assert a["suffix_seed"] != b["suffix_seed"]
    assert a["suffix_return"] != b["suffix_return"]


def panels():
    return {name: {variant: {"suffix_return": 10., "option_return": 1.}
        for variant in spec.VARIANTS} for name in spec.PANELS}


def test_crossfit_selector_cannot_pick_using_scored_panel():
    rows = panels()
    rows["A"]["axis0_plus"]["suffix_return"] += 3.
    rows["B"]["axis0_plus"]["suffix_return"] -= 2.
    rows["B"]["axis1_plus"]["suffix_return"] += 8.
    rows["A"]["axis1_plus"]["suffix_return"] -= .5
    gain, selected = experiment.crossfit_gain(rows, "suffix_return")
    assert selected == {"A": "axis0_plus", "B": "axis1_plus"}
    assert gain == -1.25
    assert experiment.crossfit_gain(panels(), "suffix_return") == (0., {"A": "zero", "B": "zero"})


def test_full_suffix_and_option_credit_are_distinct_with_known_gradients():
    rows = panels()
    slopes = np.arange(1, 9, dtype=np.float64)
    for panel in rows.values():
        for i, slope in enumerate(slopes):
            panel[f"axis{i}_plus"]["suffix_return"] += slope * spec.EPSILON
            panel[f"axis{i}_minus"]["suffix_return"] -= slope * spec.EPSILON
        panel["axis0_minus"]["option_return"] += 1.
    query = {"panels": rows, "gradients": experiment.gradients(rows)}
    for values in query["gradients"].values():
        np.testing.assert_allclose(values, slopes, atol=1e-13)
    effect = experiment.effects(50, [query, query])
    assert effect["50/suffix_gradient_cosine"] == pytest.approx(1.)
    assert effect["50/suffix_gradient_dot"] == pytest.approx(np.mean(slopes**2))
    assert effect["50/crossfit_suffix_gain"] == pytest.approx(8 * spec.EPSILON)
    assert effect["50/suffix_minus_option_selection"] == pytest.approx(9 * spec.EPSILON)


def test_reduced_run_counts_credit_and_rejects_noise_or_suffix_mutation(sources, tmp_path):
    models, predictor, calibration, states = sources
    arguments = spec.arguments
    short = lambda root, **kw: SimpleNamespace(**{**vars(arguments(root, **kw)), "horizon": 300})
    source = experiment.base.source
    with patch.object(spec, "arguments", side_effect=short), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(spec, "source_result", return_value=tmp_path / "source.json"), \
            patch.object(source, "qualify", side_effect=lambda cell, **kw: cell), \
            patch.object(source, "load_source", return_value=(models, predictor, {}, calibration)), \
            patch.object(experiment.base, "load_lower_state", side_effect=lambda root, period, **kw: states[str(period)]), \
            patch.object(source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        experiment.write_json(tmp_path / "source.json", {})
        cell = experiment.run(410011, preflight=True, output=tmp_path / "result.json")
        assert cell["cost"] == spec.budget(preflight=True)
        assert cell["cost"]["native_episodes"] == 72
        assert experiment.aggregate([cell], preflight=True)["local_plan_gain_gate"] == "mechanical_only"
        assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.npz"))
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["queries"][0]["panels"]["B"]["zero"]["suffix_seed"] += 1
        with pytest.raises(ValueError, match="noise roles"):
            experiment.qualify(bad, preflight=True)
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["queries"][0]["panels"]["A"]["axis0_plus"]["tail_return"] += .01
        with pytest.raises(AssertionError):
            experiment.qualify(bad, preflight=True)


def test_gate_needs_gain_and_repeatability_without_a_duplicate_forecast_endpoint():
    endpoints = {key: {"ci": [.1, .2]} for key in spec.ENDPOINTS}
    with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
        assert experiment.aggregate([], preflight=False)["local_plan_gain_gate"] == "supported_both_periods"
        endpoints["50/crossfit_suffix_gain"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["local_plan_gain_gate"] == "partial"
        endpoints["100/suffix_gradient_dot"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["local_plan_gain_gate"] == "not_supported"


def test_frozen_full_budget_fresh_seed_roles_and_dynamic_scheduler():
    seen = set()
    for pref in (True, False):
        for root in spec.roots(preflight=pref):
            for query in spec.seed_roles(root, preflight=pref)["queries"]:
                seeds = [query["scenario_seed"], query["prefix_noise_seed"], *query["suffix_noise_seeds"].values()]
                assert not seen.intersection(seeds)
                seen.update(seeds)
                assert all(query["start"] % period == 0 for period in spec.PERIODS)
            task = task_specification("stage117_unit", root, preflight=pref)
            assert task["require_node"] is None
            assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]
            assert task["cpu"] == spec.options(preflight=pref)["workers"] + 1
            assert task["result_dir"].endswith("/completion")
            assert "--preflight" in task["cmd"] if pref else "--preflight" not in task["cmd"]
    assert spec.budget(preflight=False)["native_episodes"] * 8 == 6912
    assert spec.budget(preflight=False)["native_steps"] * 8 == 8294400
    assert spec.budget(preflight=True)["native_steps"] == 21600
    assert len(qualification_task("stage117_unit", preflight=False)["wait_for_files"]) == 8
    assert len(spec.ENDPOINTS) == 8
    with pytest.raises(ValueError, match="all frozen roots"):
        experiment.aggregate([], preflight=False)
