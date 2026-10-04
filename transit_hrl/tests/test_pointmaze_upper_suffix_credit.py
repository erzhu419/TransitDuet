import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_upper_suffix_credit as experiment
from scripts import pointmaze_upper_suffix_credit_stage118_spec as spec
from scripts.submit_pointmaze_upper_suffix_credit_stage118_scheduleurm import task_specification, qualification_task
import test_pointmaze_optional_plan as fixture
from test_pointmaze_control_response import Float32Task
from test_pointmaze_update_isolation import ImmediatePool


def test_suffix_credit_is_decision_aligned_not_current_option_or_episode_broadcast(monkeypatch):
    class Batch:
        def __init__(self, values):
            self.reward = np.asarray(values, dtype=np.float32)
    pairs = [{"lower_batches": [Batch([1, 2]), Batch([3, 5])],
        "upper_batches": [Batch([1, 2]), Batch([3, 5])],
        "rows": [{"episode_return": 3}, {"episode_return": 8}]}]
    captured = []
    monkeypatch.setattr(experiment.base.independent, "exact_returns", lambda batch, gamma: np.asarray([batch.reward.sum()]))
    monkeypatch.setattr(experiment.base, "concat_level_batches", lambda batches: SimpleNamespace(state=np.zeros((4, 390), dtype=np.float32)))

    def fake_score(actor, upper, signals, **kw):
        captured.append(np.asarray(signals["scenario"]))
        return {"scenario": np.ones(3)}, {"actor_score_forward_batches": 1, "actor_score_backward_batches": 2}

    monkeypatch.setattr(experiment.base.lower_training, "residual_actor_gradients", fake_score)
    cost = {key: 0 for key in ("objective_checks", "mc_calls", "actor_score_forward_batches", "actor_score_backward_batches")}
    for method in spec.METHODS:
        score = experiment.score_upper(object(), {"A": pairs, "B": pairs}, horizon=4, period=2, cost=cost, method=method)
        assert score["cross_batch_gradient_cosine"] == pytest.approx(1.)
    np.testing.assert_array_equal(captured[0], [-2, -3, 2, 3])
    np.testing.assert_array_equal(captured[2], [-5, -3, 5, 3])
    assert captured[2][0] != captured[2][1]
    assert cost == {"objective_checks": 8, "mc_calls": 8, "actor_score_forward_batches": 4, "actor_score_backward_batches": 8}


def test_initial_credit_control_rejects_different_native_rollouts():
    batch = SimpleNamespace(state=np.zeros((2, 3)), action=np.zeros((2, 8)), reward=np.array([1., 2.]), old_logp=np.zeros(2))
    pair = {"upper_batches": [batch], "rows": [{"episode_return": 3.}]}
    groups = {method: {panel: [copy.deepcopy(pair)] for panel in ("A", "B")} for method in spec.METHODS}
    cost = {"initial_credit_pair_checks": 0}
    experiment.check_initial_credit_pairing(groups, cost)
    assert cost["initial_credit_pair_checks"] == 2
    groups["suffix"]["B"][0]["upper_batches"][0].action[0, 0] = .01
    with pytest.raises(AssertionError):
        experiment.check_initial_credit_pairing(groups, cost)


@pytest.mark.parametrize("preflight", [True, False])
def test_reduced_real_updates_frozen_lower_final_only_evaluation_and_exact_budget(tmp_path, preflight):
    torch.set_num_threads(1)
    models, predictor, _, calibration = fixture.OptionalPlanTest().sources()
    states = {}
    for period, model in models.items():
        actor = experiment.base.lower_training.branch(model)
        with torch.no_grad():
            actor.readout.weight[0, 392] = .8
            actor.readout.weight[1, 393] = .6
        states[period] = copy.deepcopy(actor.state_dict())
    before = {key: copy.deepcopy(model.state_dict()) for key, model in models.items()}
    lower_before = copy.deepcopy(states)
    options, arguments = spec.options, spec.arguments
    small = lambda **kw: {**options(**kw), "updates": 1, "credit_scenarios_per_batch": 2, "evaluation_episodes": 1, "workers": 1}
    short = lambda root, **kw: SimpleNamespace(**{**vars(arguments(root, **kw)), "horizon": 100})
    source = experiment.base.source
    experiment.write_json(tmp_path / "source.json", {})
    with patch.object(spec, "options", side_effect=small), patch.object(spec, "arguments", side_effect=short), \
            patch.object(experiment, "check_prerequisite"), patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(spec, "source_result", return_value=tmp_path / "source.json"), \
            patch.object(source, "qualify", side_effect=lambda cell, **kw: cell), \
            patch.object(source, "load_source", return_value=(models, predictor, {}, calibration)), \
            patch.object(experiment.base, "load_lower_state", side_effect=lambda root, period, **kw: states[str(period)]), \
            patch.object(source.native.joint, "_make_task", side_effect=lambda **kw: Float32Task()), \
            patch.object(source.native.joint, "pointmaze_goal_bounds", return_value=(-2*np.ones(2), 2*np.ones(2))):
        cell = experiment.run(410011, preflight=preflight, output=tmp_path / "result.json")
        assert cell["cost"] == spec.budget(preflight=preflight)
        assert cell["cost"]["initial_credit_pair_checks"] == 8
        assert cell["cost"]["residual_parameter_updates"] == 4
        assert len(list(tmp_path.rglob("*.pt"))) == (0 if preflight else 4)
        assert not list(tmp_path.rglob("*.npz"))
        for period, group in cell["groups"].items():
            source.native.curves.support.assert_frozen(models[period], before[period])
            for method in spec.METHODS:
                assert len(group["histories"][method]) == 1
                assert group["histories"][method][0]["geometry"]["gradient_norm"] > 0
                assert group["evaluation"][method][0]["upper_calls"] == 100 // int(period)
            assert group["evaluation"]["forecast"][0]["upper_calls"] == 0
            assert group["evaluation"]["suffix_blinded"][0]["episode_return"] == group["evaluation"]["forecast"][0]["episode_return"]
        torch.testing.assert_close(states, lower_before, atol=0, rtol=0)
        if preflight:
            assert experiment.aggregate([cell], preflight=True)["upper_credit_gain_gate"] == "mechanical_only"
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["histories"]["suffix"] = []
        with pytest.raises(ValueError, match="non-final iteration"):
            experiment.qualify(bad, preflight=preflight)
        bad = copy.deepcopy(cell)
        bad["groups"]["50"]["evaluation"]["suffix_blinded"][0]["episode_return"] += 1.
        with pytest.raises(ValueError, match="blinded plan"):
            experiment.qualify(bad, preflight=preflight)


def test_prerequisite_requires_full_not_preflight_plan_headroom(tmp_path):
    path = tmp_path / "qualification_summary.json"
    with patch.object(spec, "prerequisite_summary", return_value=path):
        experiment.write_json(path, {"protocol": "pointmaze_local_plan_gain_stage117_v1", "status": "complete", "local_plan_gain_gate": "supported_both_periods"})
        experiment.check_prerequisite()
        experiment.write_json(path, {"protocol": "pointmaze_local_plan_gain_stage117_v1", "status": "preflight_passed", "local_plan_gain_gate": "mechanical_only"})
        with pytest.raises(ValueError, match="full Stage117"):
            experiment.check_prerequisite()


def test_frozen_rosters_budget_dynamic_scheduler_and_primary_endpoints():
    seen = set()
    for preflight in (True, False):
        for root in spec.roots(preflight=preflight):
            roster = spec.seed_roles(root, preflight=preflight)
            seeds = [seed for row in roster["training_rounds"] for rows in row.values() for q in rows
                     for seed in [q["scenario_seed"], *q["noise_seeds"]]] + roster["native_evaluation"]
            assert len(set(seeds)) == len(seeds)
            assert not seen.intersection(seeds)
            seen.update(seeds)
            task = task_specification("stage118_unit", root, preflight=preflight)
            assert task["require_node"] is None
            assert task["allowed_nodes"] == [f"node{i:03d}" for i in range(1, 7)]
            assert task["cpu"] == spec.options(preflight=preflight)["workers"] + 1
            assert task["result_dir"].endswith("/completion")
    assert spec.budget(preflight=False)["native_episodes"] * 8 == 18432
    assert spec.budget(preflight=False)["native_steps"] * 8 == 22118400
    assert spec.budget(preflight=True)["native_steps"] == 28800
    assert len(qualification_task("stage118_unit", preflight=False)["wait_for_files"]) == 8
    assert len(spec.ENDPOINTS) == 4
    assert all("blinded" not in key for key in spec.ENDPOINTS)
    with pytest.raises(ValueError, match="all frozen roots"):
        experiment.aggregate([], preflight=False)


def test_gain_gate_requires_both_periods_and_credit_control():
    endpoints = {key: {"ci": [.1, .2]} for key in spec.ENDPOINTS}
    with patch.object(experiment.statistics, "aggregate", side_effect=lambda *a, **kw: {"endpoints": copy.deepcopy(endpoints)}):
        assert experiment.aggregate([], preflight=False)["upper_credit_gain_gate"] == "supported_both_periods"
        endpoints["50/suffix_minus_option"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["upper_credit_gain_gate"] == "partial"
        endpoints["100/suffix_minus_forecast"]["ci"] = [-.1, .2]
        assert experiment.aggregate([], preflight=False)["upper_credit_gain_gate"] == "not_supported"
