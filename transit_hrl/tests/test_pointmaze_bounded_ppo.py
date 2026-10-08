import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_bounded_ppo as experiment
from scripts import pointmaze_bounded_ppo_stage137_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def test_projection_preserves_causal_rows_bias_std_and_is_idempotent():
    torch.manual_seed(137)
    displacement = {"net.0.weight": torch.randn(8, 392), "net.0.bias": torch.randn(8), "log_std": torch.zeros(8)}
    before = copy.deepcopy(displacement)
    projected, outside = experiment.project_displacement(displacement)
    again, _ = experiment.project_displacement(projected)
    p = experiment.causal_summary_projection()
    np.testing.assert_allclose(projected["net.0.weight"].double().numpy() @ p.T,
        before["net.0.weight"].double().numpy() @ p.T, atol=1e-6, rtol=0)
    torch.testing.assert_close(projected, again, atol=2e-7, rtol=0)
    torch.testing.assert_close(displacement, before, atol=0, rtol=0)
    torch.testing.assert_close(projected["net.0.bias"], before["net.0.bias"], atol=0, rtol=0)
    torch.testing.assert_close(projected["log_std"], before["log_std"], atol=0, rtol=0)
    assert 0 < outside < 1


def test_candidates_match_KL_both_signs_and_leave_initial_policy_unchanged(source_data):
    trainer = warm_trainer(source_data)
    actor = trainer.upper_actor
    before = copy.deepcopy(actor.state_dict())
    updated = copy.deepcopy(before)
    torch.manual_seed(137)
    updated["net.0.weight"] += .001*torch.randn_like(updated["net.0.weight"])
    updated["net.0.bias"] += .03
    states = torch.randn(64, 392).numpy()
    candidates, geometry, work = experiment.candidates(actor, updated, states)
    assert set(candidates) == set(spec.VARIANTS)-{"source_forecast"}
    assert geometry["raw_adam_policy"]["empirical_KL"] > 10*spec.FISHER_RADIUS
    for method in spec.METHODS:
        for sign in ("plus", "minus"):
            np.testing.assert_allclose(geometry[method+"_"+sign+"_policy"]["empirical_KL"], spec.FISHER_RADIUS, atol=1e-8, rtol=0)
            torch.testing.assert_close(candidates[method+"_"+sign]["log_std"], before["log_std"], atol=0, rtol=0)
        for key in before:
            torch.testing.assert_close((candidates[method+"_plus"][key]+candidates[method+"_minus"][key])/2,
                before[key], atol=1e-7, rtol=0)
    torch.testing.assert_close(actor.state_dict(), before, atol=0, rtol=0)
    assert work == {"fisher_jvp_batches": 2, "exact_kl_forward_batches": 4, "policy_geometry_forward_batches": 7}


def test_rosters_budget_and_tracking_sign_keep_replay_separate():
    assert spec.budget()["native_episodes"] == 480
    assert spec.budget()["native_steps"] == 576000
    assert spec.budget()["upper_actor_optimizer_steps"] == 4
    assert spec.budget()["lower_actor_optimizer_steps"] == 0
    assert spec.budget()["checkpoint_writes"] == spec.budget()["native_trace_writes"] == 0
    assert spec.budget()["policy_geometry_forward_batches"] == 14
    for root in spec.ROOTS:
        roles = spec.seed_roles(root)
        assert roles["replayed_training"] == spec.source.seed_roles(root)["training"]
        assert not set(roles["evaluation"]).intersection(spec.source.seed_roles(root)["evaluation"])
    rows = [{v: {**dict.fromkeys(spec.METRICS, 0.), "episode_return": i,
        "tracking_squared_error_integral": 100-i} for i, v in enumerate(spec.VARIANTS)}]
    summary = experiment.evaluation_summary(rows)
    for a,b in spec.CONTRASTS:
        key = f"{a}_minus_{b}"
        assert summary["effects"][key] == summary["tracking_error_reduction_positive_is_better"][key]


def test_reduced_runner_reproduces_recorded_update_and_exact_budget(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    warm_path, prior_path = tmp_path/"warm"/"result.json", tmp_path/"prior"/"result.json"
    warm_cached = {"status": "complete", "protocol": spec.source.source.PROTOCOL, "root": root,
        "cost": spec.source.source.budget(), "inherited_source_cost": spec.source.source.source.budget(), "groups": {}}
    for period in spec.PERIODS:
        upper = experiment.joint.weights(warm_trainer(source_data))["upper_actor"]
        warm_cached["groups"][str(period)] = {"selection": {"method": "refresh"},
            "training_native_return_fits": {"refresh": {"pooled": {"scale": .4}}}}
        path = warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": spec.source.source.PROTOCOL, "root": root, "period": period,
            "method": "refresh", "fit": {"scale": .4}, "weights": upper}, path)
    experiment.write_json(warm_path, warm_cached)
    with patch.object(spec.source, "arguments", return_value=args), patch.object(spec, "arguments", return_value=args), \
            patch.object(spec.source, "SCENARIOS", 2), patch.object(spec.source, "EVALUATION_EPISODES", 2), \
            patch.object(spec, "EVALUATION_EPISODES", 2), \
            patch.object(spec.source, "source_result", return_value=warm_path), \
            patch.object(spec, "source_result", return_value=prior_path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r,p,**kw: teachers[str(p)]), \
            patch.object(experiment.warm, "ProcessPoolExecutor", ImmediatePool), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        prior = experiment.warm.run(root, prior_path)
        result = experiment.run(root, tmp_path/"output"/"result.json")
        assert result["cost"] == spec.budget() and result["cost"]["native_episodes"] == 36
        assert result["inherited_Stage136_cost"] == prior["cost"]
        assert (tmp_path/"output"/"completion"/"ready.json").is_file()
        assert not list((tmp_path/"output").rglob("*.pt"))
        for group in result["groups"].values():
            assert group["training_replay_and_original_update"] == "passed"
            assert group["deployed_lower_critics_teacher_and_std_frozen"] == "passed"
        prior["groups"]["50"]["training_returns"]["sampled"][0] += .01
        experiment.write_json(prior_path, prior)
        with pytest.raises(AssertionError):
            experiment.run(root, tmp_path/"bad"/"result.json")
