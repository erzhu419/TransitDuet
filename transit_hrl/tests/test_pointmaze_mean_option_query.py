import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_mean_option_query as experiment
from scripts import pointmaze_mean_option_query_stage141_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def job_for(source_data, period, panel="A", role=None):
    models, predictor, calibrations, args, teachers = source_data
    initial = experiment.previous.scale_initial(experiment.joint.weights(warm_trainer(source_data)), "reduced")
    return (experiment.joint.weights(models[str(period)]), teachers[str(period)], initial,
        role or spec.seed_roles(spec.ROOTS[0])["replayed_training"][0], panel, period,
        predictor, calibrations[str(period)]["envelope"])


def test_antithetic_query_gradient_has_correct_sign_and_cancels_quadratic_term():
    center, gradient = np.array([.4, -.2]), np.array([3., -2.])
    curvature = np.array([[2., .1], [.1, 4.]])
    innovations = np.sqrt(2)*np.eye(2)
    std = .05
    def value(x): return x@gradient+.5*x@curvature@x
    plus = [value(center+std*e) for e in innovations]
    minus = [value(center-std*e) for e in innovations]
    targets = experiment.paired_query_gradient(plus, minus, innovations, std)
    np.testing.assert_allclose(targets.mean(0), gradient+curvature@center, atol=1e-13, rtol=0)
    np.testing.assert_allclose(experiment.paired_query_gradient(minus, plus, innovations, std), -targets, atol=0, rtol=0)


@pytest.mark.parametrize("period,episodes", [(50, 5), (100, 3)])
def test_mean_queries_pair_prefix_future_mean_and_full_cost(source_data, period, episodes):
    models, predictor, calibrations, args, teachers = source_data
    job = job_for(source_data, period)
    before = copy.deepcopy(job[2])
    with patch.object(experiment.joint.source.native, "_WORKER", (models[str(period)], args)), native_patch(), bounds_patch():
        result = experiment.worker_mean_query(job)
    assert result["cost"]["native_episodes"] == episodes
    assert result["cost"]["native_steps"] == episodes*args.horizon
    assert result["cost"]["counterfactual_episodes"] == episodes-1
    assert result["cost"]["mean_query_label_pairs"] == args.horizon//period
    assert result["cost"]["credit_checks"] == episodes
    assert result["path"]["prefix_future_mean_and_noise_pairing"] == "passed"
    assert result["gradient"].shape == (args.horizon//period, spec.source.ppo.UPPER_ACTION_DIM)
    assert np.isfinite(result["gradient"]).all() and np.any(result["gradient"] != 0)
    torch.testing.assert_close(job[2], before, atol=0, rtol=0)


def test_mean_fit_keeps_fixed_std_and_both_mean_steps(source_data):
    models, predictor, calibrations, args, teachers = source_data
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"], args)), native_patch(), bounds_patch():
        rows = [experiment.worker_mean_query(job_for(source_data, 50, p)) for p in spec.PANELS]
    trainer = warm_trainer(source_data)
    initial = job_for(source_data, 50)[2]
    experiment.joint.load_weights(trainer, initial)
    actors, report, cost = experiment.mean_candidates(trainer, rows)
    for sign in ("plus", "minus"):
        np.testing.assert_allclose(report["mean_step_RMS"][sign], spec.MEAN_STEP_RMS, atol=1e-8, rtol=0)
        np.testing.assert_allclose(report["radius"]["exact_kl"][sign], spec.RADIUS, rtol=1e-5, atol=0)
        torch.testing.assert_close(actors[sign]["log_std"], initial["upper_actor"]["log_std"], atol=0, rtol=0)
    assert np.isfinite(report["noise_fold_gradient_cosine"])
    assert cost["empirical_fisher_solves"] == 1 and cost["upper_candidate_weight_steps"] == 2
    torch.testing.assert_close(experiment.joint.weights(trainer), initial, atol=0, rtol=0)
    assert not trainer.upper_actor_optimizer.state and not trainer.upper_value_optimizer.state


def test_budget_keeps_query_pairs_replay_and_inherited_seed_roles_separate():
    budget = spec.budget()
    assert budget["native_episodes"] == 1920 and budget["native_steps"] == 2304000
    assert budget["replay_episodes"] == budget["collection_episodes"] == 32
    assert budget["counterfactual_episodes"] == 1152 and budget["mean_query_label_pairs"] == 576
    assert budget["evaluation_episodes"] == 704 and budget["evaluation_alias_assignments"] == 64
    assert budget["credit_checks"] == 1216 and budget["empirical_fisher_solves"] == 4
    assert budget["score_gradient_batches"] == 2 and budget["mean_score_forward_batches"] == 2
    assert budget["upper_actor_optimizer_steps"] == budget["upper_value_optimizer_steps"] == 0
    seen = set()
    for root in spec.ROOTS:
        role, old = spec.seed_roles(root), spec.source.seed_roles(root)
        assert role["replayed_training"] == old["training"]
        prior = {s for r in old["training"] for s in (r["scenario_seed"], *r["noise_seeds"].values())} | set(old["evaluation"])
        assert not set(role["evaluation"]).intersection(prior | seen)
        seen.update(role["evaluation"])


def test_native_run_replays_credit_counts_actual_queries_and_writes_only_json(source_data, tmp_path):
    models, predictor, calibrations, args, teachers = source_data
    root = spec.ROOTS[0]
    source_path, warm_path = tmp_path/"prior"/"result.json", tmp_path/"warm"/"result.json"
    with patch.object(spec, "arguments", return_value=args), patch.object(spec.source, "SCENARIOS", 2), \
            patch.object(spec, "EVALUATION_EPISODES", 2), patch.object(spec, "source_result", return_value=source_path), \
            patch.object(spec, "warm_result", return_value=warm_path), \
            patch.object(experiment.warm.spec, "source_result", return_value=warm_path), \
            patch.object(experiment.joint.source, "load_source", return_value=(models, predictor, {}, calibrations)), \
            patch.object(experiment.joint.base, "load_lower_state", side_effect=lambda r, p, **kw: teachers[str(p)]), \
            patch.object(experiment, "ProcessPoolExecutor", ImmediatePool), native_patch(), bounds_patch():
        cached = {"status": "complete", "protocol": spec.source.PROTOCOL, "root": root,
            "cost": spec.source.budget(), "seed_roles": spec.source.seed_roles(root), "groups": {},
            "inherited_Stage139_cost": {"retained": True}}
        warm = {"status": "complete", "protocol": spec.source.warm_source.PROTOCOL, "root": root,
            "cost": spec.source.warm_source.budget(), "groups": {}}
        for period in spec.PERIODS:
            roles = spec.seed_roles(root)["replayed_training"]
            with patch.object(experiment.joint.source.native, "_WORKER", (models[str(period)], args)):
                rows = [experiment.credit.worker_credit(job_for(source_data, period, p, r)) for r in roles for p in spec.PANELS]
            cached["groups"][str(period)] = {"training": {"reduced": {"counterfactual_paths": [r["path"] for r in rows]}}}
            warm["groups"][str(period)] = {"selection": {"method": "refresh"},
                "training_native_return_fits": {"refresh": {"pooled": {"scale": .4}}}}
            checkpoint = warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol": spec.source.warm_source.PROTOCOL, "root": root, "period": period, "method": "refresh",
                "fit": {"scale": .4}, "weights": experiment.joint.weights(warm_trainer(source_data))["upper_actor"]}, checkpoint)
        experiment.write_json(source_path, cached); experiment.write_json(warm_path, warm)
        result = experiment.run(root, tmp_path/"output"/"result.json")
        assert result["cost"] == spec.budget() and result["cost"]["native_episodes"] == 84
        assert result["cost"]["native_steps"] == 8400 and result["cost"]["mean_query_label_pairs"] == 12
        assert result["inherited_Stage140_cost"] == cached["cost"]
        assert result["inherited_earlier_source_cost"]["inherited_Stage139_cost"] == {"retained": True}
        assert (tmp_path/"output"/"completion"/"ready.json").is_file()
        assert not list((tmp_path/"output").rglob("*.pt")) and not list((tmp_path/"output").rglob("*.npz"))
        for group in result["groups"].values():
            assert group["sampled_credit_replay"] == group["frozen_deployment_and_noise_pairing"] == "passed"
            assert len(group["mean_query_paths"]) == 4
            assert group["sampled_minus_mean_return"]["source_forecast"]["paired_differences"] == [0., 0.]
