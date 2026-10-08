import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_option_counterfactual_credit as experiment
from scripts import pointmaze_option_credit_stage138_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_warm_start_joint import warm_trainer, native_patch, bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def job_for(source_data, period, panel="A"):
    models, predictor, calibrations, args, teachers = source_data
    return (experiment.joint.weights(models[str(period)]), teachers[str(period)],
        experiment.joint.weights(warm_trainer(source_data)), spec.seed_roles(spec.ROOTS[0])["training"][0],
        panel, period, predictor, calibrations[str(period)]["envelope"])


def test_override_with_original_draw_is_identical_and_none_preserves_default(source_data):
    models, predictor, calibrations, args, teachers = source_data
    trainer = warm_trainer(source_data)
    call = dict(args=args, seed=138001, noise_seed=138002, arm="joint", period=50,
        predictor=predictor, envelope=calibrations["50"]["envelope"], collect=True)
    with native_patch(), bounds_patch():
        batch, row, audit = experiment.joint.native_episode(trainer, **call)
        for override in (None, {"step": 50, "action": batch.upper.action[1]}):
            replay, replay_row, replay_audit = experiment.joint.native_episode(trainer, **call, upper_override=override)
            assert row == replay_row
            for level in ("upper", "lower"):
                for name in ("state", "action", "reward", "duration", "done", "old_logp", "old_value"):
                    np.testing.assert_array_equal(getattr(getattr(batch, level), name), getattr(getattr(replay, level), name))
            for name in audit:
                np.testing.assert_array_equal(audit[name], replay_audit[name])


@pytest.mark.parametrize("period,episodes", [(50,3), (100,2)])
def test_option_credit_counts_full_replays_and_keeps_only_sampled_batch(source_data, period, episodes):
    models, predictor, calibrations, args, teachers = source_data
    job = job_for(source_data, period)
    initial = copy.deepcopy(job[2])
    with patch.object(experiment.joint.source.native, "_WORKER", (models[str(period)],args)), native_patch(), bounds_patch():
        row = experiment.worker_credit(job)
    assert row["batch"].size == args.horizon//period
    assert row["cost"]["native_episodes"] == episodes
    assert row["cost"]["native_steps"] == episodes*args.horizon
    assert row["cost"]["counterfactual_episodes"] == episodes-1
    assert row["cost"]["credit_checks"] == episodes
    assert row["cost"]["checkpoint_writes"] == row["cost"]["native_trace_writes"] == 0
    np.testing.assert_allclose(row["credit"], row["path"]["sampled_return"]-
        np.asarray(row["path"]["mean_option_baseline_returns"]), atol=0, rtol=0)
    assert any(abs(x) > 1e-6 for x in row["credit"])
    torch.testing.assert_close(job[2], initial, atol=0, rtol=0)


def test_actor_credit_changes_actor_not_critic_targets_or_update(source_data):
    models, predictor, calibrations, args, teachers = source_data
    with patch.object(experiment.joint.source.native, "_WORKER", (models["50"],args)), native_patch(), bounds_patch():
        rows = [experiment.worker_credit(job_for(source_data,50,panel)) for panel in spec.PANELS]
    initial = job_for(source_data,50)[2]
    trainers = {}
    for method in spec.METHODS:
        trainer = experiment.joint.make_trainer(models["50"],teachers["50"],args)
        experiment.joint.load_weights(trainer,initial)
        report = experiment.update_upper(trainer,rows,root=spec.ROOTS[0],period=50,method=method)
        assert report["parameter_delta_rms"]["upper_actor"] > 0
        assert report["parameter_delta_rms"]["upper_value"] > 0
        assert report["parameter_delta_rms"]["lower_actor"] == report["parameter_delta_rms"]["lower_value"] == 0
        torch.testing.assert_close(trainer.upper_actor.log_std,initial["upper_actor"]["log_std"],atol=0,rtol=0)
        trainers[method] = trainer
    torch.testing.assert_close(trainers["critic"].upper_value.state_dict(),
        trainers["option_credit"].upper_value.state_dict(),atol=0,rtol=0)
    torch.testing.assert_close(trainers["critic"].upper_value_optimizer.state_dict(),
        trainers["option_credit"].upper_value_optimizer.state_dict(),atol=0,rtol=0)
    assert not torch.equal(trainers["critic"].upper_actor.net[0].weight,trainers["option_credit"].upper_actor.net[0].weight)


def test_rosters_exact_budget_and_tracking_sign():
    expected = spec.budget()
    assert expected["native_episodes"] == 992 and expected["native_steps"] == 1190400
    assert expected["collection_episodes"] == 32 and expected["counterfactual_episodes"] == 576
    assert expected["evaluation_episodes"] == 384 and expected["credit_checks"] == 608
    assert expected["upper_actor_optimizer_steps"] == expected["upper_value_optimizer_steps"] == 8
    assert expected["lower_actor_optimizer_steps"] == expected["lower_value_optimizer_steps"] == 0
    assert expected["checkpoint_writes"] == expected["native_trace_writes"] == 0
    assert spec.WORKERS == 16
    seen = set()
    for root in spec.ROOTS:
        roles = spec.seed_roles(root)
        seeds = [s for r in roles["training"] for s in (r["scenario_seed"],*r["noise_seeds"].values())]+roles["evaluation"]
        assert len(seeds) == len(set(seeds)) and not seen.intersection(seeds)
        seen.update(seeds)
        old = spec.source.seed_roles(root)
        previous = [s for r in old["replayed_training"] for s in (r["scenario_seed"],*r["noise_seeds"])] + old["evaluation"]
        assert not set(seeds).intersection(previous)
    rows = [{v:{**dict.fromkeys(spec.METRICS,0.),"episode_return":i,
        "tracking_squared_error_integral":100-i} for i,v in enumerate(spec.VARIANTS)}]
    summary = experiment.evaluation_summary(rows)
    assert summary["effects"] == summary["tracking_error_reduction_positive_is_better"]
    assert all("ci" not in e for e in summary["effects"].values())


def test_reduced_native_runner_has_exact_cost_and_scalar_artifacts(source_data,tmp_path):
    models,predictor,calibrations,args,teachers = source_data
    root = spec.ROOTS[0]
    source_path,warm_path = tmp_path/"source"/"result.json",tmp_path/"warm"/"result.json"
    warm_spec = spec.source.source.source
    cached = {"status":"complete","protocol":spec.source.PROTOCOL,"root":root,"cost":spec.source.budget()}
    warm_cached = {"status":"complete","protocol":warm_spec.PROTOCOL,"root":root,"cost":warm_spec.budget(),
        "inherited_source_cost":warm_spec.source.budget(),"groups":{}}
    for period in spec.PERIODS:
        warm_cached["groups"][str(period)] = {"selection":{"method":"refresh"},
            "training_native_return_fits":{"refresh":{"pooled":{"scale":.4}}}}
        checkpoint = warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"protocol":warm_spec.PROTOCOL,"root":root,"period":period,"method":"refresh",
            "fit":{"scale":.4},"weights":job_for(source_data,period)[2]["upper_actor"]},checkpoint)
    experiment.write_json(source_path,cached)
    experiment.write_json(warm_path,warm_cached)
    with patch.object(spec,"arguments",return_value=args),patch.object(spec,"SCENARIOS",2), \
            patch.object(spec,"EVALUATION_EPISODES",2),patch.object(spec,"source_result",return_value=source_path), \
            patch.object(spec,"warm_result",return_value=warm_path), \
            patch.object(experiment.warm.spec,"source_result",return_value=warm_path), \
            patch.object(experiment.joint.source,"load_source",return_value=(models,predictor,{},calibrations)), \
            patch.object(experiment.joint.base,"load_lower_state",side_effect=lambda r,p,**kw:teachers[str(p)]), \
            patch.object(experiment,"ProcessPoolExecutor",ImmediatePool),native_patch(),bounds_patch():
        output = tmp_path/"output"/"result.json"
        result = experiment.run(root,output)
        assert result["cost"] == spec.budget() and result["cost"]["native_episodes"] == 44
        assert result["cost"]["native_steps"] == 4400
        assert result["cost"]["counterfactual_episodes"] == 12
        assert result["inherited_Stage137_cost"] == cached["cost"]
        assert result["inherited_Stage135_cost"] == warm_cached["cost"]
        assert (output.parent/"completion"/"ready.json").is_file()
        assert not list(output.parent.rglob("*.pt")) and not list(output.parent.rglob("*.npz"))
        for group in result["groups"].values():
            assert group["shared_critic_targets_and_update"] == "passed"
            assert group["deployed_lower_critics_teacher_and_std_frozen"] == "passed"
            assert len(group["counterfactual_paths"]) == 4
            assert all("batch" not in path and "state" not in path for path in group["counterfactual_paths"])
            for method in spec.METHODS:
                for sign in ("plus","minus"):
                    np.testing.assert_allclose(group["geometry"][method]["exact_kl"][sign],spec.FISHER_RADIUS,atol=1e-8,rtol=0)
