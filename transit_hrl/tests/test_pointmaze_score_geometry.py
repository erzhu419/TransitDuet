import copy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from freq_hrl.experiments import pointmaze_score_geometry as experiment
from freq_hrl.experiments.pointmaze_native_direction import parameter_tangents
from scripts import pointmaze_score_geometry_stage139_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_warm_start_joint import warm_trainer,native_patch,bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def credit_rows(source_data):
    models,predictor,calibrations,args,teachers=source_data
    initial=experiment.joint.weights(warm_trainer(source_data))
    roles=spec.source.seed_roles(spec.ROOTS[0])["training"][:2]
    jobs=[(experiment.joint.weights(models["50"]),teachers["50"],initial,r,p,50,predictor,calibrations["50"]["envelope"])
        for r in roles for p in spec.source.PANELS]
    with patch.object(experiment.joint.source.native,"_WORKER",(models["50"],args)),native_patch(),bounds_patch():
        return [experiment.previous.worker_credit(j) for j in jobs]


def test_stochastic_evaluation_matches_collected_path_without_a_training_batch(source_data):
    models,predictor,calibrations,args,teachers=source_data
    trainer=warm_trainer(source_data)
    call=dict(args=args,seed=139001,noise_seed=139002,arm="joint",period=50,
        predictor=predictor,envelope=calibrations["50"]["envelope"])
    with native_patch(),bounds_patch():
        batch,sampled,audit=experiment.joint.native_episode(trainer,**call,collect=True)
        empty,replay,replay_audit=experiment.joint.native_episode(trainer,**call,collect=False,sample_upper=True)
        _,mean,mean_audit=experiment.joint.native_episode(trainer,**call,collect=False)
    assert batch.upper.size==2 and empty is None
    assert sampled==replay and sampled["upper_sample"] and not mean["upper_sample"]
    assert sampled["episode_return"] != mean["episode_return"]
    for key in audit:np.testing.assert_array_equal(audit[key],replay_audit[key])
    np.testing.assert_array_equal(audit["measurements"],mean_audit["measurements"])
    np.testing.assert_allclose(audit["innovations"],mean_audit["innovations"],atol=3e-5,rtol=0)
    assert mean_audit["upper_innovations"].size==0


def test_score_is_the_autograd_ascent_and_natural_direction_uses_same_signal(source_data):
    rows=credit_rows(source_data)
    trainer=warm_trainer(source_data)
    batch=experiment.concat_level_batches([r["batch"] for r in rows])
    credit=np.asarray([r["credit"] for r in rows],dtype=np.float32).reshape(-1)
    before=copy.deepcopy(trainer.upper_actor.state_dict())
    vectors,report=experiment.score_directions(trainer.upper_actor,batch,credit,clip_ratio=trainer.config.clip_ratio)
    score=parameter_tangents(trainer.upper_actor,vectors["score"])
    with torch.no_grad():
        dist=trainer.upper_actor.distribution(torch.as_tensor(batch.state))
        ratio=torch.exp(dist.log_prob(torch.as_tensor(batch.action)).sum(-1)-torch.as_tensor(batch.old_logp))
        signal=(torch.as_tensor(trainer._normalize(credit))[:,None]*ratio[:,None]*
            (torch.as_tensor(batch.action)-dist.mean)/dist.stddev.square())
    expected_weight=signal.T@torch.as_tensor(batch.state)/batch.size
    torch.testing.assert_close(score["net.0.weight"],expected_weight,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(score["net.0.bias"],signal.mean(0),atol=2e-6,rtol=2e-5)
    assert not score["log_std"].any()
    assert vectors["score"]@vectors["natural_score"]>0
    assert report["damping"]==1. and report["max_logp_replay_error"]<experiment.joint.spec.LOGP_REPLAY_TOLERANCE
    torch.testing.assert_close(trainer.upper_actor.state_dict(),before,atol=0,rtol=0)


def test_candidates_replay_Adam_and_match_KL_for_all_geometries(source_data):
    rows=credit_rows(source_data)
    models,predictor,calibrations,args,teachers=source_data
    trainer=warm_trainer(source_data)
    initial=experiment.joint.weights(trainer)
    twin=warm_trainer(source_data)
    cached=experiment.previous.update_upper(twin,rows,root=spec.ROOTS[0],period=50,method="option_credit")
    uppers,report,cost=experiment.candidates(trainer,rows,spec.ROOTS[0],50,cached)
    assert set(uppers)==set(spec.VARIANTS)-{"source_forecast"}
    assert report["original_update_replay"]==cached
    assert cost=={"empirical_fisher_solves":1,"fisher_jvp_batches":3,"exact_kl_forward_batches":6}
    for method in spec.METHODS:
        for sign in ("plus","minus"):
            np.testing.assert_allclose(report["radii"][method]["exact_kl"][sign],spec.FISHER_RADIUS,atol=1e-8,rtol=0)
            torch.testing.assert_close(uppers[method+"_"+sign]["log_std"],initial["upper_actor"]["log_std"],atol=0,rtol=0)
        for key in uppers["warm_start"]:
            torch.testing.assert_close((uppers[method+"_plus"][key]+uppers[method+"_minus"][key])/2,
                initial["upper_actor"][key],atol=1e-7,rtol=0)
    torch.testing.assert_close(experiment.joint.weights(trainer),initial,atol=0,rtol=0)


def test_budget_roster_forecast_alias_and_metric_sign():
    expected=spec.budget()
    assert expected["native_episodes"]==992 and expected["native_steps"]==1190400
    assert expected["replay_episodes"]==32 and expected["counterfactual_episodes"]==0
    assert expected["evaluation_episodes"]==960 and expected["evaluation_alias_assignments"]==64
    assert expected["upper_actor_optimizer_steps"]==expected["upper_value_optimizer_steps"]==4
    assert expected["score_gradient_batches"]==10 and expected["empirical_fisher_solves"]==2
    assert expected["checkpoint_writes"]==expected["native_trace_writes"]==0
    for root in spec.ROOTS:
        old=spec.source.seed_roles(root)
        roles=spec.seed_roles(root)
        assert roles["replayed_training"]==old["training"]
        assert not set(roles["evaluation"]).intersection(old["evaluation"])
    rows=[{mode:{v:{**dict.fromkeys(spec.METRICS,0.),"episode_return":i,
        "tracking_squared_error_integral":100-i} for i,v in enumerate(spec.VARIANTS)} for mode in spec.MODES}]
    summary=experiment.evaluation_summary(rows)
    for mode in spec.MODES:
        assert summary["evaluation"][mode]["effects"]==summary["evaluation"][mode]["tracking_error_reduction_positive_is_better"]
    assert summary["sampled_minus_mean_return"]["source_forecast"]["mean"]==0


def test_reduced_runner_restores_cached_credit_and_counts_actual_dual_mode_episodes(source_data,tmp_path):
    models,predictor,calibrations,args,teachers=source_data
    root=spec.ROOTS[0]
    warm_path,prior_path=tmp_path/"warm"/"result.json",tmp_path/"prior"/"result.json"
    earlier_path=tmp_path/"earlier"/"result.json"
    warm_spec=spec.source.source.source.source
    warm_cached={"status":"complete","protocol":warm_spec.PROTOCOL,"root":root,"cost":warm_spec.budget(),
        "inherited_source_cost":warm_spec.source.budget(),"groups":{}}
    for period in spec.PERIODS:
        warm_cached["groups"][str(period)]={"selection":{"method":"refresh"},
            "training_native_return_fits":{"refresh":{"pooled":{"scale":.4}}}}
        checkpoint=warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"protocol":warm_spec.PROTOCOL,"root":root,"period":period,"method":"refresh",
            "fit":{"scale":.4},"weights":experiment.joint.weights(warm_trainer(source_data))["upper_actor"]},checkpoint)
    experiment.write_json(warm_path,warm_cached)
    experiment.write_json(earlier_path,{"status":"complete","protocol":spec.source.source.PROTOCOL,"root":root,
        "cost":spec.source.source.budget()})
    with patch.object(spec,"arguments",return_value=args),patch.object(spec,"EVALUATION_EPISODES",2), \
            patch.object(spec,"source_result",return_value=prior_path),patch.object(spec,"warm_result",return_value=warm_path), \
            patch.object(spec.source,"arguments",return_value=args),patch.object(spec.source,"SCENARIOS",2), \
            patch.object(spec.source,"EVALUATION_EPISODES",2),patch.object(spec.source,"source_result",return_value=earlier_path), \
            patch.object(spec.source,"warm_result",return_value=warm_path), \
            patch.object(experiment.warm.spec,"source_result",return_value=warm_path), \
            patch.object(experiment.joint.source,"load_source",return_value=(models,predictor,{},calibrations)), \
            patch.object(experiment.joint.base,"load_lower_state",side_effect=lambda r,p,**kw:teachers[str(p)]), \
            patch.object(experiment.previous,"ProcessPoolExecutor",ImmediatePool), \
            patch.object(experiment,"ProcessPoolExecutor",ImmediatePool),native_patch(),bounds_patch():
        prior=experiment.previous.run(root,prior_path)
        output=tmp_path/"output"/"result.json"
        result=experiment.run(root,output)
        assert result["cost"]==spec.budget() and result["cost"]["native_episodes"]==68
        assert result["cost"]["evaluation_alias_assignments"]==4
        assert result["inherited_Stage138_cost"]==prior["cost"]
        assert (output.parent/"completion"/"ready.json").is_file()
        assert not list(output.parent.rglob("*.pt")) and not list(output.parent.rglob("*.npz"))
        for group in result["groups"].values():
            assert group["original_training_and_update_replay"]=="passed"
            assert group["frozen_deployment_and_mode_noise_pairing"]=="passed"
            assert group["sampled_minus_mean_return"]["source_forecast"]["paired_differences"]==[0.,0.]
            assert set(group["evaluation"])==set(spec.MODES)
        prior["groups"]["50"]["counterfactual_paths"][0]["sampled_return"]+=.1
        experiment.write_json(prior_path,prior)
        with pytest.raises(AssertionError):experiment.run(root,tmp_path/"bad"/"result.json")
