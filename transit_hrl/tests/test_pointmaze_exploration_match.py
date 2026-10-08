import copy
from unittest.mock import patch

import numpy as np
import torch

from freq_hrl.experiments import pointmaze_exploration_match as experiment
from scripts import pointmaze_exploration_match_stage140_spec as spec
from test_pointmaze_joint_reference import source_data
from test_pointmaze_warm_start_joint import warm_trainer,native_patch,bounds_patch
from test_pointmaze_update_isolation import ImmediatePool


def training_rows(source_data,scale):
    models,predictor,calibrations,args,teachers=source_data
    origin=experiment.joint.weights(warm_trainer(source_data))
    initial=experiment.scale_initial(origin,scale)
    roles=spec.seed_roles(spec.ROOTS[0])["training"][:2]
    with patch.object(experiment.joint.source.native,"_WORKER",(models["50"],args)),native_patch(),bounds_patch():
        return initial,[experiment.credit.worker_credit((experiment.joint.weights(models["50"]),teachers["50"],initial,
            r,p,50,predictor,calibrations["50"]["envelope"])) for r in roles for p in spec.PANELS]


def test_scale_initial_changes_only_std_and_keeps_source_frozen(source_data):
    origin=experiment.joint.weights(warm_trainer(source_data))
    before=copy.deepcopy(origin)
    reduced=experiment.scale_initial(origin,"reduced")
    for name in experiment.joint.NETWORKS:
        for key in origin[name]:
            if (name,key)==("upper_actor","log_std"):
                np.testing.assert_allclose(reduced[name][key].exp().numpy(),.05,atol=1e-8,rtol=0)
            else:torch.testing.assert_close(reduced[name][key],origin[name][key],atol=0,rtol=0)
    torch.testing.assert_close(origin,before,atol=0,rtol=0)


def test_std_specific_credits_and_radii_have_equal_mean_steps(source_data):
    models,predictor,calibrations,args,teachers=source_data
    outcomes={}
    for scale in spec.SCALES:
        initial,rows=training_rows(source_data,scale)
        trainer=warm_trainer(source_data)
        experiment.joint.load_weights(trainer,initial)
        uppers,report,cost=experiment.natural_candidates(trainer,rows,scale)
        assert set(uppers)=={"warm_start","natural_plus","natural_minus"}
        assert cost=={"empirical_fisher_solves":1,"fisher_jvp_batches":1,"exact_kl_forward_batches":2,
            "policy_geometry_forward_batches":3,"upper_candidate_weight_steps":2}
        for sign in ("plus","minus"):
            np.testing.assert_allclose(report["radius"]["exact_kl"][sign],spec.radius(scale),atol=0,rtol=1e-5)
            np.testing.assert_allclose(report["mean_step_RMS"][sign],spec.MEAN_STEP_RMS,atol=1e-8,rtol=0)
        torch.testing.assert_close(experiment.joint.weights(trainer),initial,atol=0,rtol=0)
        assert not trainer.upper_actor_optimizer.state and not trainer.upper_value_optimizer.state
        outcomes[scale]=rows
    assert [r["path"]["noise_seed"] for r in outcomes["original"]]==[r["path"]["noise_seed"] for r in outcomes["reduced"]]
    assert not np.array_equal(outcomes["original"][0]["credit"],outcomes["reduced"][0]["credit"])


def test_budget_and_rosters_count_queries_and_actual_aliases():
    budget=spec.budget()
    assert budget["native_episodes"]==1984 and budget["native_steps"]==2380800
    assert budget["collection_episodes"]==64 and budget["counterfactual_episodes"]==1152
    assert budget["evaluation_episodes"]==768 and budget["evaluation_alias_assignments"]==256
    assert budget["credit_checks"]==1216 and budget["empirical_fisher_solves"]==4
    assert budget["score_gradient_batches"]==20 and budget["policy_geometry_forward_batches"]==12
    assert budget["upper_actor_optimizer_steps"]==budget["upper_value_optimizer_steps"]==0
    assert budget["checkpoint_writes"]==budget["native_trace_writes"]==0
    np.testing.assert_allclose(spec.radius("reduced"),9*spec.radius("original"),atol=0,rtol=1e-14)
    assert sum(experiment.aliased(s,m,v) for s in spec.SCALES for m in spec.MODES for v in spec.VARIANTS)==4
    seen=set()
    for root in spec.ROOTS:
        roles=spec.seed_roles(root)
        seeds=[s for r in roles["training"] for s in (r["scenario_seed"],*r["noise_seeds"].values())]+roles["evaluation"]
        assert len(set(seeds))==len(seeds) and not seen.intersection(seeds)
        seen.update(seeds)
        old=spec.source.seed_roles(root)
        old_seeds=[s for r in old["replayed_training"] for s in (r["scenario_seed"],*r["noise_seeds"].values())]+old["evaluation"]
        assert not set(seeds).intersection(old_seeds)


def test_report_separates_less_exploration_loss_from_better_learning():
    values={"original":{"mean":(0,10,11,9),"sampled":(0,5,6,4)},
        "reduced":{"mean":(0,10,12,8),"sampled":(0,9,11,7)}}
    scene={s:{m:{v:{**dict.fromkeys(spec.METRICS,0.),"episode_return":x,
        "tracking_squared_error_integral":100-x} for v,x in zip(spec.VARIANTS,values[s][m])} for m in spec.MODES} for s in spec.SCALES}
    summary=experiment.evaluation_summary([scene])
    assert summary["reduced_minus_original_return"]["sampled"]["warm_start"]["mean"]==4
    assert summary["reduced_minus_original_return"]["sampled"]["natural_plus"]["mean"]==5
    assert summary["reduced_minus_original_learning_increment"]["sampled"]["mean"]==1
    assert summary["scales"]["reduced"]["sampled_minus_mean_return"]["warm_start"]["mean"]==-1
    for scale in spec.SCALES:
        for mode in spec.MODES:
            table=summary["scales"][scale]["evaluation"][mode]
            assert table["effects"]==table["tracking_error_reduction_positive_is_better"]


def test_reduced_native_runner_counts_fresh_std_queries_and_shared_baselines(source_data,tmp_path):
    models,predictor,calibrations,args,teachers=source_data
    root=spec.ROOTS[0]
    prior_path,warm_path=tmp_path/"prior"/"result.json",tmp_path/"warm"/"result.json"
    prior={"status":"complete","protocol":spec.source.PROTOCOL,"root":root,"cost":spec.source.budget(),
        "inherited_Stage138_cost":spec.source.source.budget(),"inherited_earlier_source_cost":{"retained":True}}
    cached={"status":"complete","protocol":spec.warm_source.PROTOCOL,"root":root,"cost":spec.warm_source.budget(),"groups":{}}
    for period in spec.PERIODS:
        cached["groups"][str(period)]={"selection":{"method":"refresh"},
            "training_native_return_fits":{"refresh":{"pooled":{"scale":.4}}}}
        path=warm_path.parent/"final_weights"/f"period_{period}_refresh_upper.pt"
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({"protocol":spec.warm_source.PROTOCOL,"root":root,"period":period,"method":"refresh",
            "fit":{"scale":.4},"weights":experiment.joint.weights(warm_trainer(source_data))["upper_actor"]},path)
    experiment.write_json(prior_path,prior); experiment.write_json(warm_path,cached)
    with patch.object(spec,"arguments",return_value=args),patch.object(spec,"SCENARIOS",2), \
            patch.object(spec,"EVALUATION_EPISODES",2),patch.object(spec,"source_result",return_value=prior_path), \
            patch.object(spec,"warm_result",return_value=warm_path), \
            patch.object(experiment.warm.spec,"source_result",return_value=warm_path), \
            patch.object(experiment.joint.source,"load_source",return_value=(models,predictor,{},calibrations)), \
            patch.object(experiment.joint.base,"load_lower_state",side_effect=lambda r,p,**kw:teachers[str(p)]), \
            patch.object(experiment,"ProcessPoolExecutor",ImmediatePool),native_patch(),bounds_patch():
        output=tmp_path/"output"/"result.json"
        result=experiment.run(root,output)
        assert result["cost"]==spec.budget() and result["cost"]["native_episodes"]==88
        assert result["cost"]["counterfactual_episodes"]==24 and result["cost"]["evaluation_alias_assignments"]==16
        assert result["inherited_Stage139_cost"]==prior["cost"]
        assert (output.parent/"completion"/"ready.json").is_file()
        assert not list(output.parent.rglob("*.pt")) and not list(output.parent.rglob("*.npz"))
        for group in result["groups"].values():
            assert group["lower_critics_and_mode_noise_pairing"]=="passed"
            for scale in spec.SCALES:
                assert len(group["training"][scale]["counterfactual_paths"])==4
                assert group["scales"][scale]["sampled_minus_mean_return"]["source_forecast"]["paired_differences"]==[0.,0.]
            assert group["reduced_minus_original_return"]["mean"]["warm_start"]["paired_differences"]==[0.,0.]
