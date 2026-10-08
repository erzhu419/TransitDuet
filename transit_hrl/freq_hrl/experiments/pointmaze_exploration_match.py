"""Pair upper exploration scales without changing mean-output update RMS."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_score_geometry as previous
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_exploration_match_stage140_spec as spec

joint,warm,credit = previous.joint,previous.warm,previous.previous


def scale_initial(initial,scale):
    state=copy.deepcopy(initial)
    state["upper_actor"]["log_std"].fill_(float(np.log(spec.STDS[scale])))
    return state


def natural_candidates(trainer,rows,scale):
    before=joint.weights(trainer)
    batch=concat_level_batches([r["batch"] for r in rows])
    signal=np.asarray([r["credit"] for r in rows],dtype=np.float32).reshape(-1)
    vectors,geometry=previous.score_directions(trainer.upper_actor,batch,signal,clip_ratio=trainer.config.clip_ratio)
    actors,radius,work=matched_perturbations(trainer.upper_actor,batch.state,-vectors["natural_score"],
        delta=spec.radius(scale),chunk_size=joint.spec.MINIBATCH)
    weights={"warm_start":before["upper_actor"]}
    with torch.no_grad():
        original=trainer.upper_actor.distribution(torch.as_tensor(batch.state)).mean.double()
        steps={}
        for sign,actor in actors.items():
            mean=actor.distribution(torch.as_tensor(batch.state)).mean.double()
            steps[sign]=float((mean-original).square().mean().sqrt())
            np.testing.assert_allclose(steps[sign],spec.MEAN_STEP_RMS,atol=1e-8,rtol=0)
            torch.testing.assert_close(actor.log_std,trainer.upper_actor.log_std,atol=0,rtol=0)
            weights["natural_"+sign]=copy.deepcopy(actor.state_dict())
    torch.testing.assert_close(joint.weights(trainer),before,atol=0,rtol=0)
    return weights,{"geometry":geometry,"radius":radius,"mean_step_RMS":steps}, {
        **work,"empirical_fisher_solves":1,"policy_geometry_forward_batches":3,"upper_candidate_weight_steps":2}


def worker_evaluate(job):
    source_weights,teacher,initials,uppers,seed,period,predictor,envelope=job
    model,args=joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer=joint.make_trainer(model,teacher,args)
    output,common,upper_common={},None,None
    for scale in spec.SCALES:
        joint.load_weights(trainer,initials[scale])
        modes={}
        for mode in spec.MODES:
            rows={}
            for variant in spec.VARIANTS:
                if variant=="source_forecast" and (scale,mode)!=(spec.SCALES[0],"mean"):
                    rows[variant]=output[spec.SCALES[0]]["mean"][variant] if scale!=spec.SCALES[0] else modes["mean"][variant]
                    continue
                if variant=="warm_start" and mode=="mean" and scale!=spec.SCALES[0]:
                    rows[variant]=output[spec.SCALES[0]][mode][variant]
                    continue
                trainer.upper_actor.load_state_dict(uppers[scale]["warm_start" if variant=="source_forecast" else variant])
                batch,row,audit=joint.native_episode(trainer,args=args,seed=seed,noise_seed=seed,
                    arm="forecast" if variant=="source_forecast" else "joint",period=period,
                    predictor=predictor,envelope=envelope,collect=False,sample_upper=mode=="sampled")
                assert batch is None
                if common is None:common=audit
                else:
                    np.testing.assert_array_equal(audit["measurements"],common["measurements"])
                    np.testing.assert_allclose(audit["innovations"],common["innovations"],atol=3e-5,rtol=0)
                if mode=="sampled":
                    if upper_common is None:upper_common=audit["upper_innovations"]
                    else:np.testing.assert_allclose(audit["upper_innovations"],upper_common,atol=3e-5,rtol=0)
                rows[variant]={**row,"variant":variant}
            modes[mode]=rows
        output[scale]=modes
        for name in joint.NETWORKS[1:]:
            torch.testing.assert_close(getattr(trainer,name).state_dict(),initials[scale][name],atol=0,rtol=0)
        torch.testing.assert_close(trainer.upper_actor.log_std,initials[scale]["upper_actor"]["log_std"],atol=0,rtol=0)
    return output


def aliased(scale,mode,variant):
    return (variant=="source_forecast" and (scale,mode)!=(spec.SCALES[0],"mean")) or (
        variant=="warm_start" and mode=="mean" and scale!=spec.SCALES[0])


def paired_effect(values):
    return {"mean":float(np.mean(values)),"paired_differences":values}


def evaluation_summary(scenes):
    scales={}
    for scale in spec.SCALES:
        modes={}
        for mode in spec.MODES:
            rows=[r[scale][mode] for r in scenes]
            tables={name:{f"{a}_minus_{b}":paired_effect([sign*(r[a][metric]-r[b][metric]) for r in rows])
                for a,b in spec.CONTRASTS} for name,metric,sign in (
                    ("effects","episode_return",1),("tracking_error_reduction_positive_is_better","tracking_squared_error_integral",-1))}
            modes[mode]={**tables,"mean_metrics":{v:{k:float(np.mean([r[v][k] for r in rows])) for k in spec.METRICS} for v in spec.VARIANTS}}
        mismatch={v:paired_effect([r[scale]["sampled"][v]["episode_return"]-r[scale]["mean"][v]["episode_return"] for r in scenes]) for v in spec.VARIANTS}
        scales[scale]={"evaluation":modes,"sampled_minus_mean_return":mismatch}
    between={mode:{v:paired_effect([r["reduced"][mode][v]["episode_return"]-r["original"][mode][v]["episode_return"] for r in scenes]) for v in spec.VARIANTS} for mode in spec.MODES}
    increments={mode:paired_effect([(r["reduced"][mode]["natural_plus"]["episode_return"]-r["reduced"][mode]["warm_start"]["episode_return"])-
        (r["original"][mode]["natural_plus"]["episode_return"]-r["original"][mode]["warm_start"]["episode_return"]) for r in scenes]) for mode in spec.MODES}
    return {"scales":scales,"reduced_minus_original_return":between,"reduced_minus_original_learning_increment":increments}


def run(root,output):
    cached=json.loads(spec.source_result(root).read_text())
    warm_cached=json.loads(spec.warm_result(root).read_text())
    if root not in spec.ROOTS or (cached["status"],cached["protocol"],cached["root"],cached["cost"]) != (
            "complete",spec.source.PROTOCOL,root,spec.source.budget()):
        raise ValueError("Stage140 requires the completed Stage139 diagnosis")
    if (warm_cached["status"],warm_cached["protocol"],warm_cached["root"],warm_cached["cost"]) != (
            "complete",spec.warm_source.PROTOCOL,root,spec.warm_source.budget()):
        raise ValueError("Stage140 warm source differs from Stage135")
    models,predictor,_,calibrations=joint.source.load_source(root)
    args,roles=spec.arguments(root),spec.seed_roles(root)
    cost,groups=dict.fromkeys(spec.budget(),0),{}
    cost.update(source_cell_loads=2,lower_checkpoint_loads=len(spec.PERIODS),upper_checkpoint_loads=len(spec.PERIODS))
    with ProcessPoolExecutor(max_workers=spec.WORKERS,mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker,initargs=(models["50"].config,args)) as pool:
        for period in spec.PERIODS:
            model=models[str(period)]
            snapshot=copy.deepcopy(model.state_dict())
            teacher=joint.base.load_lower_state(root,period,protocol=joint.spec)
            trainer=joint.make_trainer(model,teacher,args)
            upper,provenance=warm.load_selected_upper(warm_cached,root,period)
            torch.testing.assert_close(upper["log_std"],trainer.upper_actor.log_std,atol=0,rtol=0)
            trainer.upper_actor.load_state_dict(upper)
            origin=joint.weights(trainer)
            initials={s:scale_initial(origin,s) for s in spec.SCALES}
            source_weights,envelope=joint.weights(model),calibrations[str(period)]["envelope"]
            jobs=[(source_weights,teacher,initials[s],r,p,period,predictor,envelope)
                for s in spec.SCALES for r in roles["training"] for p in spec.PANELS]
            collected=list(pool.map(credit.worker_credit,jobs))
            for job,row in zip(jobs,collected):
                role,panel=job[3:5]
                if (row["path"]["scenario_seed"],row["path"]["noise_seed"],row["path"]["panel"]) != (
                        role["scenario_seed"],role["noise_seeds"][panel],panel):
                    raise ValueError("Stage140 training scenario/noise roster changed")
                for k,v in row["cost"].items():cost[k]+=v
            paths=len(roles["training"])*len(spec.PANELS)
            uppers,learning={},{}
            for index,scale in enumerate(spec.SCALES):
                rows=collected[index*paths:(index+1)*paths]
                joint.load_weights(trainer,initials[scale])
                diagnostic=credit.score_diagnostics(trainer,rows)
                uppers[scale],fit,work=natural_candidates(trainer,rows,scale)
                for k,v in work.items():cost[k]+=v
                chunks=int(np.ceil(paths*(args.horizon//period)/joint.spec.MINIBATCH))
                folds=int(np.ceil(paths/2*(args.horizon//period)/joint.spec.MINIBATCH))
                cost["score_gradient_batches"]+=chunks+4*folds
                cost["mean_score_forward_batches"]+=chunks
                learning[scale]={"upper_std":spec.STDS[scale],"diagnostic":diagnostic,"learning":fit,
                    "counterfactual_paths":[r["path"] for r in rows],"weights_and_pairing":"passed"}
            scenes=list(pool.map(worker_evaluate,[(source_weights,teacher,initials,uppers,seed,period,predictor,envelope) for seed in roles["evaluation"]]))
            for seed,scene in zip(roles["evaluation"],scenes):
                for scale in spec.SCALES:
                    for mode in spec.MODES:
                        for variant,row in scene[scale][mode].items():
                            if (row["seed"],row["noise_seed"],row["upper_sample"]) != (
                                    seed,seed,mode=="sampled" and variant!="source_forecast"):
                                raise ValueError("Stage140 evaluation roster or sampling changed")
                            if aliased(scale,mode,variant):cost["evaluation_alias_assignments"]+=1
                            else:warm.count_row(cost,row,"evaluation")
            joint.source.native.curves.support.assert_frozen(model,snapshot)
            groups[str(period)]={"selected_source":provenance,"training":learning,"lower_critics_and_mode_noise_pairing":"passed",
                **evaluation_summary(scenes)}
            print(f"root={root} period={period}: matched-step exploration evaluation complete",flush=True)
    if cost!=spec.budget():raise ValueError(f"Stage140 measured native/learning budget changed: {cost}")
    result={"status":"complete","protocol":spec.PROTOCOL,"root":root,"contract":spec.contract(),"seed_roles":roles,
        "cost":cost,"groups":groups,"kind":"exploration_and_mean_objective_development_not_joint_HRL_confirmation",
        "inherited_Stage139_cost":cached["cost"],"inherited_Stage135_cost":warm_cached["cost"],
        "inherited_earlier_source_cost":{"Stage138":cached["inherited_Stage138_cost"],"earlier":cached["inherited_earlier_source_cost"]}}
    write_json(output,result)
    write_json(output.parent/"completion"/"ready.json",{"status":"complete","protocol":spec.PROTOCOL})
    return result
