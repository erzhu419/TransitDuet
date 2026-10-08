"""Use cached native option credit to separate score geometry and deployment."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_mean_geometry import native_mean_directions
from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_option_counterfactual_credit as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts import pointmaze_score_geometry_stage139_spec as spec

joint, warm = previous.joint, previous.warm


def worker_replay(job):
    source_weights,teacher,initial,role,panel,period,predictor,envelope,cached = job
    model,args=joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer=joint.make_trainer(model,teacher,args)
    joint.load_weights(trainer,initial)
    batch,row,_=joint.native_episode(trainer,args=args,seed=role["scenario_seed"],noise_seed=role["noise_seeds"][panel],
        arm="joint",period=period,predictor=predictor,envelope=envelope,collect=True)
    if (cached["scenario_seed"],cached["panel"],cached["noise_seed"],cached["decision_steps"],cached["pairing_and_policy_freeze"]) != (
            row["seed"],panel,row["noise_seed"],row["decision_steps"],"passed"):
        raise ValueError("Stage139 cached single-option label roster changed")
    np.testing.assert_allclose(row["episode_return"],cached["sampled_return"],atol=1e-8,rtol=0)
    credit=cached["sampled_return"]-np.asarray(cached["mean_option_baseline_returns"])
    if len(credit) != batch.upper.size:
        raise ValueError("Stage139 cached credit decision count changed")
    torch.testing.assert_close(joint.weights(trainer),initial,atol=0,rtol=0)
    return {"batch":batch.upper,"credit":credit,"row":row}


def score_directions(actor,batch,credit,*,clip_ratio):
    gradient,error=previous.loss_gradient(actor,batch,credit,clip_ratio=clip_ratio)
    offset,mapped=0,{}
    for name,p in actor.named_parameters():
        if p.requires_grad:
            mapped[name]=-gradient[offset:offset+p.numel()].reshape(p.shape)
            offset+=p.numel()
        else:
            mapped[name]=np.zeros(p.shape)
    score=np.concatenate([mapped[n].ravel() for n,_ in actor.named_parameters()])
    normalized=joint.FrequencySeparatedActorCriticPPO._normalize(credit)
    targets=[]
    for start in range(0,batch.size,joint.spec.MINIBATCH):
        stop=start+joint.spec.MINIBATCH
        with torch.no_grad():
            dist=actor.distribution(torch.as_tensor(batch.state[start:stop]))
            actions=torch.as_tensor(batch.action[start:stop])
            ratio=torch.exp(dist.log_prob(actions).sum(-1)-torch.as_tensor(batch.old_logp[start:stop]))
            targets.append((torch.as_tensor(normalized[start:stop,None])*ratio[:,None]*
                (actions-dist.mean)/dist.stddev.square()).double().numpy())
    directions,geometry=native_mean_directions(batch.state,{"pooled":np.concatenate(targets)},
        actor.log_std.detach().exp().numpy(),damping=spec.DAMPING)
    natural_map={"net.0.weight":directions["pooled"]["weight"],"net.0.bias":directions["pooled"]["bias"]}
    natural=np.concatenate([natural_map[n].ravel() if n in natural_map else np.zeros(p.numel()) for n,p in actor.named_parameters()])
    return {"score":score,"natural_score":natural},{**geometry,"max_logp_replay_error":error}


def candidates(trainer,rows,root,period,cached):
    initial=joint.weights(trainer)
    update=previous.update_upper(trainer,rows,root=root,period=period,method="option_credit")
    if (update["optimizer_seed"],update["optimizer_steps"]) != (cached["optimizer_seed"],cached["optimizer_steps"]):
        raise ValueError("Stage139 original option-credit optimizer changed")
    np.testing.assert_allclose([update["parameter_delta_rms"][n] for n in joint.NETWORKS],
        [cached["parameter_delta_rms"][n] for n in joint.NETWORKS],atol=1e-10,rtol=0)
    adam=np.concatenate([(trainer.upper_actor.state_dict()[n]-initial["upper_actor"][n]).numpy().ravel()
        for n,_ in trainer.upper_actor.named_parameters()])
    joint.load_weights(trainer,initial)
    batch=concat_level_batches([r["batch"] for r in rows])
    credit=np.asarray([r["credit"] for r in rows],dtype=np.float32).reshape(-1)
    vectors,geometry=score_directions(trainer.upper_actor,batch,credit,clip_ratio=trainer.config.clip_ratio)
    vectors["adam"]=adam
    weights,radii={"warm_start":initial["upper_actor"]},{}
    cost={"empirical_fisher_solves":1,"fisher_jvp_batches":0,"exact_kl_forward_batches":0}
    for method in spec.METHODS:
        actors,radius,work=matched_perturbations(trainer.upper_actor,batch.state,-vectors[method],
            delta=spec.FISHER_RADIUS,chunk_size=joint.spec.MINIBATCH)
        for sign,actor in actors.items():weights[method+"_"+sign]=copy.deepcopy(actor.state_dict())
        radii[method]=radius
        for k,v in work.items():cost[k]+=v
    torch.testing.assert_close(joint.weights(trainer),initial,atol=0,rtol=0)
    geometry.update(cosines={f"{a}_to_{b}":cosine(vectors[a],vectors[b])
        for a,b in (("adam","score"),("adam","natural_score"),("score","natural_score"))},
        initial_loss_descent_inner_product={m:float(vectors["score"]@vectors[m]) for m in spec.METHODS})
    return weights,{"original_update_replay":update,"geometry":geometry,"radii":radii},cost


def worker_evaluate(job):
    source_weights,teacher,initial,uppers,seed,period,predictor,envelope=job
    model,args=joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer=joint.make_trainer(model,teacher,args)
    joint.load_weights(trainer,initial)
    output,common={},None
    for mode in spec.MODES:
        rows,upper_common={},None
        for variant in spec.VARIANTS:
            if mode=="sampled" and variant=="source_forecast":
                rows[variant]=output["mean"][variant]
                continue
            trainer.upper_actor.load_state_dict(uppers["warm_start" if variant=="source_forecast" else variant])
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
        output[mode]=rows
    for name in joint.NETWORKS[1:]:
        torch.testing.assert_close(getattr(trainer,name).state_dict(),initial[name],atol=0,rtol=0)
    torch.testing.assert_close(trainer.upper_actor.log_std,initial["upper_actor"]["log_std"],atol=0,rtol=0)
    return output


def evaluation_summary(rows):
    modes={}
    for mode in spec.MODES:
        scenes=[r[mode] for r in rows]
        effects,tracking={},{}
        for a,b in spec.CONTRASTS:
            for metric,out,sign in (("episode_return",effects,1),("tracking_squared_error_integral",tracking,-1)):
                differences=[sign*(r[a][metric]-r[b][metric]) for r in scenes]
                out[f"{a}_minus_{b}"]={"mean":float(np.mean(differences)),"paired_differences":differences}
        modes[mode]={"effects":effects,"tracking_error_reduction_positive_is_better":tracking,
            "mean_metrics":{v:{k:float(np.mean([r[v][k] for r in scenes])) for k in spec.METRICS} for v in spec.VARIANTS}}
    mismatch={}
    for variant in spec.VARIANTS:
        differences=[r["sampled"][variant]["episode_return"]-r["mean"][variant]["episode_return"] for r in rows]
        mismatch[variant]={"mean":float(np.mean(differences)),"paired_differences":differences}
    return {"evaluation":modes,"sampled_minus_mean_return":mismatch}


def run(root,output):
    cached=json.loads(spec.source_result(root).read_text())
    warm_cached=json.loads(spec.warm_result(root).read_text())
    if root not in spec.ROOTS or (cached["status"],cached["protocol"],cached["root"],cached["cost"],cached["seed_roles"]) != (
            "complete",spec.source.PROTOCOL,root,spec.source.budget(),spec.source.seed_roles(root)):
        raise ValueError("Stage139 requires the completed Stage138 label protocol")
    warm_spec=spec.source.source.source.source
    if (warm_cached["status"],warm_cached["protocol"],warm_cached["root"],warm_cached["cost"]) != (
            "complete",warm_spec.PROTOCOL,root,warm_spec.budget()):
        raise ValueError("Stage139 warm source differs from Stage135")
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
            initial,source_weights=joint.weights(trainer),joint.weights(model)
            envelope,prior=calibrations[str(period)]["envelope"],cached["groups"][str(period)]
            jobs=[(source_weights,teacher,initial,role,panel,period,predictor,envelope,prior["counterfactual_paths"][i])
                for i,(role,panel) in enumerate((r,p) for r in roles["replayed_training"] for p in spec.source.PANELS)]
            rows=list(pool.map(worker_replay,jobs))
            for row in rows:
                warm.count_row(cost,row["row"],"replay"); cost["credit_checks"]+=1
            diagnostic=previous.score_diagnostics(trainer,rows)
            np.testing.assert_allclose([diagnostic[m][k] for m in spec.source.METHODS for k in diagnostic[m]],
                [prior["diagnostic"][m][k] for m in spec.source.METHODS for k in diagnostic[m]],atol=1e-10,rtol=0)
            uppers,learning,work=candidates(trainer,rows,root,period,prior["updates"]["option_credit"])
            for k,v in work.items():cost[k]+=v
            chunks=int(np.ceil(len(rows)*(args.horizon//period)/joint.spec.MINIBATCH))
            fold_chunks=int(np.ceil(len(rows)/2*(args.horizon//period)/joint.spec.MINIBATCH))
            cost["score_gradient_batches"]+=chunks+4*fold_chunks
            cost["mean_score_forward_batches"]+=chunks
            cost["upper_candidate_weight_steps"]+=2*len(spec.METHODS)
            for k in ("upper_actor_optimizer_steps","upper_value_optimizer_steps"):
                cost[k]+=learning["original_update_replay"]["optimizer_steps"][k]
            scenes=list(pool.map(worker_evaluate,[(source_weights,teacher,initial,uppers,seed,period,predictor,envelope) for seed in roles["evaluation"]]))
            for seed,scene in zip(roles["evaluation"],scenes):
                for mode in spec.MODES:
                    for variant,row in scene[mode].items():
                        if (row["seed"],row["noise_seed"],row["upper_sample"]) != (
                                seed,seed,mode=="sampled" and variant!="source_forecast"):
                            raise ValueError("Stage139 evaluation roster or sampling changed")
                        if mode=="sampled" and variant=="source_forecast":cost["evaluation_alias_assignments"]+=1
                        else:warm.count_row(cost,row,"evaluation")
            joint.source.native.curves.support.assert_frozen(model,snapshot)
            groups[str(period)]={"selected_source":provenance,"replayed_credit_diagnostic":diagnostic,
                "learning":learning,"original_training_and_update_replay":"passed",
                "frozen_deployment_and_mode_noise_pairing":"passed",**evaluation_summary(scenes)}
            print(f"root={root} period={period}: score geometry and dual-mode evaluation complete",flush=True)
    if cost != spec.budget():raise ValueError(f"Stage139 measured budget changed: {cost}")
    result={"status":"complete","protocol":spec.PROTOCOL,"root":root,"contract":spec.contract(),"seed_roles":roles,
        "cost":cost,"groups":groups,"kind":"optimizer_and_deployment_diagnosis_not_joint_HRL_confirmation",
        "inherited_Stage138_cost":cached["cost"],"inherited_Stage135_cost":warm_cached["cost"],
        "inherited_earlier_source_cost":{"Stage137":cached["inherited_Stage137_cost"],"earlier":cached["inherited_earlier_source_cost"]}}
    write_json(output,result)
    write_json(output.parent/"completion"/"ready.json",{"status":"complete","protocol":spec.PROTOCOL})
    return result
