"""Train upper credit on single-option native counterfactual differences."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_level_batches
from . import pointmaze_bounded_ppo as previous
from .pointmaze_actor_credit import cosine
from .pointmaze_native_direction import matched_perturbations
from .pointmaze_root_response import write_json
from scripts.diagnose_pointmaze_joint_reference_credit_stage122 import loss_gradient
from scripts import pointmaze_option_credit_stage138_spec as spec

joint, warm = previous.joint, previous.warm


def scalar_means(actor, states):
    with torch.no_grad():
        return np.stack([actor.distribution(torch.as_tensor(s).view(1, -1)).mean[0].numpy() for s in states])


def worker_credit(job):
    source_weights, teacher, initial, role, panel, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    joint.load_weights(trainer, initial)
    call = dict(args=args, seed=role["scenario_seed"], noise_seed=role["noise_seeds"][panel],
        arm="joint", period=period, predictor=predictor, envelope=envelope, collect=True)
    batch, row, audit = joint.native_episode(trainer, **call)
    means = scalar_means(trainer.upper_actor, batch.upper.state)
    counterfactual, baseline_returns = [], []
    counts = dict.fromkeys(spec.budget(), 0)
    warm.count_row(counts, row, "collection")
    for index, step in enumerate(row["decision_steps"]):
        baseline, control, control_audit = joint.native_episode(trainer, **call,
            upper_override={"step": step, "action": means[index]})
        np.testing.assert_array_equal(baseline.upper.state[:index+1], batch.upper.state[:index+1])
        np.testing.assert_array_equal(baseline.upper.action[:index], batch.upper.action[:index])
        for key in ("state", "action", "reward"):
            np.testing.assert_array_equal(getattr(baseline.lower,key)[:step], getattr(batch.lower,key)[:step])
        np.testing.assert_array_equal(baseline.upper.action[index], means[index])
        np.testing.assert_array_equal(control_audit["measurements"], audit["measurements"])
        np.testing.assert_allclose(control_audit["innovations"], audit["innovations"], atol=3e-5, rtol=0)
        mean = scalar_means(trainer.upper_actor, baseline.upper.state)
        keep = np.arange(batch.upper.size) != index
        std = trainer.upper_actor.log_std.detach().exp().numpy()
        np.testing.assert_allclose(((baseline.upper.action-mean)/std)[keep],
            ((batch.upper.action-means)/std)[keep], atol=3e-5, rtol=0)
        counterfactual.append(row["episode_return"]-control["episode_return"])
        baseline_returns.append(control["episode_return"])
        warm.count_row(counts, control, "counterfactual")
        counts["credit_checks"] += 1
    torch.testing.assert_close(joint.weights(trainer), initial, atol=0, rtol=0)
    return {"batch": batch.upper, "credit": counterfactual, "cost": counts,
        "path": {"scenario_seed": role["scenario_seed"], "panel": panel, "noise_seed": role["noise_seeds"][panel],
            "sampled_return": row["episode_return"], "mean_option_baseline_returns": baseline_returns,
            "decision_steps": row["decision_steps"], "pairing_and_policy_freeze": "passed"}}


def score_diagnostics(trainer, rows):
    batches = [r["batch"] for r in rows]
    signals = {"critic": np.stack([trainer._gae(b.reward,b.done,b.duration,b.old_value)[0] for b in batches]),
        "option_credit": np.asarray([r["credit"] for r in rows], dtype=np.float64)}
    report = {}
    for method, signal in signals.items():
        gradients, errors = [], []
        for fold in (0,1):
            batch = concat_level_batches(batches[fold::2])
            gradient, error = loss_gradient(trainer.upper_actor, batch, signal[fold::2].reshape(-1),
                clip_ratio=trainer.config.clip_ratio)
            gradients.append(gradient); errors.append(error)
        variance = float(signal.var())
        report[method] = {"advantage_std": float(signal.std()),
            "phase_mean_variance_fraction": float(signal.mean(0).var()/variance) if variance else 0.,
            "noise_fold_gradient_cosine": cosine(*gradients), "max_logp_replay_error": max(errors)}
    return report


def update_upper(trainer, rows, *, root, period, method):
    batch = concat_level_batches([r["batch"] for r in rows])
    advantage = None if method == "critic" else np.asarray([r["credit"] for r in rows], dtype=np.float32).reshape(-1)
    before = joint.weights(trainer)
    np.random.seed(spec.optimizer_seed(root,period)); torch.manual_seed(spec.optimizer_seed(root,period))
    metrics = trainer._update_level(level="upper", batch=batch, actor=trainer.upper_actor,
        value_net=trainer.upper_value, actor_optimizer=trainer.upper_actor_optimizer,
        value_optimizer=trainer.upper_value_optimizer, actor_advantage=advantage)
    delta = {name: joint.parameter_delta(before[name], getattr(trainer,name).state_dict()) for name in joint.NETWORKS}
    if delta["lower_actor"] != 0 or delta["lower_value"] != 0:
        raise ValueError("Stage138 upper credit updated lower")
    return {"optimizer_seed": spec.optimizer_seed(root,period), "parameter_delta_rms": delta,
        "optimizer_steps": {k:int(v) for k,v in metrics.items() if k.endswith("optimizer_steps")},
        "upper_policy_loss": float(metrics["upper_policy_loss"]), "upper_value_loss": float(metrics["upper_value_loss"])}


def run(root, output):
    cached = json.loads(spec.source_result(root).read_text())
    warm_cached = json.loads(spec.warm_result(root).read_text())
    if (root not in spec.ROOTS or (cached["protocol"],cached["status"],cached["root"],cached["cost"]) != (
            spec.source.PROTOCOL,"complete",root,spec.source.budget())):
        raise ValueError("Stage138 requires the completed Stage137 diagnosis")
    if (warm_cached["protocol"],warm_cached["status"],warm_cached["root"],warm_cached["cost"]) != (
            spec.source.source.source.PROTOCOL,"complete",root,spec.source.source.source.budget()):
        raise ValueError("Stage138 warm source differs from Stage135")
    models,predictor,_,calibrations = joint.source.load_source(root)
    args,roles = spec.arguments(root),spec.seed_roles(root)
    groups,cost = {},dict.fromkeys(spec.budget(),0)
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
            envelope=calibrations[str(period)]["envelope"]
            jobs=[(source_weights,teacher,initial,role,panel,period,predictor,envelope)
                for role in roles["training"] for panel in spec.PANELS]
            rows=list(pool.map(worker_credit,jobs))
            for registered,row in zip(jobs,rows):
                role,panel=registered[3:5]
                if (row["path"]["scenario_seed"],row["path"]["panel"],row["path"]["noise_seed"]) != (
                        role["scenario_seed"],panel,role["noise_seeds"][panel]):
                    raise ValueError("Stage138 training scenario/noise order changed")
                for k,v in row["cost"].items(): cost[k]+=v
            diagnostic=score_diagnostics(trainer,rows)
            cost["score_gradient_batches"]+=4*int(np.ceil(spec.SCENARIOS*(args.horizon//period)/joint.spec.MINIBATCH))
            candidates,updates,geometry={"warm_start":upper},{},{}
            critics=[]
            states=np.concatenate([r["batch"].state for r in rows])
            for method in spec.METHODS:
                current=joint.make_trainer(model,teacher,args)
                joint.load_weights(current,initial)
                updates[method]=update_upper(current,rows,root=root,period=period,method=method)
                for k in ("upper_actor_optimizer_steps","upper_value_optimizer_steps"):
                    cost[k]+=updates[method]["optimizer_steps"][k]
                critics.append(copy.deepcopy(current.upper_value.state_dict()))
                vector=np.concatenate([(current.upper_actor.state_dict()[n]-upper[n]).numpy().ravel()
                    for n,_ in trainer.upper_actor.named_parameters()])
                bounded,radius,work=matched_perturbations(trainer.upper_actor,states,-vector,
                    delta=spec.FISHER_RADIUS,chunk_size=joint.spec.MINIBATCH)
                for sign,actor in bounded.items(): candidates[method+"_"+sign]=copy.deepcopy(actor.state_dict())
                geometry[method]=radius
                for k,v in work.items(): cost[k]+=v
            torch.testing.assert_close(critics[0],critics[1],atol=0,rtol=0)
            evaluation_jobs=[(source_weights,teacher,initial,candidates,seed,period,predictor,envelope) for seed in roles["evaluation"]]
            # The native evaluator accepts the explicit registered roster; all values/lower remain initial.
            evaluation=list(pool.map(worker_evaluate,evaluation_jobs))
            for seed,scene in zip(roles["evaluation"],evaluation):
                for row in scene.values():
                    if row["seed"] != seed or row["noise_seed"] != seed: raise ValueError("Stage138 evaluation roster changed")
                    warm.count_row(cost,row,"evaluation")
            joint.source.native.curves.support.assert_frozen(model,snapshot)
            groups[str(period)]={"selected_source":provenance,"diagnostic":diagnostic,"updates":updates,
                "geometry":geometry,"counterfactual_paths":[r["path"] for r in rows],
                "shared_critic_targets_and_update":"passed","deployed_lower_critics_teacher_and_std_frozen":"passed",
                **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: native option credit evaluation complete",flush=True)
    if cost != spec.budget(): raise ValueError(f"Stage138 measured native/optimizer budget changed: {cost}")
    result={"status":"complete","protocol":spec.PROTOCOL,"root":root,"contract":spec.contract(),
        "seed_roles":roles,"cost":cost,"groups":groups,"kind":"native_option_credit_development_not_joint_HRL_confirmation",
        "inherited_Stage137_cost":cached["cost"],"inherited_Stage135_cost":warm_cached["cost"],
        "inherited_earlier_source_cost":warm_cached["inherited_source_cost"]}
    write_json(output,result)
    write_json(output.parent/"completion"/"ready.json",{"status":"complete","protocol":spec.PROTOCOL})
    return result


def worker_evaluate(job):
    source_weights,teacher,initial,candidates,seed,period,predictor,envelope=job
    model,args=joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer=joint.make_trainer(model,teacher,args)
    joint.load_weights(trainer,initial)
    rows,common={},None
    for variant in spec.VARIANTS:
        trainer.upper_actor.load_state_dict(candidates["warm_start" if variant=="source_forecast" else variant])
        _,row,audit=joint.native_episode(trainer,args=args,seed=seed,noise_seed=seed,
            arm="forecast" if variant=="source_forecast" else "joint",period=period,
            predictor=predictor,envelope=envelope,collect=False)
        if common is not None:
            np.testing.assert_array_equal(audit["measurements"],common["measurements"])
            np.testing.assert_allclose(audit["innovations"],common["innovations"],atol=3e-5,rtol=0)
        else: common=audit
        rows[variant]={**row,"variant":variant}
    for name in ("lower_actor","lower_value","upper_value"):
        torch.testing.assert_close(getattr(trainer,name).state_dict(),initial[name],atol=0,rtol=0)
    return rows


def evaluation_summary(rows):
    effects,tracking={},{}
    for a,b in spec.CONTRASTS:
        for metric,output,sign in (("episode_return",effects,1),("tracking_squared_error_integral",tracking,-1)):
            paired=[sign*(r[a][metric]-r[b][metric]) for r in rows]
            output[f"{a}_minus_{b}"]={"mean":float(np.mean(paired)),"paired_differences":paired}
    return {"effects":effects,"tracking_error_reduction_positive_is_better":tracking,
        "mean_metrics":{v:{k:float(np.mean([r[v][k] for r in rows])) for k in spec.METRICS} for v in spec.VARIANTS}}
