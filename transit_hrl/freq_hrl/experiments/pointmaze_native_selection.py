"""Select an actual fitted continuation on separate native validation scenes."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_compact_continuation as kernel
from .pointmaze_root_response import write_json
from scripts import pointmaze_native_selection_stage134_spec as spec

joint, curvature = kernel.joint, kernel.curvature


def select_native_candidate(rows):
    gains = {"two_step": 0.}
    paired = {}
    for method in spec.METHODS:
        paired[method] = [r[method]["episode_return"] - r["two_step"]["episode_return"] for r in rows]
        gains[method] = float(np.mean(paired[method]))
    selected = max(gains, key=gains.get)
    return {"method": selected, "mean_incremental_returns": gains, "paired_differences": paired,
        "selection_data": "separate_native_validation_at_actual_pooled_fit_weights"}


def run(root, output, *, protocol_spec=spec):
    spec = protocol_spec
    output = output.resolve()
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != ("complete", spec.source.PROTOCOL, root, spec.source.budget()):
        raise ValueError("Native selection requires the completed two-step source")
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, cost, groups = spec.arguments(root), dict.fromkeys(spec.budget(), 0), {}
    cost["source_cell_loads"] = 1
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            origin = copy.deepcopy(trainer.upper_actor.state_dict())
            path = spec.source_result(root).parent/"final_weights"/f"period_{period}_refresh_upper.pt"
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if (saved["protocol"], saved["root"], saved["period"], saved["method"], saved["fit"]) != (
                    spec.source.PROTOCOL, root, period, "refresh", cached["groups"][p]["training_native_return_fits"]["refresh"]["pooled"]):
                raise ValueError("Native selection source upper changed")
            current = saved["weights"]
            trainer.upper_actor.load_state_dict(current)
            torch.testing.assert_close(current["log_std"], origin["log_std"], atol=0, rtol=0)
            cost["upper_checkpoint_loads"] += 1
            envelope, roster = calibrations[p]["envelope"], spec.queries(root, period)
            jobs = [(joint.weights(model), teacher, current, q, period, predictor, envelope) for q in roster]
            labels = []
            for index, row in enumerate(pool.map(kernel.worker_label, jobs), 1):
                labels.append(row)
                cost["label_queries"] += 1
                cost["label_episodes"] += row["native_episodes"]
                for k in ("native_episodes", "native_steps", "native_upper_calls", "native_donor_response_calls"):
                    cost[k] += row[k]
                if index % 16 == 0 or index == len(roster):
                    print(f"root={root} period={period}: current-policy labels {index}/{len(roster)}", flush=True)
            directions, learning, work = kernel.learn_continuations(trainer.upper_actor, origin, labels, protocol_spec=spec)
            for k, v in work.items():
                cost[k] += v
            variants = [("zero", current, "joint"), *[(m+"_"+sign, directions[m][sign], "joint") for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel], period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(curvature.worker_group, jobs))
                for rows in training[panel]:
                    curvature.count_group(cost, rows, "training")
            fits = {m: {panel: kernel.fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = kernel.fit_method(training["A"]+training["B"], m)
            cost["native_return_fits"] += 6
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                variants = [("zero", current, "joint"), *[(m, curvature.scaled_upper(current, directions[m]["plus"], fits[m][train_panel]["scale"]), "joint") for m in spec.METHODS]]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel], period, predictor, envelope) for r in roles]
                gains = {m: [] for m in spec.METHODS}
                for i, rows in enumerate(pool.map(curvature.worker_group, jobs)):
                    curvature.count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][i]["zero"]["episode_return"]:
                        raise ValueError("Native selection crossfit replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"]-rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v} for m, v in gains.items()}
            plus = {m: curvature.scaled_upper(current, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = curvature.scaled_upper(current, directions["refresh"]["minus"], fits["refresh"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 11
            variants = [("two_step", current, "joint"), *[(m, plus[m], "joint") for m in spec.METHODS]]
            validation_roles, validation = spec.validation_roles(root), []
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel], period, predictor, envelope) for r in validation_roles]
                for rows in pool.map(curvature.worker_group, jobs):
                    curvature.count_group(cost, rows, "selection")
                    validation.append(rows)
            selection = select_native_candidate(validation)
            cost["native_policy_selections"] += 1
            print(f"root={root} period={period}: validation selected {selection['method']}", flush=True)
            variants = [("source_flat", origin, "flat"), ("source_forecast", origin, "forecast"), ("two_step", current, "joint"),
                ("refresh", plus["refresh"], "joint"), ("stale", plus["stale"], "joint"),
                ("refresh_descent", minus, "joint"), ("refresh_blinded", plus["refresh"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(curvature.worker_group, jobs))
            for rows in evaluation:
                curvature.count_group(cost, rows, "evaluation")
                if rows["refresh_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Native selection blinded upper changed forecast")
                # The chosen policy has already executed on this exact paired scene.
                rows["selected"] = {**rows[selection["method"]], "variant": "selected"}
                cost["evaluation_alias_assignments"] += 1
            checkpoints = {}
            for method in spec.METHODS:
                path = output.parent/"final_weights"/f"period_{period}_{method}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "method": method,
                    "fit": fits[method]["pooled"], "weights": plus[method]}, path)
                cost["checkpoint_writes"] += 1
                checkpoints[method] = str(path)
            checkpoints["two_step"] = str(spec.source_result(root).parent/"final_weights"/f"period_{period}_refresh_upper.pt")
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"learning": learning, "native_labels": [{k:v for k,v in r.items() if k != "state"} for r in labels],
                "training_roles": roles, "training_native_return_fits": fits, "training_crossfit": crossfit,
                "validation_roles": validation_roles, "selection": selection, "selected_upper_checkpoint": checkpoints[selection["method"]],
                "evaluation_seeds": seeds, "lower_and_critics_frozen": "passed", **kernel.evaluation_summary(evaluation, protocol_spec=spec)}
            print(f"root={root} period={period}: native selection evaluation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Native selection measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "source_protocol": spec.source.PROTOCOL, "inherited_source_cost": cached["cost"],
        "minimum_episode_gain": spec.MINIMUM_GAIN, "kind": spec.EVIDENCE_ROLE}
    write_json(output, result)
    write_json(output.parent/"completion"/"ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native selection result written", flush=True)
    return result
