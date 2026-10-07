"""Replicate the frozen compact mean method with new native labels and roots."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_credit_transfer as source
from .pointmaze_native_geometry import fit_method
from .statistics import bootstrap_mean_ci
from .pointmaze_root_response import write_json
from scripts import pointmaze_compact_replication_stage129_spec as spec

joint, curvature = source.joint, source.curvature


def run(root, output):
    output = output.resolve()
    models, predictor, _, calibrations = joint.source.load_source(root)
    args, cost, groups = spec.arguments(root), dict.fromkeys(spec.budget(), 0), {}
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            p, model = str(period), models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=joint.spec)
            trainer = joint.make_trainer(model, teacher, args)
            initial = copy.deepcopy(trainer.upper_actor.state_dict())
            envelope, roster = calibrations[p]["envelope"], spec.queries(root, period)
            jobs = [(joint.weights(model), teacher, q, period, predictor, envelope) for q in roster]
            labels = []
            for index, row in enumerate(pool.map(source.source.worker_label, jobs), 1):
                labels.append(row)
                cost["label_queries"] += 1
                cost["label_episodes"] += row["native_episodes"]
                for key in ("native_episodes", "native_steps", "native_upper_calls", "native_donor_response_calls"):
                    cost[key] += row[key]
                if index % 16 == 0 or index == len(roster):
                    print(f"root={root} period={period}: new native labels {index}/{len(roster)}", flush=True)
            if [r["query"] for r in labels] != roster:
                raise ValueError("Stage129 fresh native credit roster changed")
            directions, learning, work = source.learn_directions(trainer.upper_actor, labels)
            for key, value in work.items():
                cost[key] += value
            variants = [("zero", initial, "forecast"), *[(m + "_" + sign, directions[m][sign], "joint")
                for m in spec.METHODS for sign in ("plus", "minus")]]
            roles, training = spec.training_roles(root), {}
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                    period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(curvature.worker_group, jobs))
                for rows in training[panel]:
                    curvature.count_group(cost, rows, "training")
            fits = {m: {panel: fit_method(training[panel], m) for panel in spec.PANELS} for m in spec.METHODS}
            for m in spec.METHODS:
                fits[m]["pooled"] = fit_method(training["A"] + training["B"], m)
            cost["native_return_fits"] += 6
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                variants = [("zero", initial, "forecast"), *[(m, curvature.scaled_upper(initial, directions[m]["plus"],
                    fits[m][train_panel]["scale"]), "joint") for m in spec.METHODS]]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel],
                    period, predictor, envelope) for r in roles]
                gains = {m: [] for m in spec.METHODS}
                for index, rows in enumerate(pool.map(curvature.worker_group, jobs)):
                    curvature.count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][index]["zero"]["episode_return"]:
                        raise ValueError("Stage129 crossfit zero replay changed")
                    for m in spec.METHODS:
                        gains[m].append(rows[m]["episode_return"] - rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {m: {"mean_gain": float(np.mean(v)), "paired_differences": v} for m, v in gains.items()}
            plus = {m: curvature.scaled_upper(initial, directions[m]["plus"], fits[m]["pooled"]["scale"]) for m in spec.METHODS}
            minus = curvature.scaled_upper(initial, directions["compact"]["minus"], fits["compact"]["pooled"]["scale"])
            cost["upper_candidate_weight_steps"] += 11
            variants = [("source_flat", initial, "flat"), ("source_forecast", initial, "forecast"),
                ("raw", plus["raw"], "joint"), ("compact", plus["compact"], "joint"),
                ("compact_descent", minus, "joint"), ("compact_blinded", plus["compact"], "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(curvature.worker_group, jobs))
            for rows in evaluation:
                curvature.count_group(cost, rows, "evaluation")
                if rows["compact_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
                    raise ValueError("Stage129 blinded upper must exactly execute forecast")
            for method in spec.METHODS:
                path = output.parent / "final_weights" / f"period_{period}_{method}_upper.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "method": method,
                    "fit": fits[method]["pooled"], "weights": plus[method]}, path)
                cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"learning": learning, "native_labels": [{k: v for k, v in r.items() if k != "state"} for r in labels],
                "training_roles": roles, "training_native_return_fits": fits, "training_crossfit": crossfit,
                "evaluation_seeds": seeds, "lower_and_critics_frozen": "passed", **source.evaluation_summary(evaluation)}
            print(f"root={root} period={period}: frozen-method replication complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage129 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "minimum_episode_gain": spec.MINIMUM_GAIN, "kind": "frozen_compact_upper_replication_not_joint_HRL_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: compact native upper replication result written", flush=True)
    return result


def aggregate(cells):
    if len(cells) != len(spec.ROOTS) or {c["root"] for c in cells} != set(spec.ROOTS):
        raise ValueError("Stage129 requires exactly the six additional roots, without development roots")
    cells = sorted(cells, key=lambda c: c["root"])
    values = {key: [] for key in spec.ENDPOINTS}
    root_rows = []
    for c in cells:
        if (c["status"], c["protocol"], c["cost"]) != ("complete", spec.PROTOCOL, spec.budget()):
            raise ValueError("Stage129 protocol or budget changed")
        means = {}
        for period in spec.PERIODS:
            g = c["groups"][str(period)]
            if (g["evaluation_seeds"] != spec.evaluation_seeds(c["root"])
                    or g["training_roles"] != spec.training_roles(c["root"])
                    or g["lower_and_critics_frozen"] != "passed"):
                raise ValueError("Stage129 paired roster or lower/critic freeze changed")
            if g["effects"]["compact_minus_compact_blinded"] != g["effects"]["compact_minus_source_forecast"]:
                raise ValueError("Stage129 blinded forecast identity changed")
            for a, b in spec.PRIMARY_CONTRASTS:
                key = f"{period}/{a}_minus_{b}"
                e = g["effects"][f"{a}_minus_{b}"]
                paired = np.asarray(e["paired_differences"], dtype=np.float64)
                if paired.shape != (spec.EVALUATION_EPISODES,) or not np.isfinite(paired).all():
                    raise ValueError("Stage129 primary paired differences changed")
                np.testing.assert_allclose(e["mean"], paired.mean(), atol=1e-12, rtol=0)
                values[key].append(float(paired.mean()))
                means[key] = float(paired.mean())
        root_rows.append({"root": c["root"], "effects": means})
    endpoints = {}
    for index, (key, x) in enumerate(values.items()):
        ci = bootstrap_mean_ci(x, n_boot=spec.BOOTSTRAP_DRAWS, seed=spec.BOOTSTRAP_SEED + index,
            alpha=.05 / len(spec.ENDPOINTS))
        threshold = spec.MINIMUM_GAIN if key.endswith("source_forecast") else 0.
        endpoints[key] = {"mean": float(np.mean(x)), "ci": list(ci), "root_means": x,
            "threshold": threshold, "positive_gain_supported": ci[0] > 0., "threshold_supported": ci[0] > threshold}
    per_period = {str(p): all(endpoints[f"{p}/{a}_minus_{b}"]["threshold_supported"] for a, b in spec.PRIMARY_CONTRASTS)
        for p in spec.PERIODS}
    return {"status": "complete", "protocol": spec.PROTOCOL, "root_rows": root_rows, "endpoints": endpoints,
        "period_material_replication_gate": per_period, "material_replication_gate": "supported_both_periods" if all(per_period.values()) else "not_closed",
        "statistics": {"independent_unit": "source_policy_optimizer_root", "n_independent": len(cells),
            "paired_scenarios_per_root_period": spec.EVALUATION_EPISODES, "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
            "familywise_alpha": .05, "bonferroni_family_size": len(spec.ENDPOINTS)},
        "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget()},
        "evidence_role": "frozen_method_new_labels_and_scenes_on_six_additional_source_roots",
        "prior_joint_HRL_and_frequency_specific_gates": "unchanged"}
