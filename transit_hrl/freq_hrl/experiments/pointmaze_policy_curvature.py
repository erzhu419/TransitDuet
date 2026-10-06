"""Learn a full-policy native-return step along the fixed Stage124 direction."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing as mp

import numpy as np
import torch

from freq_hrl.rl.native_return_step import fit_native_return_step
from . import pointmaze_native_upper_step as source
from .pointmaze_root_response import write_json
from scripts import pointmaze_policy_curvature_stage125_spec as spec

joint = source.joint


def scaled_upper(initial, direction, scale):
    torch.testing.assert_close(initial["log_std"], direction["log_std"], atol=0, rtol=0)
    return {k: initial[k] + scale * (direction[k] - initial[k]) for k in initial}


def load_directions(root, period):
    states = {}
    for sign in ("plus", "minus"):
        path = spec.source_result(root).parent / "final_weights" / f"period_{period}_{sign}_upper.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        if (data["protocol"], data["root"], data["period"], data["sign"], data["fisher_radius"]) != (
                spec.source.PROTOCOL, root, period, sign, spec.source.FISHER_RADIUS):
            raise ValueError("Stage125 source upper direction changed")
        states[sign] = data["weights"]
    return states


def worker_group(job):
    source_weights, teacher, variants, seed, noise, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_weights)
    trainer = joint.make_trainer(model, teacher, args)
    initial, rows, common = joint.weights(trainer), {}, None
    for name, upper, arm in variants:
        trainer.upper_actor.load_state_dict(upper)
        _, row, audit = joint.native_episode(trainer, args=args, seed=seed, noise_seed=noise,
            arm=arm, period=period, predictor=predictor, envelope=envelope, collect=False)
        if common is None:
            common = audit
        else:
            np.testing.assert_array_equal(audit["measurements"], common["measurements"])
            np.testing.assert_allclose(audit["innovations"], common["innovations"], atol=3e-5, rtol=0)
        row["variant"] = name
        rows[name] = row
    for name in ("lower_actor", "upper_value", "lower_value"):
        torch.testing.assert_close(getattr(trainer, name).state_dict(), initial[name], atol=0, rtol=0)
    if "curvature_blinded" in rows and rows["curvature_blinded"]["episode_return"] != rows["source_forecast"]["episode_return"]:
        raise ValueError("Stage125 blinded upper must exactly execute forecast")
    return rows


def count_group(cost, rows, phase):
    cost[phase + "_pair_groups"] += 1
    for row in rows.values():
        cost[phase + "_episodes"] += 1
        cost["native_episodes"] += 1
        for key, field in (("native_steps", "episode_length"), ("native_upper_calls", "upper_calls"),
                ("native_donor_response_calls", "reference_donor_calls")):
            cost[key] += row[field]


def fit_panel(rows):
    return fit_native_return_step(*[[r[k]["episode_return"] for r in rows] for k in ("zero", "plus", "minus")])


def evaluation_summary(rows):
    metrics = ("episode_return", "reference_correction_rms", "reference_correction_peak", "plan_delta_rms", "upper_mean_rms")
    means = {v: {k: float(np.mean([r[v][k] for r in rows])) for k in metrics} for v in spec.VARIANTS}
    effects = {}
    for a, b in spec.CONTRASTS:
        difference = [r[a]["episode_return"] - r[b]["episode_return"] for r in rows]
        effects[f"{a}_minus_{b}"] = {"mean": float(np.mean(difference)), "paired_differences": difference}
    return {"mean_metrics": means, "effects": effects}


def run(root, output):
    output = output.resolve()
    cached = json.loads(spec.source_result(root).read_text())
    if (cached["status"], cached["protocol"], cached["root"], cached["cost"]) != (
            "complete", spec.source.PROTOCOL, root, spec.source.budget()):
        raise ValueError("Stage125 requires the completed Stage124 source")
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
            initial = copy.deepcopy(trainer.upper_actor.state_dict())
            directions = load_directions(root, period)
            cost["upper_checkpoint_loads"] += 2
            torch.testing.assert_close(scaled_upper(directions["plus"], directions["minus"], .5), initial, atol=0, rtol=0)
            envelope = calibrations[p]["envelope"]
            variants = [("zero", initial, "forecast"), ("plus", directions["plus"], "joint"), ("minus", directions["minus"], "joint")]
            training = {}
            roles = spec.training_roles(root)
            for panel in spec.PANELS:
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][panel],
                    period, predictor, envelope) for r in roles]
                training[panel] = list(pool.map(worker_group, jobs))
                for rows in training[panel]:
                    count_group(cost, rows, "training")
            fits = {panel: fit_panel(training[panel]) for panel in spec.PANELS}
            fits["pooled"] = fit_panel(training["A"] + training["B"])
            cost["native_return_fits"] += 3
            crossfit = {}
            for train_panel, test_panel in (("A", "B"), ("B", "A")):
                scaled = scaled_upper(initial, directions["plus"], fits[train_panel]["scale"])
                variants = [("zero", initial, "forecast"), ("scaled", scaled, "joint")]
                jobs = [(joint.weights(model), teacher, variants, r["scenario_seed"], r["noise_seeds"][test_panel],
                    period, predictor, envelope) for r in roles]
                gains = []
                for index, rows in enumerate(pool.map(worker_group, jobs)):
                    count_group(cost, rows, "crossfit")
                    if rows["zero"]["episode_return"] != training[test_panel][index]["zero"]["episode_return"]:
                        raise ValueError("Stage125 crossfit zero replay changed")
                    gains.append(rows["scaled"]["episode_return"] - rows["zero"]["episode_return"])
                crossfit[f"{train_panel}_to_{test_panel}"] = {"scale": fits[train_panel]["scale"],
                    "mean_gain": float(np.mean(gains)), "paired_differences": gains}
            scale = fits["pooled"]["scale"]
            plus = scaled_upper(initial, directions["plus"], scale)
            minus = scaled_upper(initial, directions["minus"], scale)
            cost["upper_candidate_weight_steps"] += 4
            variants = [("source_flat", initial, "flat"), ("source_forecast", initial, "forecast"),
                ("unit_ascent", directions["plus"], "joint"), ("curvature_ascent", plus, "joint"),
                ("curvature_descent", minus, "joint"), ("curvature_blinded", plus, "forecast")]
            seeds = spec.evaluation_seeds(root)
            jobs = [(joint.weights(model), teacher, variants, seed, seed, period, predictor, envelope) for seed in seeds]
            evaluation = list(pool.map(worker_group, jobs))
            for rows in evaluation:
                count_group(cost, rows, "evaluation")
            path = output.parent / "final_weights" / f"period_{period}_curvature_upper.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"protocol": spec.PROTOCOL, "root": root, "period": period, "source_run": spec.SOURCE_RUN,
                "fit": fits["pooled"], "weights": plus}, path)
            cost["checkpoint_writes"] += 1
            joint.source.native.curves.support.assert_frozen(model, before)
            groups[p] = {"training_roles": roles, "training_native_return_fits": fits,
                "training_crossfit": crossfit, "evaluation_seeds": seeds, "lower_and_critics_frozen": "passed",
                "upper_weight_delta_rms": joint.parameter_delta(initial, plus), **evaluation_summary(evaluation)}
            print(f"root={root} period={period}: scale={scale:.6g}, fresh native evaluation complete", flush=True)
    if cost != spec.budget():
        raise ValueError(f"Stage125 measured budget changed: {cost}")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root, "groups": groups, "cost": cost,
        "inherited_Stage124_cost": cached["cost"], "inherited_Stage123_cost": cached["inherited_Stage123_cost"],
        "minimum_episode_gain": spec.MINIMUM_GAIN,
        "kind": "training_only_native_policy_curvature_development_not_joint_HRL_confirmation"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native policy-curvature step result written", flush=True)
    return result
