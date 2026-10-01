"""Fit discounted remaining-mass values without changing either actor."""

from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import replace
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import concat_hierarchical_batches
from . import pointmaze_actor_credit as signals
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_matched_upper as previous
from . import pointmaze_joint_renewal as joint
from . import pointmaze_value_targets as values
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_horizon_value_stage67_spec as spec


def discounted_mass(remaining, gamma):
    n = np.asarray(remaining, dtype=np.float64)
    mass = n if gamma == 1. else -np.expm1(n * math.log(gamma)) / (1. - gamma)
    return np.asarray(mass, dtype=np.float32)


class FactoredValueFit(values.ValueFit):
    def __init__(self, model, horizon):
        super().__init__(model, "mc_normalized")
        self.treatment, self.horizon = spec.CANDIDATE, horizon

    def mass(self, state):
        return discounted_mass(np.rint(np.asarray(state)[..., -1] * self.horizon), self.model.config.gamma)

    def predictions(self, lower):
        rate = continuing.episode_predictions(self.model, lower)
        return (self.mass(lower.value_state) * (self.location + self.scale * rate)).astype(np.float32)

    def initialize_rate_frame(self, lower, mc):
        if self.initialized or self.model.lower_value_optimizer.state:
            raise ValueError("rate frame requires the unchanged clone and empty value Adam")
        mass = self.mass(lower.value_state)
        rate = mc / mass
        self.location, self.scale = float(rate.mean()), max(float(rate.std()), 1e-6)
        self.initialized = True
        full_mass = float(discounted_mass(self.horizon, self.model.config.gamma))
        with torch.no_grad():
            head = self.model.lower_value.net[-1]
            head.weight.div_(full_mass * self.scale)
            head.bias.div_(full_mass).sub_(self.location).div_(self.scale)
        prediction, expected = self.predictions(lower), mass / full_mass * lower.old_value
        np.testing.assert_allclose(prediction, expected, atol=2e-4, rtol=0)
        return float(np.max(np.abs(prediction - expected)))

    def update(self, lower, target, **kwargs):
        return super().update(lower, target / self.mass(lower.value_state), **kwargs)

    def public_state(self):
        raise ValueError("factored values require remaining mass; not ordinary ValueNet public weights")

    def checkpoint(self):
        return {"parameterization": "discounted_remaining_mass", "horizon": self.horizon,
            "gamma": self.model.config.gamma, "location": self.location, "scale": self.scale,
            "value_training_state": self.model.lower_value.state_dict(),
            "value_optimizer_training_units": self.model.lower_value_optimizer.state_dict()}

    @classmethod
    def restore(cls, model, saved):
        if saved["parameterization"] != "discounted_remaining_mass" or model.config.gamma != saved["gamma"]:
            raise ValueError("factored critic parameterization/gamma changed")
        fit = cls(model, saved["horizon"])
        fit.location, fit.scale, fit.initialized = saved["location"], saved["scale"], True
        model.lower_value.load_state_dict(saved["value_training_state"])
        model.lower_value_optimizer.load_state_dict(saved["value_optimizer_training_units"])
        return fit


def control_predictions(fit, lower, clone):
    public = copy.deepcopy(clone)
    public.lower_value.load_state_dict(fit.public_state())
    return continuing.episode_predictions(public, lower)


def replay(root, *, preflight, output):
    file = spec.training_result(root, preflight=preflight)
    source, reference = json.loads(file.read_text()), json.loads(spec.source_result(root, preflight=preflight).read_text())
    values.qualify(reference, preflight=preflight)
    if ((source["status"], source["root"], source["preflight"]) != ("complete", root, preflight)
            or source["contract"] != spec.source.source.source.source.previous.contract()):
        raise ValueError("Stage67 requires the completed unchanged Stage57 source")
    roles, opt = spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight)
    if set(roles["calibration"]).intersection(roles["first_training_probe"]):
        raise ValueError("Stage67 calibration/probe overlap")
    clones, _, initialization = previous.load_source(root, preflight=preflight)
    args, cost, groups, started = spec.arguments(root, preflight=preflight), dict.fromkeys(spec.budget(preflight=preflight), 0), {}, time.monotonic()
    cost.update(source_clone_loads=len(clones), forecaster_loads=1)
    directory, out_raw = file.parent.with_name(file.parent.name + "_raw"), raw_directory(output)
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"), initializer=diagnostics.init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        def batch(clone, period, arm, phase, item):
            path = directory / str(period) / arm / phase / str(item["iteration"]) / "training"
            weights = joint.inference_weights(clone)
            pairs = list(pool.map(diagnostics.worker_reconstruct,
                [(weights, str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in item["rows"]]))
            for (_, row), old in zip(pairs, item["rows"]):
                if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                    raise ValueError("Stage67 archived action or reward changed")
                cost["archive_episodes"] += 1
                cost["reconstructed_lower_calls"] += row["lower_calls"]
                cost["reconstructed_upper_calls"] += row["upper_calls"]
                cost["archive_network_checks"] += 1
            b = concat_hierarchical_batches([b for b, _ in pairs])
            lower = continuing.episode_batch(b, b.lower.old_value, args.horizon).lower
            # The clock is part of the archived causal value state, not a new label.
            remaining = (args.horizon - np.arange(lower.size) % args.horizon) / args.horizon
            np.testing.assert_allclose(lower.value_state[:, -1], remaining, atol=1e-7, rtol=0)
            mc = values.monte_carlo_returns(lower, clone.config.gamma)
            cost["mc_target_calls"] += 1
            return lower, mc

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                fits = {"mc_normalized": values.ValueFit(copy.deepcopy(clone), "mc_normalized"),
                    spec.CANDIDATE: FactoredValueFit(copy.deepcopy(clone), args.horizon)}
                history, errors = [], {}
                items = source["calibration"][p][arm]["history"]
                if [r["seed"] for item in items for r in item["rows"]] != roles["calibration"]:
                    raise ValueError("Stage67 calibration roster changed")
                for item in items:
                    lower, mc = batch(clone, period, arm, "warmup", item)
                    if item["iteration"] == 1:
                        errors["mc_normalized"] = fits["mc_normalized"].initialize_frame(float(mc.mean()), max(float(mc.std()), 1e-6), lower)
                        errors[spec.CANDIDATE] = fits[spec.CANDIDATE].initialize_rate_frame(lower, mc)
                        cost["normalization_frame_fits"] += len(fits)
                        cost["initialization_value_rows"] += len(fits) * lower.size
                    updates = {t: fit.update(lower, mc, root=root, period=period, iteration=item["iteration"]) for t, fit in fits.items()}
                    cost["calibration_updates"] += len(fits)
                    history.append({"iteration": item["iteration"], "updates": updates})
                for fit in fits.values():
                    for name in ("lower_actor", "lower_actor_optimizer", "upper_actor", "upper_value", "upper_actor_optimizer", "upper_value_optimizer"):
                        torch.testing.assert_close(getattr(fit.model, name).state_dict(), getattr(clone, name).state_dict(), atol=0, rtol=0)
                        cost["frozen_state_checks"] += 1
                control = fits["mc_normalized"]
                ref = reference["groups"][p][arm]["treatments"]["mc_normalized"]
                saved = torch.load(ref["checkpoint"], map_location="cpu", weights_only=False)
                for actual, expected in ((control.model.lower_value.state_dict(), saved["value_training_state"]),
                        (control.model.lower_value_optimizer.state_dict(), saved["value_optimizer_training_units"]),
                        (control.public_state(), saved["public_value_state"])):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                if (control.location, control.scale) != (saved["location"], saved["scale"]):
                    raise ValueError("Stage67 control normalization changed")
                cost["source_critic_loads"] += 1
                first = source["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training_probe"]:
                    raise ValueError("Stage67 probe roster changed")
                lower, mc = batch(clone, period, arm, "train", first)
                rows = {}
                for t, fit in fits.items():
                    prediction = control_predictions(fit, lower, clone) if t == "mc_normalized" else fit.predictions(lower)
                    row = signals.diagnose(fit.model, replace(lower, old_value=prediction), mc, horizon=args.horizon)
                    if t == "mc_normalized" and row["value_mc"] != ref["episode_mc"]:
                        raise ValueError("Stage67 probe differs from actual Stage64 control")
                    cost["probe_value_rows"] += lower.size
                    cost["probe_gae_calls"] += 1
                    cost["frozen_model_checks"] += 1
                    for key in ("actor_score_forward_batches", "actor_score_backward_batches"):
                        cost[key] += row["score_cost"][key]
                    rows[t] = {**row, "frame": {"location": fit.location, "scale": fit.scale, "iteration": 1, "sample_count": lower.size}}
                cost["source_control_checks"] += 1
                checkpoint = out_raw / p / arm / spec.CANDIDATE / "critic.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "arm": arm,
                    **fits[spec.CANDIDATE].checkpoint()}, checkpoint)
                cost["critic_checkpoint_writes"] += 1
                groups[p][arm] = {"control_reproduction": "passed", "frozen_actor_upper_and_Adam": "passed",
                    "initialization_error": errors, "history": history, "treatments": rows, "candidate_checkpoint": str(checkpoint)}
                print(f"finite horizon {root}/period{period}/{arm}: control/Adam exact; actor/upper frozen", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "cost": cost,
        "source_initialization": initialization, "groups": groups, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "preflight": preflight})
    return result


def qualify(cell, *, preflight):
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL or cell["contract"] != spec.contract()
            or cell["root"] not in spec.roots(preflight=preflight) or cell["preflight"] != preflight
            or cell["options"] != spec.options(preflight=preflight) or cell["seed_roles"] != spec.seed_roles(cell["root"], preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight) or set(cell["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage67 frozen protocol, roster or cost changed")
    cfg, opt = cell["source_initialization"]["config"], cell["options"]
    n = spec.arguments(cell["root"], preflight=preflight).horizon * opt["rollouts_per_iteration"]
    expected = max(1, cfg["epochs"]) * math.ceil(n / cfg["minibatch_size"])
    compact, totals, failures = copy.deepcopy(cell), dict.fromkeys(spec.TREATMENTS, 0), []
    for p, arms in compact["groups"].items():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage67 arm roster changed")
        for arm, group in arms.items():
            if (group["control_reproduction"] != "passed" or group["frozen_actor_upper_and_Adam"] != "passed"
                    or set(group["treatments"]) != set(spec.TREATMENTS) or set(group["initialization_error"]) != set(spec.TREATMENTS)
                    or max(group["initialization_error"].values()) > 2e-4
                    or [h["iteration"] for h in group["history"]] != list(range(1, opt["critic_warmup_iterations"] + 1))):
                raise ValueError("Stage67 pairing or initialization changed")
            for item in group["history"]:
                if set(item["updates"]) != set(spec.TREATMENTS):
                    raise ValueError("Stage67 calibration treatment changed")
                for t, update in item["updates"].items():
                    if update["value_optimizer_steps"] != expected or update["value_forward_minibatches"] != expected:
                        raise ValueError("Stage67 value step budget changed")
                    totals[t] += expected
            for row in group["treatments"].values():
                if (row["model_and_Adam_unchanged"] != "passed" or row["frame"]["iteration"] != 1
                        or row["frame"]["sample_count"] != n or row["frame"]["scale"] <= 0):
                    raise ValueError("Stage67 probe changed model or fitted a probe frame")
            a, b = [group["treatments"][t] for t in spec.TREATMENTS]
            ta, tb = [r["windows"]["last_decile"]["value_mc"] for r in (a, b)]
            ev = b["value_mc"]["explained_variance"]
            if (ev is None or ev < .10 or b["value_mc"]["mse"] >= a["value_mc"]["mse"]
                    or tb["mse"] >= ta["mse"] or abs(tb["bias"]) >= abs(ta["bias"])):
                failures.append({"root": cell["root"], "period": int(p), "arm": arm})
            group["training_summary"] = {t: {k: float(np.mean([h["updates"][t][k] for h in group["history"]]))
                for k in ("loss_training_units", "preclip_gradient_norm_mean", "gradient_clip_fraction")} for t in spec.TREATMENTS}
            group.pop("history")
            group.pop("candidate_checkpoint")
    return compact, totals, failures


def aggregate(cells, *, preflight):
    if len({c["root"] for c in cells}) != len(cells) or {c["root"] for c in cells} != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage67 requires the complete frozen root roster")
    rows, totals, failures, credit_failures = [], dict.fromkeys(spec.TREATMENTS, 0), [], []
    for cell in sorted(cells, key=lambda c: c["root"]):
        row, steps, failed = qualify(cell, preflight=preflight)
        rows.append(row)
        failures.extend(failed)
        for t in totals:
            totals[t] += steps[t]
    comparisons = {}
    for p in spec.PERIODS:
        for arm in spec.TRAIN_POLICIES:
            group = {}
            for t in spec.TREATMENTS:
                samples = [r["groups"][str(p)][arm]["treatments"][t] for r in rows]
                def avg(fn):
                    v = [fn(s) for s in samples]
                    return None if any(x is None for x in v) else float(np.mean(v))
                group[t] = {"global_mse": avg(lambda s: s["value_mc"]["mse"]),
                    "tail_mse": avg(lambda s: s["windows"]["last_decile"]["value_mc"]["mse"]),
                    "tail_bias": avg(lambda s: s["windows"]["last_decile"]["value_mc"]["bias"]),
                    "sign_disagreement": avg(lambda s: s["credit_alignment"]["normalized_sign_disagreement"]),
                    "mean_cosine": avg(lambda s: s["gradient"]["mean"]["credit_cosine"]),
                    "log_std_cosine": avg(lambda s: s["gradient"]["log_std"]["credit_cosine"])}
                if t == spec.CANDIDATE:
                    for s, r in zip(samples, rows):
                        cosine = s["gradient"]["mean"]["credit_cosine"]
                        if cosine is None or cosine <= 0:
                            credit_failures.append({"root": r["root"], "period": p, "arm": arm, "reason": "nonpositive_mean_gradient"})
            a, b = [group[t] for t in spec.TREATMENTS]
            if (any(b[k] is None or a[k] is None or b[k] < a[k] for k in ("mean_cosine", "log_std_cosine"))
                    or b["sign_disagreement"] > a["sign_disagreement"]):
                credit_failures.append({"period": p, "arm": arm, "reason": "equal_root_credit_worse"})
            comparisons[f"{p}/{arm}"] = group
    return {"status": "preflight_passed" if preflight else "complete", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "mechanical_gate": "passed", "fit_gate": "failed" if failures else "passed",
        "credit_gate": "failed" if credit_failures else "passed", "fit_failures": failures, "credit_failures": credit_failures,
        "comparisons": comparisons, "root_rows": rows, "value_optimizer_steps": totals,
        "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget(preflight=preflight)},
        "native_trial_prerequisite": "not_applicable_preflight" if preflight else "hold" if failures or credit_failures else "passed_critic_only",
        "performance_claim": "none_archive_value_and_credit_only"}
