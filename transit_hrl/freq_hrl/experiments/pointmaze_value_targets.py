"""Paired value-only target/scale ablation on frozen native archives."""

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import multiprocessing as mp
import time

import numpy as np
import torch

from freq_hrl.rl.smdp_actor_critic import FrequencySeparatedActorCriticPPO, concat_hierarchical_batches
from . import pointmaze_update_diagnostics as diagnostics
from . import pointmaze_continuing_credit as continuing
from . import pointmaze_matched_upper as previous
from . import pointmaze_joint_renewal as joint
from . import pointmaze_credit_diagnostics as credit
from .pointmaze_critic_calibration import monte_carlo_returns
from .pointmaze_root_response import raw_directory, write_json
from scripts import pointmaze_value_targets_stage64_spec as spec


class ValueFit:
    def __init__(self, model, treatment):
        if treatment not in spec.TREATMENTS:
            raise ValueError("unregistered value treatment")
        self.model, self.treatment = model, treatment
        self.location, self.scale = 0., 1.
        self.initialized = False

    def initialize_frame(self, location, scale, lower):
        if not self.treatment.endswith("normalized") or self.initialized:
            raise ValueError("normalization frame must be fitted once on calibration")
        if self.model.lower_value_optimizer.state:
            raise ValueError("Stage55 clone value Adam must be empty; do not reset or relabel its moments")
        self.location, self.scale = float(location), float(scale)
        self.initialized = True
        with torch.no_grad():
            head = self.model.lower_value.net[-1]
            head.weight.div_(self.scale)
            head.bias.sub_(self.location).div_(self.scale)
        public = copy.deepcopy(self.model)
        public.lower_value.load_state_dict(self.public_state())
        values = continuing.episode_predictions(public, lower)
        error = float(np.max(np.abs(values - lower.old_value)))
        np.testing.assert_allclose(values, lower.old_value, atol=2e-4, rtol=0)
        return error

    def public_state(self):
        state = copy.deepcopy(self.model.lower_value.state_dict())
        head = f"net.{len(self.model.lower_value.net) - 1}"
        state[head + ".weight"] *= self.scale
        state[head + ".bias"] = state[head + ".bias"] * self.scale + self.location
        return state

    def update(self, lower, target, *, root, period, iteration):
        model, cfg = self.model, self.model.config
        state = torch.as_tensor(lower.value_state, dtype=torch.float32, device=model.device)
        target_t = (torch.as_tensor(target, dtype=torch.float32, device=model.device) - self.location) / self.scale
        indices, count, losses, norms = np.arange(lower.size), 0, [], []
        np.random.seed(spec.source.source.source.previous.shuffle_seed(root, period, iteration, phase="warmup", level="lower"))
        minibatch = min(cfg.minibatch_size, lower.size)
        for _ in range(max(1, cfg.epochs)):
            np.random.shuffle(indices)
            for start in range(0, lower.size, minibatch):
                idx = torch.as_tensor(indices[start:start + minibatch], dtype=torch.long, device=model.device)
                loss = torch.mean((model.lower_value(state[idx]) - target_t[idx]) ** 2)
                model.lower_value_optimizer.zero_grad()
                (float(cfg.value_coef) * loss).backward()
                norm = torch.nn.utils.clip_grad_norm_(model.lower_value.parameters(), cfg.max_grad_norm)
                model.lower_value_optimizer.step()
                count += 1
                losses.append(float(loss.detach()))
                norms.append(float(norm))
        return {"value_optimizer_steps": count, "value_forward_minibatches": count,
            "loss_training_units": float(np.mean(losses)), "preclip_gradient_norm_mean": float(np.mean(norms)),
            "gradient_clip_fraction": float(np.mean(np.asarray(norms) > cfg.max_grad_norm))}


_WORKERS = None


def init_worker(config, args):
    global _WORKERS
    diagnostics.init_worker(config, args)
    _WORKERS = {t: FrequencySeparatedActorCriticPPO(config) for t in spec.TREATMENTS[1:]}


def worker_batch(job):
    weights, path, seed, period = job
    batch, row = diagnostics.worker_reconstruct((weights["gae_raw"], path, seed, period))
    values = {"gae_raw": batch.lower.old_value}
    for treatment, model in _WORKERS.items():
        model.load_state_dict(weights[treatment])
        values[treatment] = continuing.episode_predictions(model, batch.lower)
        torch.testing.assert_close(joint.inference_weights(model), weights[treatment], atol=0, rtol=0)
    return batch, values, {**row, "extra_value_calls": 3 * batch.lower.size, "network_checks": 4}


def representation(fit, lower):
    net = fit.model.lower_value.net
    counts, sums, squares, total, forwards = [0, 0], [None, None], [None, None], 0, 0
    with torch.no_grad():
        for start in range(0, lower.size, spec.VALUE_BATCH_SIZE):
            x = torch.as_tensor(lower.value_state[start:start + spec.VALUE_BATCH_SIZE], dtype=torch.float32, device=fit.model.device)
            total += len(x)
            layer_id = 0
            for layer in net:
                x = layer(x)
                if isinstance(layer, torch.nn.Tanh):
                    counts[layer_id] += int((x.abs() >= .99).sum())
                    z, z2 = x.double().sum(0), x.double().square().sum(0)
                    sums[layer_id] = z if sums[layer_id] is None else sums[layer_id] + z
                    squares[layer_id] = z2 if squares[layer_id] is None else squares[layer_id] + z2
                    layer_id += 1
            forwards += 1
    return {"tanh_saturation_fraction": [counts[i] / (total * sums[i].numel()) for i in range(2)],
        "hidden_unit_std_mean": [float((squares[i] / total - (sums[i] / total).square()).clamp_min(0).sqrt().mean()) for i in range(2)]}, forwards


def replay(root, *, preflight, output):
    file = spec.training_result(root, preflight=preflight)
    source = json.loads(file.read_text())
    reference = json.loads(spec.source_result(root, preflight=preflight).read_text())
    if (any((c["status"], c["root"], c["preflight"]) != ("complete", root, preflight) for c in (source, reference))
            or reference["contract"] != spec.source.contract()
            or source["contract"] != spec.source.source.source.previous.contract()):
        raise ValueError("Stage64 requires frozen completed Stage57/63 sources")
    roles, opt, budget = spec.seed_roles(root, preflight=preflight), spec.options(preflight=preflight), spec.budget(preflight=preflight)
    if set(roles["calibration"]).intersection(roles["first_training_probe"]):
        raise ValueError("Stage64 calibration/probe overlap")
    clones, _, initialization = previous.load_source(root, preflight=preflight)
    args, costs, started = spec.arguments(root, preflight=preflight), dict.fromkeys(budget, 0), time.monotonic()
    costs.update(source_clone_loads=len(clones), forecaster_loads=1)
    directory, out_raw, groups = file.parent.with_name(file.parent.name + "_raw"), raw_directory(output), {}
    with ProcessPoolExecutor(max_workers=opt["workers"], mp_context=mp.get_context("spawn"), initializer=init_worker,
            initargs=(clones[str(spec.PERIODS[0])].config, args)) as pool:
        def batches(fits, clone, period, arm, phase, item):
            weights = {}
            for t, fit in fits.items():
                weights[t] = joint.inference_weights(clone)
                weights[t]["lower_value"] = fit.public_state()
            path = directory / str(period) / arm / phase / str(item["iteration"]) / "training"
            pairs = list(pool.map(worker_batch, [(weights, str(path / f"episode_{r['seed']}.npz"), r["seed"], period) for r in item["rows"]]))
            for (_, _, row), old in zip(pairs, item["rows"]):
                if row["episode_return"] != old["episode_return"] or row["action_check"] != "passed":
                    raise ValueError("Stage64 source action or reward changed")
                for key, source_key in (("archive_episodes", None), ("reconstructed_lower_calls", "lower_calls"),
                        ("reconstructed_upper_calls", "upper_calls"), ("extra_critic_scalar_calls", "extra_value_calls"),
                        ("archive_network_checks", "network_checks")):
                    costs[key] += 1 if source_key is None else row[source_key]
            native = concat_hierarchical_batches([b for b, _, _ in pairs])
            return {t: continuing.episode_batch(native, np.concatenate([values[t] for _, values, _ in pairs]), args.horizon).lower for t in fits}

        for period in spec.PERIODS:
            p, clone = str(period), clones[str(period)]
            groups[p] = {}
            for arm in spec.TRAIN_POLICIES:
                fits = {t: ValueFit(copy.deepcopy(clone), t) for t in spec.TREATMENTS}
                history, frame, initialization_errors = [], None, {}
                items = source["calibration"][p][arm]["history"]
                if [r["seed"] for item in items for r in item["rows"]] != roles["calibration"]:
                    raise ValueError("Stage64 calibration roster changed")
                for item in items:
                    lower = batches(fits, clone, period, arm, "warmup", item)
                    mc = monte_carlo_returns(lower["gae_raw"], clone.config.gamma)
                    costs["mc_target_calls"] += 1
                    if frame is None:
                        frame = {"location": float(mc.mean()), "scale": max(float(mc.std()), 1e-6),
                            "iteration": item["iteration"], "sample_count": len(mc), "source": "first_calibration_episode_MC_only"}
                        costs["normalization_frame_fits"] += 1
                        for t in ("gae_normalized", "mc_normalized"):
                            initialization_errors[t] = fits[t].initialize_frame(frame["location"], frame["scale"], lower[t])
                            costs["initialization_checks"] += 1
                            costs["initialization_value_rows"] += len(mc)
                    updates = {}
                    for t, fit in fits.items():
                        target = mc
                        if t.startswith("gae"):
                            b = lower[t]
                            _, target = fit.model._gae(b.reward, b.done, b.duration, b.old_value)
                            costs["calibration_gae_calls"] += 1
                        else:
                            costs["mc_supervised_updates"] += 1
                        updates[t] = fit.update(lower[t], target, root=root, period=period, iteration=item["iteration"])
                        costs["calibration_updates"] += 1
                    history.append({"iteration": item["iteration"], "updates": updates})
                for fit in fits.values():
                    for name in ("lower_actor", "lower_actor_optimizer", "upper_actor", "upper_value", "upper_actor_optimizer", "upper_value_optimizer"):
                        torch.testing.assert_close(getattr(fit.model, name).state_dict(), getattr(clone, name).state_dict(), atol=0, rtol=0)
                        costs["frozen_state_checks"] += 1
                first = source["training"][p][arm]["history"][0]
                if [r["seed"] for r in first["rows"]] != roles["first_training_probe"]:
                    raise ValueError("Stage64 probe roster changed")
                lower = batches(fits, clone, period, arm, "train", first)
                mc = monte_carlo_returns(lower["gae_raw"], clone.config.gamma)
                costs["mc_target_calls"] += 1
                control = credit.value_metrics(lower["gae_raw"].old_value, mc)
                if control != reference["comparisons"][p][arm]["critic_probe"]["episode_value_episode_mc"]:
                    raise ValueError("Stage64 raw GAE critic probe differs from exact Stage63")
                advantages, treatments = {}, {}
                for t, fit in fits.items():
                    b = lower[t]
                    advantages[t], _ = fit.model._gae(b.reward, b.done, b.duration, b.old_value)
                    costs["probe_gae_calls"] += 1
                    hidden, forwards = representation(fit, b)
                    costs["representation_forward_batches"] += forwards
                    checkpoint = out_raw / p / arm / t / "critic.pt"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"protocol": spec.EXPERIMENT_PROTOCOL, "root": root, "period": period, "arm": arm, "treatment": t,
                        "config": clone.state_dict()["config"], "location": fit.location, "scale": fit.scale,
                        "value_training_state": fit.model.lower_value.state_dict(),
                        "value_optimizer_training_units": fit.model.lower_value_optimizer.state_dict(),
                        "public_value_state": fit.public_state()}, checkpoint)
                    costs["critic_checkpoint_writes"] += 1
                    treatments[t] = {"episode_mc": credit.value_metrics(b.old_value, mc), "representation": hidden, "checkpoint": str(checkpoint)}
                for t in fits:
                    treatments[t]["advantage_vs_gae_raw"] = credit.alignment(advantages["gae_raw"], advantages[t])
                groups[p][arm] = {"control_reproduction": "passed", "frozen_actor_upper_and_Adam": "passed",
                    "frame": frame, "initialization_public_error": initialization_errors, "history": history, "treatments": treatments}
                print(f"value targets {root}/period{period}/{arm}: raw GAE exact, actor/upper frozen", flush=True)
    result = {"status": "complete", "protocol": spec.EXPERIMENT_PROTOCOL, "contract": spec.contract(), "root": root,
        "preflight": preflight, "options": opt, "seed_roles": roles, "budget": budget, "cost": costs,
        "source_initialization": initialization, "source_reference": str(spec.source_result(root, preflight=preflight)),
        "groups": groups, "wall_seconds": time.monotonic() - started}
    qualify(result, preflight=preflight)
    write_json(output, result)
    return result


def qualify(c, *, preflight):
    if (c["status"] != "complete" or c["protocol"] != spec.EXPERIMENT_PROTOCOL or c["contract"] != spec.contract()
            or c["root"] not in spec.roots(preflight=preflight) or c["preflight"] != preflight
            or c["options"] != spec.options(preflight=preflight) or c["seed_roles"] != spec.seed_roles(c["root"], preflight=preflight)
            or c["budget"] != spec.budget(preflight=preflight) or c["cost"] != c["budget"]
            or set(c["groups"]) != {str(p) for p in spec.PERIODS}):
        raise ValueError("Stage64 protocol, source roster or costs changed")
    opt, cfg = c["options"], c["source_initialization"]["config"]
    n = opt["rollouts_per_iteration"] * spec.arguments(c["root"], preflight=preflight).horizon
    expected = max(1, cfg["epochs"]) * math.ceil(n / cfg["minibatch_size"])
    totals, failures, groups = dict.fromkeys(spec.TREATMENTS, 0), [], copy.deepcopy(c["groups"])
    for p, arms in groups.items():
        if set(arms) != set(spec.TRAIN_POLICIES):
            raise ValueError("Stage64 arm roster changed")
        for arm, cell in arms.items():
            frame = cell["frame"]
            if (cell["control_reproduction"] != "passed" or cell["frozen_actor_upper_and_Adam"] != "passed"
                    or set(cell["treatments"]) != set(spec.TREATMENTS)
                    or frame["iteration"] != 1 or frame["sample_count"] != n or frame["source"] != "first_calibration_episode_MC_only"
                    or frame["scale"] <= 0 or set(cell["initialization_public_error"]) != {"gae_normalized", "mc_normalized"}
                    or max(cell["initialization_public_error"].values()) > 2e-4
                    or [h["iteration"] for h in cell["history"]] != list(range(1, opt["critic_warmup_iterations"] + 1))):
                raise ValueError("Stage64 normalization initialization or held-out pairing changed")
            for item in cell["history"]:
                if set(item["updates"]) != set(spec.TREATMENTS):
                    raise ValueError("Stage64 factorial roster changed")
                for t, u in item["updates"].items():
                    if u["value_optimizer_steps"] != expected or u["value_forward_minibatches"] != expected:
                        raise ValueError("Stage64 value optimizer/forward budget changed")
                    totals[t] += expected
            fit = cell["treatments"][spec.CANDIDATE]["episode_mc"]
            if (fit["explained_variance"] is None or fit["explained_variance"] < spec.MIN_PROBE_EV
                    or fit["mse"] >= cell["treatments"]["gae_raw"]["episode_mc"]["mse"]):
                failures.append({"root": c["root"], "period": int(p), "arm": arm})
            for t in cell["treatments"].values():
                t.pop("checkpoint")
            cell["training_summary"] = {t: {key: float(np.mean([h["updates"][t][key] for h in cell["history"]]))
                for key in ("loss_training_units", "preclip_gradient_norm_mean", "gradient_clip_fraction")} for t in spec.TREATMENTS}
            cell.pop("history")
    return {"root": c["root"], "groups": groups}, totals, failures


def aggregate(cells, *, preflight):
    by_root = {c["root"]: c for c in cells}
    if len(by_root) != len(cells) or set(by_root) != set(spec.roots(preflight=preflight)):
        raise ValueError("Stage64 requires all frozen roots")
    rows, totals, failures = [], dict.fromkeys(spec.TREATMENTS, 0), []
    for root in spec.roots(preflight=preflight):
        row, counts, failed = qualify(by_root[root], preflight=preflight)
        rows.append(row)
        failures.extend(failed)
        for t in totals:
            totals[t] += counts[t]
    return {"status": "preflight_passed" if preflight else "fixed_archive_valid", "protocol": spec.EXPERIMENT_PROTOCOL,
        "contract": spec.contract(), "root_rows": rows, "cost": {k: sum(c["cost"][k] for c in cells) for k in spec.budget(preflight=preflight)},
        "value_optimizer_steps": totals, "supervised_MC_optimizer_steps": totals["mc_raw"] + totals["mc_normalized"],
        "candidate_fit_failures": failures, "mechanical_gate": "passed", "candidate_fit_gate": "failed" if failures else "passed",
        "native_trial_prerequisite": "not_applicable_preflight" if preflight else "hold" if failures else "passed_critic_only_actor_test_still_required",
        "performance_claim": "none_critic_archive_only"}
