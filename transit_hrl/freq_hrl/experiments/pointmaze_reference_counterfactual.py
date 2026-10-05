"""Native full-suffix credit with the exact joint reference controller frozen."""

from concurrent.futures import ProcessPoolExecutor
import copy
import multiprocessing as mp

import numpy as np
import torch

from . import pointmaze_joint_reference as joint
from .pointmaze_actor_credit import cosine
from .pointmaze_root_response import write_json
from scripts import pointmaze_reference_counterfactual_stage123_spec as spec


def intervention_episode(trainer, *, args, query, panel, action_delta, period, predictor, envelope):
    start = query["start"]
    if start % period or not 0 <= start <= args.horizon - period:
        raise ValueError("Stage123 requires a complete decision-aligned option")
    seeds = joint.source.scenario.spec.noise_seeds
    prefix_seed = seeds(args.optimizer_seed, query["scenario_seed"], query["prefix_noise_seed"])[1]
    suffix_seed = seeds(args.optimizer_seed, query["scenario_seed"], query["suffix_noise_seeds"][panel])[1]
    plan = joint.wide.WideBernsteinPlan(predictor, period, args.maximum_subgoal_delta, envelope)
    task = joint.source.native.joint._make_task(env_id=args.env_id, seed=query["scenario_seed"],
        horizon=args.horizon, **joint.source.native.joint._task_options(args))
    rewards, measurements, innovations, commands = [], [], [], []
    upper_state, query_state = None, None
    peak, reference_square = 0., 0.
    try:
        obs = task.reset()
        low, high = joint.source.native.joint.pointmaze_goal_bounds(task.environment)
        scale = joint.source.native.joint.scale_for(args)
        history = joint.source.native.joint.PointMazeRegimeFeatureBuilder(time_scale=scale)
        history.reset(obs)
        for step in range(args.horizon):
            feedback = joint.source.baseline.flat_state(history, obs)
            clock = joint.clocks(step, period, args.horizon)
            measurements.append(obs.task_measurement.copy())
            if step % period == 0:
                state = np.r_[history.upper_state(obs, oracle_context=None), clock].astype(np.float32)
                with torch.inference_mode():
                    action = trainer.upper_actor.distribution(torch.as_tensor(state).view(1, -1)).mean[0].numpy().copy()
                if step == start:
                    upper_state = state.copy()
                    query_state = np.r_[feedback, obs.achieved_goal].copy()
                    action += action_delta
                plan.decode(action=action, observation=obs, history=history, step=step, world_low=low, world_high=high)
            age = step % period
            reference = plan(observation=obs, history=history, subgoal=None, age=age,
                step=step, world_low=low, world_high=high)
            velocity = plan.actor_context(age=age, step=step, horizon=args.horizon)
            forecast_position = plan.base_points[age]
            forecast_velocity = (plan.base_points[age + 1] - forecast_position) / scale.dt_seconds
            delta = np.r_[reference - forecast_position, velocity - forecast_velocity].astype(np.float32)
            # The teacher still receives forecast advice, exactly as in Stage121.
            teacher_state = joint.source.advice_state(feedback, obs, forecast_position, forecast_velocity)
            state = torch.as_tensor(np.r_[teacher_state, delta, clock].astype(np.float32)).view(1, -1)
            with torch.inference_mode():
                torch.manual_seed((prefix_seed if step < start else suffix_seed) + step)
                distribution, correction, _ = trainer.lower_actor.components(state)
                raw = distribution.sample()[0]
                innovations.append(((raw - distribution.mean[0]) / distribution.stddev[0]).numpy())
                peak = max(peak, float(correction.abs().max()))
                reference_square += float(correction.double().square().sum())
            command = joint.source.native.joint.squash_box_action(raw.numpy(), task.action_low, task.action_high)
            commands.append(command.copy())
            obs, reward, terminated, truncated, _ = task.step(command)
            if bool(terminated or truncated) and step + 1 != args.horizon:
                raise RuntimeError("Stage123 path ended before the fixed horizon")
            rewards.append(float(reward))
            history.update(obs)
        reward = np.asarray(rewards, dtype=np.float64)
        row = {"episode_return": float(reward.sum()), "prefix_return": float(reward[:start].sum()),
            "suffix_return": float(reward[start:].sum()), "option_return": float(reward[start:start + period].sum()),
            "tail_return": float(reward[start + period:].sum()), "episode_length": args.horizon,
            "upper_calls": args.horizon // period, "reference_correction_peak": peak,
            "reference_correction_rms": float(np.sqrt(reference_square / (2 * args.horizon)))}
        audit = {"state": upper_state, "query_state": query_state, "prefix_rewards": reward[:start],
            "commands": np.asarray(commands), "measurements": np.asarray(measurements), "innovations": np.asarray(innovations)}
        return row, audit
    finally:
        task.environment.close()


def gradients(panels):
    return {panel: np.asarray([(rows[f"axis{i}_plus"]["suffix_return"] - rows[f"axis{i}_minus"]["suffix_return"])
        / (2 * spec.EPSILON) for i in range(spec.ACTION_DIM)]) for panel, rows in panels.items()}


def worker_query(job):
    source_state, teacher_state, query, period, predictor, envelope = job
    model, args = joint.source.native._WORKER
    model.load_state_dict(source_state)
    trainer = joint.make_trainer(model, teacher_state, args)
    snapshot = joint.weights(trainer)
    common, audits, panels, episodes = None, {}, {}, 0
    max_noise_error, max_peak = 0., 0.

    def episode(panel, delta):
        nonlocal common, episodes, max_noise_error, max_peak
        row, audit = intervention_episode(trainer, args=args, query=query, panel=panel,
            action_delta=delta, period=period, predictor=predictor, envelope=envelope)
        if common is None:
            common = audit
        else:
            for key in ("state", "query_state", "prefix_rewards", "measurements"):
                np.testing.assert_array_equal(audit[key], common[key])
            np.testing.assert_array_equal(audit["commands"][:query["start"]], common["commands"][:query["start"]])
        if panel in audits:
            error = float(np.abs(audit["innovations"] - audits[panel]["innovations"]).max())
            max_noise_error = max(max_noise_error, error)
            np.testing.assert_allclose(audit["innovations"], audits[panel]["innovations"], atol=3e-5, rtol=0)
        else:
            audits[panel] = audit
        np.testing.assert_allclose(row["episode_return"], row["prefix_return"] + row["suffix_return"], atol=1e-9, rtol=0)
        np.testing.assert_allclose(row["suffix_return"], row["option_return"] + row["tail_return"], atol=1e-9, rtol=0)
        if row["reference_correction_peak"] > joint.spec.REFERENCE_LIMIT + 1e-8:
            raise ValueError("Stage123 changed the registered reference authority")
        max_peak = max(max_peak, row["reference_correction_peak"])
        episodes += 1
        return row

    for panel in spec.PANELS:
        panels[panel] = {}
        for variant in spec.VARIANTS:
            delta = np.zeros(spec.ACTION_DIM, dtype=np.float32)
            if variant != "zero":
                axis, sign = variant.split("_")
                delta[int(axis[4:])] = spec.EPSILON if sign == "plus" else -spec.EPSILON
            panels[panel][variant] = episode(panel, delta)
    native = gradients(panels)
    crossfit = {}
    for fit, test in (("A", "B"), ("B", "A")):
        norm = np.linalg.norm(native[fit])
        delta = spec.EPSILON * native[fit] / norm if norm > 0 else np.zeros(spec.ACTION_DIM)
        plus, minus = episode(test, delta), episode(test, -delta)
        zero = panels[test]["zero"]["suffix_return"]
        crossfit[f"{fit}_to_{test}"] = {"plus_minus": plus["suffix_return"] - minus["suffix_return"],
            "plus_zero": plus["suffix_return"] - zero, "minus_zero": minus["suffix_return"] - zero}
    torch.testing.assert_close(joint.weights(trainer), snapshot, atol=0, rtol=0)
    return {"query": query, "state": common["state"], "gradients": {k: v.tolist() for k, v in native.items()},
        "native_gradient_cosine": cosine(native["A"], native["B"]), "crossfit": crossfit,
        "coordinate_suffix_returns": {p: {v: r["suffix_return"] for v, r in rows.items()} for p, rows in panels.items()},
        "prefix_suffix_noise_and_authority_checks": "passed", "policy_freeze": "passed",
        "max_innovation_error": max_noise_error, "max_reference_peak": max_peak,
        "native_episodes": episodes, "native_steps": episodes * args.horizon,
        "native_donor_response_calls": 2 * episodes * args.horizon,
        "native_upper_calls": episodes * (args.horizon // period)}


def pullback(trainer, results):
    states = torch.as_tensor(np.asarray([r["state"] for r in results], dtype=np.float32))
    actor = trainer.upper_actor
    parameters = [p for p in actor.parameters() if p.requires_grad]
    gradients = {}
    for panel in spec.PANELS:
        signal = torch.as_tensor(np.asarray([r["gradients"][panel] for r in results], dtype=np.float32))
        loss = -(actor.distribution(states).mean * signal).sum() / len(results)
        values = torch.autograd.grad(loss, parameters)
        gradients[panel] = np.concatenate([v.detach().double().numpy().ravel() for v in values])
    native = {p: np.asarray([r["gradients"][p] for r in results]).ravel() for p in spec.PANELS}
    return {"native_gradient_cosine": cosine(native["A"], native["B"]),
        "actor_pullback_cosine": cosine(gradients["A"], gradients["B"]),
        "native_gradient_rms": {p: float(np.sqrt(np.square(g).mean())) for p, g in native.items()},
        "crossfit_plus_zero": float(np.mean([v["plus_zero"] for r in results for v in r["crossfit"].values()])),
        "crossfit_plus_minus": float(np.mean([v["plus_minus"] for r in results for v in r["crossfit"].values()]))}


def run(root, output):
    if root not in spec.ROOTS:
        raise ValueError("Stage123 keeps the first two registered roots")
    args = spec.arguments(root)
    models, predictor, _, calibrations = joint.source.load_source(root)
    groups, cost = {}, dict.fromkeys(spec.budget(), 0)
    roster = spec.queries(root)
    with ProcessPoolExecutor(max_workers=spec.WORKERS, mp_context=mp.get_context("spawn"),
            initializer=joint.source.native.init_worker, initargs=(models["50"].config, args)) as pool:
        for period in spec.PERIODS:
            model = models[str(period)]
            before = copy.deepcopy(model.state_dict())
            teacher = joint.base.load_lower_state(root, period, protocol=spec.source)
            jobs = [(joint.weights(model), teacher, q, period, predictor, calibrations[str(period)]["envelope"]) for q in roster]
            results = list(pool.map(worker_query, jobs))
            if [r["query"] for r in results] != roster:
                raise ValueError("Stage123 native query roster changed")
            trainer = joint.make_trainer(model, teacher, args)
            snapshot = joint.weights(trainer)
            summary = pullback(trainer, results)
            torch.testing.assert_close(joint.weights(trainer), snapshot, atol=0, rtol=0)
            joint.source.native.curves.support.assert_frozen(model, before)
            for result in results:
                cost["queries"] += 1
                for key in ("native_episodes", "native_steps", "native_donor_response_calls", "native_upper_calls"):
                    cost[key] += result[key]
            cost["actor_pullback_forward_batches"] += 2
            cost["actor_pullback_backward_batches"] += 2
            groups[str(period)] = {"summary": summary, "queries": [{k: v for k, v in r.items() if k != "state"} for r in results]}
            print(f"root={root} period={period}: native reference credit complete", flush=True)
    if cost != spec.budget():
        raise ValueError("Stage123 measured native budget changed")
    result = {"status": "complete", "protocol": spec.PROTOCOL, "root": root,
        "kind": "native_reference_counterfactual_credit_diagnosis_not_learned_policy_validation",
        "source_protocol": spec.source.EXPERIMENT_PROTOCOL, "epsilon": spec.EPSILON,
        "cost": cost, "groups": groups, "teacher_upper_lower_and_source_frozen": "passed"}
    write_json(output, result)
    write_json(output.parent / "completion" / "ready.json", {"status": "complete", "protocol": spec.PROTOCOL})
    print("Eval complete: native reference counterfactual probe written", flush=True)
    return result
