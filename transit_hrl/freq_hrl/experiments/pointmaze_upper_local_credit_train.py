"""Train the Stage113 upper residual with decision-aligned local option credit."""
import numpy as np

from . import pointmaze_feasible_credit as statistics
from . import pointmaze_upper_residual_train as base
from scripts import pointmaze_upper_local_credit_train_stage114_spec as spec


def score_upper_local(actor, pair_groups, *, horizon, period, cost, protocol=spec):
    """Use each upper decision's option return instead of repeating one episode score."""
    gradients, states, score_costs, signal_rms = {}, [], {}, {}
    for name, pairs in pair_groups.items():
        episode_returns = np.asarray([[base.independent.exact_returns(batch, 1.)
            for batch in pair["lower_batches"]] for pair in pairs], dtype=np.float64)
        local_returns = np.asarray([[batch.reward for batch in pair["upper_batches"]]
            for pair in pairs], dtype=np.float64)
        for pair_returns, local_pair, pair in zip(episode_returns, local_returns, pairs):
            for values, local_values, row in zip(pair_returns, local_pair, pair["rows"]):
                np.testing.assert_allclose(values[0], row["episode_return"], atol=.002, rtol=0)
                np.testing.assert_allclose(np.sum(local_values), row["episode_return"], atol=.002, rtol=0)
                cost["objective_checks"] += 1; cost["mc_calls"] += 1
        signal = base.source.scenario.leave_other_out(local_returns).reshape(-1)
        upper = base.concat_level_batches(batch for pair in pairs for batch in pair["upper_batches"])
        scored, score_cost = base.lower_training.residual_actor_gradients(actor, upper, {"scenario": signal},
            clip_ratio=.2, chunk_size=protocol.CHUNK_SIZE)
        gradients[name] = scored["scenario"]
        states.append(upper.state)
        score_costs[name] = score_cost
        signal_rms[name] = float(np.sqrt(np.square(signal).mean()))
        cost["actor_score_forward_batches"] += score_cost["actor_score_forward_batches"]
        cost["actor_score_backward_batches"] += score_cost["actor_score_backward_batches"]
    return {"gradients": gradients, "states": np.concatenate(states), "score_costs": score_costs,
        "signal_rms": signal_rms}


def qualify(cell, *, preflight):
    root = cell["root"]
    if (cell["status"] != "complete" or cell["protocol"] != spec.EXPERIMENT_PROTOCOL
            or cell["contract"] != spec.contract() or root not in spec.roots(preflight=preflight)
            or cell["preflight"] != preflight or cell["seed_roles"] != spec.seed_roles(root, preflight=preflight)
            or cell["cost"] != spec.budget(preflight=preflight)
            or set(cell["groups"]) != {str(p) for p in spec.PERIODS}
            or any(cell.get(key, 0) for key in ("optimizer_steps", "critic_fits", "native_trace_writes"))):
        raise ValueError("Stage114 protocol, source, roster, budget or frozen path changed")
    horizon = spec.arguments(root, preflight=preflight).horizon
    for period, group in cell["groups"].items():
        if group["source_and_lower_unchanged"] != "passed":
            raise ValueError("Stage114 source or lower branch freeze failed")
        effects = base.paired_effects(int(period), group["evaluation"],
            cell["seed_roles"]["native_evaluation"], protocol=spec)
        if group["effects"] != effects or not np.isfinite(list(effects.values())).all():
            raise ValueError("Stage114 paired evaluation changed")
        if set(group["evaluation"]) != set(spec.ARMS):
            raise ValueError("Stage114 evaluation roster changed")
        for variant, rows in group["evaluation"].items():
            for row in rows:
                base.check_row(row, period=int(period), horizon=horizon, variant=variant)
                if row["seed"] not in cell["seed_roles"]["native_evaluation"]:
                    raise ValueError("Stage114 evaluation seed changed")
    return cell


def aggregate(cells, *, preflight):
    result = statistics.aggregate(cells, preflight=preflight, protocol=spec, qualifier=qualify)
    passed = {} if preflight else {str(period): all(
        result["endpoints"][f"{period}/{a}_minus_{b}"]["ci"][0] > 0 for a, b in spec.CONTRASTS)
        for period in spec.PERIODS}
    result.update(upper_gain_gate="mechanical_only" if preflight else
        "supported_both_periods" if all(passed.values()) else
        "partial" if any(passed.values()) else "not_supported",
        period_upper_gain_gate=passed,
        performance_claim="forecast_anchored_upper_residual_local_option_credit_gain",
        lower_source="Stage112 learned lower branch frozen")
    return result
