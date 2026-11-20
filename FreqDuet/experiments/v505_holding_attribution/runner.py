"""Paired V504 holding intervention, matched rolling baseline and case trace."""

import argparse
from collections import Counter
from copy import copy
import heapq
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import freqduet_v7_b1 as rt
from matched_controls import desired_headways, rule_hold

PLAN = json.loads((HERE / "v504_plan.json").read_text())


def score_schedule(plans, stock, cycle, rates, last, now):
    supply = {d: sorted(s) for d, s in stock.items()}
    prev = dict(last)
    pos = dict.fromkeys(plans, 0)
    total = 0.0
    while any(pos[d] < len(plans[d]) for d in plans):
        candidates = []
        for d in plans:
            if pos[d] < len(plans[d]) and supply[d]:
                plan = plans[d][pos[d]]
                eligible = max(plan, prev[d] + 120)
                candidates.append((max(eligible, supply[d][0]), d, eligible))
        if not candidates:
            return float("inf")
        t, d, eligible = min(candidates)
        heapq.heappop(supply[d])
        heapq.heappush(supply[not d], t + cycle[d])
        if prev[d] > -9000:
            total += rates[d] * (t - prev[d]) ** 2 / 2
        total += 360 * rates[d] * (t - eligible)
        prev[d] = t
        pos[d] += 1
    return total


def rolling_action(env, now, ledger, feed):
    available = {d: rt.remaining_quota_info(env, now, d)[0] for d in (True, False)}
    ready = {d: env._ready_vehicle_count(d) for d in available}
    floor = ledger.cycle_headway_floor_s(len(env.bus_all), env)
    cycle = {d: ledger.median_nonhold_s(d) + 1 for d in available}
    stock = {d: [] for d in available}
    for bus in env.bus_all:
        d = bool(bus.direction)
        if bus.on_route:
            progress = min(1.0, max(0.0, bus.travel_distance / ledger.corridor_len_m))
            t = now + (1 - progress) * cycle[d] + max(0.0, bus.dwelling_time or 0) + max(0.0, bus.holding_time or 0)
            stock[not d].append(t)
        else:
            stock[d].append(max(now, float(bus.back_to_terminal_time or 0) + 1))
    rates = {d: max(sum(v for v, _ in feed.minute.directional(d)), 0.1) / 60
             for d in available}
    last = {d: float(env._last_dispatch_time[d]) for d in available}
    times = env.timetables

    def score(action):
        shadow = SimpleNamespace(timetables=[copy(t) for t in times], protocol=env.protocol)
        rt.apply_global_quota_timetable(shadow, now,
                                      desired_headways(action, available, ready, floor),
                                      h_bounds=(floor, 455),
                                      tail_bounds=(max(245, floor - 10), 445))
        plans = {d: sorted(max(now, float(t._freqduet_scheduled_launch))
                           for t in shadow.timetables if not t.launched and bool(t.direction) == d)[:8]
                 for d in available}
        return (score_schedule(plans, stock, cycle, rates, last, now),
                abs(float(action[1])), abs(float(action[0]) - 1))

    return min(rt.build_upper_action_candidates(), key=score)


class Observer:
    def __init__(self, trace):
        self.trace = trace
        self.stock = []
        self.releases = []
        self.denials = {}
        self.queries = Counter()
        self.trip_starts = {}

    def begin(self):
        self.stock.clear()
        self.releases.clear()
        self.denials.clear()
        self.queries.clear()
        self.trip_starts.clear()

    def environment(self, base):
        observed = self

        class ObservedEnv(base):
            def step(self, action, *args, **kwargs):
                if observed.trace and self.current_time % 60 == 0:
                    observed.stock.append([float(self.current_time), self._ready_vehicle_count(True),
                                           self._ready_vehicle_count(False)])
                return super().step(action, *args, **kwargs)

            def launch_bus(self, trip, actual_launch=None):
                old = {} if not observed.trace else {
                    b.bus_id: (b.trip_id, b.back_to_terminal_time) for b in self.bus_in_terminal}
                bus = super().launch_bus(trip, actual_launch)
                if observed.trace:
                    key = int(trip.launch_turn)
                    ready = [self._ready_vehicle_count(d) for d in (True, False)]
                    direction = int(bool(trip.direction))
                    if bus is None and key not in observed.denials:
                        observed.denials[key] = [key, direction, float(self.current_time), *ready]
                    if bus is not None:
                        previous, back = old[bus.bus_id]
                        observed.releases.append([
                            key, direction, int(bus.bus_id), float(self.current_time),
                            float(trip._freqduet_first_eligible_time), float(trip._freqduet_scheduled_launch),
                            int(previous) if previous >= 0 else None,
                            float(back) if np.isfinite(back) else None,
                            *ready,
                        ])
                return bus

        return ObservedEnv


def load_eval(arm, seed, old_run):
    if arm == "rolling":
        return None, None, "nofreq"
    base_arm = "nofreq" if arm == "trace_nofreq" else "aligned"
    if base_arm == "aligned":
        models = old_run / "v503/train" / f"aligned_{seed}/frozen_models"
    else:
        models = old_run / "v504/formal" / f"nofreq_{seed}/frozen_models"
    upper, lower, _ = rt.make_trainers(seed)
    upper.load(models / "upper_ep200.pt")
    if arm.startswith("trace_") or arm == "audit_aligned":
        lower.load(models / "lower_ep200.pt")
    return upper, lower, base_arm


def run(arm, seed, days, old_run):
    upper, lower, band = load_eval(arm, seed, old_run)
    trace = arm.startswith("trace_")
    observer = Observer(trace)
    rt.env_bus = observer.environment(rt.env_bus)
    new_upper, new_lower = rt.build_upper_state, rt.build_lower_state
    inputs = {}

    def upper_state(env, now, n_fleet, ledger, feed, *args):
        if arm == "rolling":
            inputs["a"] = rolling_action(env, now, ledger, feed)
        return new_upper(env, now, n_fleet, ledger, feed, *args)

    def lower_state(env, bus, now, length, ledger, feed, *args):
        vec, ctx = new_lower(env, bus, now, length, ledger, feed, *args)
        limit = ledger.remaining_budget_s[(int(bus.bus_id), int(bus.trip_id))]
        inputs["lower"] = (bus, ctx, feed, limit)
        return vec, ctx

    rt.build_upper_state, rt.build_lower_state = upper_state, lower_state
    bins = rt.LOWER_ACTION_BINS
    original_lower = lower.policy_net if trace or arm == "audit_aligned" else None

    def hold(vec, deterministic):
        bus, ctx, feed, limit = inputs["lower"]
        c = observer.queries
        c["queries"] += 1
        c["mask_removes_bin"] += limit < max(bins) - 1e-6
        c["only_zero"] += limit < min(v for v in bins if v > 0) - 1e-6
        if arm == "zero":
            want = used = 0.0
        elif original_lower is not None:
            used = float(original_lower.get_action(vec, deterministic=True))
            x = rt.torch.from_numpy(vec).float().unsqueeze(0)
            with rt.torch.no_grad():
                want = float(original_lower.action_bins[original_lower(x).argmax(dim=-1)].item())
        else:
            want = rule_hold(bus, ctx, feed.minute, bins, max(bins), rt.compute_lower_reward_and_cost)
            used = (rule_hold(bus, ctx, feed.minute, bins, limit, rt.compute_lower_reward_and_cost)
                    if want > limit + 1e-6 else want)
        c["preferred_masked"] += want > limit + 1e-6
        c["nonzero"] += used > 0
        return used

    def trainer(fn):
        return SimpleNamespace(policy_net=SimpleNamespace(get_action=fn))

    if arm == "rolling":
        upper = trainer(lambda _s, deterministic: inputs["a"])
    lower = trainer(hold)
    corridor = HERE / "corridors" / PLAN["corridor"]
    meta = json.loads((corridor / "corridor_meta.json").read_text())
    rows = []
    for i, day in enumerate(days, 1):
        observer.begin()
        row = rt.run_b1_episode(corridor_dir=corridor, corridor_meta=meta, day_seed=day,
                               upper_trainer=upper, lower_trainer=lower, lower_replay=None,
                               eval_mode=True, band_mode=band)
        row["budget_counts"] = {k: int(v) for k, v in observer.queries.items()}
        if trace:
            row["trace"] = {"stock": observer.stock, "dispatch": observer.releases,
                            "denials": list(observer.denials.values())}
        rows.append(row)
        print("SCHEDULEURM_PROGRESS " + json.dumps({"current": i, "total": len(days), "unit": "day"}),
              flush=True)
    return {"arm": arm, "seed": seed, "per_day": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=("zero", "rule", "rolling", "trace_aligned", "trace_nofreq",
                                     "audit_aligned"), required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--days", type=int, nargs="+", required=True)
    p.add_argument("--old-run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = run(a.arm, a.seed, a.days, a.old_run)
    rt.write_json(a.output / "summary/results.json", result)
