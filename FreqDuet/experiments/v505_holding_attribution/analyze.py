"""Aggregate the frozen holding comparisons and selected case summaries."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
V504 = HERE.parent / "v504_corrected_comparators"
sys.path.insert(0, str(V504))
from study import METRICS, PLAN, PRIMARY, read, unpack


def comparisons(arms, seeds, days, controls):
    rows = {k: np.array([[[r["metrics"][m] for m in METRICS] for r in g] for g in v])
            for k, v in arms.items()}
    n = PLAN["bootstrap_draws"]
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    i = rng.integers(len(seeds), size=(n, len(seeds), 1))
    j = rng.integers(len(days), size=(n, 1, len(days)))
    out = {}
    for k in controls:
        delta = rows["aligned"] - rows[k]
        bounds = np.quantile(delta[i, j].mean(axis=(1, 2)), [.025, .975], axis=0).T
        out[k] = {
            "joint_pass": bool(bounds[0, 1] < 0 and (bounds[1:len(PRIMARY), 1] <= 0).all()),
            "metrics": {
                m: dict(mean=float(delta[:, :, q].mean()), ci95=bounds[q].tolist(),
                        by_seed=dict(zip(seeds, delta[:, :, q].mean(axis=1).tolist())),
                        by_day=dict(zip(days, delta[:, :, q].mean(axis=0).tolist())))
                for q, m in enumerate(METRICS)},
        }
    return out


def case(row):
    tr = row["trace"]
    dispatched = {r[0]: r for r in tr["dispatch"]}
    runtimes = defaultdict(list)
    for r in dispatched.values():
        if r[6] in dispatched and r[7] is not None:
            past = dispatched[r[6]]
            assert past[2] == r[2] and past[1] != r[1] and past[3] <= r[7] <= r[3]
            runtimes[str(past[1])].append([r[7] - past[3], r[3] - r[7]])
    blocked = [(r[2], dispatched[r[0]][3], r[1], r[0]) for r in tr["denials"]]
    pending = [list(s) for s in sorted(blocked)]
    return {
        "metrics": row["metrics"], "budget_counts": row["budget_counts"],
        "first_denial": min(blocked), "last_recovery": max(r[1] for r in blocked),
        "denied_trips_by_terminal": {str(d): sum(r[2] == d for r in blocked) for d in (0, 1)},
        "blocking_intervals": pending,
        "matched_previous_trip_times": {
            d: {"n": len(a), "trip_elapsed_s": np.quantile(np.array(a)[:, 0], [.1, .5, .9]).tolist(),
                "realized_terminal_idle_s": np.quantile(np.array(a)[:, 1], [.1, .5, .9]).tolist()}
            for d, a in sorted(runtimes.items())},
    }


def table():
    arms, seeds, days = unpack(V504 / "confirmation", "formal")
    public = np.array([[r["metrics"]["passengers_generated"] for r in g] for g in arms["aligned"]])
    metric = lambda series, name: np.array([[r["metrics"][name] for r in g] for g in series])
    delta = metric(arms["manual"], METRICS[0]) - metric(arms["aligned"], METRICS[0])
    out = {"journey_savings_vs_manual_person_hours_per_day": float((public * delta).mean() / 3600)}
    out["v504_means"] = {k: {m: float(metric(v, m).mean()) for m in METRICS}
                         for k, v in arms.items()}
    out["v504_exploratory_nofreq_minus_manual"] = comparisons(
        {"aligned": arms["nofreq"], "manual": arms["manual"]}, seeds, days, ["manual"])["manual"]
    root = HERE / "confirmation"
    for arm in ("trace_aligned", "trace_nofreq"):
        path = root / f"{arm}_50314_0/results.json"
        if path.exists():
            r, = read(path)["per_day"]
            out.setdefault("case", {})[arm] = case(r)
    for name in ("field", "field_vehicle", "field_crosswalk"):
        f = root / f"{name}.json"
        if f.exists():
            out[name] = read(f)
    raw = [read(p) for p in sorted(root.glob("formal/*/results.json"))]
    if not raw:
        return out
    assert len(raw) == 26
    cases = defaultdict(dict)
    for item in raw:
        key = (item["arm"], item["seed"])
        for r in item["per_day"]:
            assert r["day"] not in cases[key]
            cases[key][r["day"]] = r
    for a in ("zero", "rule", "rolling", "audit_aligned"):
        group = []
        for s, origin in zip(seeds, arms["aligned"]):
            key = a, (None if a == "rolling" else s)
            assert set(cases[key]) == set(days)
            rows = [cases[key][d] for d in days]
            for r, ref in zip(rows, origin):
                m = r["metrics"]
                assert r["scenario_tape_id"] == ref["scenario_tape_id"]
                assert m["passengers_generated"] == ref["metrics"]["passengers_generated"]
                assert m["trips_launched"] == m["trips_completed"] == 262 and m["physical_buses"] == 14
                assert np.isfinite([m[k] for k in METRICS]).all()
                if a == "zero":
                    assert m["hold_s_per_trip"] == 0
                if a == "audit_aligned":
                    assert m == ref["metrics"], (s, r["day"])
            group.append(rows)
        arms[a] = group
    out["v505"] = comparisons(arms, seeds, days, ["zero", "rule", "rolling"])
    out["v505_means"] = {k: {m: float(metric(v, m).mean()) for m in METRICS}
                         for k, v in arms.items() if k in out["v505"]}
    for a in (*out["v505"], "audit_aligned"):
        groups = arms[a][:1] if a == "rolling" else arms[a]
        rows = [r["budget_counts"] for g in groups for r in g]
        out.setdefault("v505_budget_counts", {})[a] = {
            k: sum(r.get(k, 0) for r in rows) for k in ("queries", "mask_removes_bin", "only_zero",
                                                       "preferred_masked", "nonzero")}
    return out


if __name__ == "__main__":
    print(json.dumps(table(), indent=2))
