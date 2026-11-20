"""V504 frozen runtime entry and paired result aggregation."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_text())
PRIMARY = ("restricted_journey_s_per_pax", "raw_cv2", "readiness_delay_s_per_trip",
           "public_abs_deviation_s_per_trip")
METRICS = PRIMARY + (
    "restricted_wait_s_per_pax", "restricted_in_vehicle_s_per_pax",
    "public_late_s_per_trip", "public_early_s_per_trip", "hold_s_per_trip",
    "readiness_delay_p95_s", "passenger_unserved_rate",
)


def read(p):
    return json.loads(p.read_text())


def check(runtime, old_run):
    from frequency.demand_frequency import DemandFrequencyTracker
    from matched_controls import dispatch_metrics, raw_lower, raw_upper, rule_plan

    prior = read(HERE / "v503_plan.json")
    assert PLAN["seeds"] == prior["training_seeds"] and PLAN["episodes"] == prior["episodes"]
    assert [PLAN["smoke_day"]] == prior["smoke"]["days"]
    assert PLAN["smoke_seed"] == prior["smoke"]["seed"]
    assert PLAN["smoke_episodes"] == prior["smoke"]["episodes"]
    used = set(prior["evaluation_days"]) | set(prior["smoke"]["days"])
    used |= {runtime.training_day(s, ep) for s, n in [
        (PLAN["smoke_seed"], PLAN["smoke_episodes"]),
        *((s, PLAN["episodes"]) for s in PLAN["seeds"]),
    ] for ep in range(1, n + 1)}
    assert len(set(PLAN["days"])) == 16 and used.isdisjoint(PLAN["days"])
    for phase, seed, ep in [
        ("smoke", PLAN["smoke_seed"], PLAN["smoke_episodes"]),
        *(("train", s, PLAN["episodes"]) for s in PLAN["seeds"]),
    ]:
        for layer in ("upper", "lower"):
            assert (old_run / phase / f"aligned_{seed}/frozen_models/{layer}_ep{ep}.pt").is_file()

    key = (1, True)
    tracker = DemandFrequencyTracker(update_interval_s=20, bin_sec=60, method="harmonic")
    stations = [SimpleNamespace(station_id=i, station_type=1, od={}, direction=d)
                for d in (True, False) for i in (1, 2, 3)]
    feed = runtime.CausalHeadwayAPCRateFeed(SimpleNamespace(stations=stations,
                                                          frequency_tracker=tracker))
    initial = feed.minute.station(*key)
    for rate, close in [(1, False), (2, False), (6, True)]:
        feed.rate_per_min[key] = rate
        tracker.update({})
        assert feed.minute.station(*key)[0] == (3 if close else initial[0])
    assert tracker.total_updates == 1 and feed.minute.station(*key)[1] == 3 - initial[0]
    assert len(raw_upper(feed.minute, 9330)) == 8 and len(raw_lower(feed.minute, 1, [2, 3], True)) == 4
    actions = runtime.build_upper_action_candidates()
    assert actions.shape == (15, 3) and np.all(actions[:, 2] == 0.5)
    plan = rule_plan(actions, {True: 360, False: 360}, {True: 2, False: 2},
                     250, feed.minute, 9330)
    assert any(np.array_equal(plan, a) for a in actions)
    assert runtime.LOWER_ACTION_BINS == [0, 5, 10, 15, 20, 25]
    trip = SimpleNamespace(launched=True, _actual_launch_time=75, _freqduet_base_launch=100)
    late = SimpleNamespace(launched=True, _actual_launch_time=115, _freqduet_base_launch=100)
    assert dispatch_metrics([trip, late]) == dict(public_abs_deviation_s_per_trip=20,
                                                  public_late_s_per_trip=7.5,
                                                  public_early_s_per_trip=12.5)
    return {"passed": True}


def unpack(root, phase):
    smoke = phase == "smoke"
    days = [PLAN["smoke_day"]] if smoke else PLAN["days"]
    seeds = [PLAN["smoke_seed"]] if smoke else PLAN["seeds"]
    episodes = PLAN["smoke_episodes"] if smoke else PLAN["episodes"]

    def rows(arm, seed):
        name = "manual" if arm == "manual" else f"{arm}_{seed}"
        item = read(root / name / "results.json")
        assert item["arm"] == arm and item["seed"] == seed
        if arm == "nofreq":
            assert item["updates"] == [(episodes - 2) * 32, (episodes - 2) * 64]
        out = item["per_day"]
        assert [r["day"] for r in out] == days
        for r in out:
            m = r["metrics"]
            assert m["physical_buses"] == 14
            assert m["trips_launched"] == m["trips_completed"] == 262
            assert np.isfinite([m[k] for k in METRICS]).all()
            assert np.isclose(m[PRIMARY[3]], m["public_late_s_per_trip"] + m["public_early_s_per_trip"])
        return out

    arms = {a: [rows(a, s) for s in seeds] for a in ("aligned", "nofreq")}
    arms["manual"] = [rows("manual", None)] * len(seeds)
    for group in arms.values():
        for a in group:
            for ref, row in zip(arms["aligned"][0], a):
                assert (ref["scenario_tape_id"], ref["metrics"]["passengers_generated"]) == (
                    row["scenario_tape_id"], row["metrics"]["passengers_generated"])
    return arms, seeds, days


def aggregate(root, smoke=False, old_smoke=None):
    arms, seeds, days = unpack(root, "smoke" if smoke else "formal")
    if smoke:
        assert old_smoke is not None
        old = read(old_smoke)["checkpoint_evals"][f"ep{PLAN['smoke_episodes']}"]["per_day"][0]
        current = arms["aligned"][0][0]
        assert old["scenario_tape_id"] == current["scenario_tape_id"]
        assert all(current["metrics"][k] == v for k, v in old["metrics"].items())
        return {"passed": True, "paths": 6, "old_aligned_metrics_exact": True}

    arrays = {a: np.array([[[r["metrics"][k] for k in METRICS] for r in group]
                           for group in groups]) for a, groups in arms.items()}
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    n = PLAN["bootstrap_draws"]
    ii = rng.integers(len(seeds), size=(n, len(seeds), 1))
    jj = rng.integers(len(days), size=(n, 1, len(days)))
    effects = {}
    for arm in ("nofreq", "manual"):
        delta = arrays["aligned"] - arrays[arm]
        ci = np.quantile(delta[ii, jj].mean(axis=(1, 2)), [0.025, 0.975], axis=0).T
        effects[f"aligned_minus_{arm}"] = {
            "joint_pass": bool(ci[0, 1] < 0 and (ci[1:len(PRIMARY), 1] <= 0).all()),
            "metrics": {
                k: dict(mean=float(delta[:, :, i].mean()), ci95=ci[i].tolist(),
                        by_seed=dict(zip(seeds, delta[:, :, i].mean(axis=1).tolist())),
                        by_day=dict(zip(days, delta[:, :, i].mean(axis=0).tolist())))
                for i, k in enumerate(METRICS)
            }
        }
    return {"seeds": seeds, "days": days, "comparisons": effects}


def run(runtime, args):
    phase, arm, seed = args.phase, args.arm, args.seed
    if phase == "check":
        runtime.write_json(args.output / "summary/results.json", check(runtime, args.old_run))
        return
    smoke = phase == "smoke"
    days = [PLAN["smoke_day"]] if smoke else PLAN["days"]
    episodes = PLAN["smoke_episodes"] if smoke else PLAN["episodes"]
    assert phase in ("smoke", "formal") and arm in ("aligned", "manual", "nofreq")
    assert (seed is None if arm == "manual" else
            seed in ([PLAN["smoke_seed"]] if smoke else PLAN["seeds"]))
    args.output.mkdir(parents=True, exist_ok=False)
    name = PLAN["corridor"]
    corridor = HERE / "corridors" / name
    result = {"arm": arm, "seed": seed}
    if arm == "nofreq":
        runtime.train_seed(corridor_dir=corridor, corridor_=name, seed=seed, episodes=episodes,
                           val_days=days, band_mode=arm, ablation="none", ckpt_dir=args.output / "ckpt",
                           result_dir=args.output / "train_record", resume_path=None)
        trained = read(args.output / "train_record/results.json")
        assert trained["completed"] and trained["val_days"] == days
        result["updates"] = [trained["upper_updates"], trained["lower_updates"]]
        result["per_day"] = trained["checkpoint_evals"][f"ep{episodes}"]["per_day"]
    else:
        upper = lower = None
        if arm == "aligned":
            models = args.old_run / ("smoke" if smoke else "train") / f"aligned_{seed}/frozen_models"
            upper, lower, _ = runtime.make_trainers(seed)
            upper.load(models / f"upper_ep{episodes}.pt")
            lower.load(models / f"lower_ep{episodes}.pt")
        meta = read(corridor / "corridor_meta.json")
        result["per_day"] = []
        for i, day in enumerate(days, start=1):
            result["per_day"].append(runtime.run_b1_episode(
                corridor_dir=corridor, corridor_meta=meta, day_seed=day,
                upper_trainer=upper, lower_trainer=lower, lower_replay=None,
                eval_mode=True, band_mode=arm))
            print("SCHEDULEURM_PROGRESS " + json.dumps(
                {"current": i, "total": len(days), "unit": "day"}), flush=True)
    runtime.write_json(args.output / "summary/results.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("check", "smoke", "formal", "aggregate", "check-smoke"))
    p.add_argument("--arm")
    p.add_argument("--seed", type=int)
    p.add_argument("--old-run", type=Path)
    p.add_argument("--old-smoke", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.phase in ("aggregate", "check-smoke"):
        print(json.dumps(aggregate(args.output, args.phase == "check-smoke", args.old_smoke), indent=2))
        return
    sys.path.insert(0, str(HERE))
    import freqduet_v7_b1 as runtime
    run(runtime, args)


if __name__ == "__main__":
    main()
