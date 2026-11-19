"""Frozen V503 driver and small-result checks."""

import argparse
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "plan.json").read_text())
ENDPOINTS = ("restricted_journey_s_per_pax", "raw_cv2",
             "readiness_delay_s_per_trip", "hold_s_per_trip")


def read(path):
    return json.loads(path.read_text())


def check(runtime, corridors):
    from frequency.demand_frequency import DemandFrequencyTracker

    key = 1, True
    tracker = DemandFrequencyTracker(update_interval_s=20, bin_sec=60, method="harmonic")
    stations = [
        SimpleNamespace(station_type=1, od={}, station_id=i, direction=direction)
        for direction in (True, False) for i in (1, 2, 3)
    ]
    env = SimpleNamespace(frequency_tracker=tracker, stations=stations)
    feed = runtime.CausalHeadwayAPCRateFeed(env)
    for rate in (1.0, 2.0, 6.0):
        feed.rate_per_min[key] = rate
        tracker.update({})
    assert tracker.total_updates == 1 and tracker._get_local_state(key).fast == 3.0
    for _ in range(3):
        tracker.update({})
    assert tracker.total_updates == 2 and tracker._get_local_state(key).fast == 6.0

    for station in stations:
        st = tracker._get_local_state((station.station_id, station.direction))
        st.low, st.prev_low = 1.0, 0.5
        st.high, st.prev_high = -0.6, -0.4
    cells = {}
    for arm in PLAN["arms"]:
        ub = runtime.band_for_head(arm, upper=True)
        lb = runtime.band_for_head(arm, upper=False)
        cells[arm] = (ub, lb)
        upper = runtime.upper_band(env, ub, 9_330)
        lower = runtime.lower_band(tracker, 1, [2, 3], True, lb)
        assert len(upper) == 8 and len(lower) == 4
        assert np.sign(upper[0]) == (1 if arm == "aligned" else -1)
        assert np.sign(lower[0]) == (-1 if arm == "aligned" else 1)
    assert cells["aligned"] == ("low", "high") and cells["swapped"] == ("high", "low")
    assert "last_left_behind_count" not in runtime.build_lower_state.__code__.co_names
    assert runtime.LOWER_STATE_DIM == 19 and runtime.UPPER_STATE_DIM == 21
    actions = runtime.build_upper_action_candidates()
    assert actions.shape == (15, 3) and np.all(actions[:, 2] == 0.50)
    used_train = {
        runtime.training_day(s, ep) for s, episodes in (
            (PLAN["smoke"]["seed"], PLAN["smoke"]["episodes"]),
            *((s, PLAN["episodes"]) for s in PLAN["training_seeds"]),
        ) for ep in range(1, episodes + 1)
    }
    used_eval = set(PLAN["evaluation_days"])
    smoke_eval = set(PLAN["smoke"]["days"])
    assert len(used_train) == 803 and len(used_eval) == 16
    assert not (used_train & used_eval or used_train & smoke_eval or used_eval & smoke_eval)
    config = read(corridors / PLAN["corridor"] / "config.json")
    assert config["passenger_state_update_freq"] == 20
    print("V503 preflight: APC units, both band mappings, mask layout, and day splits verified", flush=True)


def pair(root, seed, episodes, days):
    matrix = []
    for arm in PLAN["arms"]:
        a = read(root / f"{arm}_{seed}/results.json")
        assert a["completed"] and a["band_mode"] == arm and a["seed"] == seed
        assert a["episodes"] == episodes and a["val_days"] == days
        assert a["upper_updates"] == (episodes - 2) * 32, (arm, seed)
        assert a["lower_updates"] == (episodes - 2) * 64, (arm, seed)
        rows = a["checkpoint_evals"][f"ep{episodes}"]["per_day"]
        assert [r["day"] for r in rows] == days
        assert all(r["metrics"]["physical_buses"] == 14 and
                   r["metrics"]["trips_launched"] == r["metrics"]["trips_completed"] == 262
                   for r in rows)
        matrix.append(rows)
    for a, b in zip(*matrix):
        assert a["scenario_tape_id"] == b["scenario_tape_id"]
        assert a["metrics"]["passengers_generated"] == b["metrics"]["passengers_generated"]
    return np.asarray([[a["metrics"][k] - b["metrics"][k] for k in ENDPOINTS]
                       for a, b in zip(*matrix)])


def aggregate(root, smoke):
    if smoke:
        p = PLAN["smoke"]
        pair(root, p["seed"], p["episodes"], p["days"])
        return {"complete": True, "smoke_only": True, "launches": 8}
    seeds, days = PLAN["training_seeds"], PLAN["evaluation_days"]
    delta = np.stack([pair(root, s, PLAN["episodes"], days) for s in seeds])
    rng = np.random.default_rng(PLAN["bootstrap"]["seed"])
    n = PLAN["bootstrap"]["draws"]
    ds = rng.integers(len(seeds), size=(n, len(seeds), 1))
    dd = rng.integers(len(days), size=(n, 1, len(days)))
    draws = delta[ds, dd].mean(axis=(1, 2))
    intervals = np.quantile(draws, [0.025, 0.975], axis=0).T
    effects = {
        k: {"mean": float(delta[:, :, i].mean()), "ci95": intervals[i].tolist(),
            "by_seed": dict(zip(seeds, delta[:, :, i].mean(axis=1).tolist())),
            "by_day": dict(zip(days, delta[:, :, i].mean(axis=0).tolist()))}
        for i, k in enumerate(ENDPOINTS)
    }
    return {"complete": True, "seeds": seeds, "days": days,
            "contrast": "aligned_minus_swapped", "effects": effects,
            "passed": bool(intervals[0, 1] < 0 and (intervals[1:3, 1] <= 0).all())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("check", "smoke", "train", "check-smoke", "aggregate"))
    p.add_argument("--arm", choices=PLAN["arms"])
    p.add_argument("--seed", type=int)
    p.add_argument("--corridors", type=Path, default=HERE / "corridors")
    p.add_argument("--runtime", type=Path, default=HERE)
    p.add_argument("--output", type=Path)
    p.add_argument("--reviews", type=Path)
    args = p.parse_args()
    if args.phase in ("check-smoke", "aggregate"):
        print(json.dumps(aggregate(args.reviews, args.phase == "check-smoke"), indent=2))
        return
    sys.path.insert(0, str(args.runtime))
    import freqduet_v7_b1 as runtime

    if args.phase == "check":
        check(runtime, args.corridors)
        return
    if args.phase == "smoke":
        seed, episodes, days = (PLAN["smoke"][k] for k in ("seed", "episodes", "days"))
    else:
        seed, episodes, days = args.seed, PLAN["episodes"], PLAN["evaluation_days"]
        if seed not in PLAN["training_seeds"]:
            p.error("seed is outside the frozen matrix")
    if args.output is None or args.arm is None:
        p.error("run needs an output directory and arm")
    if (args.output / "summary").exists() or (args.output / "ckpt").exists():
        p.error("run output already exists")
    runtime.train_seed(corridor_dir=args.corridors / PLAN["corridor"], corridor_=PLAN["corridor"],
                       seed=seed, episodes=episodes, val_days=days, band_mode=args.arm,
                       ablation="none", ckpt_dir=args.output / "ckpt",
                       result_dir=args.output / "summary", resume_path=None)


if __name__ == "__main__":
    main()
