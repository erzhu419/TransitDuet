"""Stage and manually dispatch only the frozen V505 phases."""

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
V504 = HERE.parent / "v504_corrected_comparators"
sys.path.insert(0, str(V504))
import launch as previous
from study import PLAN

scheduler = previous.scheduler
BASE = previous.V504.with_name("FreqDuet-v505-holding-attribution")
SOURCE = BASE / "source01"
OLD = {"v503": previous.V503 / "run01", "v504": previous.RUN}
TAIL = 53020003


def stage(source=SOURCE):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as a:
        a.add(HERE / "runner.py", arcname="runner.py")
        a.add(V504 / "plan.json", arcname="v504_plan.json")
    payload = base64.b64encode(buf.getvalue()).decode()
    script = f"""
from pathlib import Path
import base64, io, tarfile, tempfile
dst = Path({str(source)!r})
old = Path({str(previous.SOURCE)!r})
dst.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=dst.parent) as tmp:
    work = Path(tmp) / "release"
    work.mkdir()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as ar:
        ar.extractall(work)
    for name in ("corridors", "freqduet_src", "freqduet_v7_b1.py", "matched_controls.py"):
        target = (old / name)
        assert target.exists(), target
        (work / name).symlink_to(target, target_is_directory=target.is_dir())
    for old_name, root in {dict((k, str(v)) for k, v in OLD.items())!r}.items():
        (work / old_name).symlink_to(root, target_is_directory=True)
    compile((work / "runner.py").read_text(), "runner.py", "exec")
    work.rename(dst)
print(dst)
"""
    rc, out, err = scheduler.run_on("node004", shlex.join([previous.PYTHON, "-B", "-c", script]),
                                    timeout=60, check=False)
    if rc:
        raise RuntimeError(err[-2000:])
    print(out, end="", flush=True)


def spec(phase, arm, seed, days, shard=0):
    label = f"{arm}_{seed or 0}_{shard}"
    out = BASE / "run01" / phase / label
    source = SOURCE if phase == "smoke" else BASE / "source02"
    cmd = [previous.PYTHON, "-B", "-I", "-u", str(source / "runner.py"),
           "--old-run", str(source), "--output", str(out), "--arm", arm,
           "--days", *map(str, days)]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    task = previous.task("formal", "manual")
    task.update(description=f"FreqDuet V505 {phase} {label}", signature=f"FreqDuet/v505_r01/{phase}/{label}",
                cwd=str(source), cmd=shlex.join(cmd), result_dir=str(out / "summary"),
                local_result_dir=str(HERE / "review" / phase / label))
    return task


def receipt(phase):
    return HERE / f"submission_{phase}.json"


def field_task():
    out = BASE / "run01/field/route111"
    path = BASE / "field_runtime.py"
    payload = base64.b64encode((HERE / "field_runtime.py").read_bytes()).decode()
    install = f"from pathlib import Path; import base64; Path({str(path)!r}).write_bytes(base64.b64decode({payload!r}))"
    rc, _, err = scheduler.run_on(
        "node004", shlex.join([previous.PYTHON, "-B", "-c", install]), timeout=45, check=False)
    if rc:
        raise RuntimeError(err[-1500:])
    t = previous.task("formal", "manual")
    t.update(description="FreqDuet Route 111 independent Oct 2025 stop event audit",
             signature="FreqDuet/v505_r01/field/route111", cwd=str(BASE),
             cmd=shlex.join([previous.PYTHON, "-B", "-I", "-u", str(path), str(out)]),
             result_dir=str(out / "summary"),
             local_result_dir=str(HERE / "review/field/route111"))
    return t


def gateway_field_task(repair=False):
    root = Path("/tmp/freqduet-v505-field")
    phase = "field_repair" if repair else "field_gateway"
    path = root / (phase + ".py")
    out = root / ("route111-repair" if repair else "route111")
    payload = base64.b64encode((HERE / "field_runtime.py").read_bytes()).decode()
    install = (f"from pathlib import Path; import base64; p=Path({str(path)!r}); "
               f"p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(base64.b64decode({payload!r}))")
    rc, _, err = scheduler.run_on("jtl110gpu2", shlex.join(["python3", "-B", "-c", install]),
                                  timeout=45, check=False)
    if rc:
        raise RuntimeError(err[-1500:])
    t = previous.task("formal", "manual")
    t.update(description="FreqDuet Route 111 independent stop event audit on download gateway",
             signature=f"FreqDuet/v505_r01/{phase}/route111", cwd=str(root),
             allowed_nodes=["jtl110gpu2"],
             cmd=shlex.join(["python3", "-B", "-u", str(path), str(out)] +
                            ([str(root / "route111/mbta-2025.zip")] if repair else [])),
             result_dir=str(out / "summary"),
             local_result_dir=str(HERE / "review" / phase / "route111"))
    return t


def vehicle_task(crosswalk=False):
    root = Path("/tmp/freqduet-v505-field/vehicle")
    phase = "field_crosswalk" if crosswalk else "field_vehicle"
    script = "vehicle_crosswalk.py" if crosswalk else "vehicle_runtime.py"
    out = root / ("crosswalk" if crosswalk else "audit")
    files = {n: base64.b64encode((HERE / n).read_bytes()).decode()
             for n in ("vehicle_runtime.py", "field_runtime.py", "vehicle_crosswalk.py")}
    setup = f"""from pathlib import Path
import base64
p = Path({str(root)!r})
p.mkdir(parents=True, exist_ok=True)
for name, data in {files!r}.items():
 (p/name).write_bytes(base64.b64decode(data))
"""
    rc, _, err = scheduler.run_on("jtl110gpu2", shlex.join(["python3", "-B", "-c", setup]),
                                  timeout=45, check=False)
    if rc:
        raise RuntimeError(err[-1500:])
    t = previous.task("formal", "manual")
    t["env"]["PYTHONPATH"] = str(root.parent / "lib")
    args = [str(root / "audit"), str(root.parent / "route111/mbta-2025.zip")] if crosswalk else []
    t.update(description="FreqDuet independent same-vehicle Route 111 AVL audit",
             signature=f"FreqDuet/v505_r01/{phase}/route111", cwd=str(root),
             allowed_nodes=["jtl110gpu2"],
             cmd=shlex.join(["python3", "-B", "-u", str(root / script), *args, str(out)]),
             result_dir=str(out / "summary"),
             local_result_dir=str(HERE / "review" / phase / "route111"))
    return t


def collect(phase, excluded=()):
    ids = {t["id"] for t in previous.read(receipt(phase))["tasks"]}
    ids -= set(excluded)
    ts = [t for t in scheduler.load_state()["tasks"] if t["id"] in ids]
    assert len(ts) == len(ids)
    pending = [f"{t['id']} {t['status']}" for t in ts if t["status"] != "done"]
    if pending:
        raise SystemExit("; ".join(pending))
    rows = []
    for t in ts:
        f = Path(t["local_result_dir"]) / "results.json"
        if not f.is_file():
            remote = str(Path(t["result_dir"]) / "results.json")
            rc, out, err = scheduler.run_on(t["node"], shlex.join(["cat", remote]),
                                            timeout=30, check=False)
            if rc:
                raise RuntimeError(err[-1000:])
            item = json.loads(out)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps(item, separators=(",", ":")) + "\n")
        rows.append(previous.read(f))
    return rows


def check_smoke(rows):
    for item in rows:
        row, = item["per_day"]
        arm = item["arm"]
        m = row["metrics"]
        assert m["trips_launched"] == m["trips_completed"] == 262 and m["physical_buses"] == 14
        assert row["budget_counts"]["queries"] > 0
        origin = arm.removeprefix("trace_") if arm.startswith("trace_") else "aligned"
        seed = item["seed"] or 50314
        reference = next(r for r in previous.read(V504 / "confirmation" / f"{origin}_{seed}/results.json")["per_day"]
                         if r["day"] == row["day"])
        assert row["scenario_tape_id"] == reference["scenario_tape_id"]
        assert m["passengers_generated"] == reference["metrics"]["passengers_generated"]
        if arm.startswith("trace_"):
            assert m == reference["metrics"], arm
            releases = row["trace"]["dispatch"]
            assert len(releases) == 262 and len({r[0] for r in releases}) == 262
            assert row["trace"]["stock"] and row["trace"]["denials"]
            assert abs(sum(r[3] - r[4] for r in releases) / 262 - m["readiness_delay_s_per_trip"]) < 1e-9
        if arm == "zero":
            assert m["hold_s_per_trip"] == 0 and row["budget_counts"]["nonzero"] == 0
    assert {r["arm"] for r in rows} == {"trace_aligned", "trace_nofreq", "zero", "rolling"}
    print("Smoke passed: both anomaly paths exact, roster and pairing verified", flush=True)


def verified_smoke():
    rows = collect("smoke", excluded=("t224998", "t224999")) + collect("trace_repair")
    check_smoke(rows)
    return rows


def submit(phase, tasks):
    dest = receipt(phase)
    if dest.exists():
        raise SystemExit(f"{dest} already exists")
    signatures = {t["signature"] for t in tasks}
    assert not any(t.get("signature") in signatures for t in scheduler.load_state()["tasks"])
    cmd = [sys.executable, scheduler.__file__]
    p = subprocess.run(cmd + ["submit-jsonl", "--stdin", "--trusted", "--json",
                              "--lock-timeout", "600"], input=json.dumps(tasks), text=True,
                       capture_output=True, check=True)
    entries = [{k: t[k] for k in ("id", "signature")} for t in json.loads(p.stdout)["submitted"]]
    dest.write_text(json.dumps({"tasks": entries}, indent=2) + "\n")
    assert len(entries) == len(tasks)
    print(entries, flush=True)
    ids = [arg for e in entries for arg in ("--task-id", e["id"])]
    subprocess.run(cmd + ["dispatch", "--lock-timeout", "600", *ids], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("stage", "smoke", "trace_repair", "formal", "collect",
                                     "collect-smoke", "field", "field_gateway", "field_repair", "collect-field",
                                     "field_vehicle", "collect-vehicle", "field_crosswalk",
                                     "collect-crosswalk"))
    phase = p.parse_args().phase
    if phase == "stage":
        stage()
    elif phase == "field":
        submit(phase, [field_task()])
    elif phase == "field_gateway":
        submit(phase, [gateway_field_task()])
    elif phase == "field_repair":
        submit(phase, [gateway_field_task(repair=True)])
    elif phase == "collect-field":
        print(json.dumps(collect("field_repair"), indent=2))
    elif phase == "field_vehicle":
        submit(phase, [vehicle_task()])
    elif phase == "collect-vehicle":
        print(json.dumps(collect("field_vehicle"), indent=2))
    elif phase == "field_crosswalk":
        submit(phase, [vehicle_task(crosswalk=True)])
    elif phase == "collect-crosswalk":
        print(json.dumps(collect("field_crosswalk"), indent=2))
    elif phase == "smoke":
        submit(phase, [spec(phase, arm, 50314 if arm != "rolling" else None,
                            [TAIL if arm.startswith("trace_") else PLAN["days"][0]])
                       for arm in ("trace_aligned", "trace_nofreq", "zero", "rolling")])
    elif phase == "trace_repair":
        stage(BASE / "source02")
        submit(phase, [spec(phase, arm, 50314, [TAIL]) for arm in ("trace_aligned", "trace_nofreq")])
    elif phase == "collect-smoke":
        verified_smoke()
    elif phase == "formal":
        verified_smoke()
        tasks = []
        for s, offset in enumerate((0, 8)):
            days = PLAN["days"][offset:offset + 8]
            tasks.extend(spec(phase, arm, seed, days, s) for seed in PLAN["seeds"]
                         for arm in ("zero", "rule", "audit_aligned"))
            tasks.append(spec(phase, "rolling", None, days, s))
        submit(phase, tasks)
    else:
        verified_smoke()
        print(f"Collected {len(collect('formal'))} paired shards")
