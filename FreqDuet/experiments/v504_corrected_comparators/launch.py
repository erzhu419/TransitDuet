"""Stage the isolated V504 runtime and manually dispatch each gated phase."""

import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

from study import HERE, PLAN, aggregate, read

sys.path.insert(0, "/home/erzhu419/mine_code/scheduleurm/skill")
import scheduler

EXPERIMENTS = HERE.parent
PYTHON = read(EXPERIMENTS / "v400_independent_confirmation/deployment.json")["python"]
V502 = Path(read(EXPERIMENTS / "v502_pathb_trc_matrix/staged_source02.json")["source"])
V503 = V502.parent.parent / "FreqDuet-v503-matched-routing"
V504 = V503.with_name("FreqDuet-v504-corrected-comparators")
SOURCE, RUN = V504 / "source01", V504 / "run01"
NODES = [f"node{i:03}" for i in range(1, 7)]
REVIEW = HERE / "review"
OLD_SMOKE = (EXPERIMENTS / "v503_matched_routing/review/smoke"
             / f"aligned_{PLAN['smoke_seed']}/results.json")


def stage():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name in ("matched_controls.py", "study.py", "plan.json", "runtime.patch"):
            archive.add(HERE / name, arcname=name)
        archive.add(EXPERIMENTS / "v503_matched_routing/plan.json", arcname="v503_plan.json")
    payload = base64.b64encode(buf.getvalue()).decode()
    script = f"""
from pathlib import Path
import base64, io, subprocess, tarfile, tempfile
old = Path({str(V503 / "source01")!r})
target = Path({str(SOURCE)!r})
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    raise SystemExit("V504 server source already exists")
with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
    work = Path(tmp) / "source"
    work.mkdir()
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode({payload!r}))) as ar:
        ar.extractall(work)
    subprocess.run(["patch", "--batch", "--fuzz=0", "-o", str(work / "freqduet_v7_b1.py"),
                    str(old / "freqduet_v7_b1.py"), str(work / "runtime.patch")], check=True)
    for name in ("freqduet_src", "corridors"):
        origin = (old / name).resolve(strict=True)
        (work / name).symlink_to(origin, target_is_directory=True)
    work.rename(target)
print(target)
"""
    rc, stdout, stderr = scheduler.run_on(
        "node004", shlex.join([PYTHON, "-B", "-c", script]), timeout=90, check=False)
    if rc:
        raise RuntimeError(stderr[-2200:])
    print(stdout, end="")


def task(phase, arm=None, seed=None):
    label = arm if seed is None else f"{arm}_{seed}"
    label = label or "preflight"
    output = RUN / phase / label
    cmd = [PYTHON, "-B", "-I", "-u", str(SOURCE / "study.py"), phase,
           "--old-run", str(V503 / "run01"), "--output", str(output)]
    if arm:
        cmd += ["--arm", arm]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return {
        "description": f"FreqDuet V504 {phase} {label}",
        "signature": f"FreqDuet/v504_r01/{phase}/{label}",
        "project": "FreqDuet", "priority": "high", "allowed_nodes": NODES,
        "cpu": 1, "ram_mb": 3072, "vram": 0, "allow_cpu_training": arm == "nofreq",
        "cpu_training_justification": "Matched CPU transit simulator and small SAC heads.",
        "no_auto_requeue": True, "allow_no_ckpt": True, "skip_launch_staging": True,
        "cwd": str(SOURCE),
        "env": {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                   "MKL_NUM_THREADS", "TORCH_NUM_THREADS")}
               | {"SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
        "cmd": shlex.join(cmd),
        "result_dir": str(output / "summary"),
        "local_result_dir": str(REVIEW / phase / label),
    }


def receipt(phase):
    return HERE / f"submission_{phase}.json"


def collect(phase):
    submitted = read(receipt(phase))["tasks"]
    ids = {t["id"] for t in submitted}
    status = {t["id"]: t for t in scheduler.load_state()["tasks"] if t["id"] in ids}
    assert set(status) == ids
    if any(t["status"] != "done" for t in status.values()):
        raise SystemExit(", ".join(f"{i}: {t['status']}" for i, t in status.items()))
    for t in status.values():
        path = Path(t["local_result_dir"]) / "results.json"
        if not path.is_file():
            remote = Path(t["result_dir"]) / "results.json"
            rc, out, err = scheduler.run_on(t["node"], shlex.join(["cat", str(remote)]),
                                            timeout=30, check=False)
            if rc:
                raise RuntimeError(err[-1200:])
            data = json.loads(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n")
    return REVIEW / phase


def dispatch(phase):
    if receipt(phase).exists():
        raise SystemExit(f"{phase} task receipt already exists")
    if phase == "check":
        specs = [task("check")]
    else:
        if phase == "smoke":
            assert read(collect("check") / "preflight/results.json")["passed"]
            seeds = [PLAN["smoke_seed"]]
        else:
            print(aggregate(collect("smoke"), smoke=True, old_smoke=OLD_SMOKE), flush=True)
            seeds = PLAN["seeds"]
        specs = [task(phase, arm, s) for s in seeds for arm in ("aligned", "nofreq")]
        specs.append(task(phase, "manual"))
    signatures = {t["signature"] for t in specs}
    assert not any(t.get("signature") in signatures for t in scheduler.load_state()["tasks"])
    base = [sys.executable, scheduler.__file__]
    proc = subprocess.run(base + ["submit-jsonl", "--stdin", "--trusted", "--json",
                                  "--lock-timeout", "600"],
                          input=json.dumps(specs), text=True, capture_output=True, check=True)
    tasks = [{k: t[k] for k in ("id", "signature")} for t in json.loads(proc.stdout)["submitted"]]
    receipt(phase).write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")
    assert len(tasks) == len(specs), proc.stdout
    print(tasks, flush=True)
    ids = [arg for t in tasks for arg in ("--task-id", t["id"])]
    subprocess.run(base + ["dispatch", "--lock-timeout", "600", *ids], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("stage", "check", "smoke", "formal", "report"))
    phase = p.parse_args().phase
    if phase == "stage":
        stage()
        dispatch("check")
    elif phase == "report":
        print(json.dumps(aggregate(collect("formal")), indent=2))
    else:
        dispatch(phase)
