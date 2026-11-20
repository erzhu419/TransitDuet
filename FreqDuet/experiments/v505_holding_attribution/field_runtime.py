"""Independent check of Route 111 end-to-end times using MBTA stop events."""

import csv
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
import io
import json
from pathlib import Path
import sys
from urllib.request import urlretrieve
from zipfile import ZipFile

SOURCE = "https://www.arcgis.com/sharing/rest/content/items/924df13d845f4907bb6a6c3ed380d57a/data"


def time_s(value):
    value = value.strip()
    if "T" in value:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        except ValueError:
            pass
    for fmt in ("%I:%M %p", "%I:%M:%S %p", "%H:%M:%S", "%H:%M"):
        try:
            dt = datetime.strptime(value, fmt)
        except ValueError:
            continue
        return dt.hour * 3600 + dt.minute * 60 + dt.second
    return None


def date_key(value):
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(value.strip().split(" ")[0], fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unknown MBTA service date: {value!r}")


def quantiles(values):
    values = sorted(values)
    result = []
    for q in (.1, .5, .9):
        j = (len(values) - 1) * q
        i = int(j)
        result.append(values[i] + (j - i) * (values[min(i + 1, len(values) - 1)] - values[i]))
    return result


def measure(archive):
    kept = defaultdict(lambda: [[], []])
    dates = set()
    with ZipFile(archive) as z:
        member, = [n for n in z.namelist() if "2025-10" in n and n.endswith(".csv")]
        with z.open(member) as binary:
            rows = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig"))
            cols = {k.lower().replace(" ", "_"): k for k in rows.fieldnames}
            for r in rows:
                if r[cols["route_id"]].strip() != "111":
                    continue
                raw_date = r[cols["service_date"]]
                if raw_date not in dates:
                    dates.add(raw_date)
                if weekday(raw_date) >= 5:
                    continue
                point = r[cols["point_type"]].strip().lower().replace(" ", "")
                if point not in {"startpoint", "endpoint"}:
                    continue
                t = clock(r[cols["actual"]])
                if t is not None:
                    key = (raw_date, r[cols["direction_id"]], r[cols["half_trip_id"]])
                    slot = 0 if point == "startpoint" else 1
                    kept[key][slot].append((t, r[cols["stop_id"]], r[cols["standard_type"]]))
    by_pair = defaultdict(list)
    standards = Counter()
    for (d, direction, _), (starts, ends) in kept.items():
        if len(set(starts)) != 1 or len(set(ends)) != 1:
            continue
        start, s_stop, standard = starts[0]
        end, e_stop, _ = ends[0]
        if 6 * 3600 <= start < 19 * 3600 and 0 < end - start <= 7200:
            by_pair[direction, s_stop, e_stop].append((d, end - start))
            standards[direction, s_stop, e_stop, standard] += 1
    directions = {}
    for direction in sorted(set(k[0] for k in by_pair)):
        key = max((k for k in by_pair if k[0] == direction), key=lambda k: len(by_pair[k]))
        group = by_pair[key]
        durations = [t for _, t in group]
        directions[direction] = dict(origin=key[1], destination=key[2],
                                     trips=len(group), weekdays=len({d for d, _ in group}),
                                     runtime_s=quantiles(durations),
                                     service_standard={k[3]: n for k, n in standards.items() if k[:3] == key})
    if len(directions) != 2:
        raise ValueError(f"Expected two directions; received {directions}")
    return {"source": SOURCE, "member": member, "directions": directions,
            "fields": list(cols), "has_vehicle_id": "vehicle_id" in cols,
            "has_block_id": "block_id" in cols}

weekday = lru_cache(maxsize=None)(lambda s: date_key(s).weekday())
clock = lru_cache(maxsize=None)(time_s)

if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    archive = Path(sys.argv[2]) if len(sys.argv) == 3 else output / "mbta-2025.zip"
    if len(sys.argv) == 2:
        urlretrieve(SOURCE, archive)
    summary = measure(archive)
    folder = output / "summary"
    folder.mkdir()
    (folder / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
