"""Limited Route 111 vehicle-ID turnaround check from public AVL snapshots."""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import duckdb
from field_runtime import quantiles

HOST = "https://busobservatory-lake.s3.amazonaws.com/feeds/mbta_all/"
SUFFIX = ("06_18:02:38", "07_18:02:48", "08_18:02:44", "09_18:02:50",
          "10_18:02:50", "11_18:02:41")
FILES = [f"COMPACTED_mbta_all_2025-10-{name}.parquet" for name in SUFFIX]
COLUMNS = ("vehicle.vehicle.id", "vehicle.trip.trip_id", "vehicle.trip.route_id",
           "vehicle.trip.direction_id", "vehicle.timestamp", "vehicle.stop_id", "vehicle.current_status")
EASTERN = ZoneInfo("America/New_York")


def download_range(name, span):
    req = Request(HOST + name, headers={"Range": f"bytes={span}"})
    with urlopen(req, timeout=45) as r:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers["Content-Range"])
        first, last, size = map(int, match.groups())
        b = r.read()
        if len(b) != last - first + 1:
            raise ValueError(f"Short range {span} of {name}")
    return first, size, b


def selected_columns(connection, out):
    # Fetch only relevant column pages, while keeping the publisher's footer at its original offset.
    with ThreadPoolExecutor(max_workers=24) as pool:
        footers = list(pool.map(lambda name: download_range(name, "-65536"), FILES))
        jobs = []
        paths = []
        for name, (offset, size, blob) in zip(FILES, footers):
            p = out / name
            with p.open("wb") as f:
                f.write(b"PAR1")
                f.seek(offset)
                f.write(blob)
                f.truncate(size)
            paths.append(str(p))
            rows = connection.execute(
                "SELECT path_in_schema, data_page_offset, dictionary_page_offset, total_compressed_size "
                "FROM parquet_metadata(?)", [str(p)]).fetchall()
            assert set(COLUMNS) <= {r[0] for r in rows}
            for column, data, dictionary, length in rows:
                if column not in COLUMNS:
                    continue
                start = dictionary if dictionary and dictionary > 0 else data
                limit = start + length
                jobs.extend((p, name, f"{i}-{min(i+524288, limit)-1}")
                            for i in range(start, limit, 524288))

        def fill(job):
            path, name, span = job
            offset, _, data = download_range(name, span)
            with path.open("r+b", buffering=0) as f:
                f.seek(offset)
                f.write(data)

        for _ in pool.map(fill, jobs):
            pass
    return paths


def summarize(c, paths):
    col = [f'"{name}"' for name in COLUMNS]
    sql = f"""
    WITH p AS (
      SELECT {col[0]} AS bus, {col[1]} AS trip, {col[2]} AS route, {col[3]} AS dir,
             CAST(epoch({col[4]}) AS BIGINT) AS tick, {col[5]} AS stop, {col[6]} AS status
      FROM read_parquet(?)
      WHERE {col[0]} IS NOT NULL AND {col[1]} IS NOT NULL AND {col[4]} IS NOT NULL
    ), buses AS (SELECT DISTINCT bus FROM p WHERE route = '111')
    SELECT DISTINCT p.* FROM p JOIN buses USING(bus) ORDER BY bus,tick
    """
    groups = defaultdict(dict)
    route_endpoints = {1: ("5547", "8309"), 0: ("8309", "55471")}
    for bus, trip, route, direction, timestamp, stop, status in c.execute(sql, [paths]).fetchall():
        dt = datetime.fromtimestamp(timestamp, EASTERN)
        if not (6 <= dt.day <= 10 and 6 <= dt.hour < 19):
            continue
        key = (trip, route, direction)
        item = groups[bus, dt.date().isoformat()].setdefault(key, [timestamp, timestamp, []])
        item[1] = timestamp
        if route == "111" and status == 1 and stop in ("5547", "55471", "8309"):
            item[2].append((timestamp, stop))
    stays = defaultdict(list)
    covered = set()
    pair_days = defaultdict(set)
    valid = defaultdict(int)
    for (_, day), trips in groups.items():
        ordered = sorted(trips.items(), key=lambda t: t[1][0])

        def terminals(key, times):
            if key[1] != "111" or key[2] not in route_endpoints:
                return None
            start, end = route_endpoints[key[2]]
            starts = [t for t, s in times[2] if s == start]
            ends = [t for t, s in times[2] if s == end]
            if starts and ends and max(starts) < min(ends):
                return max(starts), min(ends)
            return None

        completed = [terminals(k, times) for k, times in ordered]
        for idx, endpoints in enumerate(completed):
            if endpoints:
                covered.add(day)
                valid[str(int(ordered[idx][0][2]))] += 1
            if idx == 0 or not endpoints or not completed[idx-1]:
                continue
            direction = int(ordered[idx][0][2])
            previous = int(ordered[idx-1][0][2])
            if direction == previous or ordered[idx-1][1][1] > ordered[idx][1][0]:
                continue
            elapsed = endpoints[0] - completed[idx-1][1]
            if 0 < elapsed <= 3600:
                terminal = "Haymarket" if direction == 0 else "Woodlawn"
                stays[terminal].append(elapsed)
                pair_days[terminal].add(day)
    return {"source": "Cornell Tech Bus Observatory (CC BY-NC 4.0)", "files": FILES,
            "observed_days": sorted(covered), "complete_stopped_trips_by_direction": dict(valid),
            "turnaround": {k: {"n": len(v), "days": len(pair_days[k]), "seconds": quantiles(v)}
                           for k, v in sorted(stays.items())}}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    con = duckdb.connect(config={"threads": 1})
    paths = selected_columns(con, out)
    result = summarize(con, paths)
    dest = out / "summary"
    dest.mkdir()
    (dest / "results.json").write_text(json.dumps(result, indent=2) + "\n")
