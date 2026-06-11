"""Venue-grade LOBSTER sample validation for L2/L3 replay.

LOBSTER sample files are reconstructed from NASDAQ TotalView-ITCH messages and
come as paired message and order-book snapshot CSVs. This module converts a
small public sample into the existing manifest-driven L2/L3 validation schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

from freq_hrl.experiments.trading.order_book_large_replay_manifest_validation import (
    run_manifest_validation,
)


HF_BASE = "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main"
SAMPLE_TICKER = "AMZN"
SAMPLE_SESSION = "2012-06-21"
SAMPLE_LEVELS = 1
DEFAULT_SYMBOLS = ("AMZN", "GOOG", "AAPL")
DEFAULT_SESSIONS = (SAMPLE_SESSION,)


def _sample_paths(symbol: str, session: str = SAMPLE_SESSION, levels: int = SAMPLE_LEVELS) -> tuple[str, str]:
    sample_dir = f"LOBSTER_SampleFile_{symbol}_{session}_{levels}"
    sample_prefix = f"{symbol}_{session}_34200000_57600000"
    message = f"{sample_dir}/{sample_prefix}_message_{levels}.csv"
    orderbook = f"{sample_dir}/{sample_prefix}_orderbook_{levels}.csv"
    return message, orderbook


def _download(url_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urlretrieve(f"{HF_BASE}/{url_path}", dest)
    return dest


def download_lobster_sample(
    raw_dir: Path,
    *,
    symbol: str = SAMPLE_TICKER,
    session: str = SAMPLE_SESSION,
    levels: int = SAMPLE_LEVELS,
) -> tuple[Path, Path]:
    message_path, orderbook_path = _sample_paths(symbol, session=session, levels=int(levels))
    message = _download(message_path, raw_dir / Path(message_path).name)
    orderbook = _download(orderbook_path, raw_dir / Path(orderbook_path).name)
    return message, orderbook


def _price(value: str) -> float:
    raw = float(value)
    return raw / 10000.0 if abs(raw) > 10000.0 else raw


def _side(direction: str) -> str:
    return "bid" if float(direction) >= 0.0 else "ask"


def _event_type(value: str) -> str | None:
    code = int(float(value))
    if code == 1:
        return "add"
    if code in {2, 3}:
        return "cancel"
    if code in {4, 5}:
        return "trade"
    return None


def convert_lobster_pair(
    *,
    message_csv: Path,
    orderbook_csv: Path,
    output_dir: Path,
    max_rows: int = 2400,
    venue: str = "XNAS",
    symbol: str = SAMPLE_TICKER,
    session: str = SAMPLE_SESSION,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    l2_path = output_dir / f"{symbol}_{session}_lobster_l2.csv"
    l3_path = output_dir / f"{symbol}_{session}_lobster_l3.csv"
    manifest_path = output_dir / "manifest.json"
    max_rows = max(8, int(max_rows))

    with message_csv.open("r", newline="", encoding="utf-8") as f:
        messages = list(csv.reader(f))[:max_rows]
    with orderbook_csv.open("r", newline="", encoding="utf-8") as f:
        books = list(csv.reader(f))[:max_rows]
    n_rows = min(len(messages), len(books), max_rows)
    if n_rows < 8:
        raise ValueError("LOBSTER sample has fewer than 8 paired rows")

    with l2_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "ask_price_1",
                "ask_size_1",
                "bid_price_1",
                "bid_size_1",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for msg, book in zip(messages[:n_rows], books[:n_rows]):
            if len(msg) < 1 or len(book) < 4:
                continue
            writer.writerow({
                "timestamp": float(msg[0]),
                "ask_price_1": _price(book[0]),
                "ask_size_1": float(book[1]),
                "bid_price_1": _price(book[2]),
                "bid_size_1": float(book[3]),
            })

    valid_l3 = 0
    with l3_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "event_type", "side", "price", "size", "order_id"],
            lineterminator="\n",
        )
        writer.writeheader()
        for idx, msg in enumerate(messages[:n_rows]):
            if len(msg) < 6:
                continue
            event_type = _event_type(msg[1])
            if event_type is None:
                continue
            order_id = str(msg[2])
            if order_id in {"0", "0.0", ""}:
                order_id = f"hidden_{idx}"
            writer.writerow({
                "timestamp": float(msg[0]),
                "event_type": event_type,
                "side": _side(msg[5]),
                "price": _price(msg[4]),
                "size": float(msg[3]),
                "order_id": order_id,
            })
            valid_l3 += 1
    if valid_l3 < 8:
        raise ValueError("LOBSTER conversion produced fewer than 8 valid L3 events")

    manifest = {
        "datasets": [
            {
                "kind": "l2",
                "path": l2_path.name,
                "venue": venue,
                "symbol": symbol,
                "session": session,
                "source_id": f"lobster_{symbol}_{session}_l2",
                "source_type": "venue_grade",
                "feed_level": "l2_snapshot",
                "matching_semantics": "price_time",
            },
            {
                "kind": "l3",
                "path": l3_path.name,
                "venue": venue,
                "symbol": symbol,
                "session": session,
                "source_id": f"lobster_{symbol}_{session}_l3",
                "source_type": "venue_grade",
                "feed_level": "l3_event",
                "matching_semantics": "price_time",
            },
        ],
        "source": (
            "LOBSTER public sample reconstructed from NASDAQ TotalView-ITCH; "
            "downloaded from totalorganfailure/lobster-data on Hugging Face."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "l2_csv": l2_path,
        "l3_csv": l3_path,
        "rows": n_rows,
        "l3_events": valid_l3,
    }


def run_lobster_sample_validation(
    output_dir: Path,
    *,
    raw_dir: Path,
    symbols: list[str],
    sessions: list[str],
    max_rows: int,
    methods: list[str],
    steps: int,
    min_pairs: int,
) -> dict[str, Any]:
    converted_root = output_dir / "converted"
    converted_root.mkdir(parents=True, exist_ok=True)
    conversion_rows: list[dict[str, Any]] = []
    manifest_datasets: list[dict[str, Any]] = []
    for symbol in symbols:
        symbol = str(symbol).upper()
        for session in sessions:
            session = str(session)
            raw_message, raw_orderbook = download_lobster_sample(
                raw_dir,
                symbol=symbol,
                session=session,
            )
            converted = convert_lobster_pair(
                message_csv=raw_message,
                orderbook_csv=raw_orderbook,
                output_dir=converted_root / symbol,
                max_rows=int(max_rows),
                symbol=symbol,
                session=session,
            )
            symbol_manifest = json.loads(Path(converted["manifest"]).read_text(encoding="utf-8"))
            for entry in symbol_manifest["datasets"]:
                item = dict(entry)
                item["path"] = f"{symbol}/{item['path']}"
                manifest_datasets.append(item)
            conversion_rows.append({
                "symbol": symbol,
                "session": session,
                "rows": int(converted["rows"]),
                "l3_events": int(converted["l3_events"]),
                "l2_csv": str(converted["l2_csv"]),
                "l3_csv": str(converted["l3_csv"]),
                "raw_message_csv": str(raw_message),
                "raw_orderbook_csv": str(raw_orderbook),
            })
    combined_manifest = {
        "datasets": manifest_datasets,
        "source": (
            "Multi-symbol/multi-session LOBSTER public samples reconstructed "
            "from NASDAQ TotalView-ITCH; "
            "downloaded from totalorganfailure/lobster-data on Hugging Face."
        ),
    }
    manifest_path = converted_root / "manifest.json"
    manifest_path.write_text(json.dumps(combined_manifest, indent=2) + "\n", encoding="utf-8")
    payload = run_manifest_validation(
        output_dir,
        manifest=manifest_path,
        methods=list(methods),
        steps=int(steps),
        levels=1,
        latency_bins=[0, 2],
        execution_modes=["market", "passive_queue"],
        queue_ahead_fraction=0.5,
        min_pairs=int(min_pairs),
        require_venue_grade=True,
    )
    payload["lobster_conversion"] = conversion_rows
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("transit_hrl/data/lobster_sample_raw"))
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    parser.add_argument("--max-rows", type=int, default=2400)
    parser.add_argument("--methods", nargs="+", default=["ema", "state_space", "adaptive_wavelet", "neural_state_space"])
    parser.add_argument("--steps", type=int, default=720)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transit_hrl/results/order_book_lobster_venue_grade_multisymbol"),
    )
    args = parser.parse_args()
    payload = run_lobster_sample_validation(
        args.output_dir,
        raw_dir=args.raw_dir,
        symbols=list(args.symbols),
        sessions=list(args.sessions),
        max_rows=int(args.max_rows),
        methods=list(args.methods),
        steps=int(args.steps),
        min_pairs=int(args.min_pairs),
    )
    coverage = payload["coverage"]
    print(
        "lobster_order_book "
        f"venue_status={coverage['venue_grade_claim_status']} "
        f"pairs={coverage['venue_grade_l2_l3_session_pairs']}"
    )


if __name__ == "__main__":
    main()
