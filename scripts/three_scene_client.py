#!/usr/bin/env python3
"""Agent-facing client for staged three-scene schedule search."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path


def request(api: Path, payload: dict, timeout: float = 1800.0):
    if not (api / "READY.json").is_file():
        raise SystemExit("runner is not ready")
    identifier = f"{time.time_ns()}-{uuid.uuid4().hex}"
    temporary = api / "requests" / f".{identifier}.tmp"
    target = api / "requests" / f"{identifier}.json"
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, target)
    response = api / "responses" / target.name
    deadline = time.monotonic() + timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("runner request timed out")
        time.sleep(0.05)
    value = json.loads(response.read_text())
    if not value.get("ok"):
        raise SystemExit(value.get("error", "runner request failed"))
    return value["result"]


def parse_schedule(text: str):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != 4:
        raise SystemExit("schedule requires four comma-separated speeds")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=Path, default=Path("runner_api"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    probe = commands.add_parser("probe")
    probe.add_argument("schedule")
    score = commands.add_parser("score")
    score.add_argument("anchor_schedule_hash")
    score.add_argument("schedule")
    score.add_argument("--safe-success-probability", type=float, required=True)
    backoff = commands.add_parser("backoff")
    backoff.add_argument("schedule_hash")
    backoff.add_argument("phase")
    backoff.add_argument("evidence")
    rank = commands.add_parser("rank")
    rank.add_argument("schedule_hashes", nargs="+")
    args = parser.parse_args()
    payload = {"command": args.command}
    if args.command == "probe":
        payload["schedule"] = parse_schedule(args.schedule)
    elif args.command == "score":
        payload["anchor_schedule_hash"] = args.anchor_schedule_hash
        payload["schedule"] = parse_schedule(args.schedule)
        payload["safe_success_probability"] = args.safe_success_probability
    elif args.command == "backoff":
        payload["schedule_hash"] = args.schedule_hash
        payload["phase"] = args.phase
        payload["evidence"] = args.evidence
    elif args.command == "rank":
        payload["schedule_hashes"] = args.schedule_hashes
    print(json.dumps(request(args.api, payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
