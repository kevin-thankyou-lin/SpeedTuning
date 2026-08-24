#!/usr/bin/env python3
"""Client for the receipt-bearing staged VLM frontier server."""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path


def send(api: Path, payload: dict, timeout: float = 3600.0):
    if not (api / "READY.json").is_file():
        raise SystemExit("runner is not ready")
    name = f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    temporary = api / "requests" / f".{name}.tmp"
    target = api / "requests" / name
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, target)
    response = api / "responses" / name
    deadline = time.monotonic() + timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise SystemExit("request timed out")
        time.sleep(0.1)
    value = json.loads(response.read_text())
    if not value.get("ok"):
        raise SystemExit(value.get("error", "request failed"))
    return value["result"]


def schedule(text: str):
    values = [float(value) for value in text.split(",")]
    if len(values) != 4:
        raise SystemExit("schedule needs four comma-separated speeds")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    commands.add_parser("native")
    gate = commands.add_parser("gate")
    gate.add_argument("kind", choices=("anchor", "uniform", "repair", "promote"))
    gate.add_argument("schedule")
    gate.add_argument("--phase")
    gate.add_argument("--evidence")
    commands.add_parser("finalize")
    args = parser.parse_args()
    payload = {"command": args.command}
    if args.command == "gate":
        payload.update({"kind": args.kind, "schedule": schedule(args.schedule), "phase": args.phase, "evidence": args.evidence})
    print(json.dumps(send(args.api, payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
