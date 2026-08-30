#!/usr/bin/env python3
"""Import only immutable v34 schedule-seed state receipts after a code repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


TASKS = ("pick", "tea", "insertion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    args = parser.parse_args()

    amendment = json.loads(args.amendment.read_text())
    if amendment.get("schema") != "act-prior-causal-risk-v34-resume-amendment":
        raise RuntimeError("unexpected v34 resume amendment schema")
    if args.source_root.resolve() != Path(amendment["source_run"]).resolve():
        raise RuntimeError("v34 resume source differs from sealed amendment")
    if args.source_root.name != amendment["prior_implementation_commit"]:
        raise RuntimeError("v34 resume source commit mismatch")
    if args.implementation_commit == amendment["prior_implementation_commit"]:
        raise RuntimeError("v34 repair must use a new implementation commit")

    imported = []
    observed_counts = {}
    for task in TASKS:
        source = args.source_root / "search" / task / "search" / "states"
        paths = sorted(source.glob("*/*.json")) if source.exists() else []
        observed_counts[task] = len(paths)
        for path in paths:
            value = json.loads(path.read_text())
            if int(value.get("seed", -1)) != int(path.stem):
                raise RuntimeError(f"v34 imported state seed mismatch: {path}")
            if not isinstance(value.get("schedule"), list) or len(value["schedule"]) != 4:
                raise RuntimeError(f"v34 imported state schedule mismatch: {path}")
            relative = path.relative_to(args.source_root)
            destination = args.destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_hash = sha256(path)
            if destination.exists():
                if sha256(destination) != source_hash:
                    raise RuntimeError(f"v34 imported state collision: {destination}")
            else:
                shutil.copy2(path, destination)
            if sha256(destination) != source_hash:
                raise RuntimeError(f"v34 imported state hash mismatch: {destination}")
            imported.append({"path": str(relative), "sha256": source_hash})
    if observed_counts != amendment["prior_state_counts"]:
        raise RuntimeError(
            f"v34 source state counts changed: {observed_counts} != {amendment['prior_state_counts']}"
        )

    receipt = {
        "schema": "act-prior-causal-risk-v34-resume-import",
        "amendment_sha256": sha256(args.amendment),
        "source_root": str(args.source_root.resolve()),
        "destination_root": str(args.destination_root.resolve()),
        "implementation_commit": args.implementation_commit,
        "state_counts": observed_counts,
        "states": imported,
        "scientific_rollouts_executed_by_import": 0,
    }
    output = args.destination_root / "RESUME_IMPORT.json"
    serialized = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if output.exists() and output.read_text() != serialized:
        raise RuntimeError("v34 resume import receipt changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(json.dumps({"state_counts": observed_counts, "receipt": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
