#!/usr/bin/env python3
"""Build compact, checkpoint-bound SAIL-inspired priors on AMLFS-04."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import build_offline_artifact, canonical_sha256, sha256  # noqa: E402
from scripts.run_act_sail_warmstart_v33 import sail_phase_prior  # noqa: E402


ROOTS = {
    "pick": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-pick-3pv-wrists-20260824-v1/pick"),
    "tea": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-tea-3pv-wrists-20260823-v1/tea"),
    "insertion": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-insertion-3pv-wrists-20260824-v1/insertion"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {}
    for task, root in ROOTS.items():
        artifact = build_offline_artifact(root / "dataset", "sail_inspired_adaptive")
        prior = sail_phase_prior(artifact)
        checkpoint_hashes = {
            name: sha256(root / "checkpoints" / name)
            for name in ("policy_best.ckpt", "dataset_stats.pkl", "policy_config.json")
        }
        tasks[task] = {
            "dataset_source": str(root / "dataset"),
            "dataset_episode_count": artifact["episode_count"],
            "dataset_array_sha256": artifact["dataset_array_sha256"],
            "policy_artifacts": checkpoint_hashes,
            "offline_artifact_payload_sha256": artifact["artifact_payload_sha256"],
            "phase_prior": prior,
        }
    value = {
        "schema": "act-sail-inspired-offline-priors-v33",
        "paper_faithful_sail": False,
        "tasks": tasks,
    }
    value["payload_sha256"] = canonical_sha256(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print("V33_PRIORS_JSON=" + json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
