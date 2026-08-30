#!/usr/bin/env python3
"""Build compact, checkpoint-bound SAIL-inspired priors on AMLFS-04."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPEEDTUNING_SPEED_VALUES", "1,1.5,2,2.5,3")

from act_speed_benchmark import canonical_sha256, sha256  # noqa: E402


PHASES = ("pre_grasp", "grasp_lift", "transport", "interaction")
GRID = (1.0, 1.5, 2.0, 2.5, 3.0)


def nearest_grid(value: float, minimum: float = 1.5) -> float:
    allowed = [speed for speed in GRID if speed >= minimum]
    return min(allowed, key=lambda speed: (abs(speed - float(value)), speed))


def sail_phase_prior(artifact: dict) -> dict:
    """Map a registered offline profile evenly onto the four oracle phases."""

    source = max(
        artifact["candidates"], key=lambda item: float(item["maximum_speed"])
    )
    profile = list(map(float, source["profile"]))
    importance = list(map(float, source["importance"]))
    if len(profile) != len(importance) or len(profile) % len(PHASES):
        raise RuntimeError("offline profile must divide evenly across four phases")
    bins_per_phase = len(profile) // len(PHASES)
    phase_profile = [
        statistics.fmean(profile[bins_per_phase * i : bins_per_phase * (i + 1)])
        for i in range(len(PHASES))
    ]
    phase_importance = [
        statistics.fmean(
            importance[bins_per_phase * i : bins_per_phase * (i + 1)]
        )
        for i in range(len(PHASES))
    ]
    result = {
        "schema": "act-sail-inspired-phase-prior-v33",
        "prior_kind": "sail_inspired_offline",
        "paper_faithful_sail": False,
        "source_artifact_payload_sha256": artifact["artifact_payload_sha256"],
        "source_candidate_id": source["id"],
        "mapping": "equal_contiguous_nominal_time_bins_to_four_phases_then_nearest_common_grid",
        "source_profile_bins": len(profile),
        "bins_per_phase": bins_per_phase,
        "minimum_phase_speed": 1.5,
        "phase_order": list(PHASES),
        "phase_profile": phase_profile,
        "phase_importance": phase_importance,
        "schedule": [nearest_grid(value) for value in phase_profile],
    }
    result["prior_payload_sha256"] = canonical_sha256(result)
    return result


ROOTS = {
    "pick": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-pick-3pv-wrists-20260824-v1/pick"),
    "tea": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-tea-3pv-wrists-20260823-v1/tea"),
    "insertion": Path("/mnt/amlfs-04/home/linke/speedtuning-original-act/speedtuning-original-act-insertion-3pv-wrists-20260824-v1/insertion"),
}

OFFLINE_ROOT = Path(
    "/mnt/amlfs-04/home/linke/speedtuning-act-speed-benchmark-v1/"
    "runs/298c6d16784f228df0b1f455d0e41b4276ec5184"
)
OFFLINE_ARTIFACTS = {
    task: OFFLINE_ROOT / task / "sail_inspired_adaptive/search/offline_artifact.json"
    for task in ROOTS
}


def load_verified_offline_artifact(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    claimed = artifact.get("artifact_payload_sha256")
    payload = dict(artifact)
    payload.pop("artifact_payload_sha256", None)
    if claimed != canonical_sha256(payload):
        raise RuntimeError(f"offline artifact payload hash mismatch: {path}")
    if artifact.get("method") != "sail_inspired_adaptive":
        raise RuntimeError(f"wrong offline artifact method: {path}")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = {}
    for task, root in ROOTS.items():
        artifact_path = OFFLINE_ARTIFACTS[task]
        artifact = load_verified_offline_artifact(artifact_path)
        prior = sail_phase_prior(artifact)
        checkpoint_hashes = {
            name: sha256(root / "checkpoints" / name)
            for name in ("policy_best.ckpt", "dataset_stats.pkl", "policy_config.json")
        }
        tasks[task] = {
            "dataset_source": artifact["dataset_directory"],
            "dataset_episode_count": artifact["episode_count"],
            "dataset_array_sha256": artifact["dataset_array_sha256"],
            "policy_artifacts": checkpoint_hashes,
            "offline_artifact_source": str(artifact_path),
            "offline_artifact_file_sha256": sha256(artifact_path),
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
