"""Fail-closed one-episode overfit and same-pose closed-loop DP gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from original_act import episode_paths, set_seed, split_original_act_episodes
from original_diffusion import (
    JointRangeNormalizer,
    ModelEMA,
    ORIGINAL_DIFFUSION_CONFIG,
    OriginalDiffusionDataset,
    OriginalDiffusionPolicy,
)
from scripts.evaluate_original_diffusion import rollout
from scripts.train_original_diffusion import (
    SPLIT_SEED,
    _atomic_json,
    _checkpoint,
    _path_identity,
    cosine_warmup_multiplier,
)


def _successful_episode(paths):
    for path in sorted(paths):
        with h5py.File(path, "r") as root:
            cameras = set(root["observations/images"])
            if cameras != {"angle", "left_wrist", "right_wrist"}:
                raise ValueError(f"camera mismatch in {path}")
            if bool(root.attrs["source_success"]) and bool(root.attrs["replay_success"]):
                return (
                    path,
                    int(root.attrs["seed"]),
                    str(root.attrs["task"]),
                    np.asarray(root["object_pose"], dtype=np.float64),
                )
    raise ValueError("no successful source+replay demonstration available for smoke gate")


def _matched_smoke_contract(dataset_dir, validation_episodes=20):
    paths = episode_paths(dataset_dir)
    if len(paths) != 270:
        raise ValueError(f"expected the exact 270 ACT episodes, found {len(paths)}")
    train_paths, validation_paths = split_original_act_episodes(
        paths, seed=SPLIT_SEED, validation_count=validation_episodes
    )
    if (len(train_paths), len(validation_paths)) != (250, 20):
        raise ValueError("matched smoke requires exactly 250 train and 20 validation episodes")
    episode = _successful_episode(train_paths)
    return paths, train_paths, validation_paths, episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--updates", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-final-loss", type=float, default=0.08)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty smoke output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths, train_paths, validation_paths, episode = _matched_smoke_contract(
        args.dataset_dir
    )
    path, episode_seed, saved_task, object_pose = episode
    if saved_task != args.task:
        raise ValueError(f"task mismatch: {saved_task} != {args.task}")

    contract = {
        "schema": "original-diffusion-smoke-contract-v2",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_identity": _path_identity(paths),
        "split_seed": SPLIT_SEED,
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "normalization_fit": "same 250 training episodes as full training",
        "training_sample_stride": ORIGINAL_DIFFUSION_CONFIG["action_horizon"],
        "training_sample_semantics": "native steps 0, 8, 16, ... queried by the closed-loop evaluator",
        "overfit_episode": path.name,
        "overfit_episode_partition": "train",
        "episode_seed": episode_seed,
        "object_pose": object_pose.tolist(),
    }
    _atomic_json(args.output_dir / "contract.json", contract)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Overfit only one episode, but normalize it exactly as the executable full
    # training path does.  Fitting ranges on the one episode creates a different
    # controller whose qpos feedback clips after even a small rollout error.
    normalizer = JointRangeNormalizer.fit(train_paths)
    dataset = OriginalDiffusionDataset(
        [path],
        normalizer,
        image_size=ORIGINAL_DIFFUSION_CONFIG["image_size"],
        sample_stride=ORIGINAL_DIFFUSION_CONFIG["action_horizon"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    iterator = iter(loader)
    config = {
        **ORIGINAL_DIFFUSION_CONFIG,
        "training_sample_stride": ORIGINAL_DIFFUSION_CONFIG["action_horizon"],
    }
    model = OriginalDiffusionPolicy(config).to(device)
    ema = ModelEMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup_multiplier(step, args.updates, min(500, args.updates // 10))
    )
    recent = []
    for update in range(1, args.updates + 1):
        try:
            image, qpos, actions, is_pad = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            image, qpos, actions, is_pad = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(qpos.to(device), image.to(device), actions.to(device), is_pad.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite smoke loss at {update}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        recent.append(float(loss.detach().cpu()))
        recent = recent[-100:]
        if update == 1 or update % 250 == 0:
            print(json.dumps({"smoke_update": update, "mean_recent_loss": float(np.mean(recent))}), flush=True)

    final_loss = float(np.mean(recent))
    _checkpoint(
        args.output_dir / "overfit.pt",
        model,
        ema,
        optimizer,
        scheduler,
        normalizer,
        config,
        args.updates,
        final_loss,
    )
    overfit_passed = final_loss <= args.maximum_final_loss
    record = rollout(
        args.task,
        ema.model.eval(),
        normalizer,
        device,
        episode_seed,
        tuple(config["camera_names"]),
        object_pose=object_pose,
    )
    report = {
        "schema": "original-diffusion-smoke-gate-v2",
        "episode": path.name,
        "episode_seed": episode_seed,
        "updates": args.updates,
        "mean_final_100_loss": final_loss,
        "maximum_final_loss": args.maximum_final_loss,
        "overfit_passed": overfit_passed,
        "closed_loop": record,
        "closed_loop_passed": bool(record["success"]),
        "passed": bool(overfit_passed and record["success"]),
        "contract": contract,
    }
    _atomic_json(args.output_dir / "gate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(42)


if __name__ == "__main__":
    main()
