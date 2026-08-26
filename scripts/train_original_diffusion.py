"""Train matched Diffusion Policy on the sealed original-ACT demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

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
from scripts.evaluate_original_diffusion import checkpoint_identity, rollout


SPLIT_SEED = 1
VALIDATION_SUBSET_SEED = 1
VALIDATION_NOISE_SEED = 17


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _path_identity(paths):
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0".encode())
    return digest.hexdigest()


def cosine_warmup_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.inference_mode()
def deterministic_validation(model, loader, device, seed=0):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    losses = []
    for image, qpos, actions, is_pad in loader:
        loss = model.compute_loss(
            qpos.to(device),
            image.to(device),
            actions.to(device),
            is_pad.to(device),
            generator=generator,
        )
        losses.append(float(loss.cpu()))
    if not losses:
        raise ValueError("validation loader is empty")
    return float(np.mean(losses))


def _evaluate_milestone(
    task,
    model,
    normalizer,
    device,
    update,
    episodes,
    seed_base,
    camera_names,
    output_dir,
    checkpoint=None,
):
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"update-{update:06d}.json"
    partial = output.with_suffix(output.suffix + ".partial")
    records = []
    for index in range(episodes):
        records.append(
            rollout(
                task,
                model,
                normalizer,
                device,
                seed_base + index,
                tuple(camera_names),
            )
        )
        _atomic_json(
            partial,
            {
                "update": int(update),
                "seed_base": int(seed_base),
                "episodes": int(episodes),
                "rollouts": records,
            },
        )
    result = {
        "schema": "original-diffusion-milestone-evaluation-v1",
        "update": int(update),
        "seed_base": int(seed_base),
        "episodes": int(episodes),
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "replan_interval": int(model.action_horizon),
        "prediction_horizon": int(model.prediction_horizon),
        "observation_horizon": int(model.observation_horizon),
        "rollouts": records,
    }
    if checkpoint is not None:
        result["checkpoint"] = Path(checkpoint).name
        result["checkpoint_sha256"] = checkpoint_identity(checkpoint)
    _atomic_json(output, result)
    partial.unlink(missing_ok=True)
    return result


def _checkpoint(path, model, ema, optimizer, scheduler, normalizer, config, update, validation):
    value = {
        "schema": "original-diffusion-checkpoint-v1",
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalizer": normalizer.state_dict(),
        "config": config,
        "update": int(update),
        "validation_loss": float(validation),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--warmup-updates", type=int, default=500)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-names", nargs="+", default=("angle", "left_wrist", "right_wrist"))
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--eval-task")
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--eval-seed-base", type=int)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-updates", type=int, nargs="+", default=())
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.eval_task:
        if args.eval_output_dir is None or args.eval_seed_base is None or not args.eval_updates:
            parser.error(
                "--eval-task requires --eval-output-dir, --eval-seed-base, and --eval-updates"
            )
        if any(update < 1 or update > args.updates for update in args.eval_updates):
            parser.error("--eval-updates must be inside the training update range")
        if any(
            update != args.updates and update % args.validation_every
            for update in args.eval_updates
        ):
            parser.error("--eval-updates must coincide with deterministic validation updates")
        args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    elif args.eval_output_dir is not None or args.eval_seed_base is not None or args.eval_updates:
        parser.error("milestone evaluation arguments require --eval-task")

    paths = episode_paths(args.dataset_dir)
    if len(paths) != 270:
        raise ValueError(f"expected the exact 270 ACT episodes, found {len(paths)}")
    train_paths, validation_paths = split_original_act_episodes(
        paths, seed=SPLIT_SEED, validation_count=args.validation_episodes
    )
    if (len(train_paths), len(validation_paths)) != (250, 20):
        raise ValueError("matched experiment requires exactly 250 train and 20 validation episodes")
    normalizer = JointRangeNormalizer.fit(train_paths)
    split = {
        "schema": "original-diffusion-split-v1",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_identity": _path_identity(paths),
        "split_seed": SPLIT_SEED,
        "train": [path.name for path in train_paths],
        "validation": [path.name for path in validation_paths],
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "normalization_fit": "train episodes only; one global range per joint",
        "training_sample_stride": ORIGINAL_DIFFUSION_CONFIG["action_horizon"],
        "training_sample_semantics": "native steps 0, 8, 16, ... queried by the closed-loop evaluator",
    }
    _atomic_json(args.output_dir / "split.json", split)
    _atomic_json(
        args.output_dir / "normalizer.json",
        {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in normalizer.state_dict().items()},
    )

    config = {
        **ORIGINAL_DIFFUSION_CONFIG,
        "camera_names": list(args.camera_names),
        "pretrained_backbone": not args.no_pretrained_backbone,
        "training_sample_stride": ORIGINAL_DIFFUSION_CONFIG["action_horizon"],
    }
    train_dataset = OriginalDiffusionDataset(
        train_paths,
        normalizer,
        camera_names=args.camera_names,
        image_size=config["image_size"],
        sample_stride=config["training_sample_stride"],
    )
    validation_dataset = OriginalDiffusionDataset(
        validation_paths,
        normalizer,
        camera_names=args.camera_names,
        image_size=config["image_size"],
        sample_stride=config["training_sample_stride"],
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    validation_indices = np.random.RandomState(VALIDATION_SUBSET_SEED).choice(
        len(validation_dataset),
        size=min(len(validation_dataset), args.validation_batches * args.batch_size),
        replace=False,
    )
    validation_loader = DataLoader(
        torch.utils.data.Subset(validation_dataset, validation_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OriginalDiffusionPolicy(config).to(device)
    ema = ModelEMA(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.95, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_multiplier(step, args.updates, args.warmup_updates),
    )
    iterator = iter(train_loader)
    best = None
    history = []
    milestone_results = []
    milestone_updates = set(args.eval_updates)
    for update in range(1, args.updates + 1):
        try:
            image, qpos, actions, is_pad = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, qpos, actions, is_pad = next(iterator)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(
            qpos.to(device), image.to(device), actions.to(device), is_pad.to(device)
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at update {update}")
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update(model)
        if update == 1 or update % args.validation_every == 0 or update == args.updates:
            validation = deterministic_validation(
                ema.model, validation_loader, device, seed=VALIDATION_NOISE_SEED
            )
            record = {
                "update": update,
                "train_loss": float(loss.detach().cpu()),
                "validation_loss": validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            checkpoint = args.output_dir / f"policy_update_{update:06d}.pt"
            if update % args.checkpoint_every == 0 or update == args.updates:
                _checkpoint(checkpoint, model, ema, optimizer, scheduler, normalizer, config, update, validation)
            if best is None or validation < best[0]:
                best = (validation, update)
                _checkpoint(args.output_dir / "policy_best.pt", model, ema, optimizer, scheduler, normalizer, config, update, validation)
            _atomic_json(args.output_dir / "history.json", history)
        if args.eval_task and update in milestone_updates:
            milestone_checkpoint = args.output_dir / f"policy_update_{update:06d}.pt"
            if not milestone_checkpoint.exists():
                _checkpoint(
                    milestone_checkpoint,
                    model,
                    ema,
                    optimizer,
                    scheduler,
                    normalizer,
                    config,
                    update,
                    validation,
                )
            result = _evaluate_milestone(
                args.eval_task,
                ema.model,
                normalizer,
                device,
                update,
                args.eval_episodes,
                args.eval_seed_base,
                config["camera_names"],
                args.eval_output_dir,
                checkpoint=milestone_checkpoint,
            )
            milestone_results.append(result)
            _atomic_json(
                args.eval_output_dir / "progress.json",
                {
                    "schema": "original-diffusion-inline-training-eval-v1",
                    "task": args.eval_task,
                    "seed_base": args.eval_seed_base,
                    "episodes_per_update": args.eval_episodes,
                    "requested_updates": list(args.eval_updates),
                    "results": milestone_results,
                },
            )
            print(json.dumps({"milestone_evaluation": result}), flush=True)

    _checkpoint(
        args.output_dir / "policy_last.pt",
        model,
        ema,
        optimizer,
        scheduler,
        normalizer,
        config,
        args.updates,
        history[-1]["validation_loss"],
    )
    _atomic_json(
        args.output_dir / "training_complete.json",
        {
            "schema": "original-diffusion-training-complete-v1",
            "updates": args.updates,
            "best_validation_loss": best[0],
            "best_update": best[1],
            "split": split,
            "config": config,
            "seeds": {
                "training": args.seed,
                "loader": args.seed,
                "split": SPLIT_SEED,
                "validation_subset": VALIDATION_SUBSET_SEED,
                "validation_noise": VALIDATION_NOISE_SEED,
            },
            "milestone_evaluation": {
                "task": args.eval_task,
                "seed_base": args.eval_seed_base,
                "episodes_per_update": args.eval_episodes if args.eval_task else 0,
                "requested_updates": list(args.eval_updates),
                "completed_updates": [item["update"] for item in milestone_results],
            },
        },
    )


if __name__ == "__main__":
    main()
