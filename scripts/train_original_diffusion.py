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
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = episode_paths(args.dataset_dir)
    if len(paths) != 270:
        raise ValueError(f"expected the exact 270 ACT episodes, found {len(paths)}")
    train_paths, validation_paths = split_original_act_episodes(
        paths, seed=1, validation_count=args.validation_episodes
    )
    if (len(train_paths), len(validation_paths)) != (250, 20):
        raise ValueError("matched experiment requires exactly 250 train and 20 validation episodes")
    normalizer = JointRangeNormalizer.fit(train_paths)
    split = {
        "schema": "original-diffusion-split-v1",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_identity": _path_identity(paths),
        "train": [path.name for path in train_paths],
        "validation": [path.name for path in validation_paths],
        "train_episodes": len(train_paths),
        "validation_episodes": len(validation_paths),
        "normalization_fit": "train episodes only; one global range per joint",
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
    }
    train_dataset = OriginalDiffusionDataset(
        train_paths, normalizer, camera_names=args.camera_names, image_size=config["image_size"]
    )
    validation_dataset = OriginalDiffusionDataset(
        validation_paths, normalizer, camera_names=args.camera_names, image_size=config["image_size"]
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
    validation_indices = np.random.RandomState(1).choice(
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
            validation = deterministic_validation(ema.model, validation_loader, device, seed=17)
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
        },
    )


if __name__ == "__main__":
    main()
