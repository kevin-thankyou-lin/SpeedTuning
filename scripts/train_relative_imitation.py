"""Train ACT or Diffusion on one task's relative-joint dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from relative_imitation import create_policy, prepare_datasets


def _mean_loss(model, loader, device, batches=10):
    model.eval()
    values = []
    with torch.inference_mode():
        for index, (image, qpos, action, pad) in enumerate(loader):
            values.append(
                float(model(qpos.to(device), image.to(device), action.to(device), pad.to(device))["loss"])
            )
            if index + 1 >= batches:
                break
    model.train()
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("act", "diffusion"), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--chunk-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, validation, stats = prepare_datasets(
        args.dataset_dir, args.output_dir, args.chunk_size, args.seed
    )
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    validation_loader = DataLoader(
        validation, batch_size=args.batch_size, shuffle=False, num_workers=1
    )
    model, config = create_policy(args.kind, args.chunk_size, device, args.lr)
    optimizer = model.configure_optimizers()
    iterator = iter(train_loader)
    best = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        try:
            image, qpos, action, pad = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            image, qpos, action, pad = next(iterator)
        loss = model(qpos.to(device), image.to(device), action.to(device), pad.to(device))["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 500 == 0 or step == args.steps:
            value = _mean_loss(model, validation_loader, device)
            print(json.dumps({"step": step, "train_loss": float(loss), "validation_loss": value}), flush=True)
            if value < best:
                best = value
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "kind": args.kind,
                        "policy_config": config,
                        "stats": {key: np.asarray(item) for key, item in stats.items()},
                        "step": step,
                        "validation_loss": value,
                    },
                    args.output_dir / "best.pt",
                )
    (args.output_dir / "training_complete.json").write_text(
        json.dumps({"kind": args.kind, "steps": args.steps, "best_validation_loss": best}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
