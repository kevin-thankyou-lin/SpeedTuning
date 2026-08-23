"""Train the public-repository ACT configuration on one simulated task."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_original_act import rollout
from original_act import (
    OriginalACTDataset,
    create_original_act_policy,
    episode_paths,
    fit_original_act_stats,
    set_seed,
    split_original_act_episodes,
)


def _mean_dict(values):
    return {key: float(np.mean([float(item[key]) for item in values])) for key in values[0]}


def _run_loader(policy, loader, device, optimizer=None):
    values = []
    for image, qpos, action, is_pad in loader:
        result = policy(
            qpos.to(device), image.to(device), action.to(device), is_pad.to(device)
        )
        if optimizer is not None:
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()
        values.append({key: value.detach().cpu() for key, value in result.items()})
    return _mean_dict(values)


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _evaluate_milestone(task, policy, stats, device, epoch, episodes, seed_base, output_dir):
    policy.eval()
    records = [
        rollout(task, policy, stats, device, seed_base + index)
        for index in range(episodes)
    ]
    result = {
        "epoch": epoch,
        "episodes": episodes,
        "successes": sum(item["success"] for item in records),
        "success_rate": float(np.mean([item["success"] for item in records])),
        "seed_base": seed_base,
        "rollouts": records,
    }
    _atomic_json(output_dir / f"epoch-{epoch:04d}.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-episodes", type=int)
    parser.add_argument("--eval-task")
    parser.add_argument("--eval-output-dir", type=Path)
    parser.add_argument("--eval-seed-base", type=int)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-epochs", type=int, nargs="+", default=(500, 1000, 1500, 1900))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Upstream sets seed 1 before data loading, then the requested seed for training.
    set_seed(1)
    paths = episode_paths(args.dataset_dir)
    train_paths, validation_paths = split_original_act_episodes(
        paths, seed=1, validation_count=args.validation_episodes
    )
    stats = fit_original_act_stats(paths)
    with (args.output_dir / "dataset_stats.pkl").open("wb") as stream:
        pickle.dump(stats, stream)
    (args.output_dir / "split.json").write_text(
        json.dumps(
            {
                "train": [str(path) for path in train_paths],
                "validation": [str(path) for path in validation_paths],
                "train_episodes": len(train_paths),
                "validation_episodes": len(validation_paths),
                "normalization_fit": "all episodes, matching upstream ACT utils.py",
            },
            indent=2,
        )
        + "\n"
    )
    train_loader = DataLoader(
        OriginalACTDataset(train_paths, stats),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    validation_loader = DataLoader(
        OriginalACTDataset(validation_paths, stats),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, config = create_original_act_policy(device)
    policy.to(device)
    optimizer = policy.configure_optimizers()
    best = None
    history = []
    milestone_results = []
    if args.eval_task:
        if args.eval_output_dir is None or args.eval_seed_base is None:
            parser.error("--eval-task requires --eval-output-dir and --eval-seed-base")
        args.eval_output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.num_epochs):
        policy.eval()
        with torch.inference_mode():
            validation = _run_loader(policy, validation_loader, device)
        if best is None or validation["loss"] < best[1]:
            best = (epoch, validation["loss"], copy.deepcopy(policy.state_dict()))
        policy.train()
        train = _run_loader(policy, train_loader, device, optimizer)
        record = {"epoch": epoch, "train": train, "validation": validation}
        history.append(record)
        print(json.dumps(record), flush=True)
        if epoch % 100 == 0:
            torch.save(policy.state_dict(), args.output_dir / f"policy_epoch_{epoch}_seed_{args.seed}.ckpt")
        if args.eval_task and epoch in args.eval_epochs:
            checkpoint = args.output_dir / f"policy_epoch_{epoch}_seed_{args.seed}.ckpt"
            if not checkpoint.exists():
                torch.save(policy.state_dict(), checkpoint)
            milestone = _evaluate_milestone(
                args.eval_task,
                policy,
                stats,
                device,
                epoch,
                args.eval_episodes,
                args.eval_seed_base,
                args.eval_output_dir,
            )
            milestone_results.append(milestone)
            _atomic_json(
                args.eval_output_dir / "progress.json",
                {
                    "schema": "original-act-inline-training-eval-v1",
                    "task": args.eval_task,
                    "results": milestone_results,
                },
            )
            print(json.dumps({"milestone_evaluation": milestone}), flush=True)

    torch.save(policy.state_dict(), args.output_dir / "policy_last.ckpt")
    torch.save(best[2], args.output_dir / "policy_best.ckpt")
    torch.save(best[2], args.output_dir / f"policy_epoch_{best[0]}_seed_{args.seed}.ckpt")
    (args.output_dir / "policy_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (args.output_dir / "training_complete.json").write_text(
        json.dumps(
            {
                "num_epochs": args.num_epochs,
                "batch_size": args.batch_size,
                "best_epoch": best[0],
                "best_validation_loss": best[1],
                "optimizer_updates": args.num_epochs * len(train_loader),
                "dataset_episodes": len(paths),
                "train_episodes": len(train_paths),
                "validation_episodes": len(validation_paths),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
