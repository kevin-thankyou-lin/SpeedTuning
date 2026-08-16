#!/usr/bin/env python3
"""Run a small XIRL-style temporal cycle-consistency phase probe.

This is deliberately a diagnostic rather than a full XIRL reproduction.  A
frozen visual backbone supplies per-frame features, while a small projection
head is trained without phase labels to cycle-align successful videos.  One
fixed, offline-labelled reference video then supplies phase labels for either
monotonic correspondence or a class-balanced temporal linear head.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np

from probe_phase_correspondence import (
    augment_transform,
    build_encoder,
    correspondence_predict,
    encode_images,
    entry_errors,
    load_records,
    metrics,
    temporal_features,
)


def project_numpy(projector, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.inference_mode():
        tensor = torch.from_numpy(values).to(device)
        return projector(tensor).cpu().numpy().astype(np.float32)


def cycle_classification_loss(source, target, temperature: float):
    """Soft-match source into target and require the cycle to recover source."""
    import torch
    import torch.nn.functional as functional

    source_to_target = torch.softmax(
        -torch.cdist(source, target).square() / temperature,
        dim=1,
    )
    soft_target = source_to_target @ target
    cycle_logits = -torch.cdist(soft_target, source).square() / temperature
    expected = torch.arange(source.shape[0], device=source.device)
    return functional.cross_entropy(cycle_logits, expected)


def train_projector(
    embeddings: dict[int, np.ndarray],
    train_seeds: list[int],
    *,
    device: str,
    hidden_dim: int,
    output_dim: int,
    epochs: int,
    learning_rate: float,
    temperature: float,
    seed: int,
    pairs_per_epoch: int = 0,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    input_dim = next(iter(embeddings.values())).shape[1]

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, values):
            return functional.normalize(self.layers(values), dim=-1)

    projector = Projector().to(device)
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    tensors = {
        item: torch.from_numpy(embeddings[item]).to(device) for item in train_seeds
    }
    pairs = list(combinations(train_seeds, 2))
    if pairs_per_epoch < 0:
        raise ValueError("pairs_per_epoch must be non-negative")
    random_generator = random.Random(seed)
    best_loss = float("inf")
    best_state = None
    last_loss = None
    history = []

    for epoch in range(epochs):
        projector.train()
        optimizer.zero_grad(set_to_none=True)
        pair_losses = []
        epoch_pairs = pairs
        if pairs_per_epoch and pairs_per_epoch < len(pairs):
            epoch_pairs = random_generator.sample(pairs, pairs_per_epoch)
        for first, second in epoch_pairs:
            first_projected = projector(tensors[first])
            second_projected = projector(tensors[second])
            pair_losses.append(
                cycle_classification_loss(
                    first_projected, second_projected, temperature
                )
            )
            pair_losses.append(
                cycle_classification_loss(
                    second_projected, first_projected, temperature
                )
            )
        loss = torch.stack(pair_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=5.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if last_loss < best_loss:
            best_loss = last_loss
            best_state = copy.deepcopy(projector.state_dict())
        if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == epochs:
            history.append({"epoch": epoch + 1, "loss": last_loss})

    if best_state is None:
        raise RuntimeError("TCC training produced no checkpoint")
    projector.load_state_dict(best_state)
    projector.eval()
    return projector, {
        "best_loss": best_loss,
        "last_loss": last_loss,
        "loss_history": history,
        "pair_pool_count": len(pairs),
        "pairs_per_epoch": len(epoch_pairs),
    }


def run(args):
    import torch
    from sklearn.linear_model import LogisticRegression

    manifest, episodes = load_records(args.dataset)
    reference_seed = int(manifest["train_seed"])
    held_out_seeds = [int(item) for item in manifest["held_out_seeds"]]
    all_seeds = [reference_seed, *held_out_seeds]
    labels = sorted(
        {record["oracle_label"] for records in episodes.values() for record in records}
    )

    torch.manual_seed(args.seed)
    model, preprocess = build_encoder(
        "resnet18", args.vip_root, args.device
    )
    embeddings = {}
    for seed in all_seeds:
        paths = [args.dataset / item["image"] for item in episodes[seed]]
        embeddings[seed] = encode_images(
            model, preprocess, paths, args.device, args.batch_size
        )

    reference_paths = [
        args.dataset / item["image"] for item in episodes[reference_seed]
    ]
    augmented_reference = []
    for augmentation in range(args.augmentations):
        augmentation_seed = args.seed + 1000 + augmentation
        torch.manual_seed(augmentation_seed)
        transform = augment_transform(augmentation_seed)
        augmented_reference.append(
            encode_images(
                model,
                transform,
                reference_paths,
                args.device,
                args.batch_size,
            )
        )

    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    reference_labels = np.asarray(
        [item["oracle_label"] for item in episodes[reference_seed]]
    )
    method_names = ("tcc_correspondence", "tcc_linear_temporal_head")
    folds = {}
    aggregate = {name: {"true": [], "pred": []} for name in method_names}

    for fold_index, held_out_seed in enumerate(held_out_seeds):
        train_seeds = [
            seed for seed in all_seeds if seed != held_out_seed
        ]
        projector, training = train_projector(
            embeddings,
            train_seeds,
            device=args.device,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            temperature=args.temperature,
            seed=args.seed + fold_index,
            pairs_per_epoch=args.pairs_per_epoch,
        )
        projected_reference = project_numpy(
            projector, embeddings[reference_seed], args.device
        )
        projected_augmentations = [
            project_numpy(projector, values, args.device)
            for values in augmented_reference
        ]
        train_x = [temporal_features(projected_reference)]
        train_y = [reference_labels]
        for values in projected_augmentations:
            train_x.append(temporal_features(values))
            train_y.append(reference_labels)
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
            C=args.regularization,
        )
        classifier.fit(np.concatenate(train_x), np.concatenate(train_y))

        projected_query = project_numpy(
            projector, embeddings[held_out_seed], args.device
        )
        true_labels = np.asarray(
            [item["oracle_label"] for item in episodes[held_out_seed]]
        )
        correspondence, indices = correspondence_predict(
            projected_reference,
            reference_labels,
            projected_query,
            args.max_advance,
        )
        linear = classifier.predict(temporal_features(projected_query))
        predictions = {
            "tcc_correspondence": correspondence,
            "tcc_linear_temporal_head": linear,
        }
        fold_result = {
            "held_out_seed": held_out_seed,
            "unlabelled_tcc_train_seeds": train_seeds,
            "phase_label_source_seed": reference_seed,
            "training": training,
            "methods": {},
        }
        for name, prediction in predictions.items():
            fold_result["methods"][name] = {
                **metrics(true_labels, prediction, labels),
                "entry": entry_errors(true_labels, prediction, labels),
            }
            aggregate[name]["true"].append(true_labels)
            aggregate[name]["pred"].append(prediction)
        fold_result["methods"]["tcc_correspondence"][
            "final_reference_index"
        ] = int(indices[-1])
        folds[str(held_out_seed)] = fold_result
        print(
            json.dumps(
                {
                    "completed_fold": held_out_seed,
                    "tcc_best_loss": training["best_loss"],
                    "metrics": {
                        name: fold_result["methods"][name]
                        for name in method_names
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    aggregate_metrics = {}
    for name in method_names:
        true_labels = np.concatenate(aggregate[name]["true"])
        predicted_labels = np.concatenate(aggregate[name]["pred"])
        aggregate_metrics[name] = metrics(true_labels, predicted_labels, labels)

    result = {
        "schema": "speedtuning-offline-xirl-tcc-probe-v1",
        "method_scope": (
            "XIRL-style temporal cycle consistency on a learned projection of "
            "a frozen ImageNet ResNet18; not a full XIRL reward-learning reproduction"
        ),
        "task": manifest["task"],
        "reference_seed": reference_seed,
        "held_out_seeds": held_out_seeds,
        "fold_protocol": (
            "fixed labelled reference plus three unlabelled successful videos for "
            "TCC training; fourth successful video excluded until evaluation"
        ),
        "runtime_inputs": [
            manifest["camera"] + " camera frames",
            "short embedding history for the linear temporal head",
        ],
        "runtime_privileged_signals": False,
        "offline_oracle_labels": True,
        "tcc_phase_labels_used": False,
        "augmentations_per_reference_frame": args.augmentations,
        "augmentation_policy": "crop color-jitter mild-blur; no horizontal flip",
        "labels": labels,
        "hyperparameters": {
            "epochs": args.epochs,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "regularization": args.regularization,
            "seed": args.seed,
            "pairs_per_epoch": args.pairs_per_epoch,
        },
        "folds": folds,
        "aggregate": aggregate_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"aggregate": aggregate_metrics}, sort_keys=True), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--max-advance", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--pairs-per-epoch",
        type=int,
        default=0,
        help="Random TCC video pairs per epoch; zero uses every pair.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
