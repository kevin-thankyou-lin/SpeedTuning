#!/usr/bin/env python3
"""Evaluate one-reference phase transfer on a larger successful-video bank."""

from __future__ import annotations

import argparse
import json
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
from probe_xirl_tcc import project_numpy, train_projector


def fit_head(features, augmented_features, labels, *, seed, regularization):
    from sklearn.linear_model import LogisticRegression

    train_x = [temporal_features(features)]
    train_y = [labels]
    for values in augmented_features:
        train_x.append(temporal_features(values))
        train_y.append(labels)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        C=regularization,
    )
    classifier.fit(np.concatenate(train_x), np.concatenate(train_y))
    return classifier


def aggregate_method(items, labels):
    return metrics(
        np.concatenate([item["true"] for item in items]),
        np.concatenate([item["pred"] for item in items]),
        labels,
    )


def evaluate_split(
    seeds,
    *,
    episodes,
    embeddings,
    projected,
    reference_seed,
    reference_labels,
    raw_classifier,
    tcc_classifier,
    labels,
    max_advance,
):
    method_names = (
        "resnet_correspondence",
        "resnet_linear_temporal_head",
        "tcc_correspondence",
        "tcc_linear_temporal_head",
    )
    collected = {name: [] for name in method_names}
    per_seed = {}
    for seed in seeds:
        true_labels = np.asarray(
            [item["oracle_label"] for item in episodes[seed]]
        )
        raw_correspondence, raw_indices = correspondence_predict(
            embeddings[reference_seed],
            reference_labels,
            embeddings[seed],
            max_advance,
        )
        tcc_correspondence, tcc_indices = correspondence_predict(
            projected[reference_seed],
            reference_labels,
            projected[seed],
            max_advance,
        )
        predictions = {
            "resnet_correspondence": raw_correspondence,
            "resnet_linear_temporal_head": raw_classifier.predict(
                temporal_features(embeddings[seed])
            ),
            "tcc_correspondence": tcc_correspondence,
            "tcc_linear_temporal_head": tcc_classifier.predict(
                temporal_features(projected[seed])
            ),
        }
        per_seed[str(seed)] = {}
        for name, prediction in predictions.items():
            per_seed[str(seed)][name] = {
                **metrics(true_labels, prediction, labels),
                "entry": entry_errors(true_labels, prediction, labels),
            }
            collected[name].append({"true": true_labels, "pred": prediction})
        per_seed[str(seed)]["resnet_correspondence"][
            "final_reference_index"
        ] = int(raw_indices[-1])
        per_seed[str(seed)]["tcc_correspondence"][
            "final_reference_index"
        ] = int(tcc_indices[-1])
    return {
        "seeds": seeds,
        "per_seed": per_seed,
        "aggregate": {
            name: aggregate_method(collected[name], labels) for name in method_names
        },
    }


def run(args):
    import torch

    manifest, all_episodes = load_records(args.dataset)
    manifest_by_seed = {int(item["seed"]): item for item in manifest["episodes"]}
    successful_seeds = [
        int(item["seed"]) for item in manifest["episodes"] if item["success"]
    ]
    required = 1 + args.train_videos + args.validation_videos + args.final_videos
    if len(successful_seeds) < required:
        raise RuntimeError(
            f"need {required} successful videos, found {len(successful_seeds)}"
        )
    selected = successful_seeds[:required]
    reference_seed = selected[0]
    tcc_train_seeds = selected[: 1 + args.train_videos]
    validation_start = 1 + args.train_videos
    validation_seeds = selected[
        validation_start : validation_start + args.validation_videos
    ]
    final_seeds = selected[validation_start + args.validation_videos :]
    episodes = {seed: all_episodes[seed] for seed in selected}
    labels = sorted(
        {record["oracle_label"] for records in episodes.values() for record in records}
    )

    torch.manual_seed(args.seed)
    model, preprocess = build_encoder("resnet18", args.vip_root, args.device)
    embeddings = {}
    for index, seed in enumerate(selected):
        paths = [args.dataset / item["image"] for item in episodes[seed]]
        embeddings[seed] = encode_images(
            model, preprocess, paths, args.device, args.batch_size
        )
        print(
            json.dumps(
                {"encoded_seed": seed, "encoded": index + 1, "total": len(selected)}
            ),
            flush=True,
        )

    reference_paths = [
        args.dataset / item["image"] for item in episodes[reference_seed]
    ]
    augmented_reference = []
    for augmentation in range(args.augmentations):
        augmentation_seed = args.seed + 1000 + augmentation
        torch.manual_seed(augmentation_seed)
        augmented_reference.append(
            encode_images(
                model,
                augment_transform(augmentation_seed),
                reference_paths,
                args.device,
                args.batch_size,
            )
        )
    del model

    projector, training = train_projector(
        embeddings,
        tcc_train_seeds,
        device=args.device,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        seed=args.seed,
        pairs_per_epoch=args.pairs_per_epoch,
    )
    projected = {
        seed: project_numpy(projector, values, args.device)
        for seed, values in embeddings.items()
    }
    projected_augmentations = [
        project_numpy(projector, values, args.device)
        for values in augmented_reference
    ]
    reference_labels = np.asarray(
        [item["oracle_label"] for item in episodes[reference_seed]]
    )
    raw_classifier = fit_head(
        embeddings[reference_seed],
        augmented_reference,
        reference_labels,
        seed=args.seed,
        regularization=args.regularization,
    )
    tcc_classifier = fit_head(
        projected[reference_seed],
        projected_augmentations,
        reference_labels,
        seed=args.seed,
        regularization=args.regularization,
    )
    validation = evaluate_split(
        validation_seeds,
        episodes=episodes,
        embeddings=embeddings,
        projected=projected,
        reference_seed=reference_seed,
        reference_labels=reference_labels,
        raw_classifier=raw_classifier,
        tcc_classifier=tcc_classifier,
        labels=labels,
        max_advance=args.max_advance,
    )
    final = evaluate_split(
        final_seeds,
        episodes=episodes,
        embeddings=embeddings,
        projected=projected,
        reference_seed=reference_seed,
        reference_labels=reference_labels,
        raw_classifier=raw_classifier,
        tcc_classifier=tcc_classifier,
        labels=labels,
        max_advance=args.max_advance,
    )
    result = {
        "schema": "speedtuning-offline-xirl-bank-v1",
        "method_scope": (
            "XIRL-style TCC projection over frozen ImageNet ResNet18 embeddings; "
            "not a full XIRL reward-learning reproduction"
        ),
        "task": manifest["task"],
        "controller_sha256": manifest["controller_sha256"],
        "runtime_inputs": [
            manifest["camera"] + " camera frames",
            "short embedding history for linear temporal heads",
        ],
        "runtime_privileged_signals": False,
        "offline_oracle_labels": True,
        "tcc_phase_labels_used": False,
        "attempted_rollouts": len(manifest["episodes"]),
        "successful_rollouts": len(successful_seeds),
        "failed_attempt_seeds": [
            seed for seed, item in manifest_by_seed.items() if not item["success"]
        ],
        "reference_seed": reference_seed,
        "tcc_train_seeds": tcc_train_seeds,
        "validation_seeds": validation_seeds,
        "final_seeds": final_seeds,
        "unused_success_seeds": successful_seeds[required:],
        "split_counts": {
            "labelled_reference": 1,
            "additional_unlabelled_tcc_train": args.train_videos,
            "validation": args.validation_videos,
            "final": args.final_videos,
        },
        "labels": labels,
        "training": training,
        "hyperparameters": {
            "epochs": args.epochs,
            "pairs_per_epoch": args.pairs_per_epoch,
            "hidden_dim": args.hidden_dim,
            "output_dim": args.output_dim,
            "learning_rate": args.learning_rate,
            "temperature": args.temperature,
            "regularization": args.regularization,
            "augmentations": args.augmentations,
            "seed": args.seed,
        },
        "validation": validation,
        "final": final,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "task": manifest["task"],
                "successes": len(successful_seeds),
                "validation": validation["aggregate"],
                "final": final["aggregate"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--vip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--max-advance", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--pairs-per-epoch", type=int, default=12)
    parser.add_argument("--train-videos", type=int, default=12)
    parser.add_argument("--validation-videos", type=int, default=5)
    parser.add_argument("--final-videos", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(
        args.train_videos,
        args.validation_videos,
        args.final_videos,
        args.pairs_per_epoch,
    ) <= 0:
        parser.error("video counts and pairs-per-epoch must be positive")
    return args


def main():
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
