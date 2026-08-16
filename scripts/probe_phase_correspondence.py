#!/usr/bin/env python3
"""Evaluate one-reference visual phase transfer without runtime oracle state."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_records(dataset: Path):
    manifest = json.loads((dataset / "manifest.json").read_text())
    episodes = {}
    for item in manifest["episodes"]:
        records = json.loads((dataset / item["labels"]).read_text())
        episodes[int(item["seed"])] = records
    return manifest, episodes


def build_encoder(name: str, vip_root: Path, device: str):
    import torch
    import torchvision.transforms as transforms

    if name == "vip":
        sys.path.insert(0, str(vip_root))
        from vip.models.model_vip import VIP

        model = torch.nn.DataParallel(
            VIP(device=device, hidden_dim=1024, size=50)
        )
        checkpoint_path = Path.home() / ".vip" / "resnet50" / "model.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if not checkpoint_path.exists():
            torch.hub.download_url_to_file(
                "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt",
                str(checkpoint_path),
                progress=True,
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["vip"])
        preprocess = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )
    elif name == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        preprocess = weights.transforms()
    else:
        raise ValueError(name)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, preprocess


def augment_transform(seed: int):
    import torchvision.transforms as transforms

    random.seed(seed)
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.02),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.25),
            transforms.ToTensor(),
        ]
    )


def encode_images(model, transform, paths, device, batch_size=32, vip=False):
    import torch

    outputs = []
    for start in range(0, len(paths), batch_size):
        tensors = [transform(Image.open(path).convert("RGB")) for path in paths[start : start + batch_size]]
        batch = torch.stack(tensors).to(device)
        if vip:
            batch = batch * 255.0
        with torch.inference_mode():
            embedding = model(batch)
        outputs.append(embedding.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def temporal_features(embeddings: np.ndarray) -> np.ndarray:
    previous_1 = np.concatenate([embeddings[:1], embeddings[:-1]], axis=0)
    previous_4 = np.concatenate([np.repeat(embeddings[:1], 4, axis=0), embeddings[:-4]], axis=0)
    return np.concatenate([embeddings, embeddings - previous_1, embeddings - previous_4], axis=1)


def correspondence_predict(reference, reference_labels, query, max_advance=4):
    predictions = []
    indices = []
    previous = 0
    for embedding in query:
        stop = min(len(reference), previous + max_advance + 1)
        choices = reference[previous:stop]
        distances = np.linalg.norm(choices - embedding[None, :], axis=1)
        previous += int(np.argmin(distances))
        indices.append(previous)
        predictions.append(reference_labels[previous])
    return np.asarray(predictions), indices


def metrics(y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    true_risk = y_true != "fast"
    pred_risk = y_pred != "fast"
    return {
        "frames": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "protected_recall": float(np.mean(pred_risk[true_risk])) if np.any(true_risk) else None,
        "false_fast_rate": float(np.mean(~pred_risk[true_risk])) if np.any(true_risk) else None,
        "false_slow_rate": float(np.mean(pred_risk[~true_risk])) if np.any(~true_risk) else None,
        "confusion": {
            true_label: {
                pred_label: int(np.sum((y_true == true_label) & (y_pred == pred_label)))
                for pred_label in labels
            }
            for true_label in labels
        },
    }


def entry_errors(y_true, y_pred, labels):
    result = {}
    for label in labels:
        if label == "fast":
            continue
        true_indices = np.flatnonzero(y_true == label)
        pred_indices = np.flatnonzero(y_pred == label)
        result[label] = {
            "true_entry_frame": None if not len(true_indices) else int(true_indices[0]),
            "predicted_entry_frame": None if not len(pred_indices) else int(pred_indices[0]),
            "entry_error_frames": (
                None
                if not len(true_indices) or not len(pred_indices)
                else int(pred_indices[0] - true_indices[0])
            ),
        }
    return result


def run(args):
    import torch
    from sklearn.linear_model import LogisticRegression

    manifest, episodes = load_records(args.dataset)
    train_seed = int(manifest["train_seed"])
    labels = sorted({record["oracle_label"] for records in episodes.values() for record in records})
    model, preprocess = build_encoder(args.encoder, args.vip_root, args.device)

    embeddings = {}
    for seed, records in episodes.items():
        paths = [args.dataset / record["image"] for record in records]
        embeddings[seed] = encode_images(
            model, preprocess, paths, args.device, args.batch_size, vip=args.encoder == "vip"
        )

    train_records = episodes[train_seed]
    train_labels = np.asarray([item["oracle_label"] for item in train_records])
    augmented_x = [temporal_features(embeddings[train_seed])]
    augmented_y = [train_labels]
    train_paths = [args.dataset / item["image"] for item in train_records]
    for augmentation in range(args.augmentations):
        transform = augment_transform(args.seed + augmentation)
        encoded = encode_images(
            model, transform, train_paths, args.device, args.batch_size, vip=args.encoder == "vip"
        )
        augmented_x.append(temporal_features(encoded))
        augmented_y.append(train_labels)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=args.seed,
        C=args.regularization,
    )
    classifier.fit(np.concatenate(augmented_x), np.concatenate(augmented_y))

    methods = {"correspondence": {}, "linear_temporal_head": {}}
    aggregate = {name: {"true": [], "pred": []} for name in methods}
    for seed in manifest["held_out_seeds"]:
        seed = int(seed)
        y_true = np.asarray([item["oracle_label"] for item in episodes[seed]])
        corr_pred, corr_indices = correspondence_predict(
            embeddings[train_seed], train_labels, embeddings[seed], args.max_advance
        )
        head_pred = classifier.predict(temporal_features(embeddings[seed]))
        for name, prediction in (
            ("correspondence", corr_pred),
            ("linear_temporal_head", head_pred),
        ):
            methods[name][str(seed)] = {
                **metrics(y_true, prediction, labels),
                "entry": entry_errors(y_true, prediction, labels),
            }
            aggregate[name]["true"].append(y_true)
            aggregate[name]["pred"].append(prediction)
        methods["correspondence"][str(seed)]["final_reference_index"] = int(corr_indices[-1])

    for name in methods:
        y_true = np.concatenate(aggregate[name]["true"])
        y_pred = np.concatenate(aggregate[name]["pred"])
        methods[name]["aggregate"] = {
            **metrics(y_true, y_pred, labels),
            "entry_by_seed": {
                seed: methods[name][seed]["entry"]
                for seed in methods[name]
                if seed != "aggregate"
            },
        }

    result = {
        "schema": "speedtuning-offline-phase-probe-v1",
        "task": manifest["task"],
        "encoder": args.encoder,
        "train_seed": train_seed,
        "held_out_seeds": manifest["held_out_seeds"],
        "runtime_inputs": [manifest["camera"] + " camera frames", "short embedding history"],
        "runtime_privileged_signals": False,
        "offline_oracle_labels": True,
        "augmentations_per_reference_frame": args.augmentations,
        "augmentation_policy": "crop color-jitter mild-blur; no horizontal flip",
        "labels": labels,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--encoder", choices=("vip", "resnet18"), required=True)
    parser.add_argument("--vip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmentations", type=int, default=3)
    parser.add_argument("--max-advance", type=int, default=4)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
