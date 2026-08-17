#!/usr/bin/env python3
"""Train conservative phase heads on frozen RN18 and predicted-action intent."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_phase_correspondence import augment_transform, load_records, metrics  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_encoder(device: str):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights.transforms(), weights


def encode_paths(model, transform, paths: list[Path], device: str, batch_size: int) -> np.ndarray:
    import torch

    values = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in paths[start : start + batch_size]]
        ).to(device)
        with torch.inference_mode():
            values.append(model(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def temporal_features(values: np.ndarray) -> np.ndarray:
    previous_1 = np.concatenate([values[:1], values[:-1]])
    previous_3 = np.concatenate([np.repeat(values[:1], 3, axis=0), values[:-3]])
    return np.concatenate([values, values - previous_1, values - previous_3], axis=1)


def warp_sequence(values: np.ndarray, labels: np.ndarray, factor: int):
    indices = np.arange(0, len(values), factor, dtype=int)
    return values[indices], labels[indices]


def compose_features(method: str, visual: np.ndarray, action: np.ndarray) -> np.ndarray:
    visual_temporal = temporal_features(visual)
    action_temporal = temporal_features(action)
    if method == "visual":
        return visual_temporal
    if method == "action":
        return action_temporal
    if method == "fused":
        return np.concatenate([visual_temporal, action_temporal], axis=1)
    raise ValueError(method)


def warped_features(
    method: str,
    visual: np.ndarray,
    action: np.ndarray,
    labels: np.ndarray,
    factor: int,
):
    warped_visual, warped_labels = warp_sequence(visual, labels, factor)
    warped_action, action_labels = warp_sequence(action, labels, factor)
    if not np.array_equal(warped_labels, action_labels):
        raise RuntimeError("visual/action warp labels diverged")
    return compose_features(method, warped_visual, warped_action), warped_labels


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_risk = y_true != "fast"
    pred_risk = y_pred != "fast"
    return {
        "frames": int(len(y_true)),
        "protected_frames": int(np.sum(true_risk)),
        "protected_recall": None if not np.any(true_risk) else float(np.mean(pred_risk[true_risk])),
        "false_fast_rate": None if not np.any(true_risk) else float(np.mean(~pred_risk[true_risk])),
        "false_slow_rate": None if np.all(true_risk) else float(np.mean(pred_risk[~true_risk])),
    }


def conservative_decode(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    risk_threshold: float,
    exit_threshold: float,
    exit_stability: int,
) -> np.ndarray:
    fast_index = int(np.flatnonzero(classes == "fast")[0])
    risk_indices = np.flatnonzero(classes != "fast")
    output = []
    active = None
    fast_streak = 0
    for row in probabilities:
        risk_probability = float(row[risk_indices].sum())
        best_risk = int(risk_indices[np.argmax(row[risk_indices])])
        if active is None:
            if risk_probability >= risk_threshold:
                active = best_risk
                fast_streak = 0
        else:
            if row[fast_index] >= exit_threshold:
                fast_streak += 1
            else:
                fast_streak = 0
                if risk_probability >= risk_threshold:
                    active = best_risk
            if fast_streak >= exit_stability:
                active = None
                fast_streak = 0
        output.append("fast" if active is None else classes[active])
    return np.asarray(output)


def concatenate_sequences(items):
    return np.concatenate([item for item in items], axis=0)


def fit_model(x: np.ndarray, y: np.ndarray, labels: list[str], seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class_weight = {label: (1.0 if label == "fast" else 4.0) for label in labels}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight=class_weight,
            max_iter=3000,
            random_state=seed,
        ),
    )
    model.fit(x, y)
    return model


def tune_decoder(model, sequences, labels):
    candidates = []
    for threshold in (0.15, 0.25, 0.35, 0.45):
        for exit_threshold in (0.6, 0.75, 0.9):
            for stability in (1, 2, 3, 5):
                truth = []
                prediction = []
                for x, y in sequences:
                    pred = conservative_decode(
                        model.predict_proba(x),
                        model.classes_,
                        risk_threshold=threshold,
                        exit_threshold=exit_threshold,
                        exit_stability=stability,
                    )
                    truth.append(y)
                    prediction.append(pred)
                score = binary_metrics(concatenate_sequences(truth), concatenate_sequences(prediction))
                candidates.append(
                    {
                        "risk_threshold": threshold,
                        "exit_threshold": exit_threshold,
                        "exit_stability": stability,
                        **score,
                    }
                )
    eligible = [item for item in candidates if item["protected_recall"] >= 0.99]
    if eligible:
        selected = min(eligible, key=lambda item: (item["false_slow_rate"], -item["risk_threshold"], item["exit_stability"]))
    else:
        selected = min(candidates, key=lambda item: (-item["protected_recall"], item["false_slow_rate"]))
    return selected, candidates


def evaluate_model(model, sequences, decoder, labels):
    truth = []
    raw = []
    conservative = []
    by_speed = {}
    for seed, speed, x, y in sequences:
        raw_pred = model.predict(x)
        conservative_pred = conservative_decode(
            model.predict_proba(x),
            model.classes_,
            risk_threshold=decoder["risk_threshold"],
            exit_threshold=decoder["exit_threshold"],
            exit_stability=decoder["exit_stability"],
        )
        truth.append(y)
        raw.append(raw_pred)
        conservative.append(conservative_pred)
        by_speed.setdefault(str(speed), {"truth": [], "prediction": []})
        by_speed[str(speed)]["truth"].append(y)
        by_speed[str(speed)]["prediction"].append(conservative_pred)
    y_true = concatenate_sequences(truth)
    raw_pred = concatenate_sequences(raw)
    conservative_pred = concatenate_sequences(conservative)
    return {
        "raw_multiclass": metrics(y_true, raw_pred, labels),
        "conservative_multiclass": metrics(y_true, conservative_pred, labels),
        "conservative_binary": binary_metrics(y_true, conservative_pred),
        "conservative_by_speed": {
            speed: binary_metrics(
                concatenate_sequences(item["truth"]),
                concatenate_sequences(item["prediction"]),
            )
            for speed, item in by_speed.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--action-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-videos", type=int, default=8)
    parser.add_argument("--validation-videos", type=int, default=4)
    parser.add_argument("--final-videos", type=int, default=10)
    parser.add_argument("--augmentations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    manifest, episodes = load_records(args.dataset)
    successful = [int(item["seed"]) for item in manifest["episodes"] if item["success"]]
    required = args.train_videos + args.validation_videos + args.final_videos
    if len(successful) < required:
        raise RuntimeError(f"need {required} successful trajectories, found {len(successful)}")
    train_seeds = successful[: args.train_videos]
    validation_seeds = successful[args.train_videos : args.train_videos + args.validation_videos]
    final_seeds = successful[args.train_videos + args.validation_videos : required]
    labels = sorted({record["oracle_label"] for seed in successful for record in episodes[seed]})

    model, preprocess, weights = build_encoder(args.device)
    action_bank = np.load(args.action_features)
    embeddings = {}
    started = time.perf_counter()
    for seed in train_seeds + validation_seeds + final_seeds:
        paths = [args.dataset / record["image"] for record in episodes[seed]]
        embeddings[seed] = encode_paths(model, preprocess, paths, args.device, args.batch_size)

    # Mild visual augmentation is restricted to training trajectories.
    augmented_embeddings = {}
    for augmentation in range(args.augmentations):
        transform = augment_transform(args.seed + augmentation)
        for seed in train_seeds:
            paths = [args.dataset / record["image"] for record in episodes[seed]]
            augmented_embeddings[(seed, augmentation)] = encode_paths(
                model, transform, paths, args.device, args.batch_size
            )

    feature_banks = {}
    for seed in train_seeds + validation_seeds + final_seeds:
        feature_banks[seed] = {
            "visual": embeddings[seed],
            "action": action_bank[str(seed)],
        }

    results = {}
    args.output.mkdir(parents=True, exist_ok=False)
    for method in ("visual", "action", "fused"):
        train_x = []
        train_y = []
        for seed in train_seeds:
            y = np.asarray([record["oracle_label"] for record in episodes[seed]])
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    y,
                    speed,
                )
                train_x.append(x)
                train_y.append(warped_y)
            if method in ("visual", "fused"):
                for augmentation in range(args.augmentations):
                    augmented = compose_features(
                        method,
                        augmented_embeddings[(seed, augmentation)],
                        action_bank[str(seed)],
                    )
                    train_x.append(augmented)
                    train_y.append(y)
        classifier = fit_model(concatenate_sequences(train_x), concatenate_sequences(train_y), labels, args.seed)

        validation_sequences = []
        for seed in validation_seeds:
            y = np.asarray([record["oracle_label"] for record in episodes[seed]])
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    y,
                    speed,
                )
                validation_sequences.append((x, warped_y))
        decoder, candidates = tune_decoder(classifier, validation_sequences, labels)

        final_sequences = []
        for seed in final_seeds:
            y = np.asarray([record["oracle_label"] for record in episodes[seed]])
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    y,
                    speed,
                )
                final_sequences.append((seed, speed, x, warped_y))
        final = evaluate_model(classifier, final_sequences, decoder, labels)
        model_path = args.output / f"{method}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(classifier, handle)
        results[method] = {
            "validation_decoder": decoder,
            "validation_candidates": candidates,
            "final": final,
            "model": model_path.name,
            "model_sha256": sha256(model_path),
        }

    result = {
        "schema": "speedtuning-supervised-phase-intent-v1",
        "task": manifest["task"],
        "frozen_backbone": "ImageNet ResNet18",
        "backbone_trainable": False,
        "phase_head": "class-weighted multinomial logistic regression",
        "action_feature": "causal predicted scripted-policy chunk",
        "synthetic_speed_warps": [1, 2, 3],
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "final_seeds": final_seeds,
        "labels": labels,
        "results": results,
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "action_features_sha256": sha256(args.action_features),
        "elapsed_seconds": time.perf_counter() - started,
        "acceptance_gate": {
            "protected_recall_min": 0.99,
            "false_fast_rate_max": 0.01,
            "evaluated_on": "10 untouched trajectories at 1x, 2x, and 3x subsampling",
        },
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE").write_text(f"{sha256(result_path)}  results.json\n")
    print(
        json.dumps(
            {
                "task": manifest["task"],
                "final": {
                    method: item["final"]["conservative_binary"]
                    for method, item in results.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
