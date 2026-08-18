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
from semantic_phase import (  # noqa: E402
    FuturePhaseSequencePredictor,
    future_phase_targets,
    parse_future_offsets,
)


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


def labels_for_records(records: list[dict], mode: str) -> np.ndarray:
    if mode == "oracle-risk":
        return np.asarray([record["oracle_label"] for record in records])
    if mode == "reward-phase4":
        return np.asarray(
            [f"phase_{min(4, int(float(record['task_reward'])) + 1)}" for record in records]
        )
    if mode == "semantic-phase":
        missing = [
            index
            for index, record in enumerate(records)
            if "semantic_phase_id" not in record
        ]
        if missing:
            raise ValueError(
                "semantic-phase labels require semantic_phase_id on every record; "
                f"missing indices begin with {missing[:5]}"
            )
        return np.asarray([str(record["semantic_phase_id"]) for record in records])
    raise ValueError(mode)


def ordered_phase_metrics(truth: list[np.ndarray], prediction: list[np.ndarray]) -> dict:
    y_true = concatenate_sequences(truth)
    y_pred = concatenate_sequences(prediction)
    true_index = np.asarray([int(value.rsplit("_", 1)[1]) for value in y_true])
    pred_index = np.asarray([int(value.rsplit("_", 1)[1]) for value in y_pred])
    delta = pred_index - true_index
    backward_jumps = 0
    transitions = 0
    for sequence in prediction:
        index = np.asarray([int(value.rsplit("_", 1)[1]) for value in sequence])
        if len(index) > 1:
            backward_jumps += int(np.sum(np.diff(index) < 0))
            transitions += len(index) - 1
    return {
        "frames": int(len(y_true)),
        "mean_absolute_phase_error": float(np.mean(np.abs(delta))),
        "p90_absolute_phase_error": float(np.percentile(np.abs(delta), 90)),
        "false_advance_rate": float(np.mean(delta > 0)),
        "severe_advance_rate": float(np.mean(delta >= 2)),
        "late_phase_rate": float(np.mean(delta < 0)),
        "backward_jumps": backward_jumps,
        "backward_jump_rate": 0.0 if transitions == 0 else float(backward_jumps / transitions),
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


def monotonic_phase_decode(
    probabilities: np.ndarray,
    classes: np.ndarray,
    *,
    advance_threshold: float,
    advance_stability: int,
) -> np.ndarray:
    ordered = np.asarray(sorted(classes, key=lambda value: int(value.rsplit("_", 1)[1])))
    class_index = {value: int(np.flatnonzero(classes == value)[0]) for value in ordered}
    active = 0
    advance_streak = 0
    output = []
    for row in probabilities:
        later_probability = float(sum(row[class_index[value]] for value in ordered[active + 1 :]))
        if active < len(ordered) - 1 and later_probability >= advance_threshold:
            advance_streak += 1
            if advance_streak >= advance_stability:
                active += 1
                advance_streak = 0
        else:
            advance_streak = 0
        output.append(ordered[active])
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


def tune_phase_decoder(model, sequences):
    candidates = []
    for threshold in (0.25, 0.4, 0.55, 0.7):
        for stability in (1, 2, 3):
            truth = []
            prediction = []
            for x, y in sequences:
                pred = monotonic_phase_decode(
                    model.predict_proba(x),
                    model.classes_,
                    advance_threshold=threshold,
                    advance_stability=stability,
                )
                truth.append(y)
                prediction.append(pred)
            candidates.append(
                {
                    "advance_threshold": threshold,
                    "advance_stability": stability,
                    **ordered_phase_metrics(truth, prediction),
                }
            )
    eligible = [item for item in candidates if item["false_advance_rate"] <= 0.01]
    if eligible:
        selected = min(
            eligible,
            key=lambda item: (item["mean_absolute_phase_error"], item["late_phase_rate"]),
        )
    else:
        selected = min(
            candidates,
            key=lambda item: (item["false_advance_rate"], item["mean_absolute_phase_error"]),
        )
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


def evaluate_phase_model(model, sequences, decoder, labels):
    truth = []
    raw = []
    causal = []
    by_speed = {}
    for seed, speed, x, y in sequences:
        raw_pred = model.predict(x)
        causal_pred = monotonic_phase_decode(
            model.predict_proba(x),
            model.classes_,
            advance_threshold=decoder["advance_threshold"],
            advance_stability=decoder["advance_stability"],
        )
        truth.append(y)
        raw.append(raw_pred)
        causal.append(causal_pred)
        by_speed.setdefault(str(speed), {"truth": [], "prediction": []})
        by_speed[str(speed)]["truth"].append(y)
        by_speed[str(speed)]["prediction"].append(causal_pred)
    return {
        "raw_multiclass": metrics(
            concatenate_sequences(truth), concatenate_sequences(raw), labels
        ),
        "causal_multiclass": metrics(
            concatenate_sequences(truth), concatenate_sequences(causal), labels
        ),
        "causal_ordered": ordered_phase_metrics(truth, causal),
        "causal_by_speed": {
            speed: ordered_phase_metrics(item["truth"], item["prediction"])
            for speed, item in by_speed.items()
        },
    }


def evaluate_raw_phase_model(model, sequences, labels):
    """Evaluate named semantic IDs without assuming a universal phase order."""

    truth = []
    prediction = []
    by_speed = {}
    for _seed, speed, x, y in sequences:
        pred = model.predict(x)
        truth.append(y)
        prediction.append(pred)
        by_speed.setdefault(str(speed), {"truth": [], "prediction": []})
        by_speed[str(speed)]["truth"].append(y)
        by_speed[str(speed)]["prediction"].append(pred)
    return {
        "raw_multiclass": metrics(
            concatenate_sequences(truth), concatenate_sequences(prediction), labels
        ),
        "raw_by_speed": {
            speed: metrics(
                concatenate_sequences(item["truth"]),
                concatenate_sequences(item["prediction"]),
                labels,
            )
            for speed, item in by_speed.items()
        },
    }


def evaluate_future_phase_model(model, sequences, labels):
    """Report exact semantic-ID prediction quality at every future offset."""

    truth = []
    prediction = []
    for _seed, _speed, x, y in sequences:
        truth.append(y)
        prediction.append(model.predict(x))
    y_true = concatenate_sequences(truth)
    y_pred = concatenate_sequences(prediction)
    per_offset = {}
    for column, offset in enumerate(model.offsets):
        score = metrics(y_true[:, column], y_pred[:, column], labels)
        per_offset[str(offset)] = {
            key: score[key]
            for key in (
                "frames",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "confusion",
            )
        }
    return {
        "frames": int(y_true.shape[0]),
        "offsets": list(model.offsets),
        "element_accuracy": float(np.mean(y_true == y_pred)),
        "whole_sequence_accuracy": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "per_offset": per_offset,
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
    parser.add_argument(
        "--label-mode",
        choices=("oracle-risk", "reward-phase4", "semantic-phase"),
        default="oracle-risk",
    )
    parser.add_argument(
        "--future-offsets",
        default="0,1,2,3",
        help="Nominal policy-step offsets for future semantic phase heads.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    future_offsets = parse_future_offsets(args.future_offsets)
    manifest, episodes = load_records(args.dataset)
    successful = [int(item["seed"]) for item in manifest["episodes"] if item["success"]]
    required = args.train_videos + args.validation_videos + args.final_videos
    if len(successful) < required:
        raise RuntimeError(f"need {required} successful trajectories, found {len(successful)}")
    train_seeds = successful[: args.train_videos]
    validation_seeds = successful[args.train_videos : args.train_videos + args.validation_videos]
    final_seeds = successful[args.train_videos + args.validation_videos : required]
    labels = sorted(
        {
            label
            for seed in successful
            for label in labels_for_records(episodes[seed], args.label_mode)
        }
    )
    if args.label_mode == "reward-phase4" and labels != [
        "phase_1",
        "phase_2",
        "phase_3",
        "phase_4",
    ]:
        raise RuntimeError(f"reward-phase4 requires all four phases, found {labels}")

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
            y = labels_for_records(episodes[seed], args.label_mode)
            future_y = future_phase_targets(episodes[seed], y, future_offsets)
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    future_y,
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
                    train_y.append(future_y)
        train_features = concatenate_sequences(train_x)
        train_targets = concatenate_sequences(train_y)
        phase_models = [
            fit_model(
                train_features,
                train_targets[:, column],
                labels,
                args.seed,
            )
            for column in range(len(future_offsets))
        ]
        future_model = FuturePhaseSequencePredictor(future_offsets, phase_models)
        classifier = phase_models[0]

        validation_sequences = []
        for seed in validation_seeds:
            y = labels_for_records(episodes[seed], args.label_mode)
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    y,
                    speed,
                )
                validation_sequences.append((x, warped_y))
        if args.label_mode == "reward-phase4":
            decoder, candidates = tune_phase_decoder(classifier, validation_sequences)
        elif args.label_mode == "semantic-phase":
            decoder, candidates = {"type": "raw-semantic-phase"}, []
        else:
            decoder, candidates = tune_decoder(classifier, validation_sequences, labels)

        final_sequences = []
        future_final_sequences = []
        for seed in final_seeds:
            y = labels_for_records(episodes[seed], args.label_mode)
            future_y = future_phase_targets(episodes[seed], y, future_offsets)
            for speed in (1, 2, 3):
                x, warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    y,
                    speed,
                )
                final_sequences.append((seed, speed, x, warped_y))
                future_x, future_warped_y = warped_features(
                    method,
                    feature_banks[seed]["visual"],
                    feature_banks[seed]["action"],
                    future_y,
                    speed,
                )
                future_final_sequences.append(
                    (seed, speed, future_x, future_warped_y)
                )
        if args.label_mode == "reward-phase4":
            final = evaluate_phase_model(classifier, final_sequences, decoder, labels)
        elif args.label_mode == "semantic-phase":
            final = evaluate_raw_phase_model(classifier, final_sequences, labels)
        else:
            final = evaluate_model(classifier, final_sequences, decoder, labels)
        model_path = args.output / f"{method}.pkl"
        with model_path.open("wb") as handle:
            pickle.dump(classifier, handle)
        future_model_path = args.output / f"{method}.future-phases.pkl"
        with future_model_path.open("wb") as handle:
            pickle.dump(future_model, handle)
        results[method] = {
            "validation_decoder": decoder,
            "validation_candidates": candidates,
            "final": final,
            "model": model_path.name,
            "model_sha256": sha256(model_path),
            "future_phase_model": future_model_path.name,
            "future_phase_model_sha256": sha256(future_model_path),
            "future_phase_final": evaluate_future_phase_model(
                future_model, future_final_sequences, labels
            ),
        }

    result = {
        "schema": "speedtuning-supervised-phase-intent-v3",
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
        "label_mode": args.label_mode,
        "future_phase_offsets_policy_steps": list(future_offsets),
        "label_definition": (
            "phase_k := clipped integer task reward k-1; offline privileged labels only"
            if args.label_mode == "reward-phase4"
            else (
                "stable semantic_phase_id attached to each offline record"
                if args.label_mode == "semantic-phase"
                else "existing causal-search protected segments"
            )
        ),
        "results": results,
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "action_features_sha256": sha256(args.action_features),
        "elapsed_seconds": time.perf_counter() - started,
        "acceptance_gate": (
            {
                "false_advance_rate_max": 0.01,
                "balanced_accuracy_min": 0.90,
                "evaluated_on": "10 untouched trajectories at 1x, 2x, and 3x subsampling",
            }
            if args.label_mode == "reward-phase4"
            else (
                {
                    "status": "report-only until the semantic phase ontology is frozen",
                    "reported_metrics": [
                        "balanced_accuracy",
                        "future_phase_element_accuracy",
                        "future_phase_whole_sequence_accuracy",
                    ],
                    "evaluated_on": "10 untouched trajectories at 1x, 2x, and 3x subsampling",
                }
                if args.label_mode == "semantic-phase"
                else {
                    "protected_recall_min": 0.99,
                    "false_fast_rate_max": 0.01,
                    "evaluated_on": "10 untouched trajectories at 1x, 2x, and 3x subsampling",
                }
            )
        ),
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "COMPLETE").write_text(f"{sha256(result_path)}  results.json\n")
    print(
        json.dumps(
            {
                "task": manifest["task"],
                "final": {
                    method: item["final"][
                        "causal_ordered"
                        if args.label_mode == "reward-phase4"
                        else (
                            "raw_multiclass"
                            if args.label_mode == "semantic-phase"
                            else "conservative_binary"
                        )
                    ]
                    for method, item in results.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
