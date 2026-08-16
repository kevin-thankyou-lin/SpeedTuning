#!/usr/bin/env python3
"""Benchmark causal one-reference alignment without semantic phase labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference_alignment import OnlineReferenceAligner  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(dataset: Path):
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["semantic_segment_labels_present"]:
        raise RuntimeError("benchmark requires a segment-label-free dataset")
    trajectories = {}
    for item in manifest["trajectories"]:
        records = json.loads((dataset / item["record"]).read_text())
        trajectories[item["trajectory_id"]] = {"manifest": item, "records": records}
    return manifest, trajectories


def load_landmarks(dataset: Path, filename: str) -> dict[str, dict[int, float]]:
    landmarks: dict[str, dict[int, float]] = {}
    with (dataset / filename).open(newline="") as handle:
        for row in csv.DictReader(handle):
            landmarks.setdefault(row["query_video"], {})[
                int(row["query_frame_index"])
            ] = float(row["true_reference_position"])
    return landmarks


def load_images(dataset: Path, trajectory: dict) -> np.ndarray:
    return np.stack(
        [
            np.asarray(
                Image.open(dataset / trajectory["manifest"]["trajectory_id"] / item["image"])
                .convert("RGB")
            )
            for item in trajectory["records"]
        ]
    )


def build_model(name: str, device: str):
    import torch

    if name == "rn18_temporal_pool":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
    elif name == "r3d18":
        from torchvision.models.video import R3D_18_Weights, r3d_18

        weights = R3D_18_Weights.DEFAULT
        model = r3d_18(weights=weights)
        model.fc = torch.nn.Identity()
    else:
        raise ValueError(name)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, weights


def normalize(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def encode_rn18_frames(model, weights, images, device: str, batch_size: int):
    import torch

    transform = weights.transforms()
    outputs = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        batch = torch.stack(
            [transform(Image.fromarray(image)) for image in images[start : start + batch_size]]
        ).to(device)
        with torch.inference_mode():
            outputs.append(model(batch).cpu().numpy().astype(np.float32))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return np.concatenate(outputs), time.perf_counter() - started


def temporal_pool(frame_embeddings: np.ndarray, clip_frames: int) -> np.ndarray:
    descriptors = []
    for index in range(len(frame_embeddings)):
        start = max(0, index - clip_frames + 1)
        clip = frame_embeddings[start : index + 1]
        descriptors.append(np.concatenate([clip.mean(axis=0), clip[-1], clip[-1] - clip[0]]))
    return normalize(np.asarray(descriptors, dtype=np.float32))


def _resampled_clip(images: np.ndarray, end: int, clip_frames: int, target_frames: int):
    start = max(0, end - clip_frames + 1)
    clip = images[start : end + 1]
    if len(clip) < clip_frames:
        clip = np.concatenate([np.repeat(clip[:1], clip_frames - len(clip), axis=0), clip])
    indices = np.linspace(0, len(clip) - 1, target_frames).round().astype(int)
    return clip[indices]


def encode_r3d_clips(
    model,
    weights,
    images,
    clip_frames: int,
    device: str,
    batch_size: int,
    target_frames: int = 16,
):
    import torch

    transform = weights.transforms()
    outputs = []
    started = time.perf_counter()
    for start in range(0, len(images), batch_size):
        tensors = []
        for index in range(start, min(len(images), start + batch_size)):
            clip = _resampled_clip(images, index, clip_frames, target_frames)
            tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)
            tensors.append(transform(tensor))
        batch = torch.stack(tensors).to(device)
        with torch.inference_mode():
            outputs.append(model(batch).cpu().numpy().astype(np.float32))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return normalize(np.concatenate(outputs)), time.perf_counter() - started


def alignment_metrics(true: np.ndarray, predicted: np.ndarray, catastrophic_threshold: float):
    errors = np.abs(predicted - true)
    jumps = np.diff(predicted)
    backwards = jumps[jumps < 0]
    catastrophic = jumps[np.abs(jumps) >= catastrophic_threshold]
    return {
        "frames": int(len(true)),
        "mean_normalized_absolute_error": float(np.mean(errors)),
        "median_normalized_absolute_error": float(np.median(errors)),
        "p90_normalized_absolute_error": float(np.percentile(errors, 90)),
        "backward_jump_count": int(len(backwards)),
        "backward_jump_frequency": float(len(backwards) / max(1, len(jumps))),
        "backward_jump_total_magnitude": float(-backwards.sum()),
        "backward_jump_max_magnitude": float(-backwards.min()) if len(backwards) else 0.0,
        "catastrophic_jump_threshold": catastrophic_threshold,
        "catastrophic_jump_count": int(len(catastrophic)),
        "catastrophic_jump_frequency": float(len(catastrophic) / max(1, len(jumps))),
        "catastrophic_jump_max_magnitude": float(np.max(np.abs(catastrophic)))
        if len(catastrophic)
        else 0.0,
    }


def ambiguity_metrics(similarity: np.ndarray, predictions: np.ndarray, truth: np.ndarray):
    reference_positions = np.linspace(0.0, 1.0, similarity.shape[1])
    ambiguous = []
    margins = []
    for row in similarity:
        best = int(np.argmax(row))
        far = np.abs(reference_positions - reference_positions[best]) >= 0.15
        far_best = float(np.max(row[far])) if np.any(far) else -1.0
        margin = float(row[best] - far_best)
        margins.append(margin)
        ambiguous.append(margin < 0.02)
    ambiguous = np.asarray(ambiguous)
    errors = np.abs(predictions - truth)
    return {
        "far_state_margin_mean": float(np.mean(margins)),
        "visually_ambiguous_frame_frequency": float(np.mean(ambiguous)),
        "ambiguous_frame_median_error": float(np.median(errors[ambiguous]))
        if np.any(ambiguous)
        else None,
    }


def confidence_metrics(confidence: np.ndarray, errors: np.ndarray):
    order = np.argsort(confidence)
    quartile = max(1, len(order) // 4)
    return {
        "mean": float(np.mean(confidence)),
        "lowest_confidence_quartile_mean_error": float(np.mean(errors[order[:quartile]])),
        "highest_confidence_quartile_mean_error": float(np.mean(errors[order[-quartile:]])),
        "error_confidence_correlation": float(np.corrcoef(errors, confidence)[0, 1])
        if len(errors) > 1 and np.std(confidence) > 0
        else None,
    }


def plot_query(
    output: Path,
    task: str,
    method: str,
    window: float,
    trajectory_id: str,
    times: np.ndarray,
    truth: np.ndarray,
    raw: np.ndarray,
    causal: np.ndarray,
    confidence: np.ndarray,
    similarity: np.ndarray,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slug = f"{task}-{method}-{window:.2f}s-{trajectory_id}"
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(times, truth, color="black", linewidth=2, label="ground truth")
    axes[0].plot(times, raw, color="tab:orange", alpha=0.65, label="raw nearest neighbor")
    axes[0].plot(times, causal, color="tab:blue", linewidth=1.8, label="causal posterior")
    axes[0].set(ylabel="reference position", ylim=(-0.02, 1.02), title=slug)
    axes[0].legend(loc="upper left")
    axes[0].grid(alpha=0.2)
    axes[1].plot(times, confidence, color="tab:green")
    axes[1].set(xlabel="query time (s)", ylabel="confidence", ylim=(-0.02, 1.02))
    axes[1].grid(alpha=0.2)
    progress_path = output / f"{slug}-progress.png"
    figure.savefig(progress_path, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axis.imshow(similarity, aspect="auto", origin="lower", cmap="viridis")
    axis.set(xlabel="reference clip index", ylabel="query clip index", title=slug)
    figure.colorbar(image, ax=axis, label="cosine similarity")
    similarity_path = output / f"{slug}-similarity.png"
    figure.savefig(similarity_path, dpi=150)
    plt.close(figure)
    return progress_path, similarity_path


def evaluate_method(
    args,
    manifest,
    trajectories,
    landmarks,
    method: str,
    window_seconds: float,
    features: dict[str, np.ndarray],
    encoding_seconds: float,
):
    reference_id = manifest["reference_trajectory"]
    reference = features[reference_id]
    results = {}
    plot_paths = []
    total_alignment_seconds = 0.0
    total_query_frames = 0
    for trajectory_id, trajectory in trajectories.items():
        if trajectory_id == reference_id or not trajectory["manifest"]["success"]:
            continue
        query = features[trajectory_id]
        true = np.asarray([item["reference_position"] for item in trajectory["records"]])
        times = np.asarray([item["wall_time_s"] for item in trajectory["records"]])
        aligner = OnlineReferenceAligner(
            reference,
            max_advance=args.max_advance,
            max_backtrack=args.max_backtrack,
            emission_temperature=args.temperature,
            updates_per_second=manifest["nominal_frame_rate_hz"],
        )
        started = time.perf_counter()
        outputs = [aligner.update_embedding(value) for value in query]
        total_alignment_seconds += time.perf_counter() - started
        total_query_frames += len(query)
        predicted = np.asarray([item.reference_position for item in outputs])
        raw = np.asarray([item.raw_reference_position for item in outputs])
        confidence = np.asarray([item.confidence for item in outputs])
        similarity = query @ reference.T
        dense = alignment_metrics(true, predicted, args.catastrophic_threshold)
        raw_metrics = alignment_metrics(true, raw, args.catastrophic_threshold)
        landmark_indices = sorted(landmarks[trajectory_id])
        landmark_true = np.asarray([landmarks[trajectory_id][index] for index in landmark_indices])
        landmark_predicted = predicted[landmark_indices]
        landmark_result = alignment_metrics(
            landmark_true,
            landmark_predicted,
            args.catastrophic_threshold,
        )
        errors = np.abs(predicted - true)
        results[trajectory_id] = {
            "frames": len(query),
            "dense_oracle": dense,
            "sparse_landmarks": landmark_result,
            "raw_global_nearest_neighbor": raw_metrics,
            "confidence": confidence_metrics(confidence, errors),
            "repeated_or_visually_similar_states": ambiguity_metrics(
                similarity,
                predicted,
                true,
            ),
            "final_reference_position": float(predicted[-1]),
            "ground_truth_final_position": float(true[-1]),
        }
        if not plot_paths:
            plot_paths.extend(
                plot_query(
                    args.output / "plots",
                    manifest["task"],
                    method,
                    window_seconds,
                    trajectory_id,
                    times,
                    true,
                    raw,
                    predicted,
                    confidence,
                    similarity,
                )
            )

    dense_errors = []
    raw_errors = []
    landmark_errors = []
    for trajectory_id, trajectory in trajectories.items():
        if trajectory_id == reference_id or trajectory_id not in results:
            continue
        query = features[trajectory_id]
        true = np.asarray([item["reference_position"] for item in trajectory["records"]])
        aligner = OnlineReferenceAligner(
            reference,
            max_advance=args.max_advance,
            max_backtrack=args.max_backtrack,
            emission_temperature=args.temperature,
            updates_per_second=manifest["nominal_frame_rate_hz"],
        )
        outputs = [aligner.update_embedding(value) for value in query]
        predicted = np.asarray([item.reference_position for item in outputs])
        raw = np.asarray([item.raw_reference_position for item in outputs])
        dense_errors.extend(np.abs(predicted - true))
        raw_errors.extend(np.abs(raw - true))
        for index, target in landmarks[trajectory_id].items():
            landmark_errors.append(abs(predicted[index] - target))
    return {
        "method": method,
        "window_seconds": window_seconds,
        "successful_query_trajectories": len(results),
        "failed_query_trajectories_excluded": sum(
            item["role"] == "query" and not item["success"]
            for item in (value["manifest"] for value in trajectories.values())
        ),
        "aggregate": {
            "mean_normalized_absolute_error": float(np.mean(dense_errors)),
            "median_normalized_absolute_error": float(np.median(dense_errors)),
            "p90_normalized_absolute_error": float(np.percentile(dense_errors, 90)),
            "raw_mean_normalized_absolute_error": float(np.mean(raw_errors)),
            "raw_median_normalized_absolute_error": float(np.median(raw_errors)),
            "landmark_mean_normalized_absolute_error": float(np.mean(landmark_errors)),
            "landmark_median_normalized_absolute_error": float(np.median(landmark_errors)),
            "landmark_p90_normalized_absolute_error": float(np.percentile(landmark_errors, 90)),
        },
        "throughput": {
            "encoding_seconds": encoding_seconds,
            "encoded_frames": int(sum(len(value) for value in features.values())),
            "effective_encoding_frames_per_second": float(
                sum(len(value) for value in features.values()) / max(encoding_seconds, 1e-9)
            ),
            "alignment_seconds": total_alignment_seconds,
            "alignment_updates_per_second": float(
                total_query_frames / max(total_alignment_seconds, 1e-9)
            ),
        },
        "per_trajectory": results,
        "plots": [str(path) for path in plot_paths],
    }


def run(args):
    import torch

    args.dataset = args.dataset.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "plots").mkdir()
    manifest, trajectories = load_dataset(args.dataset)
    landmarks = load_landmarks(args.dataset, manifest["landmarks"])
    images = {
        trajectory_id: load_images(args.dataset, trajectory)
        for trajectory_id, trajectory in trajectories.items()
    }
    methods = []
    for encoder_name in args.encoders:
        model, weights = build_model(encoder_name, args.device)
        if encoder_name == "rn18_temporal_pool":
            frame_embeddings = {}
            frame_seconds = 0.0
            for trajectory_id, values in images.items():
                encoded, elapsed = encode_rn18_frames(
                    model,
                    weights,
                    values,
                    args.device,
                    args.batch_size,
                )
                frame_embeddings[trajectory_id] = encoded
                frame_seconds += elapsed
            for window_seconds in args.windows:
                clip_frames = max(
                    1,
                    int(round(window_seconds * manifest["nominal_frame_rate_hz"])),
                )
                features = {
                    trajectory_id: temporal_pool(values, clip_frames)
                    for trajectory_id, values in frame_embeddings.items()
                }
                methods.append(
                    evaluate_method(
                        args,
                        manifest,
                        trajectories,
                        landmarks,
                        encoder_name,
                        window_seconds,
                        features,
                        frame_seconds,
                    )
                )
        else:
            for window_seconds in args.video_windows:
                clip_frames = max(
                    2,
                    int(round(window_seconds * manifest["nominal_frame_rate_hz"])),
                )
                features = {}
                encoding_seconds = 0.0
                for trajectory_id, values in images.items():
                    encoded, elapsed = encode_r3d_clips(
                        model,
                        weights,
                        values,
                        clip_frames,
                        args.device,
                        args.video_batch_size,
                    )
                    features[trajectory_id] = encoded
                    encoding_seconds += elapsed
                methods.append(
                    evaluate_method(
                        args,
                        manifest,
                        trajectories,
                        landmarks,
                        encoder_name,
                        window_seconds,
                        features,
                        encoding_seconds,
                    )
                )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    result = {
        "schema": "speedtuning-online-reference-alignment-benchmark-v1",
        "task": manifest["task"],
        "hypothesis": (
            "frozen generic clip embeddings plus causal monotonic alignment can "
            "recover a reusable continuous reference coordinate"
        ),
        "semantic_segment_labels_used": False,
        "semantic_phase_classifier_trained": False,
        "task_specific_training": False,
        "future_query_frames_used": False,
        "reference_trajectory": manifest["reference_trajectory"],
        "evaluation_truth": manifest["evaluation_correspondence_truth"],
        "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
        "hyperparameters": {
            "max_advance": args.max_advance,
            "max_backtrack": args.max_backtrack,
            "emission_temperature": args.temperature,
            "catastrophic_jump_threshold": args.catastrophic_threshold,
            "windows_seconds": args.windows,
            "video_windows_seconds": args.video_windows,
        },
        "methods": methods,
    }
    result_path = args.output / "results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.output / "SHA256SUMS").write_text(
        "\n".join(
            f"{sha256(path)}  {path.relative_to(args.output)}"
            for path in sorted(args.output.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        )
        + "\n"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=("rn18_temporal_pool", "r3d18"),
        default=("rn18_temporal_pool",),
    )
    parser.add_argument("--windows", type=float, nargs="+", default=(0.25, 0.5, 1.0))
    parser.add_argument("--video-windows", type=float, nargs="+", default=(0.5,))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--video-batch-size", type=int, default=8)
    parser.add_argument("--max-advance", type=int, default=5)
    parser.add_argument("--max-backtrack", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--catastrophic-threshold", type=float, default=0.15)
    args = parser.parse_args()
    if any(value <= 0 for value in (*args.windows, *args.video_windows)):
        parser.error("clip windows must be positive")
    return args


def main() -> int:
    result = run(parse_args())
    print(json.dumps({"task": result["task"], "methods": result["methods"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
