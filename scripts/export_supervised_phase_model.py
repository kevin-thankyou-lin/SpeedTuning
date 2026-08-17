#!/usr/bin/env python3
"""Export a fitted sklearn phase head into a portable NumPy artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".receipt.json").exists():
        parser.error("output already exists")

    with args.model.open("rb") as handle:
        pipeline = pickle.load(handle)
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        classes=np.asarray(classifier.classes_),
        mean=np.asarray(scaler.mean_),
        scale=np.asarray(scaler.scale_),
        coef=np.asarray(classifier.coef_),
        intercept=np.asarray(classifier.intercept_),
    )
    receipt = {
        "schema": "speedtuning-portable-phase-head-v1",
        "source_model": str(args.model.resolve()),
        "source_model_sha256": sha256(args.model),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "classes": [str(value) for value in classifier.classes_],
        "input_features": int(scaler.n_features_in_),
        "inference": "StandardScaler then multinomial LogisticRegression in NumPy",
    }
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
