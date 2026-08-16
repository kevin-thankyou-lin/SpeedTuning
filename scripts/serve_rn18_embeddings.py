#!/usr/bin/env python3
"""Serve frozen ResNet-18 frame embeddings over a local Unix socket."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def read_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading request")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = weights.transforms()

    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.ready.parent.mkdir(parents=True, exist_ok=True)
    if args.socket.exists():
        args.socket.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(1)
    args.ready.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "device": args.device,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "embedding_dim": 512,
                "frozen": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    shutdown = False
    try:
        while not shutdown:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        header_size = struct.unpack("!I", read_exact(connection, 4))[0]
                    except EOFError:
                        break
                    header = json.loads(read_exact(connection, header_size))
                    if header["op"] == "shutdown":
                        connection.sendall(struct.pack("!I", 0))
                        shutdown = True
                        break
                    if header["op"] != "encode":
                        raise ValueError(f"unsupported operation: {header['op']}")
                    shape = tuple(int(value) for value in header["shape"])
                    payload = read_exact(connection, int(np.prod(shape)))
                    image = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
                    tensor = transform(Image.fromarray(image)).unsqueeze(0).to(args.device)
                    with torch.inference_mode():
                        embedding = model(tensor).cpu().numpy()[0].astype("<f4")
                    response = embedding.tobytes()
                    connection.sendall(struct.pack("!I", len(response)) + response)
    finally:
        server.close()
        if args.socket.exists():
            args.socket.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
