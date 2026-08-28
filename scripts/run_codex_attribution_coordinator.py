#!/usr/bin/env python3
"""Coordinate remote STRIDER evidence bundles with local Codex image agents."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.codex_agent_failure_attribution import SCHEMA, _write_json
from scripts.qwen_vlm_failure_attribution import (
    canonical_sha256,
    sha256_file,
    validate_attribution,
)


def run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=True, text=True, **kwargs)


def remote_ready(host: str, root: str) -> list[str]:
    command = (
        f"if [ -d {shlex.quote(root)} ]; then "
        f"find {shlex.quote(root)} -mindepth 2 -maxdepth 2 "
        "-type f -name READY.json -print; fi"
    )
    result = run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True)
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def remote_has_response(host: str, request_root: str) -> bool:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "test", "-f", f"{request_root}/RESPONSE.json"]
    )
    return result.returncode == 0


def checked_request(root: Path) -> dict:
    request = json.loads((root / "REQUEST.json").read_text())
    if request.get("schema") != SCHEMA:
        raise RuntimeError("unexpected attribution request schema")
    payload_hash = request.pop("payload_sha256")
    if canonical_sha256(request) != payload_hash:
        raise RuntimeError("attribution request payload hash mismatch")
    request["payload_sha256"] = payload_hash
    if request["request_id"] != root.name:
        raise RuntimeError("request directory and request ID differ")
    for image in request["images"]:
        path = root / image["path"]
        if sha256_file(path) != image["sha256"]:
            raise RuntimeError(f"image hash mismatch: {path}")
    return request


def invoke_codex(
    root: Path,
    request: dict,
    *,
    codex_binary: str,
    codex_home: Path,
    schema_path: Path,
    reasoning_effort: str,
) -> dict:
    raw_path = root / "CODEX_RAW.json"
    transcript_path = root / "CODEX_TRANSCRIPT.jsonl"
    command = [
        codex_binary,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--cd",
        str(root),
        "--model",
        request["model"],
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(raw_path),
        "--json",
    ]
    for image in request["images"]:
        command.extend(("--image", str(root / image["path"])))
    prompt = (
        request["prompt"]
        + "\nThe two attached images are chronological contact sheets: first the "
        "successful reference, then the accelerated failure. Frame captions give "
        "only time, step, and learned online phase. Do not use hidden simulator state."
    )
    command.append("-")
    environment = os.environ.copy()
    environment["CODEX_HOME"] = str(codex_home)
    with transcript_path.open("w") as transcript:
        run(
            command,
            cwd=root,
            env=environment,
            input=prompt,
            stdout=transcript,
            stderr=subprocess.STDOUT,
        )
    attribution = validate_attribution(json.loads(raw_path.read_text()))
    version = run([codex_binary, "--version"], capture_output=True).stdout.strip()
    return {
        "schema": "codex-agent-attribution-response-v1",
        "request_id": request["request_id"],
        "request_payload_sha256": request["payload_sha256"],
        "attribution": attribution,
        "coordinator": {
            "model": request["model"],
            "reasoning_effort": reasoning_effort,
            "codex_version": version,
            "ephemeral": True,
            "sandbox": "read-only",
            "user_config_ignored": True,
            "project_rules_ignored": True,
            "raw_output_sha256": sha256_file(raw_path),
            "transcript_sha256": sha256_file(transcript_path),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def upload_response(host: str, remote_root: str, response_path: Path) -> None:
    temporary = f"{remote_root}/RESPONSE.json.uploading"
    run(["scp", "-q", str(response_path), f"{host}:{temporary}"])
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "mv",
            "--",
            temporary,
            f"{remote_root}/RESPONSE.json",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-exchange-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    args = parser.parse_args()

    if not (args.codex_home / "auth.json").exists():
        raise RuntimeError("dedicated Codex home lacks auth.json")
    skills_root = args.codex_home / "skills"
    custom_skills = (
        sorted(path.name for path in skills_root.iterdir() if path.name != ".system")
        if skills_root.exists()
        else []
    )
    if custom_skills:
        raise RuntimeError(
            f"dedicated attribution Codex home contains custom skills: {custom_skills}"
        )
    args.local_root.mkdir(parents=True, exist_ok=True)
    while True:
        for ready_path in remote_ready(args.remote_host, args.remote_exchange_root):
            remote_root = str(Path(ready_path).parent)
            request_id = Path(remote_root).name
            if remote_has_response(args.remote_host, remote_root):
                continue
            local_root = args.local_root / request_id
            local_root.mkdir(parents=True, exist_ok=True)
            run(["rsync", "-a", f"{args.remote_host}:{remote_root}/", f"{local_root}/"])
            request = checked_request(local_root)
            response_path = local_root / "RESPONSE.json"
            if response_path.exists():
                response = json.loads(response_path.read_text())
                if response.get("request_payload_sha256") != request["payload_sha256"]:
                    raise RuntimeError("cached Codex response hash mismatch")
                validate_attribution(response["attribution"])
            else:
                response = invoke_codex(
                    local_root,
                    request,
                    codex_binary=args.codex_binary,
                    codex_home=args.codex_home,
                    schema_path=args.schema.resolve(),
                    reasoning_effort=args.reasoning_effort,
                )
                _write_json(response_path, response)
            upload_response(args.remote_host, remote_root, response_path)
            print(json.dumps({"completed_request": request_id, "attribution": response["attribution"]}), flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
