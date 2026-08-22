#!/usr/bin/env python3
"""Run the learned-phase three-scene VLM versus tabular comparison."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
WORKER_ENV = {
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "SPEEDTUNING_TORCH_THREADS": "2",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def start_logged(command, output_path: Path, env=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = output_path.open("w")
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, text=True, env=env)
    process._round_stream = stream
    return process


def wait_paths(paths, processes, label):
    while not all(path.exists() for path in paths):
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"{label} failed: {failed}")
        time.sleep(1)


def agent_command(agent_root: Path, runner_root: Path) -> list[str]:
    prompt = (
        "Read AGENTS.md and TASK_CONTRACT.md completely. Execute the full staged three-scene "
        "search using only this lane's runner results and MP4s. Preserve the backoff reserve, "
        "rank exactly two eligible accelerated finalists, accept the runner-selected schedule, "
        "and write terminal artifacts."
    )
    return [
        "bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin", "--ro-bind", "/etc", "/etc",
        "--dir", "/run", "--dir", "/run/systemd", "--ro-bind", "/run/systemd/resolve", "/run/systemd/resolve",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/home", "--dir", "/home/agent",
        "--dir", "/codex-home", "--ro-bind", "/home/linke/.codex/auth.json", "/codex-home/auth.json",
        "--ro-bind", "/home/linke/.nvm/versions/node/v24.14.1", "/opt/node",
        "--bind", str(agent_root), "/workspace", "--bind", str(runner_root / "api"), "/workspace/runner_api",
        "--ro-bind", str(runner_root / "public" / "media"), "/workspace/media",
        "--ro-bind", str(agent_root / "AGENTS.md"), "/workspace/AGENTS.md",
        "--ro-bind", str(agent_root / "TASK_CONTRACT.md"), "/workspace/TASK_CONTRACT.md",
        "--ro-bind", str(agent_root / "three_scene.py"), "/workspace/three_scene.py",
        "--setenv", "HOME", "/home/agent", "--setenv", "CODEX_HOME", "/codex-home",
        "--setenv", "SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt",
        "--setenv", "REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt",
        "--setenv", "PATH", "/opt/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--chdir", "/workspace", "/opt/node/bin/codex", "exec", "--ephemeral", "--ignore-user-config",
        "--ignore-rules", "--skip-git-repo-check", "--sandbox", "workspace-write", "--model", "gpt-5.6-sol",
        "--config", 'model_reasoning_effort="high"', "--config", 'service_tier="fast"',
        "--config", 'approval_policy="never"', "--json", "--output-last-message",
        "/workspace/analysis/AGENT_LAST_MESSAGE.txt", prompt,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = root / "CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    source_status = Path(contract["source_round"]) / "FINAL_STATUS.json"
    if not source_status.exists() or not json.loads(source_status.read_text()).get("terminal"):
        raise RuntimeError("source round is not terminal")
    detector_path = root / "DETECTOR.json"
    env = dict(os.environ, MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", **WORKER_ENV)
    servers = []
    workers = []
    try:
        for lane, task in contract["tasks"].items():
            runner = Path(task["vlm_runner_root"])
            command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "three_scene_server.py"),
                "--root", str(runner), "--task", task["runtime_task"],
                "--discovery-seeds", ",".join(map(str, task["discovery_seeds"])),
                "--ranking-seeds", ",".join(map(str, task["ranking_seeds"])),
                "--budget", "50", "--detector-json", str(detector_path),
            ]
            servers.append(start_logged(command, root / "logs" / f"{lane}-vlm-runner.log", env))
        wait_paths(
            [Path(task["vlm_runner_root"]) / "api" / "READY.json" for task in contract["tasks"].values()],
            servers,
            "three-scene runner bootstrap",
        )
        log("three-scene runners ready; launching blind VLM and tabular learners")
        for lane, task in contract["tasks"].items():
            runner = Path(task["vlm_runner_root"])
            private = json.loads((runner / "private" / "state.json").read_text())
            poses_path = Path(task["tabular_poses"])
            write_json(poses_path, private["discovery_poses"])
            tabular_command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "train_tabular_phase_speed.py"),
                "--config", str(REPO_ROOT / "configs" / task["config"]),
                "--task", task["runtime_task"], "--episodes", "50",
                "--seed", str(task["discovery_seeds"][0]),
                "--output", task["tabular_checkpoint"], "--report", task["tabular_report"],
                "--gamma", "0.97", "--epsilon-start", "1.0", "--epsilon-end", "0.05",
                "--success-bonus", "100", "--speed-weight", "0.01", "--speed-power", "2",
                "--speed-values", "1,1.5,2,2.5,3,3.5,4", "--frame-skip", "1",
                "--base-policy", "scripted", "--speed-observation", "external",
                "--observation-encoder-loader", "learned_phase_observation:create_learned_phase_encoder",
                "--observation-factory-kwargs", json.dumps(contract["phase_detector"]),
                "--speed-decision-mode", "phase-entry", "--terminate-on-success",
                "--object-poses-json", str(poses_path),
            ]
            workers.append(start_logged(tabular_command, root / "logs" / f"{lane}-tabular.log", env))
            agent_root = Path(task["vlm_agent_root"])
            workers.append(start_logged(
                agent_command(agent_root, runner),
                agent_root / "analysis" / "AGENT_TRACE.jsonl",
                env,
            ))
        for process in workers:
            if process.wait() != 0:
                raise RuntimeError(f"training worker failed with {process.returncode}")
        for task in contract["tasks"].values():
            if not Path(task["vlm_selection"]).exists():
                raise RuntimeError("VLM agent exited without ranked selection")
            if not Path(task["tabular_checkpoint"]).exists():
                raise RuntimeError("tabular worker exited without checkpoint")
        log("all schedules frozen; starting shared 100-state evaluation")
    finally:
        for server in servers:
            if server.poll() is None:
                server.send_signal(signal.SIGTERM)
        for server in servers:
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
            server._round_stream.close()
        for process in workers:
            process._round_stream.close()

    evaluations = []
    for lane in contract["tasks"]:
        for method in ("vlm", "tabular"):
            output = root / "results" / lane / f"{method}.json"
            process = start_logged(
                [
                    str(PYTHON), str(REPO_ROOT / "scripts" / "evaluate_one_reset_generalization.py"),
                    "--contract", str(contract_path), "--task", lane,
                    "--method", method, "--output", str(output),
                ],
                root / "logs" / f"{lane}-{method}-evaluation.log",
                env,
            )
            evaluations.append((lane, method, output, process))
    for lane, method, _, process in evaluations:
        if process.wait() != 0:
            raise RuntimeError(f"{lane} {method} evaluation failed with {process.returncode}")
        process._round_stream.close()
    results = {
        lane: {
            method: json.loads(output.read_text())
            for current_lane, method, output, _ in evaluations
            if current_lane == lane
        }
        for lane in contract["tasks"]
    }
    write_json(root / "RESULTS.json", results)
    write_json(root / "FINAL_STATUS.json", {"terminal": True, "state": "complete"})
    (root / "COMPLETE").write_text("complete\n")
    log("three-scene comparison complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
