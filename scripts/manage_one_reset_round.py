#!/usr/bin/env python3
"""Run the queued one-reset VLM versus tabular comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)

# All simulator jobs are headless. Child processes inherit these settings.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def start_logged(command, stdout_path: Path, stderr_path: Path | None = None):
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    output = stdout_path.open("w")
    error = output if stderr_path is None else stderr_path.open("w")
    process = subprocess.Popen(command, stdout=output, stderr=error, text=True)
    process._round_files = (output, error)  # keep descriptors alive
    return process


def wait_paths(paths: list[Path], processes: list[subprocess.Popen], label: str) -> None:
    while not all(path.exists() for path in paths):
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"{label} process failed: {failed}")
        time.sleep(2)


def agent_command(agent_root: Path, runner_root: Path) -> list[str]:
    prompt = (
        "Read AGENTS.md and TASK_CONTRACT.md completely. Execute the full one-reset configured-phase "
        "speed search using only this lane's runner results and MP4s. Freeze one safe successful "
        "schedule with select, then write terminal STATUS.json and FINAL_REPORT.md."
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
        "--ro-bind", str(agent_root / "one_reset.py"), "/workspace/one_reset.py",
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
    contract = json.loads((root / "CONTRACT.json").read_text())
    detector = contract.get("phase_detector")
    detector_path = root / "DETECTOR.json"
    if detector is not None:
        write_json(detector_path, detector)
    source_validation = Path(contract["source_round"]) / "manager" / "VALIDATION.json"
    log(f"queued behind source validation: {source_validation}")
    while not source_validation.exists():
        time.sleep(10)
    log("source round terminal; starting one-reset training")

    servers = []
    agents = []
    tabular = []
    try:
        for lane, task in contract["tasks"].items():
            runner_root = Path(task["vlm_runner_root"])
            server_command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "one_reset_server.py"),
                "--root", str(runner_root), "--task", task["runtime_task"],
                "--training-seed", str(task["training_seed"]), "--budget", "50",
            ]
            if detector is not None:
                server_command += ["--detector-json", str(detector_path)]
            server = start_logged(
                server_command,
                root / "logs" / f"{lane}-vlm-runner.log",
            )
            servers.append(server)
        wait_paths(
            [Path(task["vlm_runner_root"]) / "api" / "READY.json" for task in contract["tasks"].values()],
            servers,
            "VLM server bootstrap",
        )
        for lane, task in contract["tasks"].items():
            runner_root = Path(task["vlm_runner_root"])
            private = json.loads((runner_root / "private" / "state.json").read_text())
            pose = json.dumps(private["object_pose"], separators=(",", ":"))
            config = REPO_ROOT / "configs" / task["config"]
            tabular_command = [
                str(PYTHON), str(REPO_ROOT / "scripts" / "train_tabular_phase_speed.py"),
                 "--config", str(config), "--task", task["runtime_task"], "--episodes", "50",
                 "--seed", str(task["training_seed"]), "--output", task["tabular_checkpoint"],
                 "--report", task["tabular_report"], "--gamma", "0.97", "--epsilon-start", "1.0",
                 "--epsilon-end", "0.05", "--success-bonus", "100", "--speed-weight", "0.01",
                 "--speed-power", "2", "--speed-values", "1,1.5,2,2.5,3,3.5,4",
                 "--frame-skip", "1", "--base-policy", "scripted", "--speed-observation", "external",
                 "--observation-encoder-loader", (
                     "oracle_phase_observation:create_oracle_phase_encoder"
                     if detector is None
                     else "learned_phase_observation:create_learned_phase_encoder"
                 ),
                 "--speed-decision-mode", "phase-entry", "--terminate-on-success", "--object-pose", pose,
            ]
            if detector is not None:
                tabular_command += ["--observation-factory-kwargs", json.dumps(detector)]
            tabular_process = start_logged(
                tabular_command,
                root / "logs" / f"{lane}-tabular.log",
            )
            tabular.append(tabular_process)
            agent_root = Path(task["vlm_agent_root"])
            agent = start_logged(
                agent_command(agent_root, runner_root),
                agent_root / "analysis" / "AGENT_TRACE.jsonl",
                agent_root / "analysis" / "AGENT_STDERR.log",
            )
            agents.append(agent)

        for process in tabular + agents:
            if process.wait() != 0:
                raise RuntimeError(f"training process failed with {process.returncode}")
        for task in contract["tasks"].values():
            if not Path(task["vlm_selection"]).exists():
                raise RuntimeError("VLM agent exited without selecting a schedule")
            if not Path(task["tabular_checkpoint"]).exists():
                raise RuntimeError("tabular worker exited without a checkpoint")
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

    evaluations = []
    for lane in contract["tasks"]:
        for method in ("vlm", "tabular"):
            output = root / "results" / lane / f"{method}.json"
            process = start_logged(
                [str(PYTHON), str(REPO_ROOT / "scripts" / "evaluate_one_reset_generalization.py"),
                 "--contract", str(root / "CONTRACT.json"), "--task", lane,
                 "--method", method, "--output", str(output)],
                root / "logs" / f"{lane}-{method}-evaluation.log",
            )
            evaluations.append((lane, method, output, process))
    for lane, method, _, process in evaluations:
        if process.wait() != 0:
            raise RuntimeError(f"{lane} {method} evaluation failed with {process.returncode}")
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
    log("one-reset generalization experiment complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
