import json
from pathlib import Path

import numpy as np
import torch

from act_speed_benchmark import SPEED_VALUES, canonical_sha256, preregistration, sha256
from rl.rainbowDQN.network import Network
from scripts.freeze_act_controller_budget25_v22 import checked_prefix, freeze_rainbow, freeze_tabular
from speed_policy import RainbowSpeedPolicy


def test_v22_contract_has_fresh_matched_fifty_seed_banks():
    contract = json.loads(Path("experiments/act_controller_budget25_eval_v22/contract.json").read_text())
    assert contract["budget"]["inherited_training_rollouts_per_method_per_task"] == 25
    assert contract["budget"]["new_training_rollouts"] == 0
    assert contract["budget"]["new_final_rollouts"] == 300
    banks = []
    for task in ("pick", "tea", "insertion"):
        base = contract["tasks"][task]["final_seed_base"]
        bank = set(range(base, base + 50))
        assert len(bank) == 50
        banks.append(bank)
    assert banks[0].isdisjoint(banks[1])
    assert banks[0].isdisjoint(banks[2])
    assert banks[1].isdisjoint(banks[2])
    old = set(range(265000000, 265000250))
    assert all(bank.isdisjoint(old) for bank in banks)


def test_tabular_freeze_rebuilds_exact_first_25_policy():
    records = []
    for episode in range(25):
        records.append({"training_trajectory": [{"phase": episode % 4, "action": episode % 5, "reward": float(episode)}]})
    selected, evidence = freeze_tabular(records)
    assert selected["speed_values"] == list(SPEED_VALUES)
    assert len(selected["schedule"]) == 4
    assert evidence["q_values_sha256"] == canonical_sha256(selected["q_values"])
    assert sum(sum(row) for row in selected["visits"]) == 25


def test_rainbow_freeze_creates_loadable_inference_policy(tmp_path):
    support = torch.linspace(0.0, 120.0, 121)
    network = Network(4, len(SPEED_VALUES), 121, support, hidden_dim=128)
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "resume").mkdir(parents=True)
    checkpoint = source / "resume" / "episode-25.pt"
    torch.save({
        "schema": "act-speed-rainbow-resume-v1",
        "dqn": network.state_dict(),
        "decision": 77,
        "history": [np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 2],
    }, checkpoint)
    records = [{"observation_spec": {"shape": [4]}, "environment_spec": {"task": "pick"}} for _ in range(25)]
    records[-1].update(resume_checkpoint="resume/episode-25.pt", resume_checkpoint_sha256=sha256(checkpoint))
    selected, evidence = freeze_rainbow(source, records, destination)
    assert evidence["source_resume_checkpoint_sha256"] == sha256(checkpoint)
    policy = RainbowSpeedPolicy.load(Path(selected["checkpoint"]), device="cpu")
    assert policy.observation_dim == 4
    assert policy.speed_values == SPEED_VALUES


def test_checked_prefix_accepts_only_identity_matched_first_25(tmp_path):
    method = "learned_phase_tabular_rl"
    task = "pick"
    seeds = list(range(50))
    identity = {"method": method, "task_label": task}
    identity["identity_sha256"] = canonical_sha256(identity)
    (tmp_path / "states").mkdir()
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "preregistration.json").write_text(json.dumps(preregistration(method)))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"episodes": 50, "identity_sha256": identity["identity_sha256"]}))
    for seed in seeds[:25]:
        (tmp_path / "states" / f"{seed}.json").write_text(json.dumps({"seed": seed, "identity_sha256": identity["identity_sha256"]}))
    records, _ = checked_prefix(tmp_path, seeds, method, task)
    assert len(records) == 25
