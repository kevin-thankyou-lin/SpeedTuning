import json

import torch

from scripts import run_act_randomized_rainbow100_v43 as module
from rl.rainbowDQN.network import Network
from speed_policy import RainbowSpeedPolicy


def test_banks_are_exact_fresh_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_randomized_rainbow100_v43/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_seeds = []
    for spec in banks["tasks"].values():
        values = module.task_banks(spec)
        assert len(values["training"]) == 100
        assert len(values["probe"]) == 3
        assert len(values["final"]) == 50
        flattened = [seed for bank in values.values() for seed in bank]
        assert len(flattened) == len(set(flattened))
        all_seeds.extend(flattened)
    assert len(all_seeds) == len(set(all_seeds))


def test_checkpoint_chronology_and_accounting_are_exact():
    assert module.CHECKPOINT_EPISODES == tuple(range(10, 101, 10))
    assert module.PREFINAL_ROLLOUTS == 133
    assert module.PREFINAL_ROLLOUTS == 100 + 3 + 10 * 3


def test_terminal_controller_is_episode_100_and_not_probe_selected():
    value = module.controller("rainbow", checkpoint_sha256="a" * 64, episode=100)
    assert value["training_episode"] == 100
    contract = (
        module.REPO_ROOT / "experiments/act_randomized_rainbow100_v43/CONTRACT.md"
    ).read_text()
    assert "Probe outcomes cannot alter training, choose a checkpoint" in contract
    assert "episode-100 terminal policy is\nalways" in contract


def test_runtime_declares_randomized_training_and_evaluation():
    source = (
        module.REPO_ROOT / "scripts/run_act_randomized_rainbow100_v43.py"
    ).read_text()
    assert "randomize_object_pose=True" in source
    assert "fixed_three_pose" not in source
    assert "parent-v41" not in source


def test_final_methods_are_direct_comparators_without_selected_alias():
    assert module.FINAL_METHODS == ("native_1x", "uniform_2x", "rainbow")


def test_intermediate_resume_converts_to_a_loadable_greedy_policy(tmp_path):
    training = tmp_path / "training"
    states = training / "states"
    states.mkdir(parents=True)
    support = torch.linspace(0.0, 120.0, 121)
    network = Network(4, 5, 121, support, hidden_dim=128)
    resume = training / "resume" / "episode-010.pt"
    resume.parent.mkdir()
    torch.save({"dqn": network.state_dict(), "decision": 77}, resume)
    resume_hash = module.file_sha256(resume)
    for index in range(100):
        record = {
            "resume_checkpoint": "resume/episode-010.pt",
            "resume_checkpoint_sha256": resume_hash,
            "observation_spec": {"observation_dim": 4},
            "environment_spec": {"randomize_object_pose": True},
        }
        (states / f"{430000010 + index}.json").write_text(json.dumps(record))
    config = module.preregistration(
        "learned_phase_rainbow_rl", search_rollouts=100, final_rollouts=50
    )["training"]
    checkpoint, _ = module.build_checkpoint_policy(training, 10, config)
    policy = RainbowSpeedPolicy.load(checkpoint)
    assert policy.observation_dim == 4
    assert policy.checkpoint_metadata["checkpoint_after_training_episode"] == 10
