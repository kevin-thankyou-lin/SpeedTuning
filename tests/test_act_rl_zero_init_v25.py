import json
from pathlib import Path

import torch

from act_speed_benchmark import SPEED_VALUES
from rl.rainbowDQN.network import Network
from scripts.freeze_act_rl_zero_init_v25 import tabular_selected


def test_zero_init_contract_reuses_newer_bank_without_training():
    v22 = json.loads(Path("experiments/act_controller_budget25_eval_v22/contract.json").read_text())
    v25 = json.loads(Path("experiments/act_rl_zero_init_eval_v25/contract.json").read_text())
    assert v25["budget"]["new_training_rollouts"] == 0
    assert v25["budget"]["new_final_rollouts"] == 300
    assert v25["initialization"]["all_six_controllers_frozen_before_final_bank"]
    for task in ("pick", "tea", "insertion"):
        assert v25["tasks"][task]["final_seed_base"] == v22["tasks"][task]["final_seed_base"]


def test_zero_tabular_is_native_schedule():
    selected = tabular_selected()
    assert selected["schedule"] == [1.0, 1.0, 1.0, 1.0]
    assert not any(any(row) for row in selected["q_values"])
    assert not any(any(row) for row in selected["visits"])


def test_seed_fixed_untrained_network_is_reproducible_and_unit_normalized():
    def build():
        torch.manual_seed(2701)
        support = torch.linspace(0.0, 120.0, 121)
        return Network(8, len(SPEED_VALUES), 121, support, hidden_dim=32).state_dict()
    first, second = build(), build()
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert torch.equal(first["states_mean"], torch.zeros(8))
    assert torch.equal(first["states_std"], torch.ones(8))
