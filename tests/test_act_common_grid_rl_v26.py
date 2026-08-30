import json
from pathlib import Path

from act_speed_benchmark import COMMON_GRID_SPEED_VALUES, resolve_speed_values


def test_common_grid_contract_and_executable_match():
    contract = json.loads(
        Path("experiments/act_common_grid_rl_v26/contract.json").read_text()
    )
    expected = (1.0, 1.5, 2.0, 2.5, 3.0)
    assert tuple(contract["action_grid"]) == COMMON_GRID_SPEED_VALUES == expected
    assert resolve_speed_values("1,1.5,2,2.5,3") == expected
    assert all(value * 2 == round(value * 2) for value in expected)
    assert max(expected) == 3.0


def test_invalid_runtime_grid_is_rejected():
    try:
        resolve_speed_values("1,1.25,1.5,2,3")
    except RuntimeError as exc:
        assert "registered v26 common grid" in str(exc)
    else:
        raise AssertionError("unregistered action grid was accepted")


def test_fresh_banks_are_sized_and_globally_disjoint():
    contract = json.loads(
        Path("experiments/act_common_grid_rl_v26/contract.json").read_text()
    )
    banks = []
    for task in contract["tasks"].values():
        search = set(range(task["search_seed_base"], task["search_seed_base"] + 25))
        final = set(range(task["final_seed_base"], task["final_seed_base"] + 50))
        assert len(search) == 25
        assert len(final) == 50
        banks.extend((search, final))
    for index, bank in enumerate(banks):
        assert all(bank.isdisjoint(other) for other in banks[index + 1 :])
