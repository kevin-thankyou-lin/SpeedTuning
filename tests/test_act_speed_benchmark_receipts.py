import json
from pathlib import Path

import pytest

from scripts.run_act_speed_parity import (
    PARITY,
    canonical_sha256,
    immutable_json,
    load_completed_states,
)


def test_frozen_parity_banks_are_exact_and_disjoint_from_search_and_final():
    contract = json.loads(
        Path("experiments/act_speed_benchmark_v1/contract.json").read_text()
    )
    assert {label: value["expected_successes"] for label, value in PARITY.items()} == {
        "pick": 49,
        "tea": 50,
        "insertion": 49,
    }
    for label, parity in PARITY.items():
        parity_seeds = set(range(parity["seed_base"], parity["seed_base"] + 50))
        task = contract["tasks"][label]
        search = set(range(task["search_seed_base"], task["search_seed_base"] + 50))
        final = set(range(task["final_seed_base"], task["final_seed_base"] + 50))
        assert len(parity_seeds) == len(search) == len(final) == 50
        assert parity_seeds.isdisjoint(search)
        assert parity_seeds.isdisjoint(final)
        assert search.isdisjoint(final)


def test_immutable_json_never_replaces_a_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    immutable_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        immutable_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def test_resume_requires_contiguous_identity_matched_states(tmp_path):
    states = tmp_path / "states"
    identity = canonical_sha256({"task": "pick"})
    immutable_json(states / "10.json", {"seed": 10, "identity_sha256": identity})
    assert len(load_completed_states(states, [10, 11], identity)) == 1
    immutable_json(states / "12.json", {"seed": 12, "identity_sha256": identity})
    with pytest.raises(RuntimeError, match="non-contiguous"):
        load_completed_states(states, [10, 11, 12], identity)


def test_resume_rejects_wrong_identity(tmp_path):
    states = tmp_path / "states"
    immutable_json(states / "10.json", {"seed": 10, "identity_sha256": "wrong"})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        load_completed_states(states, [10], "expected")
