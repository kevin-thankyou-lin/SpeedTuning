import json

import pytest

from scripts import run_act_vlm_balanced35_extension as module


def test_extension_result_keeps_qualification_and_accounting():
    raw = {
        "completed": 20,
        "successes": 19,
        "verdict": "qualified",
        "matched_native_speedup": 2.1,
        "safety_violations": 0,
        "physics_errors": 0,
        "new_candidate_rollouts": 20,
    }
    result = module.extension_result(raw, {"RESULT.json": "a" * 64})
    assert result["candidate"] == [2.5, 1.5, 3.5, 3.5]
    assert result["qualified_for_continued_search"]
    assert result["new_native_rollouts"] == 0
    assert not result["final_bank_opened"]


def test_validate_parent_rejects_wrong_target_result(tmp_path, monkeypatch):
    parent = tmp_path
    identity = {
        "source_commit": module.PARENT_SOURCE,
        "gate_seeds": module.GATE_SEEDS,
        "reserved_final_seeds": list(range(140210000, 140210100)),
    }
    files = {
        "IDENTITY.json": identity,
        "DISCOVERY.json": {},
        "RESULT.json": {"gate": {"schedule": module.PARENT_TARGET, "completed": 20, "successes": 18, "verdict": "qualified"}},
        "COMPLETE.json": {"new_rollouts": 58, "final_bank_opened": False},
        "gate/private/state.json": {"seeds": module.GATE_SEEDS, "native": [{}] * 20},
    }
    for name, value in files.items():
        path = parent / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(module, "sha256", lambda path: "0" * 64)
    with pytest.raises(RuntimeError, match="target-gate result"):
        module.validate_parent(parent)

