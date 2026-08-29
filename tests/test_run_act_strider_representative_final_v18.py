import json
from pathlib import Path

import pytest

from scripts import run_act_strider_representative_final_v18 as v18


def test_final_banks_are_disjoint_from_v16_and_v17():
    v18_banks = json.loads(
        Path("experiments/act_strider_representative_final_v18/BANKS.json").read_text()
    )
    v16_banks = json.loads(
        Path("experiments/act_strider_codex_v16_budget48/BANKS.json").read_text()
    )
    v17_registry = json.loads(
        Path("experiments/act_strider_representative_confirmation_v17/PANELS.json").read_text()
    )
    old = set()
    for specs in v16_banks["tasks"].values():
        for bank in specs.values():
            if isinstance(bank, dict) and ("start" in bank or "seeds" in bank):
                old.update(v18.base._range(bank))
    for relative in v17_registry["tasks"].values():
        panel = json.loads(
            (Path("experiments/act_strider_representative_confirmation_v17") / relative).read_text()
        )
        old.update(panel["panel_ids"])

    fresh = set()
    for task in v18_banks["tasks"]:
        pool = v18.checked_final_pool(v18_banks, task)
        assert len(pool) == 70
        assert old.isdisjoint(pool)
        assert fresh.isdisjoint(pool)
        fresh.update(pool)


def test_checked_v17_selection_requires_unopened_bank_and_hashes(tmp_path: Path):
    identity = {"task_label": "pick"}
    selection = {
        "opens_final_bank": False,
        "selected_schedule": [2.0, 1.5, 2.0, 2.0],
        "selected_schedule_sha256": v18.v4.schedule_sha256([2.0, 1.5, 2.0, 2.0]),
    }
    (tmp_path / "IDENTITY.json").write_text(json.dumps(identity))
    (tmp_path / "SELECTION.json").write_text(json.dumps(selection))
    complete = {
        "identity_sha256": v18.v4.file_sha256(tmp_path / "IDENTITY.json"),
        "selection_sha256": v18.v4.file_sha256(tmp_path / "SELECTION.json"),
        "opens_final_bank": False,
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    _, observed, _ = v18.checked_v17_selection(tmp_path, "pick")
    assert observed == selection

    complete["opens_final_bank"] = True
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(RuntimeError, match="opened a final bank"):
        v18.checked_v17_selection(tmp_path, "pick")


class FakeLedger:
    def __init__(self, runtime, root, search_pool, final_pool):
        assert search_pool == []
        self.final_pool = final_pool

    def evaluate_final_paired(self, named):
        assert set(named) == set(v18.EXPECTED_CONTROLLERS)
        return {
            "scientific_rollouts": 150,
            "unique_controllers_evaluated": 3,
            "valid_pair_seeds": self.final_pool[:50],
        }


def test_final_evaluator_requires_three_unique_controllers(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(v18.base, "ValidVideoLedger", FakeLedger)
    result = v18.evaluate_final(
        object(), tmp_path, [2.0, 1.5, 2.0, 2.0], list(range(70))
    )
    assert result["scientific_rollouts"] == 150

    with pytest.raises(RuntimeError, match="three unique controllers"):
        v18.evaluate_final(object(), tmp_path, [1.5] * 4, list(range(70)))
