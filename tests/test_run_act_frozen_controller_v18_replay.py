import json
from pathlib import Path

import pytest

from scripts import run_act_frozen_controller_v18_replay as replay


def test_primary_seeds_are_exact_v18_final_primary_banks():
    banks = json.loads(
        Path("experiments/act_strider_representative_final_v18/BANKS.json").read_text()
    )
    assert replay.primary_seeds(banks, "pick") == list(range(265000000, 265000050))
    assert replay.primary_seeds(banks, "tea") == list(range(265000100, 265000150))
    assert replay.primary_seeds(banks, "insertion") == list(range(265000200, 265000250))


def test_only_three_frozen_controller_sources_are_registered():
    assert replay.METHOD_SOURCES == {
        "learned_phase_subtask": "866c9f436caf0a73e5e08ef83be38cbe89a23a61",
        "learned_phase_tabular_rl": "298c6d16784f228df0b1f455d0e41b4276ec5184",
        "learned_phase_rainbow_rl": "298c6d16784f228df0b1f455d0e41b4276ec5184",
    }


def test_checked_selected_binds_completion_hash(tmp_path: Path):
    selected = {
        "method": "learned_phase_subtask",
        "task_label": "pick",
        "selected_policy": {"schedule": [2.0, 1.0, 1.5, 1.0]},
    }
    selected_path = tmp_path / "selected.json"
    selected_path.write_text(json.dumps(selected))
    (tmp_path / "COMPLETE.json").write_text(
        json.dumps({"selected_sha256": replay.sha256(selected_path)})
    )
    assert replay.checked_selected(tmp_path, "learned_phase_subtask", "pick") == selected
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"selected_sha256": "bad"}))
    with pytest.raises(RuntimeError, match="hash"):
        replay.checked_selected(tmp_path, "learned_phase_subtask", "pick")
