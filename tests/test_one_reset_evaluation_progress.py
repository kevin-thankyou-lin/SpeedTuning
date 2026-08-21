import json

import pytest

from scripts.evaluate_one_reset_generalization import load_progress, write_progress


def test_evaluation_progress_round_trip_and_identity_gate(tmp_path):
    path = tmp_path / "pick-vlm.progress.json"
    identity = {"task": "pick", "method": "vlm", "evaluation_seeds": [11, 12]}
    completed = {11: {"seed": 11, "success": True, "physics_steps": 123}}

    write_progress(path, identity, [11, 12], completed)
    payload = json.loads(path.read_text())
    assert payload["completed_states"] == 1
    assert payload["terminal"] is False
    assert load_progress(path, identity) == completed

    with pytest.raises(ValueError, match="identity mismatch"):
        load_progress(path, {**identity, "method": "tabular"})


def test_terminal_progress_keeps_all_state_results(tmp_path):
    path = tmp_path / "tea-tabular.progress.json"
    identity = {"task": "tea", "method": "tabular", "evaluation_seeds": [21, 22]}
    completed = {
        21: {"seed": 21, "success": True},
        22: {"seed": 22, "success": False},
    }

    write_progress(path, identity, [21, 22], completed, terminal=True)
    payload = json.loads(path.read_text())
    assert payload["terminal"] is True
    assert payload["completed_states"] == 2
    assert [item["seed"] for item in payload["rollouts"]] == [21, 22]
