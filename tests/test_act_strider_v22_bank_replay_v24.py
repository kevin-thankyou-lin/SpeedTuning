import json
from pathlib import Path

from scripts import finalize_act_strider_v22_bank_replay_v24 as finalize
from scripts import run_act_strider_v22_bank_replay_v24 as replay


def record(success, steps, first=None):
    return {"success": success, "physics_steps": steps, "first_success_step": first}


def test_paired_counts_success_discordance_and_common_success_speed():
    value = finalize.paired(
        [record(True, 10, 8), record(True, 11, 9), record(False, 12)],
        [record(True, 12, 10), record(False, 11), record(True, 10, 9)],
    )
    assert value["pairs"] == 3
    assert value["both_success"] == 1
    assert value["strider_only_success"] == 1
    assert value["comparator_only_success"] == 1
    assert value["common_success_time_to_success"]["strider_speedup_vs_comparator"] == 1.25


def test_load_contiguous_states_rejects_suffix(tmp_path):
    root = tmp_path / "states"
    root.mkdir()
    (root / "2.json").write_text(json.dumps({"seed": 2, "identity_sha256": "x"}))
    try:
        replay.load_contiguous_states(root, [1, 2], "x")
    except RuntimeError as exc:
        assert "non-contiguous" in str(exc)
    else:
        raise AssertionError("expected non-contiguous suffix rejection")


def test_contract_freezes_exact_bank_and_schedules():
    contract = json.loads(
        Path("experiments/act_strider_v22_bank_replay_v24/contract.json").read_text()
    )
    assert contract["budget"]["new_final_rollouts"] == 150
    assert contract["budget"]["cached_v20_v22_v23_rollouts_reexecuted"] == 0
    assert contract["tasks"]["pick"]["expected_schedule"] == [2.5, 1.5, 2.0, 2.0]
    assert contract["selection"]["search_or_tuning_permitted"] is False
