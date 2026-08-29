import json
from pathlib import Path

from act_speed_benchmark import canonical_sha256, preregistration
from scripts.freeze_act_controller_budget25_v22 import checked_prefix, freeze_tabular


def test_v23_contract_reuses_v22_banks_and_adds_150_rollouts():
    v22 = json.loads(Path("experiments/act_controller_budget25_eval_v22/contract.json").read_text())
    v23 = json.loads(Path("experiments/act_tabular_budget20_eval_v23/contract.json").read_text())
    assert v23["budget"]["inherited_training_rollouts_per_task"] == 20
    assert v23["budget"]["new_training_rollouts"] == 0
    assert v23["budget"]["new_final_rollouts"] == 150
    for task in ("pick", "tea", "insertion"):
        assert v23["tasks"][task]["final_seed_base"] == v22["tasks"][task]["final_seed_base"]


def test_checked_prefix_can_stop_at_episode_20(tmp_path):
    method = "learned_phase_tabular_rl"
    task = "tea"
    seeds = list(range(50))
    identity = {"method": method, "task_label": task}
    identity["identity_sha256"] = canonical_sha256(identity)
    (tmp_path / "states").mkdir()
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    (tmp_path / "preregistration.json").write_text(json.dumps(preregistration(method)))
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"episodes": 50, "identity_sha256": identity["identity_sha256"]}))
    for seed in seeds[:20]:
        (tmp_path / "states" / f"{seed}.json").write_text(json.dumps({
            "seed": seed,
            "identity_sha256": identity["identity_sha256"],
            "training_trajectory": [{"phase": seed % 4, "action": seed % 5, "reward": float(seed)}],
        }))
    records, _ = checked_prefix(tmp_path, seeds, method, task, episodes=20)
    assert len(records) == 20
    selected, _ = freeze_tabular(records)
    assert sum(sum(row) for row in selected["visits"]) == 20
