import json
from pathlib import Path

import pytest

from scripts import create_act_controller_retrain_v20_manifest as manifest


def test_v20_contract_uses_original_search_and_v18_final_banks():
    contract = json.loads(Path("experiments/act_controller_retrain_v20/contract.json").read_text())
    assert contract["tasks"]["pick"]["search_seed_base"] == 10100000
    assert contract["tasks"]["tea"]["search_seed_base"] == 10200000
    assert contract["tasks"]["insertion"]["search_seed_base"] == 10300000
    assert contract["tasks"]["pick"]["final_seed_base"] == 265000000
    assert contract["tasks"]["tea"]["final_seed_base"] == 265000100
    assert contract["tasks"]["insertion"]["final_seed_base"] == 265000200


def test_checked_v18_receipt_requires_exact_hashes_and_seed_order(tmp_path: Path):
    root = tmp_path / "pick"
    root.mkdir()
    seeds = list(range(50))
    identity = {"task_label": "pick", "task_final_seed_pool": seeds + list(range(50, 70))}
    result = {"task_label": "pick", "final": {"valid_pair_seeds": seeds}}
    (root / "IDENTITY.json").write_text(json.dumps(identity))
    (root / "RESULT.json").write_text(json.dumps(result))
    complete = {
        "identity_sha256": manifest.sha256(root / "IDENTITY.json"),
        "result_sha256": manifest.sha256(root / "RESULT.json"),
        "simulator_invalid_pairs": 0,
        "final_scientific_rollouts": 150,
    }
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    assert manifest.checked_v18_receipt(tmp_path, "pick", seeds)["scientific_rollouts"] == 150
    complete["simulator_invalid_pairs"] = 1
    (root / "COMPLETE.json").write_text(json.dumps(complete))
    with pytest.raises(RuntimeError, match="simulator-invalid"):
        manifest.checked_v18_receipt(tmp_path, "pick", seeds)
