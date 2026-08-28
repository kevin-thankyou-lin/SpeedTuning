import json
from pathlib import Path

from scripts import run_act_strider_tea_release_v9 as v9


ROOT = Path(__file__).resolve().parents[1]


def test_replacement_contract_excludes_only_physics_error_and_has_reserves():
    contract = json.loads((ROOT / "experiments/act_strider_tea_release_v9/PHYSICS_ERROR_REPLACEMENT.json").read_text())
    assert contract["excluded_primary_seeds"] == [20171107]
    assert contract["target_valid_pairs"] == 50
    assert contract["reserve_seed_count"] == 10
    assert contract["prior_invalid_attempts_count_toward_physical_cost"] is True
    assert contract["uniform_schedule"] == v9.UNIFORM
    assert contract["delayed_release_schedule"] == v9.DELAYED_RELEASE


def test_valid_pair_runner_keeps_task_failures_and_replaces_only_physics_errors():
    source = (ROOT / "scripts/run_act_strider_tea_release_v9_valid_pairs.py").read_text()
    assert 'if "physics_error" in record' in source
    assert '"counted_in_valid_denominator": False' in source
    assert '"prior_unreceipted_physics_error_attempts": 1' in source
    assert 'len(valid_seeds) != int(contract["target_valid_pairs"])' in source
    assert '"safety_violation"' not in source.split('if physics:')[0].split('physics = []')[-1]
