import json

from scripts import run_act_three_reset_rainbow50_v42 as module


def test_banks_extend_v41_without_reexecuting_parent_rollouts():
    banks = json.loads(
        (module.REPO_ROOT / "experiments/act_three_reset_rainbow50_v42/BANKS.json").read_text()
    )
    module.validate_banks(banks)
    fresh = []
    for spec in banks["tasks"].values():
        assert len(spec["parent_training"]) == 18
        assert len(spec["extension"]) == 32
        assert len(spec["fixed_probe"]) == 3
        assert len(spec["screen"]) == 10
        assert len(spec["final"]) == 50
        values = [*spec["extension"], *spec["fixed_probe"], *spec["screen"], *spec["final"]]
        assert not set(values).intersection(spec["parent_training"])
        fresh.extend(values)
    assert len(fresh) == len(set(fresh))


def test_extension_reaches_fifty_with_registered_pose_counts():
    order = module.extension_pose_order()
    assert len(order) == 32
    assert [order.count(index) for index in range(3)] == [11, 11, 10]
    assert [6 + order.count(index) for index in range(3)] == [17, 17, 16]
    assert module.PARENT_TRAINING_EPISODES + module.EXTENSION_EPISODES == 50


def test_runtime_maps_extension_and_probe_seeds_without_randomized_overlap():
    runtime = object.__new__(module.V42Runtime)
    extension = list(range(32))
    fixed = [100, 101, 102]
    runtime.extension_seed_set = set(extension)
    runtime.fixed_probe_seed_set = set(fixed)
    mapping = dict(zip(extension, module.extension_pose_order()))
    mapping.update(dict(zip(fixed, (0, 1, 2))))
    runtime.training_pose_index = mapping
    assert [runtime.training_pose_index[seed] for seed in fixed] == [0, 1, 2]
    assert 999 not in runtime.training_pose_index


def test_parent_manifest_pins_only_v41_training_ancestry():
    path = module.REPO_ROOT / "experiments/act_three_reset_rainbow50_v42/PARENT_V41.json"
    parent = json.loads(path.read_text())
    module.validate_parent_manifest(parent)
    assert parent["implementation_commit"] == "231fef194fae83d1cc68558c33bc14ea44552b0c"
    assert "result_sha256" not in parent
    assert "heldout" not in json.dumps(parent).lower()


def test_gate_requires_fixed_three_of_three_fresh_nine_of_ten_and_no_incident():
    clean = {"physics_errors": 0, "safety_violations": 0}
    assert module.rainbow_qualifies(3, 9, clean)
    assert not module.rainbow_qualifies(2, 10, clean)
    assert not module.rainbow_qualifies(3, 8, clean)
    assert not module.rainbow_qualifies(3, 10, {"physics_errors": 1, "safety_violations": 0})


def test_new_prefinal_accounting_is_exact_and_separate_from_parent():
    assert module.NEW_PREFINAL_ROLLOUTS == 48
    assert module.NEW_PREFINAL_ROLLOUTS == 32 + 3 + 3 + 10
    contract = (
        module.REPO_ROOT / "experiments/act_three_reset_rainbow50_v42/CONTRACT.md"
    ).read_text()
    assert "no V41 rollout is re-executed" in contract
    assert "fifty untouched randomized held-out resets" in contract
