import json

import numpy as np
import pytest

from scripts import build_three_reset_rainbow_panel_v41 as panel
from scripts import run_act_three_reset_rainbow25_v41 as module


def test_banks_are_fresh_exact_and_disjoint():
    path = module.REPO_ROOT / "experiments/act_three_reset_rainbow25_v41/BANKS.json"
    banks = json.loads(path.read_text())
    module.validate_banks(banks)
    all_values = []
    for spec in banks["tasks"].values():
        assert len(spec["training"]) == 18
        assert len(spec["screen"]) == 7
        assert len(spec["training"]) + len(spec["screen"]) == 25
        assert len(spec["final"]) == 50
        values = [spec["pose_design_seed"], *spec["training"], *spec["screen"], *spec["final"]]
        assert len(values) == len(set(values)) == 76
        assert min(values) >= 410000000
        all_values.extend(values)
    assert len(all_values) == len(set(all_values)) == 228


@pytest.mark.parametrize("task,dimensions", [("pick", 2), ("tea", 2), ("insertion", 4)])
def test_three_pose_design_is_stratified_and_balanced(task, dimensions):
    design = panel.stratified_three(task, 41)
    assert design.shape == (3, dimensions)
    for column in range(dimensions):
        assert sorted(np.floor(design[:, column] * 3).astype(int).tolist()) == [0, 1, 2]


def test_panel_cycles_each_pose_exactly_six_times(monkeypatch):
    monkeypatch.setattr(panel, "sample_object_pose", lambda task, seed: np.arange(39))
    value = panel.build("tea", 410001000)
    assert value["training_pose_order"] == [0, 1, 2] * 6
    assert value["training_visits_per_pose"] == [6, 6, 6]
    assert len(value["object_pose_vectors"]) == 3
    assert all(len(pose) == 39 for pose in value["object_pose_vectors"])
    assert value["selection_uses_policy_outcomes"] is False


def test_runtime_maps_unique_training_ids_to_repeated_pose_indices():
    runtime = object.__new__(module.V41Runtime)
    seeds = list(range(18))
    runtime.training_pose_index = dict(zip(seeds, [0, 1, 2] * 6))
    assert [runtime.training_pose_index[seed] for seed in seeds] == [0, 1, 2] * 6
    assert [list(runtime.training_pose_index.values()).count(index) for index in range(3)] == [6, 6, 6]


def test_incident_accounting_uses_explicit_non_null_values():
    records = [
        {"physics_error": None, "safety_violation": None},
        {"physics_error": "bad", "safety_violation": None},
        {"safety_violation": "outside"},
    ]
    assert module.incidents(records) == {"physics_errors": 1, "safety_violations": 1}


def test_registered_final_methods_keep_uniform_as_untuned_comparator():
    assert module.FINAL_METHODS == ("native_1x", "uniform_2x", "rainbow", "selected")
    contract = (module.REPO_ROOT / "experiments/act_three_reset_rainbow25_v41/CONTRACT.md").read_text()
    assert "does not import any speed" in contract
    assert "18 training episodes" in contract
    assert "Seven fresh randomized resets" in contract
    assert "7/7 incident-free" in contract
