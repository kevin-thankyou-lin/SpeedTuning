import numpy as np

from scripts import run_act_rainbow50_scripted_correlation_v44 as v44


def test_registered_banks_are_exact_and_disjoint():
    banks = v44.checked_json(
        v44.REPO_ROOT / "experiments/act_rainbow50_scripted_correlation_v44/BANKS.json"
    )
    v44.validate_banks(banks)
    values = [seed for task in v44.TASKS for seed in v44.bank_seeds(banks, task)]
    assert len(values) == 30
    assert len(set(values)) == 30


def test_spearman_and_average_tie_ranks():
    assert np.allclose(v44.average_ranks([1, 2, 2, 4]), [1, 2.5, 2.5, 4])
    assert v44.spearman([0, 1, 2], [1, 2, 3]) == 1.0
    assert v44.spearman([0, 1, 2], [3, 2, 1]) == -1.0
    assert v44.spearman([0, 1], [2, 2]) is None


def test_phase_speed_nmi_is_one_for_bijection():
    pairs = [("a", 1.0), ("a", 1.0), ("b", 2.0), ("b", 2.0)]
    assert np.isclose(v44.normalized_mutual_information(pairs), 1.0)


def test_trace_summary_preserves_phase_and_progress_views():
    records = [{
        "decisions": [
            {"phase": "pre_grasp", "next_phase": "pre_grasp", "nominal_progress": 0.05,
             "rainbow_speed": 2.0, "chosen_speed": 2.0, "decision_physics_steps": 10},
            {"phase": "transport", "next_phase": "interaction", "nominal_progress": 0.65,
             "rainbow_speed": 1.5, "chosen_speed": 1.5, "decision_physics_steps": 4},
        ]
    }]
    summary = v44.trace_summary(records, "rainbow_speed")
    assert summary["decisions"] == 2
    assert summary["phase_speed_counts"]["pre_grasp"] == {2.0: 1}
    assert summary["progress_deciles"]["0"]["decisions"] == 1
    assert summary["progress_deciles"]["6"]["decisions"] == 1
    assert summary["phase_transition_counts"]["transport->interaction"] == 1
    assert summary["phase_native_equivalent_work"]["pre_grasp"] == 20.0
