import numpy as np

from scripts import build_tea_representative_repair_panel_v21 as panel
from scripts import run_act_strider_tea_interaction_repair_v21 as v21


def summary(successes, *, safety=0, physics=0):
    return {
        "successes": successes,
        "safety_violations": safety,
        "physics_errors": physics,
        "achieved_throughput_per_step": 1.0,
    }


def pair(incumbent, repair, **kwargs):
    return {"incumbent": summary(incumbent), "interaction_repair": summary(repair, **kwargs)}


def test_only_interaction_phase_changes():
    changed = [i for i, (old, new) in enumerate(zip(v21.INCUMBENT, v21.REPAIR)) if old != new]
    assert changed == [3]
    assert v21.INCUMBENT[3] == 2.0
    assert v21.REPAIR[3] == 1.5


def test_nested_panel_is_fresh_balanced_and_complete():
    design = panel.normalized_nested_sixteen()
    assert design.shape == (16, 2)
    assert len({tuple(row) for row in design}) == 16
    assert np.allclose(design[:4].mean(axis=0), 0.5)
    assert np.allclose(np.corrcoef(design[:4], rowvar=False), np.eye(2))
    assert np.allclose(design[:8].mean(axis=0), 0.5)
    assert np.allclose(np.corrcoef(design[:8], rowvar=False), np.eye(2))
    assert len({tuple(row) for row in design}) == 4 * 4


def test_gate_requires_strict_reliability_gain_at_sixteen():
    assert v21.gate_decision(4, pair(4, 3)) == "continue"
    assert v21.gate_decision(8, pair(8, 7)) == "continue"
    assert v21.gate_decision(16, pair(15, 16)) == "select_repair_reliability_gain"
    assert v21.gate_decision(16, pair(16, 16)) == "retain_incumbent_no_demonstrated_reliability_gain"
    assert v21.gate_decision(16, pair(16, 15)) == "retain_incumbent_repair_regression"
    assert v21.gate_decision(16, pair(16, 14)) == "reject_repair_absolute_reliability"


def test_repair_incident_rejects_at_every_stage():
    for stage in (4, 8, 16):
        assert v21.gate_decision(stage, pair(stage, stage, safety=1)) == "reject_repair_incident"
