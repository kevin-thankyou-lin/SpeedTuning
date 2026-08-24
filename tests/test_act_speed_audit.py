import json

import pytest

from scripts.audit_act_speed_benchmark import (
    DETECTOR_HASHES,
    accumulate_incidents,
    canonical_sha256,
    render_markdown,
    verify_identity,
)


def _identity(method, detector):
    value = {
        "schema": "act-speed-cell-identity-v1",
        "task_label": "pick",
        "method": method,
        "stage": "search",
        "seed_bank": {"seeds": [10, 11], "sha256": canonical_sha256([10, 11])},
        "detector": detector,
        "controller": {
            "cameras": ["angle", "left_wrist", "right_wrist"],
            "progress_clock": "nominal_policy_time",
            "per_physics_step_inference": True,
            "temporal_ensemble_m": 0.01,
            "physics_error_policy": "count_as_failure_and_continue",
        },
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def test_audit_identity_enforces_detector_boundary(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_identity("uniform_sweep", None)))
    verify_identity(
        path, task="pick", method="uniform_sweep", stage="search", seeds=[10, 11]
    )

    path.write_text(json.dumps(_identity("uniform_sweep", DETECTOR_HASHES)))
    with pytest.raises(RuntimeError, match="detector boundary"):
        verify_identity(
            path, task="pick", method="uniform_sweep", stage="search", seeds=[10, 11]
        )


def test_audit_identity_accepts_exact_phase_detector(tmp_path):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_identity("learned_phase_subtask", DETECTOR_HASHES)))
    verify_identity(
        path,
        task="pick",
        method="learned_phase_subtask",
        stage="search",
        seeds=[10, 11],
    )


def test_rendered_table_uses_success_only_speedup_and_evidence_paths():
    report = {
        "audit_path": "/evidence/audit.json",
        "manifests": {"base": {"sha256": "a"}, "repair": {"sha256": "b"}},
        "totals": {
            "search_rollouts": 900,
            "method_final_rollouts": 900,
            "native_rollouts": 150,
            "safety_violations": 0,
            "physics_errors": 0,
        },
        "results": [
            {
                "task": "pick",
                "method": "uniform_sweep",
                "successes": 50,
                "success_rate": 1.0,
                "method_mean": 100.0,
                "native_mean": 250.0,
                "speedup": 2.5,
                "safety_violations": 0,
                "physics_errors": 0,
                "manifest_path": "/evidence/manifest.json",
                "method_artifact_path": "/evidence/selected.json",
                "states_path": "/evidence/states",
            }
        ],
    }
    rendered = render_markdown(report)
    assert "2.500x" in rendered
    assert "/evidence/selected.json" in rendered
    assert "/evidence/states" in rendered


def test_incident_totals_include_search_final_and_native_stages():
    totals = {"safety_violations": 0, "physics_errors": 0}

    accumulate_incidents(totals, {"safety_violations": 0, "physics_errors": 1})
    accumulate_incidents(totals, {"safety_violations": 2, "physics_errors": 0})
    accumulate_incidents(totals, {"safety_violations": 0, "physics_errors": 0})

    assert totals == {"safety_violations": 2, "physics_errors": 1}
