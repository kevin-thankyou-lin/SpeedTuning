import pytest

from scripts.qwen_vlm_failure_attribution import (
    aggregate_attributions,
    build_prompt,
    extract_json,
    validate_attribution,
)


def record(seed, success, schedule):
    return {
        "seed": seed,
        "success": success,
        "schedule": schedule,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": schedule[0]},
            {"phase": "grasp_lift", "physics_step": 50, "speed": schedule[1]},
            {"phase": "transport", "physics_step": 100, "speed": schedule[2]},
            {"phase": "interaction", "physics_step": 150, "speed": schedule[3]},
        ],
    }


def attribution(observed, causal, confidence=0.8):
    return {
        "observed_failure_phase": observed,
        "causal_phase": causal,
        "confidence": confidence,
        "evidence": "candidate visibly diverges from the matched reference",
    }


def test_prompt_explicitly_allows_same_or_earlier_cause():
    prompt = build_prompt(
        "tea",
        record(7, True, [2, 2, 2, 2]),
        record(7, False, [2.5, 2.5, 2.5, 2.5]),
    )
    assert "may equal the observed phase or be earlier" in prompt
    assert "must never be later" in prompt
    assert "tea-bag center inside" in prompt
    assert "learned detector" in prompt


def test_prompt_requires_same_seed():
    with pytest.raises(ValueError, match="same initial-state seed"):
        build_prompt(
            "pick",
            record(1, True, [2, 2, 2, 2]),
            record(2, False, [2.5, 2.5, 2.5, 2.5]),
        )


def test_extracts_fenced_json_and_validates_causal_order():
    item = attribution("interaction", "grasp_lift")
    parsed = extract_json("```json\n" + __import__("json").dumps(item) + "\n```")
    assert validate_attribution(parsed)["causal_phase"] == "grasp_lift"
    with pytest.raises(ValueError, match="cannot occur after"):
        validate_attribution(attribution("grasp_lift", "interaction"))


def test_aggregation_uses_majority_and_ties_break_earlier():
    result = aggregate_attributions(
        [
            attribution("interaction", "transport"),
            attribution("interaction", "grasp_lift"),
        ],
        "interaction",
    )
    assert result["selected_phase"] == "grasp_lift"
    assert result["counts"] == {"grasp_lift": 1, "transport": 1}


def test_empty_attribution_fails_closed_to_registered_semantic_phase():
    result = aggregate_attributions([], "interaction")
    assert result["selected_phase"] == "interaction"
    assert result["method"] == "vlm_no_matched_pair_semantic_fallback"
