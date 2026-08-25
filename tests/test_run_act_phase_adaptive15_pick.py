from scripts import run_act_phase_adaptive15_pick as module


def test_pick_prior_is_qualitative_and_disclosed():
    assert module.PHASE_RISK_LABELS == {
        "pre_grasp": "cautious",
        "grasp_lift": "protected",
        "transport": "open",
        "interaction": "open",
    }
    assert module.PROPOSAL_RECEIPT["labels_only"]
    assert not module.PROPOSAL_RECEIPT["independent_schedule_discovery_claim"]
