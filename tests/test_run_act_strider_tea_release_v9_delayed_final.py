import json
from pathlib import Path

from scripts import run_act_strider_tea_release_v9 as v9


ROOT = Path(__file__).resolve().parents[1]


def test_delayed_release_proposal_is_frozen_before_parent_final_outcomes():
    proposal = json.loads(
        (ROOT / "experiments/act_strider_tea_release_v9/DELAYED_RELEASE_FINAL_PROPOSAL.json").read_text()
    )
    assert proposal["schedule"] == v9.DELAYED_RELEASE
    assert proposal["parent_final_outcomes_visible"] is False
    assert proposal["final_seed_count"] == 50


def test_delayed_final_runner_records_actual_videos():
    source = (
        ROOT / "scripts/run_act_strider_tea_release_v9_delayed_final.py"
    ).read_text()
    assert "VideoFinalLedger" in source
    assert '"actual_episode_videos": True' in source
    assert '"same_gpu_controller_concurrency": False' in source
    assert 'tea.SUCCESS_CRITERION_SCHEMA = "tea-cup-center-success-v1"' in source
