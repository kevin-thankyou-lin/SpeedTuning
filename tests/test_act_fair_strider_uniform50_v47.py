import json
import subprocess
import sys
from pathlib import Path

from scripts import run_act_fair_strider_uniform50_v47 as v47


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "act_fair_strider_uniform50_v47"


def report(schedule, successes, throughput):
    return {
        "controller": v47.static_controller(schedule),
        "controller_sha256": v47.static_controller(schedule)["controller_sha256"],
        "summary": {
            "episodes": 10,
            "successes": successes,
            "achieved_throughput_per_step": throughput,
            "physics_errors": 0,
            "safety_violations": 0,
        },
    }


def test_contract_is_symmetric_and_history_blind():
    text = (EXPERIMENT / "CONTRACT.md").read_text()
    normalized = " ".join(text.split())
    assert "every arm consumes exactly 50 search rollouts per task" in normalized
    assert "Neither arm may read a historical speed schedule" in normalized
    assert "same 100 untouched randomized resets" in normalized
    assert v47.SEARCH_BUDGET == 50
    assert v47.FINAL_EPISODES == 100


def test_seed_banks_expand_to_disjoint_equal_budgets():
    banks = json.loads((EXPERIMENT / "BANKS.json").read_text())
    v47.validate_banks(banks)
    all_seeds = []
    for task in v47.TASKS:
        spec = v47.expand_task_banks(banks["tasks"][task])
        for arm in v47.SEARCH_ARMS:
            assert sum(
                len(value["diagnostic"]) + 2 * len(value["paired"])
                for value in spec[arm]["rounds"]
            ) == 50
        task_unique = []
        for arm in v47.SEARCH_ARMS:
            for value in spec[arm]["rounds"]:
                task_unique += value["diagnostic"] + value["paired"]
        task_unique += spec["final"]
        assert len(task_unique) == len(set(task_unique))
        all_seeds += task_unique
    assert len(all_seeds) == len(set(all_seeds))


def test_round_selection_requires_reliability_then_throughput():
    incumbent = report([2, 2, 2, 2], 10, 0.005)
    fast = report([2.5, 2.5, 2.5, 2.5], 10, 0.0053)
    winner, qualified, status = v47.choose_round_winner(
        incumbent, fast, {"challenger_throughput_ratio": 1.06}
    )
    assert qualified and status == "challenger_promoted"
    assert winner["schedule"] == [2.5] * 4

    unreliable = report([3, 3, 3, 3], 8, 0.009)
    winner, qualified, status = v47.choose_round_winner(
        incumbent, unreliable, {"challenger_throughput_ratio": 1.8}
    )
    assert qualified and status == "incumbent_retained"
    assert winner["schedule"] == [2.0] * 4


def test_uniform_round_two_promotes_only_after_round_one_promotion():
    anchor = v47.static_controller([2, 2, 2, 2])
    first, _ = v47.uniform_proposal(0, anchor, None)
    assert first["schedule"] == [2.5] * 4
    promoted, _ = v47.uniform_proposal(
        1,
        first,
        {"winner_qualified": True, "selection_status": "challenger_promoted"},
    )
    assert promoted["schedule"] == [3.0] * 4
    backoff, _ = v47.uniform_proposal(
        1,
        anchor,
        {"winner_qualified": True, "selection_status": "incumbent_retained"},
    )
    assert backoff["schedule"] == [1.5] * 4


def test_strider_changes_one_phase_using_current_records_only():
    incumbent = v47.static_controller([2, 2, 2, 2])
    record = {
        "success": True,
        "safety_violation": None,
        "physics_steps": 120,
        "first_success_step": 120,
        "phase_decisions": [
            {"phase": "pre_grasp", "physics_step": 0, "speed": 2.0},
            {"phase": "grasp_lift", "physics_step": 10, "speed": 2.0},
            {"phase": "transport", "physics_step": 20, "speed": 2.0},
            {"phase": "interaction", "physics_step": 110, "speed": 2.0},
        ],
    }
    challenger, receipt = v47.strider_proposal(
        incumbent, [record], {incumbent["controller_sha256"]}, None
    )
    changed = [
        index
        for index, (left, right) in enumerate(
            zip(incumbent["schedule"], challenger["schedule"])
        )
        if left != right
    ]
    assert changed == [v47.PHASES.index("transport")]
    assert challenger["schedule"][changed[0]] == 2.5
    assert receipt["operation"] == "one_rung_current_run_bang_for_buck_promotion"


def test_final_methods_are_only_native_and_two_search_outputs():
    assert v47.FINAL_METHODS == (
        "native_1x",
        "uniform_selected",
        "strider_selected",
    )


def test_finalizer_script_resolves_the_current_worktree():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "finalize_act_fair_strider_uniform50_v47.py"),
            "--help",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Seal the V47 fair STRIDER-versus-uniform aggregate" in completed.stdout
