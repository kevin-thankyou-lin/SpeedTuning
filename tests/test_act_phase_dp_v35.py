import json

from scripts import run_act_phase_dp_v35 as module


def record(seed, schedule, transitions=(True, True, True, True), durations=(20, 30, 40, 50)):
    decisions = []
    step = 0
    for index, phase in enumerate(module.PHASES):
        if index > 0 and not transitions[index - 1]:
            break
        decisions.append({"phase": phase, "physics_step": step, "speed": schedule[index]})
        step += durations[index]
    success = len(decisions) == 4 and transitions[3]
    return {
        "seed": seed,
        "schedule": list(schedule),
        "success": success,
        "first_success_step": step if success else None,
        "physics_steps": step,
        "safety_violation": None,
        "physics_error": None,
        "phase_decisions": decisions,
    }


def test_orthogonal_design_is_exact_and_pairwise_balanced():
    schedules = module.orthogonal_schedules()
    module.validate_design(schedules)
    assert len(schedules) == 25
    for left in range(4):
        for right in range(left + 1, 4):
            assert len({(row[left], row[right]) for row in schedules}) == 25


def test_phase_segment_uses_next_registered_phase_and_terminal_success():
    value = record(0, [2.0] * 4, transitions=(True, True, False, False))
    assert module.phase_segment(value, 0) == {"visited": True, "progressed": True, "duration_steps": 20}
    assert module.phase_segment(value, 2) == {"visited": True, "progressed": False, "duration_steps": 40}
    assert module.phase_segment(value, 3)["visited"] is False


def test_backward_induction_picks_reliable_then_fast_actions():
    records = []
    for seed, schedule in enumerate(module.orthogonal_schedules()):
        # Every action progresses, and higher speeds take fewer steps.
        durations = tuple(int(120 / speed) for speed in schedule)
        records.append(record(seed, schedule, durations=durations))
    model = module.estimate_model(records)
    dp = module.backward_induction(model)
    assert dp["selected_schedule"] == [3.0, 3.0, 3.0, 3.0]
    assert dp["global_optimum_claimed"] is False
    assert all(model["cells"][phase][str(speed)]["assigned"] == 5 for phase in module.PHASES for speed in module.GRID)


def test_backward_induction_rejects_less_reliable_action():
    records = []
    schedules = module.orthogonal_schedules()
    for seed, schedule in enumerate(schedules):
        transitions = [True] * 4
        if schedule[2] == 3.0 and seed == next(i for i, row in enumerate(schedules) if row[2] == 3.0):
            transitions[2] = False
        durations = tuple(int(120 / speed) for speed in schedule)
        records.append(record(seed, schedule, transitions=tuple(transitions), durations=durations))
    dp = module.backward_induction(module.estimate_model(records))
    assert dp["selected_schedule"][2] != 3.0


def test_registered_banks_are_fresh_exact_and_disjoint():
    banks = json.loads((module.REPO_ROOT / "experiments/act_phase_dp_v35/BANKS.json").read_text())
    all_seeds = []
    for task in banks["tasks"].values():
        assert len(task["search"]) == 25
        assert len(task["final"]) == 50
        assert min(task["search"] + task["final"]) >= 350000000
        assert not set(task["search"]) & set(task["final"])
        all_seeds.extend(task["search"] + task["final"])
    assert len(all_seeds) == len(set(all_seeds)) == 225


def test_v34_schedules_are_frozen_on_common_grid():
    frozen = json.loads((module.REPO_ROOT / "experiments/act_phase_dp_v35/FROZEN_V34_SCHEDULES.json").read_text())
    assert set(frozen["tasks"]) == set(module.TASKS)
    for task in module.TASKS:
        assert list(module.v32.validate_schedule(frozen["tasks"][task]["schedule"]))
