from scripts import run_act_uniform_3p5_extension_v1 as module


def test_schedule_is_uniform_3p5():
    assert module.SCHEDULE == [3.5, 3.5, 3.5, 3.5]


def test_expected_final_seeds_uses_frozen_task_bank():
    banks = {
        "tasks": {
            "pick": {"final": {"start": 123, "count": 50}},
        }
    }

    assert module.expected_final_seeds(banks, "pick") == list(range(123, 173))
