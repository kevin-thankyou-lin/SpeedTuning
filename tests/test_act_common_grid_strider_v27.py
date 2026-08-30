import os
import subprocess
import sys
from pathlib import Path

from act_speed_benchmark import COMMON_GRID_SPEED_VALUES
from scripts.build_common_grid_strider_panel_v27 import build
from scripts.run_act_common_grid_strider_v27 import BASELINE, DISCOVERY


def test_candidate_set_is_task_independent_capped_common_grid():
    assert len(DISCOVERY) == 6
    assert len({tuple(value) for value in DISCOVERY}) == 6
    assert BASELINE == [1.5] * 4
    for schedule in DISCOVERY:
        assert len(schedule) == 4
        assert all(speed in COMMON_GRID_SPEED_VALUES for speed in schedule)
        assert max(schedule) <= 3.0


def test_panels_are_fresh_stratified_and_outcome_blind():
    for task in ("pick", "tea", "insertion"):
        panel = build(task)
        assert panel["selection_uses_policy_outcomes"] is False
        assert panel["selection_uses_trajectory_or_reward"] is False
        assert len(panel["panel_ids"]) == len(panel["object_pose_vectors"]) == 8
        design = panel["normalized_design"]
        dimensions = 2 if task in {"pick", "tea"} else 4
        for block in (design[:4], design[4:]):
            for column in range(dimensions):
                assert sorted(int(row[column] * 4) for row in block) == [0, 1, 2, 3]


def test_budget_is_exactly_32_per_task():
    assert len(DISCOVERY) * 4 + 2 * 4 == 32


def test_entrypoints_import_from_arbitrary_cwd(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    for name in (
        "scripts/build_common_grid_strider_panel_v27.py",
        "scripts/run_act_common_grid_strider_v27.py",
        "scripts/finalize_act_common_grid_strider_v27.py",
    ):
        completed = subprocess.run(
            [sys.executable, str(repo / name), "--help"], cwd=tmp_path,
            env=environment, text=True, capture_output=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
