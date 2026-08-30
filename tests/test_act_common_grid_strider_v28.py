import os
import subprocess
import sys
from pathlib import Path

from act_speed_benchmark import COMMON_GRID_SPEED_VALUES
from scripts.build_common_grid_strider_panel_v28 import build
from scripts.run_act_common_grid_strider_v27 import BASELINE, DISCOVERY_V28


def test_candidate_set_is_five_task_independent_schedules():
    assert len(DISCOVERY_V28) == 5
    assert len({tuple(value) for value in DISCOVERY_V28}) == 5
    assert BASELINE == [1.5] * 4
    assert [2.0] * 4 not in DISCOVERY_V28
    for schedule in DISCOVERY_V28:
        assert len(schedule) == 4
        assert all(speed in COMMON_GRID_SPEED_VALUES for speed in schedule)
        assert max(schedule) <= 3.0


def test_panels_are_fresh_three_plus_five_latin_hypercubes():
    for task in ("pick", "tea", "insertion"):
        panel = build(task)
        assert panel["selection_uses_policy_outcomes"] is False
        assert panel["selection_uses_trajectory_or_reward"] is False
        assert panel["stage_prefix_sizes"] == [3, 8]
        assert len(panel["panel_ids"]) == len(panel["object_pose_vectors"]) == 8
        design = panel["normalized_design"]
        dimensions = 2 if task in {"pick", "tea"} else 4
        for block in (design[:3], design[3:]):
            size = len(block)
            for column in range(dimensions):
                assert sorted(int(row[column] * size) for row in block) == list(range(size))


def test_budget_is_exactly_25_per_task():
    assert len(DISCOVERY_V28) * 3 + 2 * 5 == 25


def test_workflow_requires_hash_verified_v27_cache():
    repo = Path(__file__).resolve().parents[1]
    workflow = (repo / "osmo/act_common_grid_strider_v28_l40.yaml").read_text()
    assert "--v27-root \"$V27\"" in workflow
    assert "test -f \"$V27/COMPLETE.json\"" in workflow
    runner = (repo / "scripts/run_act_common_grid_strider_v27.py").read_text()
    assert "cache_source_state_sha256" in runner
    assert "v27 cached final identity mismatch" in runner


def test_shared_entrypoints_accept_v28_from_arbitrary_cwd(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    commands = (
        [sys.executable, str(repo / "scripts/build_common_grid_strider_panel_v28.py"), "--help"],
        [sys.executable, str(repo / "scripts/run_act_common_grid_strider_v27.py"), "--help"],
        [sys.executable, str(repo / "scripts/finalize_act_common_grid_strider_v27.py"), "--help"],
    )
    for command in commands:
        completed = subprocess.run(
            command, cwd=tmp_path, env=environment, text=True, capture_output=True, check=False,
        )
        assert completed.returncode == 0, completed.stderr
